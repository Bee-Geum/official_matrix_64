import torch
import triton
import triton.language as tl

@triton.jit
def rope_apply_kernel(
    qk_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    B,
    S,
    H,
    D,
    stride_b_qk, stride_s_qk, stride_h_qk, stride_d_qk,
    stride_s_cos, stride_d_cos,
    stride_s_sin, stride_d_sin,
    stride_b_out, stride_s_out, stride_h_out, stride_d_out,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Offsets for this block in the head dimension
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    mask = d_offsets < D
    
    # Base pointers for this batch, sequence position, and head
    qk_base = (pid_b * stride_b_qk + 
               pid_s * stride_s_qk + 
               pid_h * stride_h_qk)
    cos_base = pid_s * stride_s_cos
    sin_base = pid_s * stride_s_sin
    out_base = (pid_b * stride_b_out + 
                pid_s * stride_s_out + 
                pid_h * stride_h_out)
    
    # Load query/key block
    qk_vals = tl.load(qk_ptr + qk_base + d_offsets * stride_d_qk, mask=mask)
    
    # Load cos/sin frequencies
    cos_vals = tl.load(cos_ptr + cos_base + d_offsets * stride_d_cos, mask=mask)
    sin_vals = tl.load(sin_ptr + sin_base + d_offsets * stride_d_sin, mask=mask)
    
    # Split into even and odd indices for complex rotation
    # Even indices (x1) and odd indices (x2)
    x1 = qk_vals[0::2]
    x2 = qk_vals[1::2]
    c1 = cos_vals[0::2]
    c2 = cos_vals[1::2]
    s1 = sin_vals[0::2]
    s2 = sin_vals[1::2]
    
    # Apply rotation: (x1*c1 - x2*s1, x1*s2 + x2*c2)
    out1 = x1 * c1 - x2 * s1
    out2 = x1 * s2 + x2 * c2
    
    # Interleave results back
    out_vals = tl.zeros_like(qk_vals)
    out_vals = out_vals.to(tl.float32)
    out_vals = out_vals.to(tl.bfloat16)
    
    # We need to write back interleaved
    # Create output array
    out_arr = tl.zeros((BLOCK_SIZE_D,), dtype=tl.bfloat16)
    out_arr = out_arr.to(tl.float32)
    
    # Store even indices
    even_mask = d_offsets < D
    even_mask = even_mask & (d_offsets % 2 == 0)
    tl.store(out_ptr + out_base + d_offsets * stride_d_out, out1, mask=even_mask)
    
    # Store odd indices
    odd_mask = d_offsets < D
    odd_mask = odd_mask & (d_offsets % 2 == 1)
    tl.store(out_ptr + out_base + d_offsets * stride_d_out, out2, mask=odd_mask)

def run(
    query_or_key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    B, S, H, D = query_or_key.shape
    
    # Ensure tensors are on CUDA and contiguous
    assert query_or_key.is_cuda, "Input must be on CUDA"
    assert freqs_cos.is_cuda, "Frequencies must be on CUDA"
    assert freqs_sin.is_cuda, "Frequencies must be on CUDA"
    
    # Create output tensor
    output = torch.empty_like(query_or_key)
    
    # Define grid
    grid = (B, S, H)
    
    # Get strides
    stride_b_qk, stride_s_qk, stride_h_qk, stride_d_qk = query_or_key.stride()
    stride_s_cos, stride_d_cos = freqs_cos.stride()
    stride_s_sin, stride_d_sin = freqs_sin.stride()
    stride_b_out, stride_s_out, stride_h_out, stride_d_out = output.stride()
    
    # Launch kernel
    # Use block size that's a power of 2 and >= D
    BLOCK_SIZE_D = triton.next_power_of_2(D)
    if BLOCK_SIZE_D < 32:
        BLOCK_SIZE_D = 32
    if BLOCK_SIZE_D > 1024:
        BLOCK_SIZE_D = 1024
    
    rope_apply_kernel[grid](
        query_or_key,
        freqs_cos,
        freqs_sin,
        output,
        B, S, H, D,
        stride_b_qk, stride_s_qk, stride_h_qk, stride_d_qk,
        stride_s_cos, stride_d_cos,
        stride_s_sin, stride_d_sin,
        stride_b_out, stride_s_out, stride_h_out, stride_d_out,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return output

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query_or_key, freqs_cos, freqs_sin):
        return run(query_or_key, freqs_cos, freqs_sin)