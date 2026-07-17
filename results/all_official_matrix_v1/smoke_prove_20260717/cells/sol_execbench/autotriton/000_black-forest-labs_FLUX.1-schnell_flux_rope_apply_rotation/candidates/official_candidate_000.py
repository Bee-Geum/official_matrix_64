import torch
import triton
import triton.language as tl

@triton.jit
def flux_rope_apply_rotation_kernel(
    query_or_key_ptr, freqs_cos_ptr, freqs_sin_ptr, output_ptr,
    B, S, H, D, N,
    stride_b_q, stride_s_q, stride_h_q, stride_d_q,
    stride_s_c, stride_d_c,
    stride_s_s, stride_d_s,
    stride_b_out, stride_s_out, stride_h_out, stride_d_out,
    total_pairs,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= total_pairs:
        return
    
    n = pid % N
    pid_rest = pid // N
    h = pid_rest % H
    pid_rest = pid_rest // H
    s = pid_rest % S
    b = pid_rest // S
    
    off_b = b * stride_b_q
    off_s = s * stride_s_q
    off_h = h * stride_h_q
    off_d_base = 2 * n
    
    x1_ptr = query_or_key_ptr + off_b + off_s + off_h + off_d_base
    x2_ptr = query_or_key_ptr + off_b + off_s + off_h + off_d_base + 1
    x1 = tl.load(x1_ptr)
    x2 = tl.load(x2_ptr)
    
    off_s_c = s * stride_s_c
    off_d_base_c = 2 * n
    cos1_ptr = freqs_cos_ptr + off_s_c + off_d_base_c
    cos2_ptr = freqs_cos_ptr + off_s_c + off_d_base_c + 1
    sin1_ptr = freqs_sin_ptr + off_s_c + off_d_base_c
    sin2_ptr = freqs_sin_ptr + off_s_c + off_d_base_c + 1
    
    cos1 = tl.load(cos1_ptr)
    cos2 = tl.load(cos2_ptr)
    sin1 = tl.load(sin1_ptr)
    sin2 = tl.load(sin2_ptr)
    
    out1 = x1 * cos1 - x2 * sin1
    out2 = x1 * sin2 + x2 * cos2
    
    out1_ptr = output_ptr + off_b + off_s + off_h + off_d_base
    out2_ptr = output_ptr + off_b + off_s + off_h + off_d_base + 1
    tl.store(out1_ptr, out1)
    tl.store(out2_ptr, out2)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_or_key: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        assert query_or_key.shape[0:3] == freqs_cos.shape[0:2], "Batch and seq length must match for frequencies"
        assert query_or_key.shape[3] == freqs_cos.shape[1], "Head dim must match for frequencies"
        assert query_or_key.shape[3] % 2 == 0, "Head dimension must be even"
        
        B, S, H, D = query_or_key.shape
        N = D // 2
        
        output = torch.empty_like(query_or_key)
        
        if output.numel() == 0:
            return output
        
        stride_b_q, stride_s_q, stride_h_q, stride_d_q = query_or_key.stride()
        stride_s_c, stride_d_c = freqs_cos.stride()
        stride_s_s, stride_d_s = freqs_sin.stride()
        stride_b_out, stride_s_out, stride_h_out, stride_d_out = output.stride()
        
        total_pairs = B * S * H * N
        grid = (total_pairs,)
        
        flux_rope_apply_rotation_kernel[grid](
            query_or_key, freqs_cos, freqs_sin, output,
            B, S, H, D, N,
            stride_b_q, stride_s_q, stride_h_q, stride_d_q,
            stride_s_c, stride_d_c,
            stride_s_s, stride_d_s,
            stride_b_out, stride_s_out, stride_h_out, stride_d_out,
            total_pairs,
            BLOCK_SIZE=1
        )
        
        return output
