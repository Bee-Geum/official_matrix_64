import torch
import triton
import triton.language as tl

@triton.jit
def flux_rope_apply_rotation_kernel(
    query_or_key_ptr,
    freqs_cos_ptr,
    freqs_sin_ptr,
    output_ptr,
    batch_size,
    seq_len,
    num_attention_heads,
    attention_head_dim,
    half_head_dim,
    stride_qk_b, stride_qk_s, stride_qk_h, stride_qk_d,
    stride_fc_s, stride_fc_d,
    stride_out_b, stride_out_s, stride_out_h, stride_out_d,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for FLUX RoPE rotation.
    
    Each program processes a block of head_dim elements for a specific
    (batch, seq_pos, head) combination.
    """
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_s = tl.program_id(1)  # sequence position index
    pid_h = tl.program_id(2)  # head index
    
    # Check bounds
    if pid_b >= batch_size or pid_s >= seq_len or pid_h >= num_attention_heads:
        return
    
    # Create range for head_dim dimension
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    mask = d_offsets < attention_head_dim
    
    # Calculate base pointers for this (batch, seq_pos, head)
    qk_base = (pid_b * stride_qk_b + 
               pid_s * stride_qk_s + 
               pid_h * stride_qk_h)
    
    fc_base = pid_s * stride_fc_s
    
    out_base = (pid_b * stride_out_b + 
                pid_s * stride_out_s + 
                pid_h * stride_out_h)
    
    # Load query/key block
    qk_ptrs = qk_base + d_offsets * stride_qk_d
    x = tl.load(query_or_key_ptr + qk_ptrs, mask=mask, other=0.0)
    
    # Load cos and sin frequencies
    fc_ptrs = fc_base + d_offsets * stride_fc_d
    cos_val = tl.load(freqs_cos_ptr + fc_ptrs, mask=mask, other=0.0)
    sin_val = tl.load(freqs_sin_ptr + fc_ptrs, mask=mask, other=0.0)
    
    # Split into even and odd indices for complex rotation
    # Even indices (0, 2, 4, ...): x1
    # Odd indices (1, 3, 5, ...): x2
    x1 = tl.where(d_offsets % 2 == 0, x, 0.0)
    x2 = tl.where(d_offsets % 2 == 1, x, 0.0)
    
    cos1 = tl.where(d_offsets % 2 == 0, cos_val, 0.0)
    cos2 = tl.where(d_offsets % 2 == 1, cos_val, 0.0)
    sin1 = tl.where(d_offsets % 2 == 0, sin_val, 0.0)
    sin2 = tl.where(d_offsets % 2 == 1, sin_val, 0.0)
    
    # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    # For even indices: out = x1*cos1 - x2*sin1
    # For odd indices: out = x1*sin2 + x2*cos2
    out_even = x1 * cos1 - x2 * sin1
    out_odd = x1 * sin2 + x2 * cos2
    
    # Combine results
    out = tl.where(d_offsets % 2 == 0, out_even, out_odd)
    
    # Store result
    out_ptrs = out_base + d_offsets * stride_out_d
    tl.store(output_ptr + out_ptrs, out, mask=mask)

def run(
    query_or_key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to query or key tensor using Triton kernel.
    
    Args:
        query_or_key: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        freqs_cos: Cosine frequencies of shape (seq_len, head_dim)
        freqs_sin: Sine frequencies of shape (seq_len, head_dim)
        
    Returns:
        Rotated tensor of same shape as input
    """
    # Ensure tensors are on CUDA and in bfloat16
    assert query_or_key.is_cuda, "Input must be on CUDA"
    assert freqs_cos.is_cuda, "Frequencies must be on CUDA"
    assert freqs_sin.is_cuda, "Frequencies must be on CUDA"
    
    # Get dimensions
    batch_size, seq_len, num_attention_heads, attention_head_dim = query_or_key.shape
    half_head_dim = attention_head_dim // 2
    
    # Create output tensor
    output = torch.empty_like(query_or_key)
    
    # Define grid
    grid = (batch_size, seq_len, num_attention_heads)
    
    # Get strides
    stride_qk_b, stride_qk_s, stride_qk_h, stride_qk_d = query_or_key.stride()
    stride_fc_s, stride_fc_d = freqs_cos.stride()
    stride_out_b, stride_out_s, stride_out_h, stride_out_d = output.stride()
    
    # Launch kernel
    # Use block size that's a power of 2 and >= attention_head_dim
    BLOCK_SIZE_D = triton.next_power_of_2(attention_head_dim)
    
    flux_rope_apply_rotation_kernel[grid](
        query_or_key,
        freqs_cos,
        freqs_sin,
        output,
        batch_size,
        seq_len,
        num_attention_heads,
        attention_head_dim,
        half_head_dim,
        stride_qk_b, stride_qk_s, stride_qk_h, stride_qk_d,
        stride_fc_s, stride_fc_d,
        stride_out_b, stride_out_s, stride_out_h, stride_out_d,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return output

class ModelNew(torch.nn.Module):
    """Optimized model using Triton kernel for RoPE rotation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, query_or_key, freqs_cos, freqs_sin):
        return run(query_or_key, freqs_cos, freqs_sin)

# Test function to verify correctness
def test_correctness():
    """Test that the optimized implementation matches the reference."""
    import torch
    
    # Reference implementation
    @torch.no_grad()
    def reference_run(query_or_key, freqs_cos, freqs_sin):
        x_shape = query_or_key.shape
        x_reshaped = query_or_key.float().reshape(
            x_shape[0], x_shape[1], x_shape[2], -1, 2
        )
        freqs_cos_expanded = freqs_cos.unsqueeze(0).unsqueeze(2).float()
        freqs_sin_expanded = freqs_sin.unsqueeze(0).unsqueeze(2).float()
        freqs_cos_reshaped = freqs_cos_expanded.reshape(
            freqs_cos_expanded.shape[0], freqs_cos_expanded.shape[1], freqs_cos_expanded.shape[2], -1, 2
        )
        freqs_sin_reshaped = freqs_sin_expanded.reshape(
            freqs_sin_expanded.shape[0], freqs_sin_expanded.shape[1], freqs_sin_expanded.shape[2], -1, 2
        )
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        cos1 = freqs_cos_reshaped[..., 0]
        cos2 = freqs_cos_reshaped[..., 1]
        sin1 = freqs_sin_reshaped[..., 0]
        sin2 = freqs_sin_reshaped[..., 1]
        out1 = x1 * cos1 - x2 * sin1
        out2 = x1 * sin2 + x2 * cos2
        output = torch.stack([out1, out2], dim=-1)
        output = output.reshape(x_shape)
        return output.to(torch.bfloat16)
    
    # Create test data
    batch_size = 1
    seq_len = 128
    num_attention_heads = 24
    attention_head_dim = 128
    
    query_or_key = torch.randn(
        batch_size, seq_len, num_attention_heads, attention_head_dim,
        dtype=torch.bfloat16, device='cuda'
    )
    freqs_cos = torch.randn(seq_len, attention_head_dim, dtype=torch.bfloat16, device='cuda')
    freqs_sin = torch.randn(seq_len, attention_head_dim, dtype=torch.bfloat16, device='cuda')
    
    # Run both implementations
    ref_output = reference_run(query_or_key, freqs_cos, freqs_sin)
    opt_output = run(query_or_key, freqs_cos, freqs_sin)
    
    # Check correctness
    diff = torch.abs(ref_output - opt_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    # Check if within tolerance
    if max_diff < 1e-2:
        print("✓ Output matches reference within tolerance")
    else:
        print("✗ Output differs from reference")
    
    return max_diff < 1e-2

if __name__ == "__main__":
    # Run test if executed directly
    test_correctness()
