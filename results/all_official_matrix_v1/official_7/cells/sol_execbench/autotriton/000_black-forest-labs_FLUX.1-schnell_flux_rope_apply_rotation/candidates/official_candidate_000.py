import torch
import triton
import triton.language as tl

@torch.no_grad()
def run(
    query_or_key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to query or key tensor using a Triton kernel.
    
    The rotation is applied using the formula:
    - Split features into pairs (x1, x2)
    - Rotate: (x1 * cos1 - x2 * sin1, x1 * sin2 + x2 * cos2)
    
    Args:
        query_or_key: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        freqs_cos: Cosine frequencies of shape (seq_len, head_dim)
        freqs_sin: Sine frequencies of shape (seq_len, head_dim)
        
    Returns:
        Rotated tensor of same shape as input
    """
    assert query_or_key.is_cuda, "Input tensor must be on CUDA device"
    assert freqs_cos.is_cuda and freqs_sin.is_cuda, "Frequency tensors must be on CUDA device"
    assert query_or_key.shape[-1] % 2 == 0, "Head dimension must be even"

    B, S, H, D = query_or_key.shape
    total_pairs = D // 2
    grid = (B * S * H * total_pairs,)

    # Compute strides
    stride_qb, stride_qs, stride_qh, stride_qd = query_or_key.stride()
    stride_fs, stride_fd = freqs_cos.stride()
    stride_ks, stride_kd = freqs_sin.stride()
    stride_ob, stride_os, stride_oh, stride_od = query_or_key.stride()

    # Launch kernel
    _rope_apply_rotation_kernel[grid](
        query_or_key, freqs_cos, freqs_sin, query_or_key,
        B, S, H, D,
        stride_qb, stride_qs, stride_qh, stride_qd,
        stride_fs, stride_fd,
        stride_ks, stride_kd,
        stride_ob, stride_os, stride_oh, stride_od,
        D,
        total_pairs,
    )
    return query_or_key

@triton.jit
def _rope_apply_rotation_kernel(
    query_or_key_ptr, freqs_cos_ptr, freqs_sin_ptr, output_ptr,
    batch_size, seq_len, num_attention_heads, attention_head_dim,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_fs, stride_fd,
    stride_ks, stride_kd,
    stride_ob, stride_os, stride_oh, stride_od,
    D: tl.constexpr,
    total_pairs: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elems_per_batch_seq_head = num_attention_heads * total_pairs
    total_elems_per_batch = seq_len * total_elems_per_batch_seq_head

    pair_idx = pid % total_pairs
    pid_rest = pid // total_pairs
    head_idx = pid_rest % num_attention_heads
    pid_rest = pid_rest // num_attention_heads
    seq_idx = pid_rest % seq_len
    batch_idx = pid_rest // seq_len

    base_q = batch_idx * stride_qb + seq_idx * stride_qs + head_idx * stride_qh
    base_f = seq_idx * stride_fs

    x1_ptr = query_or_key_ptr + base_q + pair_idx * 2
    x2_ptr = query_or_key_ptr + base_q + pair_idx * 2 + 1
    x1 = tl.load(x1_ptr)
    x2 = tl.load(x2_ptr)

    f_cos1_ptr = freqs_cos_ptr + base_f + pair_idx * 2
    f_sin1_ptr = freqs_sin_ptr + base_f + pair_idx * 2
    f_cos2_ptr = freqs_cos_ptr + base_f + pair_idx * 2 + 1
    f_sin2_ptr = freqs_sin_ptr + base_f + pair_idx * 2 + 1

    cos1 = tl.load(f_cos1_ptr)
    sin1 = tl.load(f_sin1_ptr)
    cos2 = tl.load(f_cos2_ptr)
    sin2 = tl.load(f_sin2_ptr)

    out1 = x1 * cos1 - x2 * sin1
    out2 = x1 * sin2 + x2 * cos2

    out1_ptr = output_ptr + base_q + pair_idx * 2
    out2_ptr = output_ptr + base_q + pair_idx * 2 + 1
    tl.store(out1_ptr, out1)
    tl.store(out2_ptr, out2)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query_or_key, freqs_cos, freqs_sin):
        return run(query_or_key, freqs_cos, freqs_sin)
