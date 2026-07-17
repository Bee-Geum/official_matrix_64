import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: apply rotary position embedding on (B, S, H, D) with freqs (S, D)
# Computes, for each (b, s, h):
#   out[:, 2i  ] = x[:, 2i  ]*cos[:, 2i  ] - x[:, 2i+1]*sin[:, 2i  ] + x[:, 2i-1]*sin[:, 2i  ]
#                 + x[:, 2i  ]*cos[:, 2i  ] + x[:, 2i+1]*sin[:, 2i  ] + x[:, 2i-1]*cos[:, 2i  ]
#   out[:, 2i+1] = x[:, 2i  ]*sin[:, 2i+1] - x[:, 2i+1]*cos[:, 2i+1] + x[:, 2i+2]*sin[:, 2i+1]
#                 + x[:, 2i  ]*cos[:, 2i+1] + x[:, 2i+1]*sin[:, 2i+1] + x[:, 2i+2]*cos[:, 2i+1]
# Simplified:
#   out1 = x1*cos1 - x2*sin1
#   out2 = x1*sin2 + x2*cos2
# where (x1,x2) are paired features and (cos1,cos2,sin1,sin2) are freqs paired features.
@triton.jit
def _rope_apply_kernel(
    x_ptr,            # *bf16, shape (B, S, H, D)
    cos_ptr,          # *bf16, shape (S, D)
    sin_ptr,          # *bf16, shape (S, D)
    out_ptr,          # *bf16, shape (B, S, H, D)
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    cos_stride_s: tl.constexpr,
    cos_stride_d: tl.constexpr,
    sin_stride_s: tl.constexpr,
    sin_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_s: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    # vector of d indices
    d = tl.arange(0, BLOCK_D)
    mask = d < D

    # base offsets
    x_base = pid_b * stride_b + pid_s * stride_s + pid_h * stride_h
    out_base = pid_b * out_stride_b + pid_s * out_stride_s + pid_h * out_stride_h

    # Load x[d], cos[d], sin[d]
    x = tl.load(x_ptr + x_base + d * stride_d, mask=mask, other=0).to(tl.float32)
    cos = tl.load(cos_ptr + pid_s * cos_stride_s + d * cos_stride_d, mask=mask, other=0).to(tl.float32)
    sin = tl.load(sin_ptr + pid_s * sin_stride_s + d * sin_stride_d, mask=mask, other=0).to(tl.float32)

    # Pair processing without reshape: treat d as 2*idx + parity
    # Even lanes use (j = 2*idx): need x[j+1] = x[d+1]; Odd lanes use (j = 2*idx+1): need x[j-1] = x[d-1]
    is_even = (d % 2) == 0
    d_plus1 = d + 1
    d_minus1 = d - 1
    # mask for +/-1 shifts must stay within [0, D)
    mask_plus1 = mask & is_even & (d_plus1 < D)
    mask_minus1 = mask & (~is_even) & (d_minus1 >= 0)

    x_plus1 = tl.load(x_ptr + x_base + d_plus1 * stride_d, mask=mask_plus1, other=0).to(tl.float32)
    x_minus1 = tl.load(x_ptr + x_base + d_minus1 * stride_d, mask=mask_minus1, other=0).to(tl.float32)

    # Select shift value per lane: even -> x[d+1], odd -> x[d-1]
    x_shift = tl.where(is_even & (d_plus1 < D), x_plus1, tl.where((~is_even) & (d_minus1 >= 0), x_minus1, x))

    # Compute outputs for pairs
    # out1 at positions d = 2*idx     : result for feature 2*idx
    # out2 at positions d = 2*idx + 1: result for feature 2*idx + 1
    # But we are vectorizing over d; so for lane d:
    # If d is even: out = x*cos - x_shift*sin
    # If d is odd:  out = x*sin + x_shift*cos
    out_even = x * cos - x_shift * sin       # d even
    out_odd  = x * sin + x_shift * cos       # d odd
    out = tl.where(is_even, out_even, out_odd)

    # Store as bf16
    tl.store(out_ptr + out_base + d * out_stride_d, out.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available. Please install triton to use this implementation.")

    @torch.no_grad()
    def run(
        self,
        query_or_key: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings using Triton.

        Args:
            query_or_key: (B, S, H, D) bfloat16 tensor
            freqs_cos: (S, D) bfloat16 tensor
            freqs_sin: (S, D) bfloat16 tensor

        Returns:
            output: (B, S, H, D) bfloat16 tensor
        """
        assert query_or_key.is_cuda, "Input must be on CUDA device for Triton kernel."
        assert freqs_cos.is_cuda and freqs_sin.is_cuda, "Freqs must be on CUDA device."
        assert query_or_key.dtype == torch.bfloat16, "Expected bfloat16 input."
        assert freqs_cos.dtype == torch.bfloat16 and freqs_sin.dtype == torch.bfloat16, "Expected bfloat16 freqs."

        B, S, H, D = query_or_key.shape
        assert D % 2 == 0, f"Last dim D={D} must be even for pair-wise rotation."
        assert freqs_cos.shape == (S, D) and freqs_sin.shape == (S, D), "Freqs shape must be (S, D)."

        # Ensure contiguous
        x = query_or_key.contiguous()
        cos = freqs_cos.contiguous()
        sin = freqs_sin.contiguous()

        out = torch.empty_like(x)

        # Strides in elements
        stride_b, stride_s, stride_h, stride_d = x.stride()
        cos_stride_s, cos_stride_d = cos.stride()
        sin_stride_s, sin_stride_d = sin.stride()
        out_stride_b, out_stride_s, out_stride_h, out_stride_d = out.stride()

        BLOCK_D = 128  # matches D=128; kernel supports any D<=BLOCK_D with mask

        grid = (B, S, H)
        _rope_apply_kernel[grid](
            x, cos, sin, out,
            B, S, H, D,
            stride_b, stride_s, stride_h, stride_d,
            cos_stride_s, cos_stride_d,
            sin_stride_s, sin_stride_d,
            out_stride_b, out_stride_s, out_stride_h, out_stride_d,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )

        return out
