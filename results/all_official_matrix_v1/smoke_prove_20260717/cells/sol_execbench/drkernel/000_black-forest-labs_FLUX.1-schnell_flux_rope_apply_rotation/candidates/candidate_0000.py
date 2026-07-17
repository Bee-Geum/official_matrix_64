import math
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Kernel: apply RoPE over last dim D for each (b, s, h) row
# X: (B, S, H, D) contiguous
# Cos/Sin: (S, D) contiguous
# Out: (B, S, H, D) contiguous
@triton.jit
def _rope_apply_kernel(
    X, Cos, Sin, Out,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    stride_b: tl.constexpr, stride_s: tl.constexpr, stride_h: tl.constexpr, stride_d: tl.constexpr,
    cos_stride_s: tl.constexpr, cos_stride_d: tl.constexpr,
    sin_stride_s: tl.constexpr, sin_stride_d: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # map program id to (b, s, h)
    SH = H * S
    b = pid // SH
    rem = pid % SH
    s = rem // H
    h = rem % H

    # base offset for this (b, s, h) row in X/Out
    base = b * stride_b + s * stride_s + h * stride_h

    # vector of d in [0, D)
    d = tl.arange(0, D)
    # compute i = d // 2, parity = d & 1
    i = d // 2
    parity = d & 1  # 0 or 1

    # pointers for X and Out
    x_ptrs = X + base + d * stride_d
    out_ptrs = Out + base + d * stride_d

    # load x
    x = tl.load(x_ptrs)

    # load cos[i], sin[i]
    cos_ptrs = Cos + s * cos_stride_s + i * cos_stride_d
    sin_ptrs = Sin + s * sin_stride_s + i * sin_stride_d
    cos = tl.load(cos_ptrs)
    sin = tl.load(sin_ptrs)

    # compute out = x * cos - x * sin * parity
    # parity is 0/1, so this is branchless
    out = x * cos - x * sin * parity

    # store
    tl.store(out_ptrs, out)


class ModelNew:
    @staticmethod
    def run(query_or_key: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding using Triton.

        Args:
            query_or_key: (B, S, H, D) bfloat16
            freqs_cos: (S, D) bfloat16
            freqs_sin: (S, D) bfloat16

        Returns:
            out: (B, S, H, D) bfloat16
        """
        assert query_or_key.dtype == torch.bfloat16, "Expected bfloat16 input"
        assert freqs_cos.dtype == torch.bfloat16 and freqs_sin.dtype == torch.bfloat16, "Expected bfloat16 freqs"
        assert query_or_key.is_contiguous(), "Input must be contiguous"
        assert freqs_cos.is_contiguous() and freqs_sin.is_contiguous(), "Freqs must be contiguous"

        B, S, H, D = query_or_key.shape
        # Basic checks
        assert D == freqs_cos.shape[1] == freqs_sin.shape[1], "Mismatch in head dim"
        assert S == freqs_cos.shape[0] == freqs_sin.shape[0], "Mismatch in seq len"

        # Allocate output
        out = torch.empty_like(query_or_key)

        # Strides (in elements)
        sb, ss, sh, sd = query_or_key.stride()
        cos_ss, cos_sd = freqs_cos.stride()
        sin_ss, sin_sd = freqs_sin.stride()

        G = B * S * H
        # Choose num_warps
        num_warps = 4

        _rope_apply_kernel[
            G
        ](
            query_or_key, freqs_cos, freqs_sin, out,
            B, S, H, D,
            sb, ss, sh, sd,
            cos_ss, cos_sd,
            sin_ss, sin_sd,
            num_warps=num_warps,
        )

        return out

    # Fallback vectorized PyTorch implementation (no reshapes, no upcast)
    @staticmethod
    def _fallback(query_or_key: torch.Tensor,
                  freqs_cos: torch.Tensor,
                  freqs_sin: torch.Tensor) -> torch.Tensor:
        B, S, H, D = query_or_key.shape
        device = query_or_key.device

        # View as (B, S, H, D) without copy
        x = query_or_key
        # Compute i = d//2 and parity = d%2
        i = torch.arange(D, device=device, dtype=torch.long) // 2
        parity = torch.arange(D, device=device, dtype=torch.long) % 2

        # Gather cos[i], sin[i]: shapes (D,)
        cos_d = freqs_cos[:, i].reshape(S * D)          # (S,D) -> (SD)
        sin_d = freqs_sin[:, i].reshape(S * D)          # (S,D) -> (SD)
        # Now broadcast over (B,S,H): shapes (1,B,S,H,D)
        x_ = x.reshape(B, S, H, D)
        out_ = x_ * cos_d.view(1, S, 1, D) - x_ * sin_d.view(1, S, 1, D) * parity.view(1, 1, 1, D)
        return out_.reshape(B, S, H, D)


# Example usage:
# model = ModelNew()
# out = model.run(x, cos, sin)
