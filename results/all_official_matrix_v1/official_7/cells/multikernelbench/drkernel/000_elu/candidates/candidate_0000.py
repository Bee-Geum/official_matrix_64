import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# 1D elementwise ELU kernel: y = x if x > 0 else alpha * (exp(x) - 1)
# Computes in float32 for stability when input is f16/bf16, then casts back.
@triton.jit
def _elu_kernel(X, Y, N, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + offs, mask=mask, other=0.0)
    # Upcast to fp32 for math if needed
    x32 = x.to(tl.float32)

    # Compute ELU in fp32
    is_pos = x32 > 0.0
    neg_part = alpha * (tl.exp(x32) - 1.0)
    y32 = tl.where(is_pos, x32, neg_part)

    # Cast back to original dtype
    y = y32.to(x.dtype)
    tl.store(Y + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized ELU activation that mirrors torch.nn.functional.elu.

    Entry point as requested: ModelNew
    """
    def __init__(self, alpha: float = 1.0, block_size: int = 4096, num_warps: int = 8, num_stages: int = 2):
        """
        alpha: ELU alpha parameter (float).
        block_size: elements per program (power of two, e.g., 1024–4096).
        num_warps: Triton kernel launch parameter.
        num_stages: Triton kernel launch parameter.
        """
        super().__init__()
        self.alpha = float(alpha)
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback if Triton not available or tensor not on CUDA
        if (not _HAS_TRITON) or (not x.is_cuda):
            return torch.nn.functional.elu(x, alpha=self.alpha)

        if x.numel() == 0:
            return x

        # Supported dtypes
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            # Fallback for unsupported dtypes
            return torch.nn.functional.elu(x, alpha=self.alpha)

        # Ensure contiguous and flatten
        x_contig = x.contiguous()
        x_flat = x_contig.view(-1)
        y_flat = torch.empty_like(x_flat)

        N = x_flat.numel()
        grid = (triton.cdiv(N, self.block_size),)

        _elu_kernel[grid](
            x_flat, y_flat,
            N,
            self.alpha,
            BLOCK=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        y = y_flat.view_as(x_contig)
        return y
