import math
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Simple, bandwidth-bound elementwise ELU kernel.
# Computes: y = x if x > 0 else alpha * (exp(x) - 1)
# We upcast to float32 for the exp, then downcast to the output dtype.
@triton.jit
def _elu_kernel(x_ptr, y_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, eviction_policy='evict_last')
    x32 = x.to(tl.float32)

    # pred = x > 0
    pos = x32 > 0.0
    # neg branch: alpha * (exp(x) - 1)
    neg = alpha * (tl.exp(x32) - 1.0)
    # combine
    y32 = tl.where(pos, x32, neg)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized ELU activation that mirrors torch.nn.functional.elu numerics.
    Falls back to PyTorch on CPU.
    """
    def __init__(self, alpha: float = 1.0, block_size: int = 4096, num_warps: int = 4):
        super().__init__()
        self.alpha = float(alpha)
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if not CUDA or Triton not available
        if (not x.is_cuda) or (not _HAS_TRITON):
            return torch.nn.functional.elu(x, alpha=self.alpha)

        if not x.is_floating_point():
            raise TypeError(f"ELU expects a floating tensor, got dtype={x.dtype}")

        # Ensure contiguous for simple 1D indexing
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        n = x_contig.numel()
        # Grid: one program per block of elements
        grid = (triton.cdiv(n, self.block_size),)

        _elu_kernel[grid](
            x_contig, y,
            n,
            self.alpha,
            BLOCK=self.block_size,
            num_warps=self.num_warps,
        )

        return y.view_as(x)
