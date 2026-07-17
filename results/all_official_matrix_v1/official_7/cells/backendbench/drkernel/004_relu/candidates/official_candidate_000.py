import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# Simple, memory-bound elementwise ReLU kernel.
# Autotune over a few reasonable block sizes / num_warps.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def _relu_kernel(X, Y, N: tl.int32, BLOCK: tl.int32):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0)
    zero = tl.zeros([BLOCK], dtype=x.dtype)
    y = tl.maximum(x, zero)
    tl.store(Y + offs, y, mask=mask)


def relu_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    # Preconditions
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise RuntimeError(f"Expected CUDA tensor, got device={x.device}")
    if not x.is_floating_point():
        raise RuntimeError(f"Expected floating point tensor, got dtype={x.dtype}")

    # Ensure contiguous for coalesced access
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    # Flatten to 1D
    x_flat = x_contig.view(-1)
    y_flat = y.view(-1)
    N = x_flat.numel()

    # Launch kernel
    grid = (triton.cdiv(N, 1024),)  # grid is recomputed internally by autotune; this is a hint
    _relu_kernel(x_flat, y_flat, N, grid=grid)

    # Reshape back to original
    return y.view_as(x)
