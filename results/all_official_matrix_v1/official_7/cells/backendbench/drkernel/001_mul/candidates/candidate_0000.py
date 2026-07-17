import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Simple, fast elementwise square kernel for float32
@triton.jit
def square_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, eviction_policy="evict_last")
    y = x * x
    tl.store(Y + offs, y, mask=mask)


class MulTriton:
    @staticmethod
    def mul_kernel_impl(x: torch.Tensor) -> torch.Tensor:
        # Assumes: x is 1D, float32
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

        if x.dtype != torch.float32:
            raise TypeError(f"Expected float32, got {x.dtype}")
        if x.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got shape {tuple(x.shape)}")

        N = x.shape[0]
        y = torch.empty_like(x)

        # Choose a block size; 1024 is a good default for pointwise fp32
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        square_kernel[grid](
            x, y, N,
            BLOCK=BLOCK,
            num_warps=4,   # reasonable default for 1024 block
            num_stages=2,  # small pipeline depth is fine here
        )
        return y


# The benchmark harness will look for 'mul_kernel_impl'
mul_kernel_impl = MulTriton.mul_kernel_impl
