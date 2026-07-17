import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# Simple, bandwidth-optimal elementwise: y = x + x  =>  y = 2 * x
@triton.jit
def _double_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = x + x
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 4):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate device
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available. Install triton to use this backend.")
        if not x.is_cuda:
            raise RuntimeError(f"Expected CUDA tensor, got device={x.device}.")
        if not x.is_contiguous():
            # Could make contiguous; for now enforce for simplicity
            raise RuntimeError("Input must be contiguous for this Triton kernel.")

        # Allocate output
        y = torch.empty_like(x)

        N = x.numel()
        if N == 0:
            return y

        grid = (triton.cdiv(N, self.block_size),)

        # Launch
        _double_kernel[grid](
            x, y,
            N,
            BLOCK=self.block_size,
            num_warps=self.num_warps,
        )
        return y
