import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Simple 1D subtraction kernel: out = x - y
# Assumes float32, 1D contiguous tensors.
@triton.jit
def sub_kernel(x_ptr, y_ptr, out_ptr, N: tl.int32, BLOCK: tl.int32):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, eviction_policy='evict_last')
    y = tl.load(y_ptr + offs, mask=mask, eviction_policy='evict_last')
    tl.store(out_ptr + offs, x - y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Fallback if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (not torch.cuda.is_available()):
            # Match reference behavior: out = x - y
            return x - y

        # Validate inputs
        if x.device.type != 'cuda' or y.device.type != 'cuda':
            raise RuntimeError("ModelNew requires CUDA tensors.")
        if x.dtype != torch.float32 or y.dtype != torch.float32:
            raise RuntimeError(f"ModelNew expects float32 tensors, got {x.dtype} and {y.dtype}.")
        if x.shape != y.shape:
            raise RuntimeError(f"Shape mismatch: {x.shape} vs {y.shape}.")
        if not x.is_contiguous() or not y.is_contiguous():
            # For simplicity, enforce contiguous; could also use .contiguous()
            raise RuntimeError("ModelNew expects contiguous tensors.")

        N = x.numel()
        out = torch.empty_like(x)

        # Launch configuration
        BLOCK = self.block_size
        grid = (triton.cdiv(N, BLOCK),)

        sub_kernel[grid](
            x, y, out,
            N,
            BLOCK=BLOCK,
            num_warps=4,   # good default for 1024 block
            num_stages=2,  # simple pipeline
        )

        return out
