import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Simple, fast ReLU kernel: y = max(x, 0)
@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0)
    # Use tl.maximum to preserve dtype; 0 is cast to x's dtype
    y = tl.maximum(x, 0)
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-backed ReLU module. Numerically matches torch.relu with atol=rtol=1e-2.
    - Uses a Triton kernel on CUDA tensors.
    - Falls back to torch.relu on CPU or if Triton is not available.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback for non-CUDA or missing Triton
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            return torch.relu(x)

        # Ensure contiguous for simple 1D indexing
        if not x.is_contiguous():
            x = x.contiguous()

        # Allocate output
        y = torch.empty_like(x)

        # Flatten to 1D
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        n = x_flat.numel()

        # Choose a reasonable block size; 1024 or 2048 are good defaults
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)

        # Launch kernel
        relu_kernel[grid](
            x_flat, y_flat,
            n,
            BLOCK=BLOCK,
            num_warps=4,     # good for memory-bound elementwise ops
            num_stages=2,
        )
        return y


# The following are kept identical to your reference for convenience/testing.

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')  # put on GPU to use Triton
    return [x]

def get_init_inputs():
    return []