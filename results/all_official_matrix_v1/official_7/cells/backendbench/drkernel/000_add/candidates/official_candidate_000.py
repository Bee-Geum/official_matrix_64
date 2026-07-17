import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Simple, fast pointwise kernel: y = x + 1
@triton.jit
def add_one_kernel(x_ptr, y_ptr, N: tl.int32, BLOCK: tl.int32):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = x + 1.0
    tl.store(y_ptr + offs, y, mask=mask)


def add_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    # Validate environment
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    if not x.is_cuda:
        raise ValueError(f"Expected CUDA tensor, got device={x.device}")

    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported dtype {x.dtype}; expected float32/float16/bfloat16")

    # Ensure contiguous for simple 1D indexing
    if not x.is_contiguous():
        x = x.contiguous()

    # Flatten to 1D
    orig_shape = x.shape
    N = x.numel()
    x_flat = x.view(-1)

    # Allocate output
    y_flat = torch.empty_like(x_flat)

    # Launch configuration
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    # Launch kernel
    add_one_kernel[grid](
        x_flat, y_flat,
        N,
        BLOCK=BLOCK,
        num_warps=4,   # good default for memory-bound pointwise
        num_stages=2,  # small pipeline depth is fine here
    )

    # Reshape back
    return y_flat.view(orig_shape)
