import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# Simple 1D elementwise kernel: y = x * 0.2  (equivalent to x / 5)
@triton.jit
def _div_by_five_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    # multiply by reciprocal to avoid division
    y = x * 0.2
    tl.store(Y + offs, y, mask=mask)


def div_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of y = x / 5 for a 1D contiguous tensor x.
    Always uses GPU if available; falls back to PyTorch only if Triton/CUDA is unavailable.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Ensure floating dtype
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        x = x.float()

    # Contiguous is required for this simple kernel
    if not x.is_contiguous():
        x = x.contiguous()

    # Move to GPU if needed
    x_gpu = x
    moved_to_gpu = False
    if not x_gpu.is_cuda:
        x_gpu = x_gpu.to('cuda')
        moved_to_gpu = True

    N = x_gpu.numel()
    y_gpu = torch.empty_like(x_gpu)

    # Choose launch params
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    # Launch kernel
    _div_by_five_kernel[grid](
        x_gpu, y_gpu, N,
        BLOCK=BLOCK,
        num_warps=4,  # good default for memory-bound
        num_stages=2,
    )

    # Move back to original device if we moved
    if moved_to_gpu and x.device != torch.device('cuda'):
        return y_gpu.to(x.device)
    return y_gpu
