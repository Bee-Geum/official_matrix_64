import triton
import triton.language as tl

# Elementwise tanh kernel.
# Computes y = tanh(x) using a numerically stable formulation:
# tanh(x) = 1 - 2 / (exp(2x) + 1)
@triton.jit
def tanh_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load as input dtype, then upcast to float32 for math
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # Compute tanh via stable formula
    two_x = 2.0 * x32
    e = tl.exp(two_x)              # exp(2x)
    y32 = 1.0 - 2.0 / (e + 1.0)    # 1 - 2 / (exp + 1)

    # Cast back to input dtype and store
    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


def tanh_kernel_impl(x):
    """
    Triton implementation of tanh.
    Entry point expected by the benchmark.
    """
    if not x.is_cuda:
        raise RuntimeError("tanh_kernel_impl expects a CUDA tensor")

    # Ensure contiguous for coalesced access
    was_contig = x.is_contiguous()
    x_c = x.contiguous()

    N = x_c.numel()
    y = torch.empty_like(x_c)

    # Choose block and grid
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    # Launch kernel
    tanh_kernel[grid](
        x_c, y, N,
        BLOCK=BLOCK,
        num_warps=4,   # reasonable default for pointwise
        num_stages=2,
    )

    # Restore original shape (y is already same shape as x_c)
    if not was_contig:
        # If the original was non-contiguous, return with same shape but possibly different strides.
        # torch.empty_like on non-contig returns contig; we can view as original shape.
        return y.view(x.shape)
    return y
