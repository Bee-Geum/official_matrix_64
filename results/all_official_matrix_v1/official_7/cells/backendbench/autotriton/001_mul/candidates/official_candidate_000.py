import torch
import triton
import triton.language as tl

@triton.jit
def triton_mul_kernel(x_ptrs, y_ptrs, out_ptrs, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0) 
    # Offset for each thread block
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    # Mask for valid elements in each thread block
    x_mask = offsets < xnumel
    # Load x and y
    x = tl.load(x_ptrs + offsets, mask=x_mask)
    y = tl.load(y_ptrs + offsets, mask=x_mask)
    # Compute x * y
    out = x * y
    # Store the result
    tl.store(out_ptrs + offsets, out, mask=x_mask)

def mul_kernel_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the element-wise multiplication of two tensors `x` and `y`.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor, must be broadcastable with `x`.

    Returns:
        torch.Tensor: Resulting tensor after applying element-wise multiplication.
    """
    output = torch.empty_like(x)
    # Number of elements per thread block
    XBLOCK = 1024
    # Number of thread blocks
    grid = (triton.cdiv(x.numel(), XBLOCK),)
    # Launch triton_mul_kernel
    triton_mul_kernel[grid](x, y, output, x.numel(), XBLOCK)
    return output
