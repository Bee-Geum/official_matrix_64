import torch
import triton
import triton.language as tl

@triton.jit
def triton_div_kernel(a_ptrs, b_ptrs, out_ptrs, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0) 
    # Offset for each thread block
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    # Mask for valid elements in each thread block
    x_mask = offsets < xnumel
    # Load a and b
    a = tl.load(a_ptrs + offsets, mask=x_mask)
    b = tl.load(b_ptrs + offsets, mask=x_mask)
    # Compute a / b
    out = a / b
    # Store the result
    tl.store(out_ptrs + offsets, out, mask=x_mask)

def div_kernel_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the element-wise division of two tensors `a` and `b`.
    
    Args:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor, must be broadcastable with `a`.
    
    Returns:
        torch.Tensor: Resulting tensor after applying element-wise division.
    """
    output = torch.empty_like(a)
    # Number of elements per thread block
    XBLOCK = 1024
    # Number of thread blocks
    grid = (triton.cdiv(a.numel(), XBLOCK),)
    # Launch triton_div_kernel
    triton_div_kernel[grid](a, b, output, a.numel(), XBLOCK)
    return output
