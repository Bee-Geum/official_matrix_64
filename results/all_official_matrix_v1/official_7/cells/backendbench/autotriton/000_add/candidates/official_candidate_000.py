import torch
import triton
import triton.language as tl

@triton.jit
def triton_add(x_ptrs, y_ptrs, out_ptrs, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0) 
    # Offset for each thread block
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    # Mask for valid elements in each thread block
    x_mask = offsets < xnumel
    # Load x and y
    x = tl.load(x_ptrs + offsets, mask=x_mask)
    y = tl.load(y_ptrs + offsets, mask=x_mask)
    # Compute x + y
    out = x + y
    # Store the result
    tl.store(out_ptrs + offsets, out, mask=x_mask)

def add_kernel_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the element-wise addition of two tensors `a` and `b` using a custom Triton kernel.
    
    Args:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor, must be broadcastable with `a`.
    
    Returns:
        torch.Tensor: Resulting tensor after applying element-wise addition.
    """
    output = torch.empty_like(a)
    # Number of elements per thread block
    XBLOCK = 1024
    # Number of thread blocks
    grid = (triton.cdiv(a.numel(), XBLOCK),)
    # Launch triton_add
    triton_add[grid](a, b, output, a.numel(), XBLOCK)
    return output

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model, computes the element-wise addition of two tensors `a` and `b`.
        
        Args:
            a (torch.Tensor): First input tensor.
            b (torch.Tensor): Second input tensor, must be broadcastable with `a`.
        
        Returns:
            torch.Tensor: Resulting tensor after applying element-wise addition.
        """
        return add_kernel_impl(a, b)
