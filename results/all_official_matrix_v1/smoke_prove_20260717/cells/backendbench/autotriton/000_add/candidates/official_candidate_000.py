import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel_impl(x_ptrs, y_ptrs, out_ptrs, xnumel, XBLOCK : tl.constexpr):
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

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor, must be broadcastable with `x`.
        """
        output = torch.empty_like(x)
        # Number of elements per thread block
        XBLOCK = 1024
        # Number of thread blocks
        grid = (triton.cdiv(x.numel(), XBLOCK),)
        # Launch triton kernel
        add_kernel_impl[grid](x, y, output, x.numel(), XBLOCK)
        return output
