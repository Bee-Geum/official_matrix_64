import torch
import triton
import triton.language as tl

@triton.jit
def triton_relu(x_ptrs, out_ptrs, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0) 
    # Offset for each thread block
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    # Mask for valid elements in each thread block
    x_mask = offsets < xnumel
    # Load x
    x = tl.load(x_ptrs + offsets, mask=x_mask)
    # Compute ReLU(x)
    out = tl.maximum(tl.zeros_like(x), x)
    # Store the result
    tl.store(out_ptrs + offsets, out, mask=x_mask)

class ModelNew(torch.nn.Module):
    """
    Simple model that performs a ReLU activation using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        output = torch.empty_like(x)
        # Number of elements per thread block
        XBLOCK = 1024
        # Number of thread blocks
        grid = (triton.cdiv(x.numel(), XBLOCK),)
        # Launch triton_relu
        triton_relu[grid](x, output, x.numel(), XBLOCK)
        return output