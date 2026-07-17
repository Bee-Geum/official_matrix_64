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

def relu_kernel_impl(x):
    output = torch.empty_like(x)
    # Number of elements per thread block
    XBLOCK = 1024
    # Number of thread blocks
    grid = (triton.cdiv(x.numel(), XBLOCK),)
    # Launch triton_relu
    triton_relu[grid](x, output, x.numel(), XBLOCK)
    return output

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu_kernel_impl(x)
