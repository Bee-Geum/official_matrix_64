import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def triton_tanh(x_ptrs, out_ptrs, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0) 
    # Offset for each thread block
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    # Mask for valid elements in each thread block
    x_mask = offsets < xnumel
    # Load x
    x = tl.load(x_ptrs + offsets, mask=x_mask)
    # Compute tanh(x)
    out = libdevice.tanh(x)
    # Store the result
    tl.store(out_ptrs + offsets, out, mask=x_mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def tanh_kernel_impl(self, x):
        output = torch.empty_like(x)
        # Number of elements per thread block
        XBLOCK = 1024
        # Number of thread blocks
        grid = (triton.cdiv(x.numel(), XBLOCK),)
        # Launch triton_tanh
        triton_tanh[grid](x, output, x.numel(), XBLOCK)
        return output

    def forward(self, x):
        return self.tanh_kernel_impl(x)
