import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptrs, out_ptrs, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0) 
    # Offset for each thread block
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    # Mask for valid elements in each thread block
    x_mask = offsets < xnumel
    # Load x
    x = tl.load(x_ptrs + offsets, mask=x_mask)
    # Compute 0.5 * x
    half_x = x * 0.5
    # Compute 0.7071067811865476 * x
    scaled_x = x * 0.7071067811865476
    # Compute erf(scaled_x)
    erf_scaled_x = tl.math.erf(scaled_x)
    # Compute 1 + erf(scaled_x)
    one_plus_erf = 1.0 + erf_scaled_x
    # Compute half_x * one_plus_erf
    out = half_x * one_plus_erf
    # Store the result
    tl.store(out_ptrs + offsets, out, mask=x_mask)

def gelu_kernel_impl(x):
    output = torch.empty_like(x)
    # Number of elements per thread block
    XBLOCK = 1024
    # Number of thread blocks
    grid = (triton.cdiv(x.numel(), XBLOCK),)
    # Launch triton kernel
    gelu_kernel[grid](x, output, x.numel(), XBLOCK)
    return output

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu_kernel_impl(x)
