import torch
import triton
import triton.language as tl

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - x.mean()

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sub_kernel_impl(x)

@triton.jit
def sub_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mean = tl.sum(x, axis=0) / n_elements
    out = x - mean
    tl.store(out_ptr + offsets, out, mask=mask)

def sub_kernel_impl(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sub_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out
