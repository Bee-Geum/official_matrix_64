import torch
import torch.nn as nn
import triton
import triton.language as tl

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x * y

@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)

def mul_kernel_impl(x, y):
    assert x.is_cuda and y.is_cuda, "Inputs must be on CUDA"
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return mul_kernel_impl(x, y)

if __name__ == "__main__":
    model_old = Model()
    model_new = ModelNew()
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    out_old = model_old(x, y)
    out_new = model_new(x, y)
    print(torch.allclose(out_old, out_new, atol=1e-2, rtol=1e-2))
    import time
    start = time.time()
    for _ in range(100):
        model_old(x, y)
    torch.cuda.synchronize()
    print("Old:", time.time() - start)
    start = time.time()
    for _ in range(100):
        model_new(x, y)
    torch.cuda.synchronize()
    print("New:", time.time() - start)

__all__ = ["ModelNew"]