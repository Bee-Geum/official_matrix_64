import torch
import triton
import triton.language as tl

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

@triton.jit
def add_kernel(
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
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

def add_kernel_impl(x, y):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        return add_kernel_impl(x, y)

if __name__ == "__main__":
    model_old = Model()
    model_new = ModelNew()
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    out_old = model_old(x, y)
    out_new = model_new(x, y)
    assert torch.allclose(out_old, out_new, atol=1e-2, rtol=1e-2)
    print("Test passed!")