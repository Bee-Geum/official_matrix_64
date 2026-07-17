import torch
import triton
import triton.language as tl

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(x)

@triton.jit
def tanh_kernel(
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
    out = tl.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)

def tanh_kernel_impl(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return tanh_kernel_impl(x)

if __name__ == "__main__":
    model_ref = Model()
    model_new = ModelNew()
    x = torch.randn(1024, 1024, device='cuda')
    with torch.no_grad():
        y_ref = model_ref(x)
        y_new = model_new(x)
    print(torch.allclose(y_ref, y_new, atol=1e-2, rtol=1e-2))
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model_ref(x)
    torch.cuda.synchronize()
    t_ref = time.time() - start
    start = time.time()
    for _ in range(100):
        _ = model_new(x)
    torch.cuda.synchronize()
    t_new = time.time() - start
    print(f"Reference: {t_ref:.4f}s, New: {t_new:.4f}s, Speedup: {t_ref/t_new:.2f}x")
