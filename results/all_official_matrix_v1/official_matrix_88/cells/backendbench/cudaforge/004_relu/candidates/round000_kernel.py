import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(y_ptr + offsets, y, mask=mask)

def relu_kernel_impl(input_tensor):
    output_tensor = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=1024)
    return output_tensor
