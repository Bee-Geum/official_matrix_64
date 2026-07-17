import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def elu_kernel(
    x_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    condition = x > 0.0
    exp_term = tl.math.exp(x) - 1.0
    result = tl.where(condition, x, alpha * exp_term)
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.view(-1)
        n_elements = x_flat.numel()
        output_flat = torch.empty_like(x_flat)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        elu_kernel[grid](x_flat, output_flat, self.alpha, n_elements, BLOCK_SIZE=1024)
        return output_flat.view(original_shape)
