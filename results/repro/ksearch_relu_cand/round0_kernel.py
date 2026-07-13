import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    """
    Optimized model that performs a ReLU activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    @staticmethod
    @triton.jit
    def relu_kernel(
        X,  # pointer to the input tensor
        Y,  # pointer to the output tensor
        N,  # number of elements
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask)
        y = tl.where(x >= 0, x, 0)
        tl.store(Y + offsets, y, mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        N = x.numel()
        Y = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
        self.relu_kernel[grid](x.contiguous(), Y.contiguous(), N, BLOCK_SIZE=1024)
        return Y

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed