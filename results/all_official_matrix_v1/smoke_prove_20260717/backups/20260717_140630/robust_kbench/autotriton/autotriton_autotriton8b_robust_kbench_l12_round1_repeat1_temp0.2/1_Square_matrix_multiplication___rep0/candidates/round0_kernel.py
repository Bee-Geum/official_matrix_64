import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a * b
    tl.store(c_ptr + offsets, c, mask=mask)

class ModelNew(torch.nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using a Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        N = A.size(0)
        C = torch.empty_like(A)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N * N, BLOCK_SIZE),)
        matmul_kernel[grid](A, B, C, N * N, BLOCK_SIZE)
        return C

def get_inputs():
    N = 2048 * 2
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []