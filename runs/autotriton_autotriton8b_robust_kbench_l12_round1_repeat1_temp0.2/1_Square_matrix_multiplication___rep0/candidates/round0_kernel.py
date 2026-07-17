import torch
import triton
import triton.language as tl

@triton.jit
def triton_matmul(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Offset for each thread block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:]
    # Mask for valid elements in each thread block
    mask = offsets < n * n
    # Calculate indices for A and B
    x = offsets % n
    y = offsets // n
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    # Iterate over k
    for k in range(0, n, BLOCK_SIZE):
        # Load A and B
        a = tl.load(a_ptr + y * n + k + tl.arange(0, BLOCK_SIZE)[:], mask=k + tl.arange(0, BLOCK_SIZE)[:] < n, other=0.0)
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_SIZE)[:, None]) * n + x, mask=k + tl.arange(0, BLOCK_SIZE)[:, None] < n, other=0.0)
        # Compute product and accumulate
        acc += tl.sum(a * b, axis=1)
    # Store the result
    tl.store(c_ptr + offsets, acc, mask=mask)

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
        n = A.size(0)
        # Ensure the output tensor is properly sized
        C = torch.empty((n, n), device=A.device, dtype=A.dtype)
        # Number of elements per thread block
        BLOCK_SIZE = 128
        # Number of thread blocks
        grid = (triton.cdiv(n * n, BLOCK_SIZE),)
        # Launch triton_matmul
        triton_matmul[grid](A, B, C, n, BLOCK_SIZE)
        return C