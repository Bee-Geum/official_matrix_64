import torch
import triton
import triton.language as tl

@triton.jit
def triton_matmul(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Offset for each thread block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:]
    # Mask for valid elements in each thread block
    mask = offsets < n
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Loop over k
    for k in range(0, n, BLOCK_SIZE):
        # Load a and b
        a = tl.load(a_ptr + offsets[:, None] * n + (k + tl.arange(0, BLOCK_SIZE)[None, :]), mask=mask[:, None] & (k + tl.arange(0, BLOCK_SIZE)[None, :] < n), other=0.0)
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_SIZE)[None, :])[:, None] * n + offsets[None, :], mask=(k + tl.arange(0, BLOCK_SIZE)[None, :] < n)[:, None] & mask[None, :], other=0.0)
        # Compute product
        prod = a @ b
        # Accumulate
        acc += prod
    # Store the result
    tl.store(c_ptr + offsets[:, None] * n + offsets[None, :], acc, mask=mask[:, None] & mask[None, :])

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
        # Ensure the input tensors are on the same device
        assert A.device == B.device, "Input tensors must be on the same device"
        # Ensure the input tensors are of the same dtype
        assert A.dtype == B.dtype, "Input tensors must be of the same dtype"
        # Ensure the input tensors are 2D
        assert A.dim() == 2 and B.dim() == 2, "Input tensors must be 2D"
        # Ensure the input tensors are square
        assert A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1], "Input tensors must be square"
        # Ensure the input tensors have the same size
        assert A.shape == B.shape, "Input tensors must have the same shape"
        
        N = A.shape[0]
        # Output tensor
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)
        # Number of elements per thread block
        BLOCK_SIZE = 128
        # Number of thread blocks
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        # Launch triton_matmul
        triton_matmul[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
        return C