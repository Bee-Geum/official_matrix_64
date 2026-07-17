import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel for H100 (sm_90).
    Uses tiling with software pipelining and group ordering for better L2 cache utilization.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to A and B blocks
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B blocks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Compute accumulator
        accumulator += tl.dot(a, b)
        
        # Move pointers to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication using Triton kernel.
    
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    
    Returns:
        Output matrix C of shape (M, N)
    """
    # Check dimensions
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D"
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Heuristics for block sizes (optimized for H100)
    # These values are tuned for 48MB L2 cache on H100
    if M <= 512 and N <= 512:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
        GROUP_SIZE_M = 8
    elif M <= 1024 and N <= 1024:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32
        GROUP_SIZE_M = 8
    elif M <= 2048 and N <= 2048:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32
        GROUP_SIZE_M = 8
    else:
        # For larger matrices, use larger blocks
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 256, 32
        GROUP_SIZE_M = 8
    
    # Ensure block sizes don't exceed matrix dimensions
    BLOCK_SIZE_M = min(BLOCK_SIZE_M, M)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, N)
    BLOCK_SIZE_K = min(BLOCK_SIZE_K, K)
    
    # Compute number of program IDs
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid = num_pid_m * num_pid_n
    
    # Launch kernel
    grid = lambda META: (num_pid,)
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)


N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N, device='cuda')
    B = torch.rand(N, N, device='cuda')
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed


# Test function to verify correctness and performance
def test_model():
    """Test the optimized model for correctness and performance."""
    import time
    
    # Create models
    model_ref = Model()
    model_new = ModelNew()
    
    # Get inputs
    A, B = get_inputs()
    
    # Warmup
    for _ in range(10):
        _ = model_ref(A, B)
        _ = model_new(A, B)
    
    # Test correctness
    torch.cuda.synchronize()
    start = time.time()
    C_ref = model_ref(A, B)
    torch.cuda.synchronize()
    ref_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    C_new = model_new(A, B)
    torch.cuda.synchronize()
    new_time = time.time() - start
    
    # Check correctness
    if torch.allclose(C_ref, C_new, atol=1e-2, rtol=1e-2):
        print(f"✓ Correctness test passed!")
        print(f"  Reference time: {ref_time:.4f}s")
        print(f"  Optimized time: {new_time:.4f}s")
        print(f"  Speedup: {ref_time/new_time:.2f}x")
    else:
        print("✗ Correctness test failed!")
        max_diff = torch.max(torch.abs(C_ref - C_new))
        print(f"  Maximum difference: {max_diff.item()}")


if __name__ == "__main__":
    # Run test if executed directly
    test_model()