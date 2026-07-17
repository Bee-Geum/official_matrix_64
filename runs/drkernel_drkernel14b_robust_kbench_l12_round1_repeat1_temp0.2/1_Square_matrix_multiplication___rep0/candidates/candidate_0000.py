import math
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Simple, robust float32 matmul kernel.
# Computes C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def _matmul_fp32_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    # Pointers for the first K-block
    A_ptr = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BK):
        a = tl.load(
            A_ptr,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        # Accumulate
        acc += tl.dot(a, b)
        # Advance pointers
        A_ptr += BK * stride_ak
        B_ptr += BK * stride_bk

    # Write back
    C_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        C_ptr,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class ModelNew(torch.nn.Module):
    """
    Triton implementation of C = A @ B for float32, shape (N, N).
    Falls back to torch.matmul if Triton/CUDA is not available.
    """
    def __init__(self, block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3):
        super().__init__()
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate inputs
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got A.dim={A.dim()}, B.dim={B.dim()}")
        if A.shape != B.shape:
            raise ValueError(f"Expected same shapes, got A.shape={A.shape}, B.shape={B.shape}")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise TypeError(f"Expected float32 dtype, got A.dtype={A.dtype}, B.dtype={B.dtype}")
        if A.device != B.device:
            raise ValueError(f"Expected same device, got A.device={A.device}, B.device={B.device}")

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError(f"Incompatible shapes: A is (*, {K}), B is ({K2}, *).")

        # If not CUDA or Triton not available, fall back
        if (not _HAS_TRITON) or (A.device.type != "cuda"):
            return torch.matmul(A, B)

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Compute grid
        grid = (triton.cdiv(M, self.block_m), triton.cdiv(N, self.block_n))

        # Launch kernel
        _matmul_fp32_kernel[
            grid
        ](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BM=self.block_m, BN=self.block_n, BK=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        return C
