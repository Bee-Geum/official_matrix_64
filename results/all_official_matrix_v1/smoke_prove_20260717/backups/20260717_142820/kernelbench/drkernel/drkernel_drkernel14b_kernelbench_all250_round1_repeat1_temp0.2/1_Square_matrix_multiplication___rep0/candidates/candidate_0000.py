import math
import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for tiling over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    A_ptr = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load with masks
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
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    # Write back
    C_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        C_ptr,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using a Triton kernel.
    Assumes float32, 2D square matrices.
    """
    assert A.dim() == 2 and B.dim() == 2, "Only 2D matrices supported"
    assert A.shape == B.shape, "This simple implementation assumes square matrices of equal shape"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "This implementation assumes float32"

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is (*, {K}), B is ({Kb}, *)"

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Strides (in elements)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Tile sizes – good starting point for fp32 on modern GPUs
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[
        grid
    ](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,  # heuristic
        num_stages=3, # heuristic
    )

    return C


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version of the reference Model that computes C = A @ B.
    Entry point as requested.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Fallback to torch if not on CUDA
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A, B)
        return _matmul_triton(A, B)
