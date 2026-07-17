import math
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Autotuned square GEMM: C = A @ B  (float32, shapes (N, N))
@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64,  'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 64,  'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64,  'BN': 64,  'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=4),
        triton.Config({'BM': 64,  'BN': 128, 'BK': 64}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 64,  'BK': 64}, num_warps=8, num_stages=4),
        triton.Config({'BM': 64,  'BN': 64,  'BK': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def _matmul_square_fp32_kernel(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    # Create accumulator
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Loop over K dimension in tiles of BK
    for kk in range(0, K, BK):
        offs_k = kk + tl.arange(0, BK)

        # Pointers for A: shape (BM, BK)
        a_ptrs = A + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        # Pointers for B: shape (BK, BN)
        b_ptrs = B + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        # Load with boundary checks
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    """
    Triton implementation of C = A @ B for float32, square matrices.
    Falls back to torch.matmul if Triton/CUDA is not available.
    """
    def __init__(self):
        super().__init__()
        if not _HAS_TRITON:
            # Nothing to init
            pass

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got A.dim={A.dim()}, B.dim={B.dim()}")
        if A.shape != B.shape:
            raise ValueError(f"Expected same shapes for A and B, got {A.shape} vs {B.shape}")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise TypeError(f"Expected float32 dtype, got {A.dtype} and {B.dtype}")
        if A.device != B.device:
            raise ValueError(f"A and B must be on the same device, got {A.device} and {B.device}")

        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (*, {K}), B is ({Kb}, *).")

        # If Triton not available or not CUDA, fallback
        if (not _HAS_TRITON) or (A.device.type != 'cuda'):
            return torch.matmul(A, B)

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Compute strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch grid
        def grid(meta):
            return (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))

        _matmul_square_fp32_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )

        return C
