# [official_matrix_64] make AutoKernel's own helpers importable outside its tree.
# AutoKernel's playbook (kernelbench/program_kb.md) tells the model to
# `from kernels.cuda._compile import compile_cuda`, which only resolves when the
# repo root is on sys.path -- true for its own bench_kb.py, false for the
# benchmark's official oracle, which runs the kernel from another directory and
# would fail at import with ModuleNotFoundError: No module named 'kernels'.
# This prepends the path only; compile_cuda itself is AutoKernel's, unmodified.
import sys as _sys
_AK_ROOT = '/home/bi_geum/official_matrix_64/third_party/autokernel'
if _AK_ROOT not in _sys.path:
    _sys.path.insert(0, _AK_ROOT)
# ---- AutoKernel output (verbatim) ----
"""
KernelBench Problem L1_P001: 1 Square matrix multiplication 
Level: 1 | Problem ID: 1
Operations: matmul
Difficulty: hard

Source: ScalingIntelligence/KernelBench
Optimized with AutoKernel (https://github.com/RightNow-AI/autokernel)

The agent optimizes ModelNew to outperform the PyTorch reference (Model).
Edit ModelNew.forward() -- use CUDA C++ via compile_cuda() or Triton @jit.
Run `uv run kernelbench/bench_kb.py` to evaluate correctness + speedup.
"""

KERNELBENCH_PROBLEM = {
    "level": 1,
    "problem_id": 1,
    "name": '1 Square matrix multiplication ',
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels.cuda._compile import compile_cuda

# Optional: use AutoKernel's CUDA compilation utility for custom CUDA C++ kernels
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}
"""

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = compile_cuda(CUDA_SRC, "matmul_cuda")
    return _mod

# ============================================================================
# Reference implementation (DO NOT MODIFY below this line)
# ============================================================================

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

# ============================================================================
# Optimized implementation (EDIT THIS)
# ============================================================================

# ModelNew must produce outputs matching Model within atol=1e-2, rtol=1e-2.
# Start by copying Model's logic, then optimize with CUDA C++ or Triton.

class ModelNew(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return _get_mod().matmul_cuda(A, B)

N = 2048 * 2