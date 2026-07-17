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

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;
    float Cvalue = 0.0f;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if (Row < N && (m * TILE_SIZE + tx) < N)
            As[ty][tx] = A[Row * N + m * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if ((m * TILE_SIZE + ty) < N && Col < N)
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + Col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            Cvalue += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Cvalue;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
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