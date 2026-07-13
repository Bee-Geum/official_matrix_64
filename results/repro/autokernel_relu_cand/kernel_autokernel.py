"""
KernelBench Problem L1_P019: 19 ReLU
Level: 1 | Problem ID: 19
Operations: unknown
Difficulty: medium

Source: ScalingIntelligence/KernelBench
Optimized with AutoKernel (https://github.com/RightNow-AI/autokernel)

The agent optimizes ModelNew to outperform the PyTorch reference (Model).
Edit ModelNew.forward() -- use CUDA C++ via compile_cuda() or Triton @jit.
Run `uv run kernelbench/bench_kb.py` to evaluate correctness + speedup.
"""

KERNELBENCH_PROBLEM = {
    "level": 1,
    "problem_id": 19,
    "name": '19 ReLU',
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

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) output[idx] = fmaxf(0.0f, input[idx]);
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.numel();
    relu_kernel<<<(N+255)/256, 256>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}
"""

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = compile_cuda(CUDA_SRC, "relu_cuda")
    return _mod

# ============================================================================
# Reference implementation (DO NOT MODIFY below this line)
# ============================================================================

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return torch.relu(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed

# ============================================================================
# Optimized implementation (EDIT THIS)
# ============================================================================

# ModelNew must produce outputs matching Model within atol=1e-2, rtol=1e-2.
# Start by copying Model's logic, then optimize with CUDA C++ or Triton.

class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return _get_mod().relu_cuda(x)

batch_size = 4096
dim = 393216