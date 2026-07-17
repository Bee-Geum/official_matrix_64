"""
KernelBench Problem L1_P000: official prompt
Level: 1 | Problem ID: 0
Operations: unknown
Difficulty: unknown

Source: ScalingIntelligence/KernelBench
Optimized with AutoKernel (https://github.com/RightNow-AI/autokernel)

The agent optimizes ModelNew to outperform the PyTorch reference (Model).
Edit ModelNew.forward() -- use CUDA C++ via compile_cuda() or Triton @jit.
Run `uv run kernelbench/bench_kb.py` to evaluate correctness + speedup.
"""

KERNELBENCH_PROBLEM = {
    "level": 1,
    "problem_id": 0,
    "name": 'official prompt',
}

import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional: use AutoKernel's CUDA compilation utility for custom CUDA C++ kernels
# from kernels.cuda._compile import compile_cuda
#
# CUDA_SRC = r"""
# #include <torch/extension.h>
# #include <cuda_runtime.h>
# #include <cuda_fp16.h>
#
# __global__ void my_kernel(const float* input, float* output, int N) {
#     int idx = blockIdx.x * blockDim.x + threadIdx.x;
#     if (idx < N) output[idx] = input[idx];
# }
#
# torch::Tensor my_op_cuda(torch::Tensor input) {
#     auto output = torch::empty_like(input);
#     int N = input.numel();
#     my_kernel<<<(N+255)/256, 256>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
#     return output;
# }
# """
# _mod = None
# def _get_mod():
#     global _mod
#     if _mod is None:
#         _mod = compile_cuda(CUDA_SRC, "my_op_cuda")
#     return _mod

# ============================================================================
# Reference implementation (DO NOT MODIFY below this line)
# ============================================================================

Benchmark: tritonbench_t
Official task: Adam
Return only source code, without Markdown fences or explanations.

Implement the official TritonBench task as a Python module and preserve its public interface.


def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

##################################################################################################################################################



# def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
#     return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

def test_Adam():
    results = {}

    # Test Case 1: Default parameters
    params1 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer1 = Adam(params1)
    results["test_case_1"] = optimizer1.defaults

    # Test Case 2: Custom learning rate
    params2 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer2 = Adam(params2, lr=0.01)
    results["test_case_2"] = optimizer2.defaults

    # Test Case 3: Custom betas
    params3 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer3 = Adam(params3, betas=(0.85, 0.95))
    results["test_case_3"] = optimizer3.defaults

    # Test Case 4: Custom weight decay
    params4 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer4 = Adam(params4, weight_decay=0.01)
    results["test_case_4"] = optimizer4.defaults

    return results

test_results = test_Adam()

# ============================================================================
# Optimized implementation (EDIT THIS)
# ============================================================================

# ModelNew must produce outputs matching Model within atol=1e-2, rtol=1e-2.
# Start by copying Model's logic, then optimize with CUDA C++ or Triton.

class ModelNew(nn.Module):
    """Optimized version -- replace forward() internals with custom kernels."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._ref = Model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # TODO: Replace with optimized implementation
        return self._ref(*args, **kwargs)

