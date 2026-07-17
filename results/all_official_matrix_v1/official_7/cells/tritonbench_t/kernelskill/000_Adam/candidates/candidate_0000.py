# <complete ModelNew code>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Adam optimizer step
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void adam_step_kernel(float* params, float* grads, float* exp_avg, float* exp_avg_sq, float* max_exp_avg_sq, int size, float lr, float beta1, float beta2, float eps, float weight_decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = grads[idx];
        if (weight_decay != 0) {
            grad += weight_decay * params[idx];
        }
        exp_avg[idx] = beta1 * exp_avg[idx] + (1 - beta1) * grad;
        exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (1 - beta2) * grad * grad;
        max_exp_avg_sq[idx] = fmaxf(max_exp_avg_sq[idx], exp_avg_sq[idx]);
        float bias_correction1 = 1 - powf(beta1, 1);
        float bias_correction2 = 1 - powf(beta2, 1);
        float step_size = lr * sqrtf(bias_correction2) / bias_correction1;
        params[idx] -= step_size * exp_avg[idx] / (sqrtf(max_exp_avg_sq[idx]) + eps);
    }
}

void adam_step_cuda(torch::Tensor params, torch::Tensor grads, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay) {
    auto size = params.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    auto stream = at::cuda::getDefaultCUDAStream();
    adam_step_kernel<<<num_blocks, block_size, 0, stream>>>(
        params.data_ptr<float>(), grads.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(), max_exp_avg_sq.data_ptr<float>(), (int)size, lr, beta1, beta2, eps, weight_decay
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

cpp_src = (
    "void adam_step_cuda(torch::Tensor params, torch::Tensor grads, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay);"
)

# Compile the inline CUDA code for Adam optimizer step
adam_step = load_inline(
    name="adam_step",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["adam_step_cuda"],
    verbose=True,
    extra_cflags=[
        "-O3",                          # High optimization
        "-std=c++17",                    # Use C++17 standard
        "--expt-relaxed-constexpr",  # CUDA specific flag
        "-lineinfo",  # Line information for debugging
        ],
    extra_ldflags=[""],
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90",  # Correct CUDA target arch
        ],
)

class AdamOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}
        for param in params:
            if id(param) not in self.state:
                self.state[id(param)] = {
                    'exp_avg': torch.zeros_like(param, device=param.device),
                    'exp_avg_sq': torch.zeros_like(param, device=param.device),
                    'max_exp_avg_sq': torch.zeros_like(param, device=param.device),
                }

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            state = self.state[id(param)]
            adam_step.adam_step_cuda(
                param, param.grad, state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq'],
                self.lr, self.betas[0], self.betas[1], self.eps, self.weight_decay
            )

def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    return AdamOptimizer(params, lr, betas, eps, weight_decay)

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
