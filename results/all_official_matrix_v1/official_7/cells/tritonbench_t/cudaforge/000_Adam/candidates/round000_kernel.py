# <complete ModelNew code>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Adam optimization
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void adam_update_kernel(float* params, float* grads, float* exp_avg, float* exp_avg_sq, float beta1, float beta2, float lr, float eps, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = grads[idx];
        float param = params[idx];
        float avg = exp_avg[idx];
        float avg_sq = exp_avg_sq[idx];

        avg = beta1 * avg + (1 - beta1) * grad;
        avg_sq = beta2 * avg_sq + (1 - beta2) * grad * grad;

        float bias_correction1 = 1 - pow(beta1, 1);
        float bias_correction2 = 1 - pow(beta2, 1);
        float step_size = lr * sqrt(bias_correction2) / bias_correction1;

        params[idx] = param - step_size * (avg / (sqrt(avg_sq) + eps));
        exp_avg[idx] = avg;
        exp_avg_sq[idx] = avg_sq;
    }
}

void adam_update_cuda(torch::Tensor params, torch::Tensor grads, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, float beta1, float beta2, float lr, float eps) {
    auto size = params.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    adam_update_kernel<<<num_blocks, block_size>>>(params.data_ptr<float>(), grads.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(), beta1, beta2, lr, eps, size);
}
"""

cpp_src = (
    "void adam_update_cuda(torch::Tensor params, torch::Tensor grads, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, float beta1, float beta2, float lr, float eps);"
)

# Compile the inline CUDA code for Adam optimization
adam_update = load_inline(
    name="adam_update",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["adam_update_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adam_update = adam_update

    def forward(self, params, grads, exp_avg, exp_avg_sq, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        self.adam_update.adam_update_cuda(params, grads, exp_avg, exp_avg_sq, betas[0], betas[1], lr, eps)
        return params
