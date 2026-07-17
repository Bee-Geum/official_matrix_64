# <complete ModelNew code>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Adam update
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void adam_update_kernel(
    float* p_ptr,
    const float* grad_ptr,
    float* exp_avg_ptr,
    float lr,
    float wd,
    float beta1,
    float beta2,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        // Load
        float p = p_ptr[idx];
        float grad = grad_ptr[idx];
        float exp_avg = exp_avg_ptr[idx];

        // Stepweight decay
        p = p * (1 - lr * wd);

        // Diff between momentum running average and grad
        float diff = exp_avg - grad;

        // Weight update
        float update = diff * beta1 + grad;

        // torch.sign
        bool can_update = update != 0;
        float update_sign = update > 0 ? -lr : lr;

        p = p + update_sign * can_update;

        // Decay the momentum running average coefficient
        exp_avg = diff * beta2 + grad;

        // Store new params and momentum running average coefficient
        p_ptr[idx] = p;
        exp_avg_ptr[idx] = exp_avg;
    }
}

torch::Tensor adam_update_cuda(
    torch::Tensor p,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    float lr,
    float wd,
    float beta1,
    float beta2
) {
    auto size = p.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    adam_update_kernel<<<num_blocks, block_size>>>(
        p.data_ptr<float>(),
        grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        lr,
        wd,
        beta1,
        beta2,
        size
    );

    return p;
}
"""

cpp_src = (
    "torch::Tensor adam_update_cuda(torch::Tensor p, torch::Tensor grad, torch::Tensor exp_avg, float lr, float wd, float beta1, float beta2);"
)

# Compile the inline CUDA code for Adam update
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

    def forward(self, p, grad, exp_avg, lr, wd, beta1, beta2):
        return self.adam_update.adam_update_cuda(p, grad, exp_avg, lr, wd, beta1, beta2)
