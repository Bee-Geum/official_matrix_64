# <complete ModelNew code>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise subtraction
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_sub_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

torch::Tensor elementwise_sub_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_sub_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

cpp_src = (
    "torch::Tensor elementwise_sub_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise subtraction
elementwise_sub = load_inline(
    name="elementwise_sub",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["elementwise_sub_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_sub = elementwise_sub

    def forward(self, a, b):
        return self.elementwise_sub.elementwise_sub_cuda(a, b)
