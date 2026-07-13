# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation with optimized branch prediction
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_optimized_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = x[idx];
        out[idx] = value * (value > 0.0f);
    }
}

torch::Tensor relu_optimized_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_optimized_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

cpp_src = (
    "torch::Tensor relu_optimized_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for optimized ReLU activation
relu_optimized = load_inline(
    name="relu_optimized",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["relu_optimized_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = relu_optimized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.relu_optimized_cuda(x)
