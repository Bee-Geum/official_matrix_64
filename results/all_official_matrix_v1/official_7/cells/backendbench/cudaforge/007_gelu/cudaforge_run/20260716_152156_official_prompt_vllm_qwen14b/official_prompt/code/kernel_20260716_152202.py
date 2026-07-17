# <complete ModelNew code>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU activation
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ float gelu_device(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrt(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
}

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_device(input[idx]);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

cpp_src = (
    "torch::Tensor gelu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for GELU activation
gelu_kernel_impl = load_inline(
    name="gelu_kernel_impl",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gelu_kernel_impl = gelu_kernel_impl

    def forward(self, x):
        return self.gelu_kernel_impl.gelu_cuda(x)
