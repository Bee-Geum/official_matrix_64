# <complete ModelNew code>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sigmoid activation
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

cpp_src = (
    "torch::Tensor sigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for sigmoid activation
sigmoid = load_inline(
    name="sigmoid",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, x):
        return self.sigmoid.sigmoid_cuda(x)
