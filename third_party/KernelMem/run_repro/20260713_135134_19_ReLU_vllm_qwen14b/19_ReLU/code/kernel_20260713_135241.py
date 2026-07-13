# <complete ModelNew code>

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ReLU activation
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(input[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    auto stream = at::cuda::getDefaultCUDAStream();
    relu_kernel<<<num_blocks, block_size, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), (int)size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

cpp_src = (
    "torch::Tensor relu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for ReLU activation
relu_op = load_inline(
    name="relu_op",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[
        "-O3",                          # High optimization
        "-std=c++17",                    # Use C++17 standard
        "-lineinfo",  # Line information for debugging
        ],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for ReLU activation.
    
    Granularity: (A) optimize a single hotspot op.
    Replaced op: torch.relu(x).
    Fusion/Library Calls: None (custom CUDA kernel used).
    Remaining in PyTorch: None (entire forward replaced).
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_op = relu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return self.relu_op.relu_cuda(x)
