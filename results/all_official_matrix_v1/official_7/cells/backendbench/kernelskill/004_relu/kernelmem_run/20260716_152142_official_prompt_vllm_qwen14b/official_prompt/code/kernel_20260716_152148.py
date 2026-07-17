import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
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
relu_kernel_impl = load_inline(
    name="relu_kernel_impl",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[
        "-O3",                          # High optimization
        "-std=c++17",                    # Use C++17 standard
        ],
    extra_ldflags=[""],
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90",  # Correct CUDA target arch
        "--expt-relaxed-constexpr",  # CUDA specific flag
        "-lineinfo",  # Line information for debugging
        ],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu_kernel_impl = relu_kernel_impl

    def forward(self, x):
        return self.relu_kernel_impl.relu_cuda(x)
