import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tanh
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    auto stream = at::cuda::getDefaultCUDAStream();
    tanh_kernel<<<num_blocks, block_size, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), (int)size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

cpp_src = (
    "torch::Tensor tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for tanh
tanh_kernel_impl = load_inline(
    name="tanh_kernel",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["tanh_cuda"],
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
        self.tanh_kernel_impl = tanh_kernel_impl

    def forward(self, input):
        return self.tanh_kernel_impl.tanh_cuda(input)
