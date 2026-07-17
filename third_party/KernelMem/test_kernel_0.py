# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void relu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = max(x[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    auto stream = at::cuda::getDefaultCUDAStream();
    relu_kernel<<<num_blocks, block_size, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), (int)size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_src = (
    "torch::Tensor relu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for ReLU activation
relu = load_inline(
    name="relu",
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
    """
    Optimized model that performs a ReLU activation using a custom CUDA kernel.
    Granularity: (A) optimize a single hotspot op
    Replaced ops: torch.relu
    Fused ops into kernels: relu_kernel
    Remaining in PyTorch: None (all computation is offloaded to CUDA)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.relu_cuda(x)
