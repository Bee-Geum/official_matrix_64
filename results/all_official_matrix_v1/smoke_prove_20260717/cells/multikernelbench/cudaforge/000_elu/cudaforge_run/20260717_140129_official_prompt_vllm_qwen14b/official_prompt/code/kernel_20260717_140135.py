# <complete ModelNew code>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ELU activation
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elu_kernel(const float* x, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = (val > 0) ? val : alpha * (exp(val) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, size);

    return out;
}
"""

cpp_src = (
    "torch::Tensor elu_cuda(torch::Tensor x, float alpha);"
)

# Compile the inline CUDA code for ELU activation
elu = load_inline(
    name="elu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["elu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.elu = elu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu.elu_cuda(x, self.alpha)
