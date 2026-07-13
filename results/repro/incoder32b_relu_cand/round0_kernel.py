import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for ReLU activation
relu_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

torch::Tensor relu_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int numel = input.numel();
    
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
    
    return output;
}
"""

# Load the CUDA extension
relu_extension = load_inline(
    name='relu_extension',
    cpp_sources=[relu_cuda_source],
    functions=['relu_forward_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        if x.is_cuda:
            return relu_extension.relu_forward_cuda(x)
        else:
            # Fallback to PyTorch's implementation for CPU
            return torch.relu(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed