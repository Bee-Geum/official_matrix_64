# <complete ModelNew code>

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for RoPE application
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void rope_kernel(
    const __half* query_or_key, const __half* freqs_cos, const __half* freqs_sin,
    __half* output, int batch_size, int seq_len, int num_attention_heads, int half_head_dim) {
    
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int pair_idx = threadIdx.x;

    if (batch_idx < batch_size && seq_idx < seq_len && head_idx < num_attention_heads && pair_idx < half_head_dim) {
        int qk_idx = ((batch_idx * seq_len + seq_idx) * num_attention_heads + head_idx) * half_head_dim * 2 + pair_idx * 2;
        int freq_idx = (seq_idx * half_head_dim + pair_idx) * 2;

        __half x1 = query_or_key[qk_idx];
        __half x2 = query_or_key[qk_idx + 1];
        __half cos1 = freqs_cos[freq_idx];
        __half cos2 = freqs_cos[freq_idx + 1];
        __half sin1 = freqs_sin[freq_idx];
        __half sin2 = freqs_sin[freq_idx + 1];

        __half out1 = x1 * cos1 - x2 * sin1;
        __half out2 = x1 * sin2 + x2 * cos2;

        int out_idx = qk_idx;
        output[out_idx] = out1;
        output[out_idx + 1] = out2;
    }
}

torch::Tensor rope_cuda(
    torch::Tensor query_or_key, torch::Tensor freqs_cos, torch::Tensor freqs_sin,
    int batch_size, int seq_len, int num_attention_heads, int half_head_dim) {
    
    auto output = torch::empty_like(query_or_key);

    const int block_size = 256;
    const dim3 grid(batch_size, seq_len, num_attention_heads);
    const dim3 block(block_size);

    auto stream = at::cuda::getDefaultCUDAStream();
    rope_kernel<<<grid, block, 0, stream>>>(
        query_or_key.data_ptr<__half>(), freqs_cos.data_ptr<__half>(), freqs_sin.data_ptr<__half>(),
        output.data_ptr<__half>(), batch_size, seq_len, num_attention_heads, half_head_dim
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

cpp_src = (
    "torch::Tensor rope_cuda(torch::Tensor query_or_key, torch::Tensor freqs_cos, torch::Tensor freqs_sin, int batch_size, int seq_len, int num_attention_heads, int half_head_dim);"
)

# Compile the inline CUDA code for RoPE application
rope = load_inline(
    name="rope",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["rope_cuda"],
    verbose=True,
    extra_cflags=[
        "-O3",                          # High optimization
        "-std=c++17",                    # Use C++17 standard
        "--expt-relaxed-constexpr",  # CUDA specific flag
        "-lineinfo",  # Line information for debugging
    ],
    extra_ldflags=[""],
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90",  # Correct CUDA target arch
    ],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope = rope

    def forward(self, query_or_key, freqs_cos, freqs_sin):
        batch_size, seq_len, num_attention_heads, attention_head_dim = query_or_key.shape
        half_head_dim = attention_head_dim // 2
        return self.rope.rope_cuda(query_or_key, freqs_cos, freqs_sin, batch_size, seq_len, num_attention_heads, half_head_dim)

# </complete ModelNew code>
