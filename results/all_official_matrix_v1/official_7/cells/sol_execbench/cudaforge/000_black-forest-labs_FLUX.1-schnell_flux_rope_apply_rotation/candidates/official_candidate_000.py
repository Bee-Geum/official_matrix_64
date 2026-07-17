# <complete ModelNew code>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for rotary position embedding
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rope_rotation_kernel(
    const __half* query_or_key, const __half* freqs_cos, const __half* freqs_sin,
    __half* output, int batch_size, int seq_len, int num_attention_heads, int half_head_dim) {
    
    int batch_idx = blockIdx.x / (seq_len * num_attention_heads);
    int seq_idx = (blockIdx.x % (seq_len * num_attention_heads)) / num_attention_heads;
    int head_idx = blockIdx.x % num_attention_heads;
    int half_dim_idx = threadIdx.x;

    if (batch_idx < batch_size && seq_idx < seq_len && head_idx < num_attention_heads && half_dim_idx < half_head_dim) {
        int qk_index = ((batch_idx * seq_len + seq_idx) * num_attention_heads + head_idx) * half_head_dim * 2 + half_dim_idx * 2;
        int freq_index = (seq_idx * half_head_dim + half_dim_idx) * 2;

        __half x1 = query_or_key[qk_index];
        __half x2 = query_or_key[qk_index + 1];
        __half cos1 = freqs_cos[freq_index];
        __half cos2 = freqs_cos[freq_index + 1];
        __half sin1 = freqs_sin[freq_index];
        __half sin2 = freqs_sin[freq_index + 1];

        __half out1 = x1 * cos1 - x2 * sin1;
        __half out2 = x1 * sin2 + x2 * cos2;

        int out_index = qk_index;
        output[out_index] = out1;
        output[out_index + 1] = out2;
    }
}

torch::Tensor rope_rotation_cuda(
    torch::Tensor query_or_key, torch::Tensor freqs_cos, torch::Tensor freqs_sin) {
    
    int batch_size = query_or_key.size(0);
    int seq_len = query_or_key.size(1);
    int num_attention_heads = query_or_key.size(2);
    int attention_head_dim = query_or_key.size(3);
    int half_head_dim = attention_head_dim / 2;

    auto output = torch::zeros_like(query_or_key);

    const int block_size = 256;
    const int num_blocks = seq_len * num_attention_heads * half_head_dim;

    rope_rotation_kernel<<<num_blocks, block_size>>>(
        query_or_key.data_ptr<__half>(), freqs_cos.data_ptr<__half>(), freqs_sin.data_ptr<__half>(),
        output.data_ptr<__half>(), batch_size, seq_len, num_attention_heads, half_head_dim);

    return output;
}
"""

cpp_src = (
    "torch::Tensor rope_rotation_cuda(torch::Tensor query_or_key, torch::Tensor freqs_cos, torch::Tensor freqs_sin);"
)

# Compile the inline CUDA code for rotary position embedding
rope_rotation = load_inline(
    name="rope_rotation",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["rope_rotation_cuda"],
    verbose=True,
    extra_cflags=["-I/usr/local/cuda/include", "-I/usr/local/cuda/include/thrust"],
    extra_ldflags=["-L/usr/local/cuda/lib64", "-lcudart"],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rope_rotation = rope_rotation

    def forward(self, query_or_key, freqs_cos, freqs_sin):
        return self.rope_rotation.rope_rotation_cuda(query_or_key, freqs_cos, freqs_sin)
