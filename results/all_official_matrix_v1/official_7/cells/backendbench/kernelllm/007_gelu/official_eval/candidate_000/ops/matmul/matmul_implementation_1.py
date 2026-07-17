import torch

def matmul_kernel_impl(*args, **kwargs):
    return torch.matmul(args[0], args[1])
