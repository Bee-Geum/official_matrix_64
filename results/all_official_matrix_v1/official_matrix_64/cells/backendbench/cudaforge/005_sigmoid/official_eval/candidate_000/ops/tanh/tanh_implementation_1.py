import torch

def tanh_kernel_impl(*args, **kwargs):
    return torch.tanh(args[0])
