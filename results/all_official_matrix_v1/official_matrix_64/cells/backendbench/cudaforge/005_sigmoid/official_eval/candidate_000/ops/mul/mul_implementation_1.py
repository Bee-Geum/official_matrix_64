import torch

def mul_kernel_impl(*args, **kwargs):
    return torch.mul(args[0], args[1])
