import torch

def add_kernel_impl(*args, **kwargs):
    return torch.add(args[0], args[1])
