import torch

def sum_kernel_impl(*args, **kwargs):
    return torch.sum(args[0])
