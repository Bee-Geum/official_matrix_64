import torch

def mean_kernel_impl(*args, **kwargs):
    return torch.mean(args[0])
