import torch

def mm_kernel_impl(*args, **kwargs):
    return torch.mm(args[0], args[1])
