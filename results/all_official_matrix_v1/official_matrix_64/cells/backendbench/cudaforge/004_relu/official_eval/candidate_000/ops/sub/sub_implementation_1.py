import torch

def sub_kernel_impl(*args, **kwargs):
    return torch.sub(args[0], args[1])
