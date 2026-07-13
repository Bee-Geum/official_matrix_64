import torch

def div_kernel_impl(*args, **kwargs):
    return torch.div(args[0], args[1])
