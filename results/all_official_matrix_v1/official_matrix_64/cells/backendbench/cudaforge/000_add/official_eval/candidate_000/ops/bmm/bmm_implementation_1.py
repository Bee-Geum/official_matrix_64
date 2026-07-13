import torch

def bmm_kernel_impl(*args, **kwargs):
    return torch.bmm(args[0], args[1])
