import torch

def softmax_kernel_impl(*args, **kwargs):
    return torch.softmax(args[0], dim=-1)
