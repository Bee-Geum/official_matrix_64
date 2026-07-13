import torch

def gelu_kernel_impl(*args, **kwargs):
    return torch.nn.functional.gelu(args[0])
