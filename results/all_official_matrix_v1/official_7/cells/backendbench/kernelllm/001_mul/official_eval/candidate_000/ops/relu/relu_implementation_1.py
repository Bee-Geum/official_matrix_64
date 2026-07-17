import torch

def relu_kernel_impl(*args, **kwargs):
    return torch.relu(args[0])
