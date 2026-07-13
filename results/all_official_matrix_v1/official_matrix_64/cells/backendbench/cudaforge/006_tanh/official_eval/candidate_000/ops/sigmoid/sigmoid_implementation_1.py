import torch

def sigmoid_kernel_impl(*args, **kwargs):
    return torch.sigmoid(args[0])
