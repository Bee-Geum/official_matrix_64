import torch
a=torch.randn(512,512,device='cuda'); b=torch.randn(512,512,device='cuda')
print((a@b).sum().item())
