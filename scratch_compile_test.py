import os, time, torch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST","12.0")
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
__global__ void relu_kernel(const float* x, float* out, int size){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<size) out[i]=max(0.0f,x[i]);}
torch::Tensor relu_cuda(torch::Tensor x){auto o=torch::empty_like(x); int n=x.numel(); relu_kernel<<<(n+255)/256,256>>>(x.data_ptr<float>(),o.data_ptr<float>(),n); return o;}
'''
t=time.time()
print("compiling...", flush=True)
m=load_inline(name="relu_probe", cpp_sources="torch::Tensor relu_cuda(torch::Tensor x);", cuda_sources=src, functions=["relu_cuda"], verbose=True)
print(f"COMPILE OK in {time.time()-t:.1f}s", flush=True)
x=torch.randn(1024,device='cuda'); y=m.relu_cuda(x)
print("RUN OK, correct=", torch.allclose(y, torch.relu(x)), flush=True)
