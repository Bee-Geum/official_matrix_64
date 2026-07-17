# [official_matrix_64] torch>=2.11 compat shim: restore classic inductor `grid`
# (KernelLLM emits torch-inductor-style code; `grid` was refactored out of
#  torch._inductor.runtime.triton_heuristics in newer torch. Kernel logic below is verbatim.)
import triton as _triton
import torch._inductor.runtime.triton_heuristics as _th
if not hasattr(_th, "grid"):
    def grid(*numels):
        def grid_fn(meta):
            return tuple(_triton.cdiv(n, meta[b]) for n, b in zip(numels, ("XBLOCK", "YBLOCK", "ZBLOCK")))
        return grid_fn
    _th.grid = grid
# ---- KernelLLM output (verbatim) ----
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_div_mean_mul_pow_sub_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0010000072759311445
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 - tmp2
    tmp4 = tmp3 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.75
    tmp8 = tmp6 * tmp7
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tmp3 / tmp11
    tl.store(out_ptr0 + x0, tmp12, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 2), (2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 2), (2, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_div_mean_mul_pow_sub_0[grid(4)](arg0_1, buf0, 4,
            XBLOCK=4, num_warps=1, num_stages=1)
        del arg0_1
    return buf0,


def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )


class AdamOptNew(torch.nn.Module):

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
        weight_decay=0, activation='relu'):
        super(AdamOptNew, self).__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.activation = activation
        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr, betas=
            self.betas, eps=self.eps, weight_decay=self.weight_decay)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
