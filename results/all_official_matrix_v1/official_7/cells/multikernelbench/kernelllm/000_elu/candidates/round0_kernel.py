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
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tl.store(out_ptr0 + x0, tmp7, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_elu_0[grid(1592529216)](arg0_1, buf0, 1592529216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]