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
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_add_pow_sqrt_sub_0(in_ptr0, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_ptr0 + (64 + r0), None)
    tmp4 = tl.load(in_ptr0 + (128 + r0), None)
    tmp8 = tl.load(in_ptr0 + (192 + r0), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp5 = tmp0 - tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = triton_helpers.min2(tmp13, 1)[:, None]
    tl.store(out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp15, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_pow_sqrt_sub_0[grid(1)](arg0_1, buf0, 1, 64,
            XBLOCK=1, num_warps=2, num_stages=1)
        del arg0_1
    return buf0,


class Point:
    """
    A point in 2D space.
    """

    def __init__(self, x: 'int', y: 'int'):
        self.x = x
        self.y = y


class DistanceNew(nn.Module):
    """
    A module for calculating distance between two points.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
