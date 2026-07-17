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
```
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_stack_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 2
    x3 = xindex // 2
    x1 = xindex // 2 % 128
    x4 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + x3, tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + x1, tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + x1, tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 - tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tl.full([1], 2, tl.int64)
    tmp15 = tl.load(in_ptr0 + x3, tmp12, eviction_policy='evict_last', other=0.0
        )
    tmp16 = tl.load(in_ptr1 + x1, tmp12, eviction_policy='evict_last', other=0.0
        )
    tmp17 = tl.load(in_ptr2 + x1, tmp12, eviction_policy='evict_last', other=0.0
        )
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + x4, tmp22, None)


@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 256
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 128 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tl.store(out_ptr1 + (y0 + 128 * x2 + 16384 * y1), tmp1, xmask & ymask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 128, 24, 128), (393216, 3072, 128, 1))
    assert_size_stride(arg1_1, (128, 128), (128, 1))
    assert_size_stride(arg2_1, (128, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128, 24, 2), (16384, 128, 2, 1),
            torch.float16)
        get_raw_stream(0)
        triton_poi_fused_stack_0[grid(16384)](arg0_1, arg1_1, arg2_1, buf0,
            16384, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        del arg2_1
        buf2 = empty_strided_cuda((1, 128, 128), (16384, 128, 1), torch.float16)
        triton_poi_fused__to_copy_1[grid(256, 128)](buf0, buf2, 256, 128,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del buf0
    return reinterpret_tensor(buf2, (1, 128, 24, 2), (16384, 128, 2, 1), 0),


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, embed_dim, freq='fixed', freq_factor=8, eps=1e-05):
        super().__init__()
        self.embed_dim = embed_dim
        self.dim = dim
        self.freq_factor = freq_factor
        self.eps = eps
        self.freq = torch.nn.Parameter(torch.Tensor(1, dim, dim))
        if freq == 'random':
            torch.nn.init.normal_(self.freq, std=0.02)
        elif freq == 'fixed':
            self.freq.data.fill_(1 / self.freq_factor)
        self.register_buffer('freq', self.freq)
        self.register_buffer('freq_inv', torch.linalg.inv(self.freq))

    def forward(self, x):
        device = x.device
        if self.dim == self.embed_dim:
            return x
        freq = self.freq
        if self.embed_dim % self.dim != 0:
            raise ValueError(
                'embed_dim must be divisible by dim for rotary embedding')
        freq = freq[:, :self.embed_dim]
        embed = x * freq
        embed = torch.stack(torch.split(embed, self.dim, dim=-1))
        cos, sin = torch.chunk(embed, 2, dim=-1)
        cos = torch.cos(cos * self.freq)
        sin = torch.sin(sin * self.freq)
        z = torch.zeros_like(cos)
        z[..., 0] = cos
        z[..., 1] = sin
        embed = torch.where(x[..., 0] < self.eps, z, embed)
        return embed


class RotaryPositionalEmbedding(RotaryEmbedding):

    def __init__(self, dim, embed_dim, freq='fixed', freq_factor=8, eps=1e-05):
        super().__init__(dim, embed_dim, freq, freq_factor, eps)
        self.register_buffer('pos', torch.arange(dim))

    def forward(self, x):
        device = x.device
        if self.dim == self.embed_dim:
            return x
        freq = self.freq
        if self.embed_dim % self.dim != 0:
            raise ValueError(
                'embed_dim must be divisible by dim for rotary embedding')
        freq = freq[:, :self.embed_dim]
        pos = self.pos
        embed = x * freq
        embed = torch.stack(torch.split(embed, self.dim, dim=-1))
        cos, sin = torch.chunk(embed, 2, dim=-1)
        cos = torch.cos(pos * freq)
        sin = torch.sin(pos * freq)
        z = torch.zeros_like(cos)
        z[..., 0] = cos
        z[..., 1] = sin
        embed = torch.where(x[..., 0] < self.eps, z, embed)
        return embed


class RotaryMultiHeadAttentionNew(nn.Module):

    def __init__(self, dim, n_head, n_model, dropout=0.1, rotary_pos_emb=
        False, block_length=None):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.n_model = n_model
        self.dropout = nn.Dropout(dropout)
        self.block_length = block_length
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.rotary_pos_emb = rotary_pos_emb
        if rotary_pos_emb:
            self.pos_embed = RotaryPositionalEmbedding(dim, dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, input_0):
        arg1_1 = self.query.weight
