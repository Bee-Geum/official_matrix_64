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
    tmp5 = tl.load(in_ptr0 + 2 * x3, tmp4, eviction_policy='evict_last',
        other=0.0)
    tmp6 = tl.load(in_ptr1 + x1, tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + x1, tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 - tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tl.full([1], 2, tl.int64)
    tmp15 = tl.load(in_ptr0 + (1 + 2 * x3), tmp12, eviction_policy=
        'evict_last', other=0.0)
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


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 128), (128, 1))
    assert_size_stride(arg1_1, (1, 128), (128, 1))
    assert_size_stride(arg2_1, (1, 128, 24, 2), (576, 48, 2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128, 24, 2), (576, 1, 24, 12), torch.
            float16)
        get_raw_stream(0)
        triton_poi_fused_stack_0[grid(576)](arg2_1, arg1_1, arg0_1, buf0, 
            576, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        del arg2_1
    return reinterpret_tensor(buf0, (1, 128, 48), (576, 48, 1), 0),


class RotaryEmbedding(nn.Module):

    def __init__(self, seq_len, embedding_dim, embedding_type='fixed', freq=
        'sine', embedding_init=None):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        self.freq = freq
        self.embedding_init = embedding_init
        if embedding_type == 'fixed':
            self.rotary_embedding = nn.Embedding(seq_len, embedding_dim)
            self.init_fixed_embedding()
        elif embedding_type == 'learnable':
            self.rotary_embedding = nn.Embedding(seq_len, embedding_dim)
            self.init_learnable_embedding()
        else:
            raise RuntimeError(
                'embedding_type must be either "fixed" or "learnable"')

    def init_fixed_embedding(self):
        freq = torch.linspace(0, 1, self.seq_len)
        if self.freq == 'sine':
            embedding = torch.sin(2 * math.pi * freq)
        elif self.freq == 'cos':
            embedding = torch.cos(2 * math.pi * freq)
        elif self.freq == 'exp':
            embedding = torch.exp(-freq)
        else:
            raise RuntimeError(
                'freq must be either "sine", "cos" or "exp"')
        self.rotary_embedding.weight.data.copy_(embedding)

    def init_learnable_embedding(self):
        nn.init.normal_(self.rotary_embedding.weight)
        if self.embedding_init is not None:
            self.rotary_embedding.weight.data.copy_(torch.from_numpy(
                self.embedding_init))

    def forward(self, q):
        return self.rotary_embedding(q)


class RotaryAttentionNew(nn.Module):

    def __init__(self, embedding_dim, num_heads, embedding_type='fixed',
        freq='sine', embedding_init=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding_type = embedding_type
        self.freq = freq
        self.embedding_init = embedding_init
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.pos_emb = RotaryEmbedding(self.seq_len, embedding_dim,
            embedding_type, freq, embedding_init)

    def forward(self, input_0):
        arg2_1 = self.pos_emb.rotary_embedding.weight
        arg1_1 = self.query_proj.weight
        arg0_1 = self.query_proj.bias
        arg4_1 = self.key_proj.weight
        arg3_1 = self.key_proj.bias
        arg6_1 = self.value_proj.weight
        arg5_1 = self.value_proj.bias
        arg7_1 = input_0
        output = call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
            arg6_1, arg7_1])
        return output[0]
