import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)

    p = p * (1.0 - lr * wd)
    diff = exp_avg - grad
    update = diff * beta1 + grad
    can_update = update != 0.0
    update_sign = tl.where(update > 0.0, -lr, lr)
    p = p + update_sign * can_update
    exp_avg = diff * beta2 + grad

    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
):
    assert all(t.is_cuda for t in (p, grad, exp_avg))
    n_elements = p.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    update_fn_kernel[grid](
        p,
        grad,
        exp_avg,
        lr,
        wd,
        beta1,
        beta2,
        n_elements,
    )

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, grad, exp_avg, lr, wd, beta1, beta2):
        update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)
        return p, exp_avg

def test_update_fn():
    n_elements = 128
    p1 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    grad1 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    exp_avg1 = torch.zeros(n_elements, device='cuda', dtype=torch.float32)

    n_elements = 1024
    p2 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    grad2 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    exp_avg2 = torch.zeros(n_elements, device='cuda', dtype=torch.float32)

    lr = 0.01
    wd = 0.01
    beta1 = 0.9
    beta2 = 0.999

    update_fn(p1, grad1, exp_avg1, lr, wd, beta1, beta2)
    update_fn(p2, grad2, exp_avg2, lr, wd, beta1, beta2)

    results = {
        "test_case_1": (p1.clone(), exp_avg1.clone()),
        "test_case_2": (p2.clone(), exp_avg2.clone())
    }

    return results

result_gold = test_update_fn()