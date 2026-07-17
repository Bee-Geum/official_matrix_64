import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
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

    # Pointers
    p = tl.load(p_ptr + offsets, mask=mask)
    g = tl.load(grad_ptr + offsets, mask=mask)
    e = tl.load(exp_avg_ptr + offsets, mask=mask)

    # Weight decay
    p = p * (1.0 - lr * wd)

    # diff and update
    diff = e - g
    upd = diff * beta1 + g

    # sign-step with mask
    pos = upd > 0
    neg = upd < 0
    nonzero = upd != 0
    sign = tl.where(pos, -1.0, tl.where(neg, 1.0, 0.0))  # -1, +1, or 0
    p = p + sign * lr * nonzero  # broadcast over vector; zero when upd==0

    # update exp_avg
    e = diff * beta2 + g

    # store
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, e, mask=mask)


def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
):
    # Basic checks
    assert p.is_cuda and grad.is_cuda and exp_avg.is_cuda, "All tensors must be CUDA"
    assert p.dtype == torch.float32 and grad.dtype == torch.float32 and exp_avg.dtype == torch.float32, \
        "This implementation currently supports float32 tensors"
    assert p.shape == grad.shape == exp_avg.shape, "Shapes must match"
    assert p.is_contiguous() and grad.is_contiguous() and exp_avg.is_contiguous(), "Tensors must be contiguous"

    n_elements = p.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    update_fn_kernel[grid](
        p, grad, exp_avg,
        lr, wd, beta1, beta2,
        n_elements,
    )


# Optional: keep the same test helper for local validation
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

    return {
        "test_case_1": (p1.clone(), exp_avg1.clone()),
        "test_case_2": (p2.clone(), exp_avg2.clone())
    }
