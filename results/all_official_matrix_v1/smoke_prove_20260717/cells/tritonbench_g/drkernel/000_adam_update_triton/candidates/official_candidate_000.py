import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,       # float32 scalar
    wd,       # float32 scalar
    beta1,    # float32 scalar
    beta2,    # float32 scalar
    n_elements: tl.constexpr,
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

    # 1) weight decay on p
    p = p * (1.0 - lr * wd)

    # 2) diff and update
    diff = e - g
    update = diff * beta1 + g

    # 3) sign-based step; move by +/- lr depending on sign of update
    # sign = 1 if update>0, -1 if update<0, 0 if update==0
    sign = tl.where(update > 0, 1.0, tl.where(update < 0, -1.0, 0.0))
    step = lr * sign
    can_update = update != 0
    p = p + step * can_update

    # 4) update exp_avg with beta2
    e_new = diff * beta2 + g

    # store
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, e_new, mask=mask)


class ModelNew:
    def __init__(self, lr: float, wd: float, beta1: float, beta2: float):
        self.lr = float(lr)
        self.wd = float(wd)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

    def update(self, p: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor):
        # Validate
        if not p.is_cuda or not grad.is_cuda or not exp_avg.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors")
        if p.dtype != torch.float32 or grad.dtype != torch.float32 or exp_avg.dtype != torch.float32:
            raise RuntimeError("ModelNew currently supports float32 only")
        if p.shape != grad.shape or p.shape != exp_avg.shape:
            raise RuntimeError("Shapes of p, grad, exp_avg must match")
        # Ensure contiguous for 1D kernel
        if not p.is_contiguous():
            p = p.contiguous()
        if not grad.is_contiguous():
            grad = grad.contiguous()
        if not exp_avg.is_contiguous():
            exp_avg = exp_avg.contiguous()

        n_elements = p.numel()

        # Grid: 1D
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        update_fn_kernel[grid](
            p,
            grad,
            exp_avg,
            self.lr,
            self.wd,
            self.beta1,
            self.beta2,
            n_elements,
        )

        return p, exp_avg


# Optional: quick test to verify numerical equivalence with the reference behavior
if __name__ == "__main__":
    # Setup
    n_elements = 1024
    p = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    grad = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    exp_avg = torch.zeros(n_elements, device='cuda', dtype=torch.float32)

    lr = 0.01
    wd = 0.01
    beta1 = 0.9
    beta2 = 0.999

    # Reference (original kernel)
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def ref_kernel(p_ptr, grad_ptr, exp_avg_ptr, lr, wd, beta1, beta2, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(p_ptr + offsets, mask=mask)
        g = tl.load(grad_ptr + offsets, mask=mask)
        e = tl.load(exp_avg_ptr + offsets, mask=mask)
        p = p * (1 - lr * wd)
        diff = e - g
        update = diff * beta1 + g
        sign = tl.where(update > 0, 1.0, tl.where(update < 0, -1.0, 0.0))
        step = lr * sign
        can = update != 0
        p = p + step * can
        e_new = diff * beta2 + g
        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(exp_avg_ptr + offsets, e_new, mask=mask)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    ref_kernel[grid](p, grad, exp_avg, lr, wd, beta1, beta2, n_elements)

    # Our ModelNew
    mn = ModelNew(lr, wd, beta1, beta2)
    p2 = p.clone()
    exp2 = exp_avg.clone()
    mn.update(p2, grad, exp2)

    # Compare
    max_p = (p - p2).abs().max().item()
    max_e = (exp_avg - exp2).abs().max().item()
    print(f"Max abs diff p: {max_p}, exp_avg: {max_e}")
