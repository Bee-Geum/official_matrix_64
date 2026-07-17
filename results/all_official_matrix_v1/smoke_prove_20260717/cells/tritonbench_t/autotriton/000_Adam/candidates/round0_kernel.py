import torch
import triton
import triton.language as tl

class AdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                if 'exp_avg_sq' not in state:
                    state['exp_avg_sq'] = torch.zeros_like(p)
                if 'step' not in state:
                    state['step'] = 0

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                p = p.contiguous()
                grad = grad.contiguous()
                exp_avg = exp_avg.contiguous()
                exp_avg_sq = exp_avg_sq.contiguous()

                n_elements = p.numel()
                if n_elements == 0:
                    continue

                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)

                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                adam_kernel[grid](
                    p, grad, exp_avg, exp_avg_sq,
                    lr, beta1, beta2, eps, bias_correction1, bias_correction2,
                    n_elements,
                    BLOCK_SIZE=1024,
                )

        return loss

@triton.jit
def adam_kernel(
    p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    lr, beta1, beta2, eps, bias_correction1, bias_correction2,
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
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

    exp_avg = exp_avg * beta1 + grad * (1 - beta1)
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2)

    denom = tl.sqrt(exp_avg_sq) / tl.sqrt(bias_correction2) + eps
    step_size = lr / bias_correction1
    update = exp_avg / denom * step_size
    p_new = p - update

    tl.store(p_ptr + offsets, p_new, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)

def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    return AdamOptimizer(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

def test_Adam():
    results = {}

    params1 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer1 = Adam(params1)
    results["test_case_1"] = optimizer1.defaults

    params2 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer2 = Adam(params2, lr=0.01)
    results["test_case_2"] = optimizer2.defaults

    params3 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer3 = Adam(params3, betas=(0.85, 0.95))
    results["test_case_3"] = optimizer3.defaults

    params4 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer4 = Adam(params4, weight_decay=0.01)
    results["test_case_4"] = optimizer4.defaults

    return results

test_results = test_Adam()