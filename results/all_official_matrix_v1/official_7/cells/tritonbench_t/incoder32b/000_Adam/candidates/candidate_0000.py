import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def adam_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    param_size,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    stride_param,
    stride_grad,
    stride_exp_avg,
    stride_exp_avg_sq,
):
    pid = tl.program_id(0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < param_size
    
    # Load data
    param = tl.load(param_ptr + offsets * stride_param, mask=mask)
    grad = tl.load(grad_ptr + offsets * stride_grad, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets * stride_exp_avg, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets * stride_exp_avg_sq, mask=mask)
    
    # Adam update
    grad_with_wd = grad + weight_decay * param
    exp_avg_new = beta1 * exp_avg + (1 - beta1) * grad_with_wd
    exp_avg_sq_new = beta2 * exp_avg_sq + (1 - beta2) * grad_with_wd * grad_with_wd
    
    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    step_size = lr / (tl.sqrt(exp_avg_sq_new / bias_correction2) + eps)
    param_new = param - step_size * (exp_avg_new / bias_correction1)
    
    # Store results
    tl.store(param_ptr + offsets * stride_param, param_new, mask=mask)
    tl.store(exp_avg_ptr + offsets * stride_exp_avg, exp_avg_new, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets * stride_exp_avg_sq, exp_avg_sq_new, mask=mask)

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        self.step_count += 1
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                param_state = self.state[p]
                
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p)
                    param_state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']
                
                param_size = p.numel()
                grid = lambda meta: (triton.cdiv(param_size, 128),)
                
                adam_kernel[grid](
                    p, d_p, exp_avg, exp_avg_sq,
                    param_size,
                    lr, beta1, beta2, eps, weight_decay, self.step_count,
                    p.stride(0) if p.dim() > 0 else 1,
                    d_p.stride(0) if d_p.dim() > 0 else 1,
                    exp_avg.stride(0) if exp_avg.dim() > 0 else 1,
                    exp_avg_sq.stride(0) if exp_avg_sq.dim() > 0 else 1,
                )
                
                param_state['step'] += 1

def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    return Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

def test_Adam():
    results = {}

    # Test Case 1: Default parameters
    params1 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer1 = Adam(params1)
    results["test_case_1"] = optimizer1.defaults

    # Test Case 2: Custom learning rate
    params2 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer2 = Adam(params2, lr=0.01)
    results["test_case_2"] = optimizer2.defaults

    # Test Case 3: Custom betas
    params3 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer3 = Adam(params3, betas=(0.85, 0.95))
    results["test_case_3"] = optimizer3.defaults

    # Test Case 4: Custom weight decay
    params4 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer4 = Adam(params4, weight_decay=0.01)
    results["test_case_4"] = optimizer4.defaults

    return results

test_results = test_Adam()