import torch
import math
from torch.optim.optimizer import Optimizer

class AdamW_AGC(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, clip_threshold=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, clip_threshold=clip_threshold)
        #self.noise_stddev = noise_stddev
        super(AdamW_AGC, self).__init__(params, defaults)

    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW_AGC does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction1

                # Compute bias-corrected second raw moment estimate
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Apply adaptive gradient clipping
                g_norm = torch.norm(grad)
                p_norm = torch.norm(p.data)
                max_norm = p_norm * group['clip_threshold']
                clip_coef = max_norm / (g_norm + 1e-6)
                if clip_coef < 1:
                    grad = grad * clip_coef

                # Add Gaussian noise
                #noise = torch.normal(0.0, self.noise_stddev, p.data.size())
                #if torch.cuda.is_available():
                    #noise = noise.cuda()

                #p.data.addcdiv_(exp_avg, denom, value=-step_size)
                #p.data.add_(noise)
                # Update parameters
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
