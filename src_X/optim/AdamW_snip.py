import math
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    """ Implements Sophia optimizer with AdamW and diagonal Hessian estimation.

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        hessian_k (int): Number of steps to use for estimating Hessian diagonal. Default: 10
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999,0.99), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1]  < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= betas[2]  < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[2]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))


        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)


    def step(self, closure=None,):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]


                # State initialization
                if len(state) == 0:

                    state['step'] = 0
                    state['hessian_step'] = 0

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['h'] = torch.zeros_like(p.data)
                    state['k'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2, beta3 = group['betas']

                state['step'] += 1
                state['hessian_step'] += 1
                hessian_estimate = self.hutchinson(p, grad)
                h = state['h']

                h.mul_(beta3).add_(1.0 - beta3, hessian_estimate)


                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                state['k'] = exp_avg_sq.sqrt().add_(1-beta3,h)

                denom = state['k'].add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction3 = 1.0 - beta3 ** state['step']
                    exp_avg_0 = exp_avg / bias_correction1
                    #exp_avg_1 = (1 - beta1) * (grad * beta1 + exp_avg_0)
                    exp_avg_1 = beta1 * exp_avg_0 + (1 - beta1) * grad
                    step_size = step_size -0.0001 * (exp_avg_1 / (denom/bias_correction3))
                    div_term = exp_avg_1.div(denom).mul(-step_size)

                    p.data.add_(div_term)



                # Calculate the division and multiply by -step_size



                # Add the result to p.data


                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

    def hutchinson(self, p, grad):
        u = torch.randn_like(grad, requires_grad=True)  # Ensure u requires gradients
        grad_dot_u = torch.sum(grad * u)
        hessian_vector_product = torch.autograd.grad(grad_dot_u, p, retain_graph=True, allow_unused=True)[0]

        if hessian_vector_product is None:
            hessian_vector_product = torch.zeros_like(p.data)

        return u * hessian_vector_product

        # Hessian diagonal estimation


        return loss
