

import torch
import torch.optim as optim

class LCMAdam(optim.Optimizer):
    def __init__(self, params, lr=0.001, initial_lr=0.005, min_lr=0.001, max_lr_change=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # 初始化优化器，设置默认参数
        defaults = dict(lr=lr, initial_lr=initial_lr, min_lr=min_lr, max_lr_change=max_lr_change,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(LCMAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 初始化状态信息
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # 更新偏置一阶矩估计
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # 更新偏置二阶矩估计
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # 权重衰减
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # 曲率控制动量:1.计算了梯度的曲率，梯度的二次平均和梯度的一次平均的平方取差值。
                # 2.将曲率限制在一个范围内。它使用了 torch.min 函数来取曲率和 1.0 之间的较小值，这个操作确保了动量值不会超过 1.0。
                curvature = exp_avg_sq - exp_avg ** 2
                momentum = torch.min(torch.abs(curvature), torch.tensor(1.0))

                # 使用动量更新梯度，以便在梯度更新中引入动量的影响
                grad.mul_(momentum)

                # 更新梯度的向量关系：将梯度标准化（归一化）为单位向量
                grad /= torch.norm(grad)

                # 线性插值
                lr = self.linear_interpolation(group, state)

                # 步长
                denom = corrected_exp_avg_sq.sqrt().add_(group['eps'])

                # 步骤
                step_size = lr / bias_correction1
                p.data.addcdiv_(-step_size, corrected_exp_avg, denom)

        return loss

    def linear_interpolation(self, group, state):

        max_lr = group['initial_lr']
        min_lr = group['min_lr']
        max_lr_change = group['max_lr_change']
        # period = 1.0001
        period = 1.1

        t = state['step'] % period
        lr = max_lr - (max_lr - min_lr) * (t / period)
        lr = max(min_lr, lr - max_lr_change)  # 确保学习率不小于最小学习率

        return lr







