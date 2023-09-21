#!/usr/bin/env python3

"""
@author: xi
@since: 2023-08-10
"""

import math

import torch
from torch.optim import adamw as _adamw

__all__ = [
    'adamw',
    'adambelief',
    'adamx'
]


def _adam_prepare(params, states, state_prefixes, options):
    params_with_grad = []
    grads = []
    exp_avgs = []
    exp_avg_sqs = []
    max_exp_avg_sqs = []
    state_steps = []

    amsgrad = options['amsgrad']

    for p, state, prefix in zip(params, states, state_prefixes):
        if p.grad is None:
            continue
        params_with_grad.append(p)

        if p.grad.is_sparse:
            raise RuntimeError('AdamW does not support sparse gradients')
        grads.append(p.grad)

        if f'{prefix}step' not in state:
            state[f'{prefix}exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state[f'{prefix}exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if amsgrad:
                state[f'{prefix}max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state[f'{prefix}step'] = torch.tensor(0.)

        exp_avgs.append(state[f'{prefix}exp_avg'])
        exp_avg_sqs.append(state[f'{prefix}exp_avg_sq'])
        if amsgrad:
            max_exp_avg_sqs.append(state[f'{prefix}max_exp_avg_sq'])
        state_steps.append(state[f'{prefix}step'])

    return (
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    )


def adamw(params, states, state_prefixes, options):
    (params_with_grad,
     grads,
     exp_avgs,
     exp_avg_sqs,
     max_exp_avg_sqs,
     state_steps) = _adam_prepare(params, states, state_prefixes, options)

    lr = options['lr']
    weight_decay = options['weight_decay']
    beta1, beta2 = options['betas']
    amsgrad = options['amsgrad']
    eps = options['eps']

    _adamw.adamw(
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=False
    )


def adambelief(params, states, state_prefixes, options):
    (params_with_grad,
     grads,
     exp_avgs,
     exp_avg_sqs,
     max_exp_avg_sqs,
     state_steps,) = _adam_prepare(params, states, state_prefixes, options)

    lr = options['lr']
    weight_decay = options['weight_decay']
    beta1, beta2 = options['betas']
    amsgrad = options['amsgrad']
    eps = options['eps']

    for i, param in enumerate(params_with_grad):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        brief = (grad - exp_avg).square_()
        exp_avg_sq.mul_(beta2).add_(brief, alpha=1 - beta2)

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = math.sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        param.addcdiv_(exp_avg, denom, value=-step_size)


def adamx(params, states, state_prefixes, options):
    (params_with_grad,
     grads,
     exp_avgs,
     exp_avg_sqs,
     max_exp_avg_sqs,
     state_steps,) = _adam_prepare(params, states, state_prefixes, options)

    lr = options['lr']
    weight_decay = options['weight_decay']
    beta1, beta2 = options['betas']
    amsgrad = options['amsgrad']
    eps = options['eps']

    for i, param in enumerate(params_with_grad):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).add_((grad * exp_avg).abs_(), alpha=1 - beta2).add_(eps)

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = math.sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt)

        param.addcdiv_(exp_avg, denom, value=-step_size)
