#!/usr/bin/env python3

"""
@author: xi
@since: 2023-08-10
"""

from typing import Tuple

import numpy as np
import torch
from torch.optim import Optimizer

__all__ = [
    'PEFTOptimizer',
    'compute_flat_shape',
    'layer_grad_norm_',
    'rms_grad_norm_',
]


class PEFTOptimizer(Optimizer):

    @staticmethod
    def is_peft_group(group):
        if 'tag' in group:
            tag = group['tag']
            if isinstance(tag, str):
                return tag.lower() in {'peft', 'low_rank'}
            else:
                return 'peft' in tag or 'low_rank' in tag
        elif 'peft' in group:
            return group['peft']
        elif 'low_rank' in group:
            return group['low_rank']
        else:
            return False

    def __init__(self, params, defaults, opt_fn) -> None:
        super().__init__(params, defaults)
        self.opt_fn = opt_fn
        self.training = True

        with torch.no_grad():
            for group in self.param_groups:
                if not self.is_peft_group(group):
                    continue

                new_params = []
                for p in group["params"]:
                    if len(p.shape) != 1:
                        continue
                    state = self._init_peft(p, group)
                    self.state[p] = state
                    if state is None:
                        continue
                    self.state[p] = state
                    new_params.append(p)
                    self._decompose(p, group, state)
                    self._compose(p, group, state)
                # group["params"] = new_params

            for group in self.param_groups:
                if not self.is_peft_group(group):
                    continue

                new_params = []
                for p in group["params"]:
                    if len(p.shape) == 1:
                        continue
                    state = self._init_peft(p, group)
                    self.state[p] = state
                    if state is None:
                        continue
                    self.state[p] = state
                    new_params.append(p)
                    self._decompose(p, group, state)
                    self._compose(p, group, state)
                # group["params"] = new_params

    def _init_peft(self, param: torch.Tensor, group):
        raise NotImplementedError()

    def _decompose(self, param: torch.Tensor, group, state):
        raise NotImplementedError()

    @torch.no_grad()
    def train(self, training=True):
        self.training = training
        for group in self.param_groups:
            if not self.is_peft_group(group):
                continue
            for p in group['params']:
                if p not in self.state:
                    continue
                state = self.state[p]
                if state is None:
                    continue
                self._compose(p, group, state)

    def eval(self):
        self.train(False)

    @torch.no_grad()
    def step(self, closure=None):
        self._pre_update()
        self._update()
        self._post_update()

    def _pre_update(self):
        for group in self.param_groups:
            if not self.is_peft_group(group):
                continue
            for p in group['params']:
                if p not in self.state:
                    state = self._init_peft(p, group)
                    self.state[p] = state
                    if state is None:
                        continue
                    self._decompose(p, group, state)
                    self._compose(p, group, state)

                state = self.state[p]
                if state is None:
                    continue
                self._gradient(p, group, state)

    def _update(self):
        for group in self.param_groups:
            if not self.is_peft_group(group):
                params = group['params']
                states = [self.state[p] for p in params]
                state_prefixes = [''] * len(params)
            else:
                params = []
                states = []
                state_prefixes = []
                for p in group['params']:
                    state = self.state[p]
                    if state is None:
                        continue
                    for name, factor in state.items():
                        if isinstance(factor, torch.Tensor) and factor.requires_grad:
                            params.append(factor)
                            states.append(state)
                            state_prefixes.append(name + '_')

            self.opt_fn(
                params=params,
                states=states,
                state_prefixes=state_prefixes,
                options=group
            )

    def _post_update(self):
        for group in self.param_groups:
            if not self.is_peft_group(group):
                continue
            for p in group['params']:
                state = self.state[p]
                if state is None:
                    continue
                self._compose(p, group, state)

    # @staticmethod
    # def _adamw_step(params, states, state_prefixes, options):
    #     params_with_grad = []
    #     grads = []
    #     exp_avgs = []
    #     exp_avg_sqs = []
    #     max_exp_avg_sqs = []
    #     state_steps = []
    #
    #     amsgrad = options['amsgrad']
    #     beta1, beta2 = options['betas']
    #
    #     for p, state, prefix in zip(params, states, state_prefixes):
    #         if p.grad is None:
    #             continue
    #         params_with_grad.append(p)
    #
    #         if p.grad.is_sparse:
    #             raise RuntimeError('AdamW does not support sparse gradients')
    #         grads.append(p.grad)
    #
    #         if f'{prefix}step' not in state:
    #             state[f'{prefix}exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
    #             state[f'{prefix}exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
    #             if amsgrad:
    #                 state[f'{prefix}max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
    #             state[f'{prefix}step'] = torch.tensor(0.)
    #
    #         exp_avgs.append(state[f'{prefix}exp_avg'])
    #         exp_avg_sqs.append(state[f'{prefix}exp_avg_sq'])
    #         if amsgrad:
    #             max_exp_avg_sqs.append(state[f'{prefix}max_exp_avg_sq'])
    #         state_steps.append(state[f'{prefix}step'])
    #
    #     adamw.adamw(
    #         params_with_grad,
    #         grads,
    #         exp_avgs,
    #         exp_avg_sqs,
    #         max_exp_avg_sqs,
    #         state_steps,
    #         amsgrad=amsgrad,
    #         beta1=beta1,
    #         beta2=beta2,
    #         lr=options['lr'],
    #         weight_decay=options['weight_decay'],
    #         eps=options['eps'],
    #         maximize=False
    #     )

    def _compose(self, param: torch.Tensor, group, state):
        raise NotImplementedError()

    def _gradient(self, param: torch.Tensor, group, state):
        raise NotImplementedError()


def compute_flat_shape(shape: Tuple) -> Tuple:
    size = int(np.prod(shape))
    factor = int(np.sqrt(size))
    while size % factor != 0:
        factor -= 1
    target_shape = (factor, size // factor)
    return target_shape


def layer_grad_norm_(p: torch.Tensor, dim: int, eps=1e-8):
    g = p.grad
    mu = g.mean(dim, keepdims=True)
    sigma = (g - mu).square_().mean(dim, keepdims=True).add_(eps).rsqrt_()
    g.sub_(mu).mul_(sigma)
    # sigma = (g - mu).square_().mean(dim, keepdims=True).sqrt_().add_(eps)
    # g.sub_(mu).div_(sigma)


def rms_grad_norm_(p: torch.Tensor, dim: int, eps=1e-8):
    g = p.grad
    rms = g.square().mean(dim, keepdim=True).add_(eps).rsqrt_()
    g.mul_(rms)
    # rms = g.square().mean(dim, keepdim=True).sqrt_().add_(eps)
    # g.div_(rms)
