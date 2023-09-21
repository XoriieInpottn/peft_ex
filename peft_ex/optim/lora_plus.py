#!/usr/bin/env python3

"""
@author: xi
@since: 2023-08-10
"""

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import init

from .common import *
from .common import PEFTOptimizer
from .functional import adamw, adambelief

__all__ = [
    'LoraPlusAdamW',
    'LoraPlusAdamBelief'
]


class LoraPlusOptimizer(PEFTOptimizer):

    def __init__(
            self,
            params,
            defaults,
            opt_fn,
            r,
            weight_dropout=0.1,
            flat_nd=False,
            square_2d=False,
            grad_norm='layer',
    ) -> None:
        self.rank = r
        self.weight_dropout = weight_dropout
        self.flat_nd = flat_nd
        self.square_2d = square_2d
        self.grad_norm = grad_norm
        super().__init__(
            params=params,
            defaults=defaults,
            opt_fn=opt_fn
        )

    def _init_peft(self, param: torch.Tensor, group):
        param_shape = param.shape
        num_dims = len(param_shape)
        state = None
        if num_dims == 2:
            state = {}
            if self.square_2d:
                target_shape = compute_flat_shape(param_shape)
                if target_shape != param_shape:
                    state['target_shape'] = target_shape
        elif num_dims > 2:
            if self.flat_nd:
                state = {'target_shape': compute_flat_shape(param_shape)}
        return state

    def _decompose(self, param, group, state):
        if 'sh' not in self.state:
            self.state['sh']['sh'] = torch.eye(self.rank, dtype=param.dtype, device=param.device, requires_grad=True)

        target_shape = state.get('target_shape')
        if target_shape is not None:
            param = param.reshape(target_shape)

        h, w = param.shape
        r = self.rank
        p0 = torch.clone(param)
        a = torch.zeros((h, r), dtype=param.dtype, device=param.device, requires_grad=True)
        b = torch.zeros((r, w), dtype=param.dtype, device=param.device, requires_grad=True)
        init.kaiming_uniform_(b, float(np.sqrt(5)))
        state['p0'] = p0
        state['a'] = a
        state['b'] = b

    def _compose(self, param: torch.Tensor, group, state):
        p0 = state['p0']
        a = state['a']
        b = state['b']
        assert 'sh' in self.state
        sh = self.state['sh']['sh']

        dp = a @ sh @ b

        if self.training and self.weight_dropout > 0:
            state['dropout'] = F.dropout(torch.ones_like(dp), self.weight_dropout)
            dp.mul_(state['dropout'])

        dp.add_(p0)

        if state.get('target_shape') is not None:
            dp = dp.reshape(param.shape)

        param[...] = dp

    def _gradient(self, param: torch.Tensor, group, state):
        g = param.grad
        param.grad = None

        target_shape = state.get('target_shape')
        if target_shape is not None:
            g = g.reshape(target_shape)

        if self.training and self.weight_dropout > 0:
            g.mul_(state['dropout'])
            del state['dropout']

        a = state['a']
        b = state['b']
        assert 'sh' in self.state
        sh = self.state['sh']['sh']

        g_ash = g @ b.T
        a.grad = g_ash @ sh.T
        b.grad = (a @ sh).T @ g
        g_sh = a.T @ g_ash
        if sh.grad is None:
            sh.grad = g_sh
        else:
            sh.grad.add_(g_sh)

        if self.grad_norm == 'layer':
            layer_grad_norm_(a, 0)
            layer_grad_norm_(b, 1)
        elif self.grad_norm == 'rms':
            rms_grad_norm_(a, 0)
            rms_grad_norm_(b, 1)

    def _update(self):
        super()._update()

        sh_state = self.state['sh']
        sh_param = sh_state['sh']
        for group in self.param_groups:
            if self.is_peft_group(group):
                self.opt_fn(
                    params=[sh_param],
                    states=[sh_state],
                    state_prefixes=['sh_'],
                    options=group
                )
                break
        sh_param.grad = None


class LoraPlusAdamW(LoraPlusOptimizer):

    def __init__(
            self,
            params,
            r,
            weight_dropout=0.1,
            flat_nd=False,
            square_2d=False,
            grad_norm='none',
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad,
        }
        super().__init__(
            params=params,
            defaults=defaults,
            opt_fn=adamw,
            r=r,
            weight_dropout=weight_dropout,
            flat_nd=flat_nd,
            square_2d=square_2d,
            grad_norm=grad_norm,
        )


class LoraPlusAdamBelief(LoraPlusOptimizer):

    def __init__(
            self,
            params,
            r,
            weight_dropout=0.1,
            flat_nd=False,
            square_2d=False,
            grad_norm='none',
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad,
        }
        super().__init__(
            params=params,
            defaults=defaults,
            opt_fn=adambelief,
            r=r,
            weight_dropout=weight_dropout,
            flat_nd=flat_nd,
            square_2d=square_2d,
            grad_norm=grad_norm,
        )
