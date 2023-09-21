#!/usr/bin/env python3

"""
@author: xi
@since: 2023-08-10
"""

import math
from typing import Union

import torch
from torch.nn import functional as F
from torch.nn import init

from .common import *
from .functional import adamw

__all__ = [
    'STAdamW'
]


def l1_decay_(x, decay):
    xs = x.sign()
    x.sub_(xs * decay)
    x[x.sign() != xs] = 0
    return x


def l2_decay_(x, decay):
    return x.mul_(1 - decay)


def lp_decay_(x, decay, p: Union[int, float]):
    assert p >= 1
    if p == 2:
        return x.mul_(1 - decay)
    elif p == 1:
        return x.sub_(decay * x.sign())
    else:
        return x.sub_(decay * x.abs().pow_(p - 1).mul_(x.sign()))


class STOptimizer(PEFTOptimizer):

    def __init__(
            self,
            params,
            defaults,
            opt_fn,
            r: int,
            max_rank: int = 1024,  # 768, 1024
            extra_decay: float = 1.2,  # 1.0, 1.5
            extra_decay_norm: Union[int, float] = 2,  # 2
            grad_norm='layer',  # 'rms', 'layer', None
            weight_dropout=0.1,
    ) -> None:
        self.rank = r
        self.max_rank = max_rank
        self.extra_decay = extra_decay
        self.extra_decay_norm = extra_decay_norm
        self.grad_norm = grad_norm
        self.weight_dropout = weight_dropout
        assert self.grad_norm in {'none', 'layer', 'rms'}
        super().__init__(
            params=params,
            defaults=defaults,
            opt_fn=opt_fn
        )

    def _init_peft(self, param: torch.Tensor, group):
        param_shape = param.shape
        num_dims = len(param_shape)
        state = None
        if num_dims >= 2:
            state = {}
            target_shape = compute_flat_shape(param_shape)
            if target_shape != param_shape:
                state['target_shape'] = target_shape
        elif num_dims == 1:
            state = {}
        return state

    def _decompose(self, param: torch.Tensor, group, state):
        num_dims = len(param.shape)
        if num_dims >= 2:
            self._decompose_nd(param, group, state)
        elif num_dims == 1:
            self._decompose_1d(param, group, state)

    def _decompose_nd(self, param: torch.Tensor, group, state):
        if 'sh' not in self.state:
            self.state['sh']['sh'] = torch.eye(self.rank, dtype=param.dtype, device=param.device, requires_grad=True)

        target_shape = state.get('target_shape')
        if target_shape is not None:
            param = param.reshape(target_shape)

        max_rank = min(*param.shape, self.max_rank)
        u, s, v = torch.svd_lowrank(param, q=max_rank)
        vh = v.mT
        d = int(s.shape[0])

        u1 = u[:, :d]
        s1 = s[:d]
        vh1 = vh[:d, :]
        a = torch.zeros((d, self.rank), dtype=param.dtype, device=param.device, requires_grad=True)
        b = torch.zeros((self.rank, d), dtype=param.dtype, device=param.device, requires_grad=True)
        std = math.sqrt(1.0 / d)
        init.trunc_normal_(b, 0, std, -2 * std, 2 * std)

        state['p0'] = param.clone()
        state['u1'] = u1
        state['s1'] = s1
        state['vh1'] = vh1
        state['a'] = a
        state['b'] = b

    def _decompose_1d(self, param: torch.Tensor, group, state):
        state['p0'] = param.clone()
        state['p1'] = torch.zeros_like(param, requires_grad=True)

    def _compose(self, param: torch.Tensor, group, state):
        num_dims = len(param.shape)
        if num_dims >= 2:
            self._compose_nd(param, group, state)
        elif num_dims == 1:
            self._compose_1d(param, group, state)

    def _compose_nd(self, param: torch.Tensor, group, state):
        p0 = state['p0']
        u1 = state['u1']
        vh1 = state['vh1']
        a = state['a']
        b = state['b']
        assert 'sh' in self.state
        sh = self.state['sh']['sh']

        dp = (u1 @ a) @ sh @ (b @ vh1)

        if self.training and self.weight_dropout > 0:
            state['dropout'] = F.dropout(torch.ones_like(dp), self.weight_dropout)
            dp.mul_(state['dropout'])

        dp.add_(p0)

        if state.get('target_shape') is not None:
            dp = dp.reshape(param.shape)

        param[...] = dp

    def _compose_1d(self, param: torch.Tensor, group, state):
        p0 = state['p0']
        p1 = state['p1']

        dp = p1.clone()

        if self.training and self.weight_dropout > 0:
            state['dropout'] = F.dropout(torch.ones_like(dp), self.weight_dropout)
            dp.mul_(state['dropout'])

        param[...] = dp.add_(p0)

    def _gradient(self, param: torch.Tensor, group, state):
        num_dims = len(param.shape)
        if num_dims >= 2:
            self._gradient_nd(param, group, state)
        elif num_dims == 1:
            self._gradient_1d(param, group, state)

    def _gradient_nd(self, param: torch.Tensor, group, state):
        g = param.grad
        # param.grad = None

        target_shape = state.get('target_shape')
        if target_shape is not None:
            g = g.reshape(target_shape)

        if self.training and self.weight_dropout > 0:
            g.mul_(state['dropout'])
            del state['dropout']

        u1 = state['u1']
        vh1 = state['vh1']
        a = state['a']
        b = state['b']
        assert 'sh' in self.state
        sh = self.state['sh']['sh']

        ua = u1 @ a
        uah = ua @ sh
        bv = b @ vh1

        g_uah = g @ bv.T
        g_ua = g_uah @ sh.T
        g_bv = uah.T @ g

        a.grad = u1.T @ g_ua
        b.grad = g_bv @ vh1.T
        if sh.grad is None:
            sh.grad = ua.T @ g_uah
        else:
            sh.grad.add_(ua.T @ g_uah)

        if self.grad_norm == 'layer':
            layer_grad_norm_(a, 0)
            layer_grad_norm_(b, 1)
        elif self.grad_norm == 'rms':
            rms_grad_norm_(a, 0)
            rms_grad_norm_(b, 1)

    def _gradient_1d(self, param: torch.Tensor, group, state):
        g = param.grad
        param.grad = None

        if self.training and self.weight_dropout > 0:
            g.mul_(state['dropout'])
            del state['dropout']

        p1 = state['p1']
        p1.grad = g

    def _update(self):
        for group in self.param_groups:
            if not self.is_peft_group(group):
                continue
            for p in group['params']:
                if (state := self.state.get(p)) is None:
                    continue
                self._extra_decay(p, group, state)

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

    def _extra_decay(self, param, group, state):
        num_dims = len(param.shape)
        if num_dims >= 2:
            self._extra_decay_nd(param, group, state)
        elif num_dims == 1:
            self._extra_decay_1d(param, group, state)

    def _extra_decay_nd(self, param, group, state):
        a = state['a']
        b = state['b']
        s1 = state['s1']

        weight_decay = group['weight_decay']
        lr = group['lr']

        singular_decay = 1 - s1 / s1.max()
        weight_decay = self.extra_decay * weight_decay
        lp_decay_(a, lr * weight_decay * singular_decay[:, None], p=self.extra_decay_norm)
        lp_decay_(b, lr * weight_decay * singular_decay, p=self.extra_decay_norm)

    def _extra_decay_1d(self, param, group, state):
        p1 = state['p1']

        weight_decay = group['weight_decay']
        lr = group['lr']

        weight_decay = self.extra_decay * weight_decay
        lp_decay_(p1, lr * weight_decay, p=self.extra_decay_norm)


class STAdamW(STOptimizer):

    def __init__(
            self,
            params,
            r: int,
            max_rank: int = 1024,  # 768, 1024
            extra_decay: float = 1.2,  # 1.0, 1.5
            extra_decay_norm: Union[int, float] = 2,  # 2
            grad_norm='layer',  # 'rms', 'layer', None
            weight_dropout=0.1,
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
            max_rank=max_rank,
            extra_decay=extra_decay,
            extra_decay_norm=extra_decay_norm,
            grad_norm=grad_norm,
            weight_dropout=weight_dropout,
        )
