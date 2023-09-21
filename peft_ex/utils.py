#!/usr/bin/env python3

"""
@author: xi
@since: 2023-09-06
"""

import torch

__all__ = [
    'layer_grad_norm_',
    'rms_grad_norm_'
]


def layer_grad_norm_(g: torch.Tensor, dim: int, eps=1e-8):
    mu = g.mean(dim, keepdims=True)
    sigma = (g - mu).square_().mean(dim, keepdims=True).add_(eps).rsqrt_()
    g.sub_(mu).mul_(sigma)
    # sigma = (g - mu).square_().mean(dim, keepdims=True).sqrt_().add_(eps)
    # g.sub_(mu).div_(sigma)


def rms_grad_norm_(g: torch.Tensor, dim: int, eps=1e-8):
    rms = g.square().mean(dim, keepdim=True).add_(eps).rsqrt_()
    g.mul_(rms)
    # rms = g.square().mean(dim, keepdim=True).sqrt_().add_(eps)
    # g.div_(rms)
