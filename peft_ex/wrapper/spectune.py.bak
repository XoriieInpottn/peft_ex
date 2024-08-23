#!/usr/bin/env python3

"""
@author: xi
@since: 2023-09-06
"""

import math
import re
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Union, Mapping, Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTuner
from torch import nn
from torch.nn import init

from .lora_ex import Decomposition, LoraExLayer
from .utils import *

__all__ = [
    'STConfig',
    'STLayer',
    'STModel',
    'STLinear',
    'STLayerNorm',
]


@dataclass
class STConfig(PeftConfig):
    target_modules: Optional[Union[List[str], str]] = field(default=None)
    r: int = field(default=2)
    max_rank: int = field(default=1024)
    weight_dropout: float = field(default=0.1)
    extra_decay: float = field(default=1.2)
    grad_norm: str = field(default='layer')
    grad_norm_eps: float = field(default=1e-8)
    sh: bool = field(default=True)
    init_weights: bool = field(default=True)


class FlatLikeSquare(object):

    def __init__(self, shape, ignore_2d=False):
        shape = tuple(int(s) for s in shape)
        self.source_shape = shape

        if (not ignore_2d) or len(shape) != 2:
            size = int(np.prod(shape))
            factor = int(np.sqrt(size))
            while size % factor != 0:
                factor -= 1
            self.target_shape = (factor, size // factor)
        else:
            self.target_shape = shape

        self.need_reshape = self.source_shape != self.target_shape

    def __call__(self, x, inverse=False):
        if inverse:
            if self.need_reshape:
                x = x.reshape(self.source_shape)
        else:
            if self.need_reshape:
                x = x.reshape(self.target_shape)
        return x


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


def _find_lr_and_weight_decay(context: Mapping[str, Any]):
    lr, weight_decay = None, None

    if 'optimizer' in context:
        largest_group = max([
            (group, len(group['params']))
            for group in context['optimizer'].param_groups
        ], key=lambda _x: _x[1])[0]
        lr = largest_group['lr']
        weight_decay = largest_group['weight_decay']

    if 'lr' in context:
        lr = context['lr']
    if 'weight_decay' in context:
        weight_decay = context['weight_decay']

    if lr is None or weight_decay is None:
        raise RuntimeError('You should bind an optimizer to the STModel by calling bind_optimizer().')
    return lr, weight_decay


class STDecompositionNd(Decomposition):

    def __init__(self, module, param_name, config: STConfig, context):
        self.module = module
        self.param_name = param_name
        self.config = config
        self.context = context

        self.a = nn.ParameterDict()
        self.b = nn.ParameterDict()

        setattr(module, f'{param_name}_spectune_A', self.a)
        setattr(module, f'{param_name}_spectune_B', self.b)

        if self.config.sh:
            self.sh = nn.ParameterDict()
            setattr(module, f'{param_name}_spectune_H', self.sh)

        with torch.no_grad():
            param = getattr(self.module, self.param_name)
            self.flat = FlatLikeSquare(param.shape)
            flat_param = self.flat(param)
            max_rank = min(*flat_param.shape, self.config.max_rank)
            u, s, v = torch.svd_lowrank(flat_param, q=max_rank)
            vh = v.mT
            d = int(s.shape[0])
            self.u = nn.Parameter(u[:, :d], requires_grad=False)
            self.s = nn.Parameter(s[:d], requires_grad=False)
            self.vh = nn.Parameter(vh[:d, :], requires_grad=False)
            setattr(module, f'{param_name}_spectune_U', self.u)
            setattr(module, f'{param_name}_spectune_S', self.s)
            setattr(module, f'{param_name}_spectune_V', self.vh)

    def decompose(self, adapter_name):
        r = self.config.r
        d = self.s.shape[0]
        param = getattr(self.module, self.param_name)
        dtype = param.dtype
        device = param.device

        a = nn.Parameter(torch.empty((d, r), dtype=dtype, device=device), requires_grad=True)
        b = nn.Parameter(torch.empty((r, d), dtype=dtype, device=device), requires_grad=True)

        if self.config.init_weights:
            init.zeros_(a)
            std = math.sqrt(1.0 / d)
            init.trunc_normal_(b, 0, std, -2 * std, 2 * std)

        if self.config.grad_norm != 'none':
            a.register_hook(partial(self._grad_hook, param=a, dim=0, eps=self.config.grad_norm_eps))
            b.register_hook(partial(self._grad_hook, param=b, dim=1, eps=self.config.grad_norm_eps))

        self.a[adapter_name] = a
        self.b[adapter_name] = b

        if self.config.sh:
            if adapter_name not in self.context:
                sh = nn.Parameter(torch.eye(r, dtype=dtype, device=device), requires_grad=True)
                self.context[adapter_name] = sh
            self.sh[adapter_name] = self.context[adapter_name]

    def compose(self, adapter_name):
        u = self.u
        vh = self.vh
        a = self.a[adapter_name]
        b = self.b[adapter_name]
        sh = self.sh[adapter_name] if self.config.sh else None

        u1 = u @ a
        v1 = b @ vh
        dp = u1 @ v1 if sh is None else u1 @ sh @ v1
        if self.module.training:
            dp = F.dropout(dp, self.config.weight_dropout)
        dp = self.flat(dp, inverse=True)
        return dp

    def _grad_hook(self, grad, param, dim, eps):
        with torch.no_grad():
            s1 = self.s
            singular_decay = 1 - s1 / s1.max()
            if dim == 0:
                singular_decay = singular_decay[:, None]
            lr, weight_decay = _find_lr_and_weight_decay(self.context)
            l2_decay_(param, lr * self.config.extra_decay * weight_decay * singular_decay)

            if self.config.grad_norm == 'layer':
                layer_grad_norm_(grad, dim, eps=eps)
            elif self.config.grad_norm == 'rms':
                rms_grad_norm_(grad, dim, eps=eps)
        return grad

    def merge(self, adapter_name):
        param = getattr(self.module, self.param_name)
        param.add_(self.compose(adapter_name))

    def unmerge(self, adapter_name):
        param = getattr(self.module, self.param_name)
        param.sub_(self.compose(adapter_name))


class STDecomposition1d(Decomposition):

    def __init__(self, module, param_name, config: STConfig, context):
        self.module = module
        self.param_name = param_name
        self.config = config
        self.context = context

        self.delta = nn.ParameterDict()
        setattr(module, f'{param_name}_spectune_D', self.delta)

    def decompose(self, adapter_name):
        param = getattr(self.module, self.param_name)
        delta = nn.Parameter(torch.empty_like(param), requires_grad=True)

        if self.config.init_weights:
            init.zeros_(delta)

        delta.register_hook(partial(self._grad_hook, adapter_name=adapter_name))

        self.delta[adapter_name] = delta

    def compose(self, adapter_name):
        delta = self.delta[adapter_name]
        if self.module.training:
            delta = F.dropout(delta, self.config.weight_dropout)
        return delta

    def _grad_hook(self, grad, adapter_name):
        param = self.delta[adapter_name]
        lr, weight_decay = _find_lr_and_weight_decay(self.context)
        l2_decay_(param, lr * self.config.extra_decay * weight_decay)

    def merge(self, adapter_name):
        param = getattr(self.module, self.param_name)
        param.add_(self.compose(adapter_name))

    def unmerge(self, adapter_name):
        param = getattr(self.module, self.param_name)
        param.sub_(self.compose(adapter_name))


class STLayer(LoraExLayer):
    pass


class STModel(BaseTuner):

    def __init__(self, model, config, adapter_name):
        self.context = {}
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _check_target_module_exists(config, key):
        if isinstance(config.target_modules, str):
            target_module_found = re.fullmatch(config.target_modules, key)
        else:
            target_module_found = any(re.fullmatch(target_key, key) for target_key in config.target_modules)
        return target_module_found

    def _mark_only_adapters_as_trainable(self) -> None:
        def _set(m):
            if isinstance(m, STLayer):
                return
            for child in m.children():
                _set(child)
            for p in m.parameters(False):
                p.requires_grad = False

        _set(self)

    def _create_and_replace(
            self,
            config,
            adapter_name,
            target,
            target_name,
            parent,
            **optional_kwargs,
    ):
        if isinstance(target, STLayer):
            target.add_adapter(adapter_name)
        else:
            if isinstance(target, nn.Linear):
                new_module = STLinear(
                    adapter_name=adapter_name,
                    config=config,
                    context=self.context,
                    in_features=target.in_features,
                    out_features=target.out_features,
                    bias=target.bias is not None,
                    device=target.weight.device,
                    dtype=target.weight.dtype
                )
                self._replace_module(parent, new_module, target_name, target, ['weight', 'bias'])
            elif isinstance(target, nn.LayerNorm):
                new_module = STLayerNorm(
                    adapter_name=adapter_name,
                    config=config,
                    context=self.context,
                    normalized_shape=target.normalized_shape,
                    eps=target.eps,
                    elementwise_affine=target.elementwise_affine,
                    device=target.weight.device if target.weight is not None else None,
                    dtype=target.weight.dtype if target.weight is not None else None
                )
                self._replace_module(parent, new_module, target_name, target, ['weight', 'bias'])
            elif isinstance(target, nn.Conv2d):
                new_module = STConv2d(
                    adapter_name=adapter_name,
                    config=config,
                    context=self.context,
                    in_channels=target.in_channels,
                    out_channels=target.out_channels,
                    kernel_size=target.kernel_size,
                    stride=target.stride,
                    padding=target.padding,
                    dilation=target.dilation,
                    groups=target.groups,
                    bias=target.bias is not None,
                    padding_mode=target.padding_mode,
                    device=target.weight.device,
                    dtype=target.weight.dtype
                )
                self._replace_module(parent, new_module, target_name, target, ['weight', 'bias'])

    @staticmethod
    def _replace_module(
            parent: nn.Module,
            new_module: nn.Module,
            target_name: str,
            target: nn.Module,
            param_names: List[str]
    ):
        device = None
        for param_name in param_names:
            if hasattr(target, param_name):
                param = getattr(target, param_name)
                if param is not None:
                    getattr(new_module, param_name).data = param
                    device = param.device
        if hasattr(target, 'state'):
            new_module.state = target.state
        if device is not None:
            new_module.to(device)
        setattr(parent, target_name, new_module)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, STLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, STLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def _prepare_adapter_config(self, peft_config, model_config):
        return peft_config

    def merge(self):
        for module in self.model.modules():
            if isinstance(module, LoraExLayer):
                module.merge()

    def merge_and_unload(self):
        raise NotImplementedError()


class STLinear(nn.Linear, STLayer):

    def __init__(
            self,
            adapter_name: str,
            config: STConfig,
            context,
            in_features: int,
            out_features: int,
            bias,
            device,
            dtype,
    ):
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        STLayer.__init__(self)
        self.config = config

        self.weight_decomposition = STDecompositionNd(self, 'weight', config, context)
        if self.bias is not None:
            self.bias_decomposition = STDecomposition1d(self, 'bias', config, context)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.add_adapter(adapter_name)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = F.linear(x, self.weight, bias=self.bias)
        elif not self.merged:
            weight = self.weight + self.weight_decomposition.compose(self.active_adapter)
            bias = None
            if self.bias is not None:
                bias = self.bias + self.bias_decomposition.compose(self.active_adapter)
            result = F.linear(x, weight, bias=bias)
        else:
            result = F.linear(x, self.weight, bias=self.bias)

        result = result.to(previous_dtype)
        return result


class STConv2d(nn.Conv2d, STLayer):

    def __init__(
            self,
            adapter_name: str,
            config: STConfig,
            context,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        STLayer.__init__(self)
        self.config = config

        self.weight_decomposition = STDecompositionNd(self, 'weight', config, context)
        if self.bias is not None:
            self.bias_decomposition = STDecomposition1d(self, 'bias', config, context)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.add_adapter(adapter_name)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._conv_forward(x, self.weight, self.bias)
        elif not self.merged:
            weight = self.weight + self.weight_decomposition.compose(self.active_adapter)
            bias = None
            if self.bias is not None:
                bias = self.bias + self.bias_decomposition.compose(self.active_adapter)
            result = self._conv_forward(x, weight, bias=bias)
        else:
            result = self._conv_forward(x, self.weight, bias=self.bias)

        result = result.to(previous_dtype)
        return result


class STLayerNorm(nn.LayerNorm, STLayer):

    def __init__(
            self,
            adapter_name: str,
            config: STConfig,
            context,
            normalized_shape,
            eps,
            elementwise_affine,
            device,
            dtype
    ):
        nn.LayerNorm.__init__(
            self,
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype
        )
        STLayer.__init__(self)
        self.config = config

        if self.weight is not None:
            self.weight_decomposition = STDecomposition1d(self, 'weight', config, context)
        if self.bias is not None:
            self.bias_decomposition = STDecomposition1d(self, 'bias', config, context)

        if self.weight is not None:
            self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.add_adapter(adapter_name)
        self.active_adapter = adapter_name

    def forward(self, x):
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif not self.merged:
            weight = None
            if self.weight is not None:
                weight = self.weight + self.weight_decomposition.compose(self.active_adapter)
            bias = None
            if self.bias is not None:
                bias = self.bias + self.bias_decomposition.compose(self.active_adapter)
            result = F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        else:
            result = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        result = result.to(previous_dtype)
        return result
