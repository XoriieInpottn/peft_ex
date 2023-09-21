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
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from torch import nn
from torch.nn import init

from .utils import *

__all__ = [
    'LoraExConfig',
    'Decomposition',
    'LoraExLayer',
    'LoraExModel',
    'LoraExLinear',
]


@dataclass
class LoraExConfig(PeftConfig):
    target_modules: Optional[Union[List[str], str]] = field(default=None)
    r: int = field(default=2)
    weight_dropout: float = field(default=0.1)
    grad_norm: str = field(default='layer')
    sh: bool = field(default=False)
    init_weights: bool = field(default=True)


class Decomposition(object):

    def decompose(self, adapter_name):
        pass

    def compose(self, adapter_name):
        pass

    def merge(self, adapter_name):
        pass

    def unmerge(self, adapter_name):
        pass


class LoraExDecompositionNd(Decomposition):

    def __init__(self, module, param_name, config: LoraExConfig, context):
        self.module = module
        self.param_name = param_name
        self.config = config
        self.context = context

        self.a = nn.ParameterDict()
        self.b = nn.ParameterDict()

        setattr(module, f'{param_name}_lora_A', self.a)
        setattr(module, f'{param_name}_lora_B', self.b)

        if self.config.sh:
            self.sh = nn.ParameterDict()
            setattr(module, f'{param_name}_lora_H', self.sh)

    def decompose(self, adapter_name):
        r = self.config.r
        param = getattr(self.module, self.param_name)
        h, w = param.shape
        dtype = param.dtype
        device = param.device

        a = nn.Parameter(torch.empty((h, r), dtype=dtype, device=device), requires_grad=True)
        b = nn.Parameter(torch.empty((r, w), dtype=dtype, device=device), requires_grad=True)

        if self.config.init_weights:
            init.zeros_(a)
            std = math.sqrt(2.0 / (h + w))
            init.trunc_normal_(b, 0, std, -2 * std, 2 * std)

        if self.config.grad_norm == 'layer':
            grad_norm_ = layer_grad_norm_
        elif self.config.grad_norm == 'rms':
            grad_norm_ = rms_grad_norm_
        else:
            grad_norm_ = None
        if grad_norm_ is not None:
            a.register_hook(partial(self._grad_hook, grad_norm_=grad_norm_, dim=0))
            b.register_hook(partial(self._grad_hook, grad_norm_=grad_norm_, dim=1))

        self.a[adapter_name] = a
        self.b[adapter_name] = b

        if self.config.sh:
            if adapter_name not in self.context:
                sh = nn.Parameter(torch.eye(r, dtype=dtype, device=device), requires_grad=True)
                self.context[adapter_name] = sh
            self.sh[adapter_name] = self.context[adapter_name]

    def compose(self, adapter_name):
        a = self.a[adapter_name]
        b = self.b[adapter_name]
        sh = self.sh[adapter_name] if self.config.sh else None

        dp = a @ b if sh is None else a @ sh @ b
        if self.module.training:
            dp = F.dropout(dp, self.config.weight_dropout)
        return dp

    @staticmethod
    def _grad_hook(grad, grad_norm_, dim):
        with torch.no_grad():
            grad_norm_(grad, dim)
        return grad

    def merge(self, adapter_name):
        param = getattr(self.module, self.param_name)
        param.add_(self.compose(adapter_name))

    def unmerge(self, adapter_name):
        param = getattr(self.module, self.param_name)
        param.sub_(self.compose(adapter_name))


class LoraExLayer(BaseTunerLayer):

    def __init__(self):
        self.merged = False
        self.disable_adapters = False
        self.decompositions = {}

    def add_adapter(self, adapter_name):
        for member in self.__dict__.values():
            if isinstance(member, Decomposition):
                member.decompose(adapter_name)

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return

        for member in self.__dict__.values():
            if isinstance(member, Decomposition):
                member.merge(self.active_adapter)

        self.merged = True

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        for member in self.__dict__.values():
            if isinstance(member, Decomposition):
                member.unmerge(self.active_adapter)

        self.merged = False


class LoraExModel(BaseTuner):

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
            if isinstance(m, LoraExLayer):
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
        if isinstance(target, LoraExLayer):
            target.add_adapter(adapter_name)
        else:
            if isinstance(target, nn.Linear):
                new_module = LoraExLinear(
                    adapter_name=adapter_name,
                    config=config,
                    context=self.context,
                    in_features=target.in_features,
                    out_features=target.out_features,
                    bias=target.bias is not None
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
            if isinstance(module, LoraExLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraExLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def _prepare_adapter_config(self, peft_config, model_config):
        return peft_config

    def merge_and_unload(self):
        raise NotImplementedError()


class LoraExLinear(nn.Linear, LoraExLayer):

    def __init__(
            self,
            adapter_name: str,
            config: LoraExConfig,
            context,
            in_features: int,
            out_features: int,
            bias,
            **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias, **kwargs)
        LoraExLayer.__init__(self)
        self.config = config

        self.weight_decomposition = LoraExDecompositionNd(self, 'weight', config, context)

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
            delta_weight = self.weight_decomposition.compose(self.active_adapter)
            result = F.linear(x, self.weight + delta_weight, bias=self.bias)
        else:
            result = F.linear(x, self.weight, bias=self.bias)

        result = result.to(previous_dtype)

        return result
