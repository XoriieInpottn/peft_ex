#!/usr/bin/env python3

"""
@author: XoriieInpottn
@since: 2024-03-10
"""

__all__ = [
    "PEFTConfig",
    "AbstractAdapter",
    "AdapterProxy",
    "Attach",
    "AbstractTunner",
]

import re
from dataclasses import dataclass, field
from typing import List, MutableMapping, Optional, Sequence, Union

from libentry import logger
from torch import nn


@dataclass
class PEFTConfig:
    target_modules: Union[str, List[str]] = field(default=".*")


class AbstractAdapter(nn.ModuleDict):

    def __init__(self, param: nn.Parameter):
        super().__init__()
        # `param` belongs to the original module, it's not a parameter of the adapter
        # do not use `self.param = param`
        self.__dict__["param"] = param

    def merge(self):
        raise NotImplementedError()

    def unmerge(self):
        raise NotImplementedError()


class AdapterProxy(nn.ModuleDict):

    def __init__(self, adapter: AbstractAdapter, adapter_name: str = "default"):
        super().__init__({adapter_name: adapter})
        self.active_adapter: str = adapter_name

    def add_adapter(self, adapter: AbstractAdapter, adapter_name: str):
        if adapter_name in self:
            raise ValueError(f"Adapter \"{adapter_name}\" exists.")
        self.update({adapter_name: adapter})

    @property
    def param(self):
        return self[self.active_adapter].param

    def forward(self):
        return self[self.active_adapter].forward()

    def merge(self):
        return self[self.active_adapter].merge()

    def unmerge(self):
        return self[self.active_adapter].unmerge()


class Attach:

    def __init__(self, module: nn.Module, param_name: str, proxy: AdapterProxy):
        self.module = module
        self.param_name = param_name
        self.proxy = proxy

        self.attach_name = f"{param_name}_lora_ex"
        self.forward_pre_hook = None
        self.forward_post_hook = None

    def attach(self):
        if hasattr(self.module, self.attach_name):
            raise RuntimeError(f"There is already a member named {self.attach_name} in the target module.")
        setattr(self.module, self.attach_name, self.proxy)

        self.forward_pre_hook = self.module.register_forward_pre_hook(self._compose_hook)
        self.forward_post_hook = self.module.register_forward_hook(self._decompose_hook)
        return self

    def _compose_hook(self, _module, _args):
        composed = self.proxy.forward()
        self.module.__dict__[self.param_name] = composed
        # print("Hook before forward().")

    def _decompose_hook(self, _module, _args, _output):
        if self.param_name in self.module.__dict__:
            del self.module.__dict__[self.param_name]
        # print("Hook after forward().")

    def detach(self):
        if hasattr(self.module, self.attach_name):
            delattr(self.module, self.attach_name)
        if self.forward_pre_hook is not None:
            self.forward_pre_hook.remove()
        if self.forward_post_hook is not None:
            self.forward_post_hook.remove()
        return self

    def __str__(self):
        return f"{type(self.module)}.{self.attach_name}"


class AbstractTunner(nn.Module):

    def __init__(self, model: nn.Module, config: PEFTConfig, adapter_name: str = "default"):
        super().__init__()
        self.model = model
        self.config = config

        regex_list = config.target_modules
        if isinstance(regex_list, str):
            regex_list = [regex_list]
        patterns = [re.compile(regex) for regex in regex_list]
        self.params = self._match_params(patterns)

        self.adapter_names = []
        self.adapter_proxies: MutableMapping[int, AdapterProxy] = {}
        self.add_adapter(adapter_name)

        self.attaches = self._create_attaches()
        self.attach()

    def _match_params(self, patterns) -> Sequence[nn.Parameter]:
        visited = {}
        for name, param in self.model.named_parameters(remove_duplicate=False):
            if not any(pattern.search(name) is not None for pattern in patterns):
                continue
            logger.info(f"Matched \"{name}\".")
            if id(param) in visited:
                continue
            visited[id(param)] = param
        return [*visited.values()]

    def add_adapter(self, adapter_name: str):
        if adapter_name in self.adapter_names:
            raise ValueError(f"Adapter \"{adapter_name}\" exists.")
        self.adapter_names.append(adapter_name)

        for param in self.params:
            adapter = self._create_adapter(param)
            if adapter is None:
                continue
            if id(param) in self.adapter_proxies:
                self.adapter_proxies[id(param)].add_adapter(adapter, adapter_name)
            else:
                self.adapter_proxies[id(param)] = AdapterProxy(adapter, adapter_name)

    def _create_adapter(self, param: nn.Parameter) -> Optional[AbstractAdapter]:
        raise NotImplementedError()

    def _create_attaches(self):
        attaches = []
        for module in self.model.modules():
            for param_name, param in module.named_parameters(recurse=False, remove_duplicate=False):
                proxy = self.adapter_proxies.get(id(param))
                if proxy is None:
                    continue
                attaches.append(Attach(module, param_name, proxy))
        return attaches

    def _mark_only_adapters_trainable(self, trainable: bool):
        for param in self.model.parameters():
            param.requires_grad = not trainable
        for proxy in self.adapter_proxies.values():
            for param in proxy.parameters():
                param.requires_grad = trainable

    def attach(self):
        for attach in self.attaches:
            attach.attach()
        self._mark_only_adapters_trainable(trainable=True)

    def detach(self):
        for attach in self.attaches:
            attach.detach()
        self._mark_only_adapters_trainable(trainable=False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def merge(self):
        for proxy in self.adapter_proxies.values():
            proxy.merge()

    def unmerge(self):
        for proxy in self.adapter_proxies.values():
            proxy.unmerge()

    def set_active_adapter(self, adapter_name: str):
        if adapter_name not in self.adapter_names:
            raise ValueError(f"Adapter \"{adapter_name}\" doesn't exist.")
        for proxy in self.adapter_proxies.values():
            proxy.active_adapter = adapter_name
