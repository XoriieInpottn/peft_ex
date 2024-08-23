#!/usr/bin/env python3

"""
@author: XoriieInpottn
@since: 2024-03-10
"""

__all__ = [
    "LoraExConfig",
    "LoraExAdapter2d",
    "LoraExAdapter1d",
    "LoraExModel",
]

from dataclasses import dataclass, field
from typing import Iterable, MutableMapping, Optional, Tuple

import torch
from libentry import logger
from torch import nn

from .common import AbstractAdapter, AbstractTunner, PEFTConfig
from .utils import layer_grad_norm_, rms_grad_norm_


@dataclass
class LoraExConfig(PEFTConfig):
    adapter_name: str = field(default="lora_ex")
    r: int = field(default=2)
    weight_dropout: float = field(default=0.0)
    grad_norm: str = field(default="layer")
    grad_norm_eps: float = field(default=1e-8)
    grad_norm_steps: int = field(default=1)
    sh: bool = field(default=True)
    force_reshape: bool = field(default=False)
    init_weights: bool = field(default=True)


class LoraExAdapter1d(AbstractAdapter):

    def __init__(self, config: LoraExConfig, param: nn.Parameter):
        super().__init__(param)
        self.config = config
        self.merged = False

        self.scale = None
        self.shift = None
        self._init()

    def _init(self):
        assert len(self.param.shape) == 1
        dtype = self.param.dtype
        device = self.param.device

        self.scale = nn.Parameter(torch.empty((), dtype=dtype, device=device), requires_grad=True)
        self.shift = nn.Parameter(torch.empty((), dtype=dtype, device=device), requires_grad=True)

        if self.config.init_weights:
            nn.init.zeros_(self.scale)
            nn.init.zeros_(self.shift)

    def merge(self):
        if self.merged:
            logger.warning("Already merged, nothing to do.")
            return

        with torch.no_grad():
            self.param.add_(self.scale * self.param + self.shift)

        self.merged = True

    def unmerge(self):
        if not self.merged:
            logger.warning("Already unmerged, nothing to do.")
            return

        with torch.no_grad():
            self.param.sub_(self.scale * self.param + self.shift)

        self.merged = False

    def forward(self):
        return (self.scale + 1.0) * self.param + self.shift


class GradNorm:

    def __init__(self, config: LoraExConfig, dim: int):
        self.config = config
        self.dim = dim

        if self.config.grad_norm == "layer":
            self.norm_fn_ = layer_grad_norm_
        elif self.config.grad_norm == "rms":
            self.norm_fn_ = rms_grad_norm_
        else:
            raise ValueError(f"Unsupported grad_norm function \"{self.config.grad_norm}\".")

        self.grad_step = 0

    def __call__(self, param: torch.Tensor):
        # print("Grad norm called.")
        with torch.no_grad():
            self.grad_step += 1
            if self.grad_step % self.config.grad_norm_steps == 0:
                # print("GradNorm", "Before: ", float(param.grad.norm()), end=", ")
                self.norm_fn_(param.grad, dim=self.dim, eps=self.config.grad_norm_eps)
                # print("After: ", float(param.grad.norm()))


class LoraExAdapter2d(AbstractAdapter):

    def __init__(self, config: LoraExConfig, param: nn.Parameter, context: MutableMapping):
        super().__init__(param)
        self.config = config
        self.context = context
        self.merged = False
        self.grad_step = 1

        self.a = None
        self.b = None
        self.sh = None

        self.hw = None
        if self.config.force_reshape or len(param.shape) != 2:
            self.hw = self.get_hw_like_square(param.shape)
        self._init()

    @staticmethod
    def get_hw_like_square(shape: Iterable[int]) -> Tuple[int, int]:
        size = 1
        for s in shape:
            size *= s
        factor = int(size ** 0.5)
        while size % factor != 0:
            factor -= 1
        return factor, size // factor

    def _init(self):
        h, w = self.param.shape if self.hw is None else self.hw
        dtype = self.param.dtype
        device = self.param.device
        r = self.config.r

        self.sh = None
        if self.config.sh:
            self.sh = self.context.get("sh")
            if self.sh is None:
                self.sh = nn.Parameter(torch.zeros((r, r), dtype=dtype, device=device), requires_grad=True)
                self.context["sh"] = self.sh

        self.a = nn.Parameter(torch.empty((h, r), dtype=dtype, device=device), requires_grad=True)
        self.b = nn.Parameter(torch.empty((r, w), dtype=dtype, device=device), requires_grad=True)

        if self.config.init_weights:
            std = (2.0 / (h + w)) ** 0.5
            if self.config.sh:
                nn.init.trunc_normal_(self.a, 0, std, -2 * std, 2 * std)
            else:
                nn.init.zeros_(self.a)
            nn.init.trunc_normal_(self.b, 0, std, -2 * std, 2 * std)

        if self.config.grad_norm is not None:
            self.a.register_post_accumulate_grad_hook(GradNorm(self.config, 0))
            self.b.register_post_accumulate_grad_hook(GradNorm(self.config, 1))

    def merge(self):
        if not self.merged:
            with torch.no_grad():
                self.param.add_(self._compute_delta())
            self.merged = True

    def unmerge(self):
        if self.merged:
            with torch.no_grad():
                self.param.sub_(self._compute_delta())
            self.merged = False

    def forward(self):
        return self.param + self._compute_delta()

    def _compute_delta(self):
        a, b, sh = self.a, self.b, self.sh
        dp = (a @ b) if sh is None else (a @ sh @ b)
        if self.training:
            dp = nn.functional.dropout(dp, self.config.weight_dropout)
        if self.hw is not None:
            dp = dp.reshape(self.param.shape)
        return dp


class LoraExModel(AbstractTunner):

    def __init__(self, model: nn.Module, config: LoraExConfig, adapter_name: str = "default"):
        self.context = {}
        super().__init__(model, config, adapter_name)
        self.config = config

    def _create_adapter(self, param: nn.Parameter) -> Optional[AbstractAdapter]:
        if len(param.shape) >= 2:
            return LoraExAdapter2d(self.config, param, self.context)
        elif len(param.shape) == 1:
            return LoraExAdapter1d(self.config, param)
        else:
            return None
