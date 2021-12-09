#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from omegaconf import MISSING
from torch.nn import Module as Model  # @manual


class LocalOptimizer:
    def __init__(self, *, model: Model, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=LocalOptimizerConfig,
            **kwargs,
        )
        self.model = model

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass


# pyre-ignore[11] Annotation `torch.optim.SGD` is not defined as a type
class LocalOptimizerSGD(LocalOptimizer, torch.optim.SGD):
    def __init__(self, *, model: Model, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=LocalOptimizerSGDConfig,
            **kwargs,
        )

        super().__init__(model=model, **kwargs)

        torch.optim.SGD.__init__(
            self=self,
            params=self.model.parameters(),
            # pyre-fixme[16]: `LocalOptimizerSGD` has no attribute `cfg`.
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @staticmethod
    def dict_config(
        lr: float = 0.001, momentum: float = 0.0, weight_decay: float = 0.0
    ) -> Dict[str, Any]:
        """Allows downstream functions to get configs given lr and momentum
        With this function, we can change implementation of
        LocalSGDOptimizer.dict_config without changing downstream code
        """
        return {
            "_target_": LocalOptimizerSGDConfig._target_,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }


class LocalOptimizerFedProx(LocalOptimizer, torch.optim.SGD):
    def __init__(
        self,
        *,
        model: Model,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=LocalOptimizerFedProxConfig,
            **kwargs,
        )

        super().__init__(model=model, **kwargs)

        torch.optim.SGD.__init__(
            self=self,
            params=self.model.parameters(),
            # pyre-fixme[16]: `LocalOptimizerFedProx` has no attribute `cfg`.
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                param_state = self.state[p]

                if "global_model" not in param_state:
                    param_state["global_model"] = torch.clone(p.data).detach()

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                d_p.add_(p.data - param_state["global_model"], alpha=self.cfg.mu)
                p.add_(d_p, alpha=-group["lr"])

        return loss


@dataclass
class LocalOptimizerConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    lr: float = 0.001
    momentum: float = 0.0
    weight_decay: float = 0.0


@dataclass
class LocalOptimizerSGDConfig(LocalOptimizerConfig):
    _target_: str = fullclassname(LocalOptimizerSGD)


@dataclass
class LocalOptimizerFedProxConfig(LocalOptimizerConfig):
    _target_: str = fullclassname(LocalOptimizerFedProx)
    mu: float = 0.0
