#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

import torch
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
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


class LocalOptimizerProximal(LocalOptimizer, torch.optim.SGD):
    def __init__(
        self,
        *,
        model: Model,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=LocalOptimizerProximalConfig,
            **kwargs,
        )

        super().__init__(model=model, **kwargs)

        torch.optim.SGD.__init__(
            self=self,
            params=self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )
        # save a copy to fixed W throughout all the local steps
        # we assume that initial training starts from the global model. If this is not the case,
        # then update global parameters using set_new_global_model function
        self.global_model = deepcopy(self.model)
        self.global_param_group = self.get_param_group(self.global_model)

    def get_param_group(self, model):
        # To get param_groups out of the global parameters, lr & momentum are the dummy parameters,
        # we don't need them
        opt = torch.optim.SGD(model.parameters(), lr=0, momentum=0)
        return opt.param_groups

    def set_new_global_model(self, model: Model):
        """
        Updates global parameters
        """
        self.global_model = deepcopy(model)
        self.global_param_group = self.get_param_group(model)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @torch.no_grad()
    # pyre-ignore[14]
    def step(self, closure=None):
        """
        Approximate local proximal gradient descent
        Args:
            init_model (IFLModel): model to initialize local training
            For CD it's the previous local model,
            For Bilevel it's the current global (server) model
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_norm_sq = 0
        for local_group, global_group in zip(self.param_groups, self.global_param_group):
            weight_decay = local_group["weight_decay"]
            momentum = local_group["momentum"]
            dampening = local_group["dampening"]
            nesterov = local_group["nesterov"]

            # p_local = v_i^t
            # p_global = w^t
            for p_local, p_global in zip(local_group["params"], global_group["params"]):
                if p_local.grad is None:
                    continue

                grad = p_local.grad
                # print("Grad ", grad)
                grad_norm_sq += torch.linalg.norm(grad) ** 2
                param_state = self.state[p_local]
                # add weight decay
                if weight_decay != 0:
                    grad = grad.add(p_local, alpha=weight_decay)

                # apply the momentum
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            grad
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                # Algorithm 1: Line 3 grad + lambda * (v_i^k - w^t)
                grad.add_(p_local.data - p_global, alpha=self.cfg.lambda_)
                # Algorithm 1: Line 3 v_i^k - lr (grad + lambda * (v_i^k - w^t))
                p_local.add_(grad, alpha=-local_group["lr"])
        return loss, grad_norm_sq


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


@dataclass
class LocalOptimizerProximalConfig(LocalOptimizerConfig):
    _target_: str = fullclassname(LocalOptimizerProximal)
    lambda_: float = 0.0
