#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This file contains optimizers for the server.

The server expects an IServerOptimizer with two abstract methods: step and zero_grad.
This interface is similar to torch.optim.Optimizer.

  Typical usage example:

  optimizer = OptimizerType.create_optimizer(model=model, config=config)
  optimizer.step()
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from flsim.optimizers.layerwise_optimizers import LAMB, LARS
from flsim.utils.config_utils import fullclassname, init_self_cfg, is_target
from omegaconf import MISSING


class IServerOptimizer(abc.ABC):
    def __init__(self, *, model: nn.Module, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=ServerOptimizerConfig,
            **kwargs,
        )
        self.model = model

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        raise NotImplementedError


# pyre-ignore[11] Annotation
class FedAvgWithLROptimizer(IServerOptimizer, torch.optim.SGD):
    def __init__(self, *, model: nn.Module, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAvgWithLROptimizerConfig,
            **kwargs,
        )

        IServerOptimizer.__init__(self, model=model, **kwargs)

        torch.optim.SGD.__init__(
            self,
            params=self.model.parameters(),
            # pyre-ignore[16] Undefined attribute
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
        )

    def step(self, closure=None):
        return torch.optim.SGD.step(self, closure)

    def zero_grad(self, set_to_none: bool = False):
        return torch.optim.SGD.zero_grad(self, set_to_none)


class FedAvgOptimizer(IServerOptimizer, torch.optim.SGD):
    def __init__(self, *, model: nn.Module, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAvgOptimizerConfig,
            **kwargs,
        )

        IServerOptimizer.__init__(self, model=model, **kwargs)

        torch.optim.SGD.__init__(
            self,
            params=self.model.parameters(),
            lr=1.0,
            momentum=0,
        )

    def step(self, closure=None):
        return torch.optim.SGD.step(self, closure)

    def zero_grad(self, set_to_none: bool = False):
        return torch.optim.SGD.zero_grad(self, set_to_none)


# pyre-ignore[11] Annotation
class FedAdamOptimizer(IServerOptimizer, torch.optim.Adam):
    def __init__(self, *, model: nn.Module, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAdamOptimizerConfig,
            **kwargs,
        )

        IServerOptimizer.__init__(self, model=model, **kwargs)

        torch.optim.Adam.__init__(
            self,
            params=self.model.parameters(),
            # pyre-ignore[16] Undefined attribute
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(self.cfg.beta1, self.cfg.beta2),
            eps=self.cfg.eps,
        )

    def step(self, closure=None):
        return torch.optim.Adam.step(self, closure)

    def zero_grad(self, set_to_none: bool = False):
        return torch.optim.Adam.zero_grad(self, set_to_none)


class FedLARSOptimizer(IServerOptimizer, LARS):
    def __init__(self, *, model: nn.Module, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedLARSOptimizerConfig,
            **kwargs,
        )

        IServerOptimizer.__init__(self, model=model, **kwargs)

        LARS.__init__(
            self,
            params=self.model.parameters(),
            # pyre-ignore[16] Undefined attribute
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            beta=self.cfg.beta,
        )

    def step(self, closure=None):
        return LARS.step(self, closure)

    def zero_grad(self, set_to_none: bool = False):
        return LARS.zero_grad(self, set_to_none)


class FedLAMBOptimizer(IServerOptimizer, LAMB):
    def __init__(self, *, model: nn.Module, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedLARSOptimizerConfig,
            **kwargs,
        )

        IServerOptimizer.__init__(self, model=model, **kwargs)

        LAMB.__init__(
            self,
            params=self.model.parameters(),
            # pyre-ignore[16] Undefined attribute
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            beta1=self.cfg.beta1,
            beta2=self.cfg.beta2,
            eps=self.cfg.eps,
        )

    def step(self, closure=None):
        return LAMB.step(self, closure)

    def zero_grad(self, set_to_none: bool = False):
        return LAMB.zero_grad(self, set_to_none)


@dataclass
class ServerOptimizerConfig:
    _target_: str = MISSING
    _recursive_: bool = False


@dataclass
class FedAvgOptimizerConfig(ServerOptimizerConfig):
    _target_: str = fullclassname(FedAvgOptimizer)


@dataclass
class FedAvgWithLROptimizerConfig(ServerOptimizerConfig):
    _target_: str = fullclassname(FedAvgWithLROptimizer)
    lr: float = 0.001
    momentum: float = 0.0


@dataclass
class FedAdamOptimizerConfig(ServerOptimizerConfig):
    _target_: str = fullclassname(FedAdamOptimizer)
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class FedLARSOptimizerConfig(ServerOptimizerConfig):
    _target_: str = fullclassname(FedLARSOptimizer)
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta: float = 0.9


@dataclass
class FedLAMBOptimizerConfig(ServerOptimizerConfig):
    _target_: str = fullclassname(FedLAMBOptimizer)
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


class OptimizerType(Enum):
    fed_avg: str = FedAvgOptimizerConfig._target_
    fed_avg_with_lr: str = FedAvgWithLROptimizerConfig._target_
    fed_adam: str = FedAdamOptimizerConfig._target_
    fed_lamb: str = FedLAMBOptimizerConfig._target_
    fed_lars: str = FedLARSOptimizerConfig._target_

    @staticmethod
    def create_optimizer(
        model: nn.Module, config: ServerOptimizerConfig
    ) -> IServerOptimizer:
        if is_target(config, FedAvgWithLROptimizerConfig):
            return torch.optim.SGD(
                model.parameters(),
                # pyre-ignore[16] Undefined attribute
                lr=config.lr,
                # pyre-ignore[16] Undefined attribute
                momentum=config.momentum,
            )
        elif is_target(config, FedAdamOptimizerConfig):
            return torch.optim.Adam(
                model.parameters(),
                lr=config.lr,
                # pyre-ignore[16] Undefined attribute
                weight_decay=config.weight_decay,
                # pyre-ignore[16] Undefined attribute
                betas=(config.beta1, config.beta2),
                # pyre-ignore[16] Undefined attribute
                eps=config.eps,
            )
        elif is_target(config, FedLARSOptimizerConfig):
            # pyre-ignore[7]
            return LARS(
                model.parameters(),
                lr=config.lr,
                # pyre-ignore[16] Undefined attribute
                beta=config.beta,
                weight_decay=config.weight_decay,
            )

        elif is_target(config, FedLAMBOptimizerConfig):
            # pyre-ignore[7]
            return LAMB(
                model.parameters(),
                lr=config.lr,
                beta1=config.beta1,
                beta2=config.beta2,
                weight_decay=config.weight_decay,
                eps=config.eps,
            )
        elif is_target(config, FedAvgOptimizerConfig):
            return torch.optim.SGD(
                model.parameters(),
                lr=1.0,
                momentum=0,
            )
        else:
            raise ValueError(
                f"Optimizer type {config._target_} not found. Please update OptimizerType.create_optimizer"
            )
