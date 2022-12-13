#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This file contains optimizers for the server.

The server expects an IServerOptimizer with two abstract methods: step and zero_grad.
This interface is similar to torch.optim.Optimizer.

  Typical usage example:

  optimizer =        self._optimizer = instantiate(
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )
  optimizer.step()
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from flsim.optimizers.layerwise_optimizers import LAMB, LARS
from flsim.utils.config_utils import fullclassname, init_self_cfg
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
    def step(self, closure, noise=None):
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
                This will in general have a lower memory footprint, and can modestly improve performance.
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
            config_class=FedLAMBOptimizerConfig,
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


class ServerFTRLOptimizer(IServerOptimizer, torch.optim.Optimizer):
    """
    :param params: parameter groups
    :param momentum: if non-zero, use DP-FTRLM
    :param record_last_noise: whether to record the last noise. for the tree completion trick.
    """

    def __init__(self, *, model: nn.Module, record_last_noise: bool, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ServerFTRLOptimizerConfig,
            **kwargs,
        )

        IServerOptimizer.__init__(self, model=model, **kwargs)
        torch.optim.Optimizer.__init__(self, params=model.parameters(), defaults={})
        # pyre-ignore[16]
        self.momentum = self.cfg.momentum
        self.lr = self.cfg.lr
        self.record_last_noise = record_last_noise

    def __setstate__(self, state):
        super(ServerFTRLOptimizer, self).__setstate__(state)

    def zero_grad(self, set_to_none: bool = False):
        return torch.optim.Optimizer.zero_grad(self, set_to_none)

    @torch.no_grad()
    def step(self, noise, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p, nz in zip(group["params"], noise):
                if p.grad is None:
                    continue
                d_p = p.grad

                param_state = self.state[p]

                if len(param_state) == 0:
                    param_state["grad_sum"] = torch.zeros_like(
                        d_p, memory_format=torch.preserve_format
                    )
                    param_state["model_sum"] = p.detach().clone(
                        memory_format=torch.preserve_format
                    )  # just record the initial model
                    param_state["momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if self.record_last_noise:
                        param_state["last_noise"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )  # record the last noise needed, in order for restarting

                gs, ms = param_state["grad_sum"], param_state["model_sum"]
                if self.momentum == 0:
                    gs.add_(d_p)
                    p.copy_(ms + (-gs - nz) / self.lr)
                else:
                    gs.add_(d_p)
                    param_state["momentum"].mul_(self.momentum).add_(gs + nz)
                    p.copy_(ms - param_state["momentum"] / self.lr)
                if self.record_last_noise:
                    param_state["last_noise"].copy_(nz)
        return loss

    @torch.no_grad()
    def restart(self, last_noise=None):
        """
        Restart the tree.
        :param last_noise: the last noise to be added. If none, use the last noise recorded.
        """
        assert last_noise is not None or self.record_last_noise
        for group in self.param_groups:
            if last_noise is None:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if len(param_state) == 0:
                        continue
                    param_state["grad_sum"].add_(
                        param_state["last_noise"]
                    )  # add the last piece of noise to the current gradient sum
            else:
                for p, nz in zip(group["params"], last_noise):
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if len(param_state) == 0:
                        continue
                    param_state["grad_sum"].add_(nz)


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


@dataclass
class ServerFTRLOptimizerConfig(ServerOptimizerConfig):
    _target_: str = fullclassname(ServerFTRLOptimizer)
    lr: float = 0.001
    momentum: float = 0.0
