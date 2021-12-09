#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import copy
from dataclasses import dataclass
from typing import Any, Optional

from flsim.interfaces.batch_metrics import IFLBatchMetrics
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from omegaconf import MISSING
from torch.optim.optimizer import Optimizer


class OptimizerScheduler(abc.ABC):
    """
    base class for local LR scheduler, enable the laerning rate
    of local optimizers for individual users during local training
    """

    def __init__(
        self,
        *,
        optimizer: Optimizer,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=OptimizerSchedulerConfig,
            **kwargs,
        )
        self.optimizer = optimizer

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def step(
        self,
        batch_metric: Optional[IFLBatchMetrics] = None,
        model: Optional[IFLModel] = None,
        data: Optional[Any] = None,
        epoch: Optional[int] = None,
    ):
        """
        interface for updating learning rate. Some learning rate scheduling methods
        rely produces multiple trial forward passes internally, e.g, line search
        methods, hence model is required in the interface.
        """
        pass

    def get_lr(self):
        lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]
        return lrs


class ConstantLRScheduler(OptimizerScheduler):
    def __init__(
        self,
        *,
        optimizer: Optimizer,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ConstantLRSchedulerConfig,
            **kwargs,
        )

        super().__init__(optimizer=optimizer, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def step(
        self,
        batch_metric: Optional[IFLBatchMetrics] = None,
        model: Optional[IFLModel] = None,
        data: Optional[Any] = None,
        epoch: Optional[int] = None,
    ):
        pass

    @property
    def lr(self):
        return self.cfg.base_lr


class LRBatchSizeNormalizer(OptimizerScheduler):
    """
    normalized the LR by number of examples in the batch
    """

    def __init__(
        self,
        *,
        optimizer: Optimizer,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=LRBatchSizeNormalizerSchedulerConfig,
            **kwargs,
        )

        super().__init__(optimizer=optimizer, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def step(
        self,
        batch_metric: Optional[IFLBatchMetrics],
        model: Optional[IFLModel] = None,
        data: Optional[Any] = None,
        epoch: Optional[int] = None,
    ):
        assert (
            batch_metric is not None
        ), "`batch_metric` param cannot be None for LRBatchSizeNormalizer"
        lr_normalizer = self._get_lr_normalizer(batch_metric)
        for param_group in self.optimizer.param_groups:
            # pyre-fixme[16]: `LRBatchSizeNormalizer` has no attribute `cfg`.
            param_group["lr"] = self.cfg.base_lr * lr_normalizer

    def _get_lr_normalizer(self, batch_metric: IFLBatchMetrics):
        # pyre-fixme[16]: `LRBatchSizeNormalizer` has no attribute `cfg`.
        return batch_metric.num_examples / self.cfg.local_lr_normalizer


class ArmijoLineSearch(OptimizerScheduler):
    """
    Classical Armijo line-search for step-size selection in optimization.
    Recent work suggests that it might also be used in stochastic over-parametrized
    setting, cf.
    "Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates"
    """

    def __init__(
        self,
        *,
        optimizer: Optimizer,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ArmijoLineSearchSchedulerConfig,
            **kwargs,
        )

        super().__init__(optimizer=optimizer, **kwargs)
        assert (
            0
            # pyre-fixme[16]: `ArmijoLineSearch` has no attribute `cfg`.
            < self.cfg.shrinking_factor
            <= 1.0
        ), "shrinking_factor must be between 0 and 1.0"
        assert 0 < self.cfg.c <= 1.0, "constant c must be between 0 and 1.0"
        assert (
            self.cfg.max_iter > 0
        ), "number of line-search iterations must be a non-negative integer"

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def step(
        self,
        batch_metric: Optional[IFLBatchMetrics],
        model: Optional[IFLModel],
        data: Optional[Any] = None,
        epoch: Optional[int] = None,
    ):
        assert (
            batch_metric is not None
        ), "`batch_metric` param cannot be None for ArmijoLineSearch"
        assert model is not None, "`model` param cannot be None for ArmijoLineSearch"

        state_dict = copy.deepcopy(
            FLModelParamUtils.get_state_dict(
                model.fl_get_module(), only_federated_params=False
            )
        )
        grad_norm_before_update = FLModelParamUtils.get_gradient_l2_norm_raw(
            model.fl_get_module()
        )
        loss_before_update = batch_metric.loss.item()
        # pyre-fixme[16]: `ArmijoLineSearch` has no attribute `cfg`.
        if self.cfg.reset:
            self._reset_lr()

        for _ in range(self.cfg.max_iter):
            FLModelParamUtils.load_state_dict(
                model.fl_get_module(), state_dict, only_federated_params=False
            )
            proposed_lr = self.get_lr()
            assert (
                len(proposed_lr) == 1
            ), "Armijo line-search only works with single param_group"
            # pyre-ignore[20]
            self.optimizer.step()
            # DO NOT compute backprop after forward here, only the forward is
            # required for step-size selection, use existent gradient direction
            new_batch_metrics = model.fl_forward(data)
            # loss if we use the proposed LR
            new_loss = new_batch_metrics.loss.item()
            if (
                float(new_loss)
                <= loss_before_update
                - self.cfg.c * proposed_lr[0] * grad_norm_before_update ** 2
            ):
                # satisfy sufficient descent, accept proposed_lr
                # and terminate line search
                break
            # reduce lr
            self._shrink_lr()

        # recover model state before the line search scatching, do the actual
        # optimizer.step() outside of the scheduler
        FLModelParamUtils.load_state_dict(
            model.fl_get_module(), state_dict, only_federated_params=False
        )

    def _shrink_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.cfg.shrinking_factor

    def _reset_lr(self):
        # reset LR back to base lr, use for resetting LR across training batches
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.cfg.base_lr


@dataclass
class OptimizerSchedulerConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    base_lr: float = 0.001


@dataclass
class ConstantLRSchedulerConfig(OptimizerSchedulerConfig):
    _target_: str = fullclassname(ConstantLRScheduler)


@dataclass
class LRBatchSizeNormalizerSchedulerConfig(OptimizerSchedulerConfig):
    _target_: str = fullclassname(LRBatchSizeNormalizer)
    local_lr_normalizer: int = 1


@dataclass
class ArmijoLineSearchSchedulerConfig(OptimizerSchedulerConfig):
    _target_: str = fullclassname(ArmijoLineSearch)
    # between (0, 1), algorithm parameter, no need to sweep usually
    shrinking_factor: float = 0.5
    # between (0, 1), algorithm parameter, no need to sweep usually
    c: float = 0.5
    # whether to reset the learning rate to base_lr in between steps
    # if False, line search for next optimizer.step() will continue
    # from the step-size found in the previous step
    reset: bool = False
    # maximum number of line-search iterations
    max_iter: int = 5
