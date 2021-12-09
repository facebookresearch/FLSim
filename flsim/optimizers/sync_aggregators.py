#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import copy
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from flsim.channels.base_channel import IdentityChannel, IFLChannel
from flsim.common.logger import Logger
from flsim.interfaces.model import IFLModel
from flsim.optimizers.layerwise_optimizers import LAMB, LARS
from flsim.reducers.base_round_reducer import (
    IFLRoundReducer,
    IFLRoundReducerConfig,
)
from flsim.reducers.base_round_reducer import RoundReducerConfig
from flsim.reducers.dp_round_reducer import DPRoundReducerConfig
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import MISSING
from omegaconf import OmegaConf, DictConfig
from torch.nn import Module as Model  # @manual


class SyncAggregator(abc.ABC):
    """
    FL global optimizer for trainers with locally aggregated model
    """

    logger: logging.Logger = Logger.get_logger("SyncAggregator")

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IFLChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SyncAggregatorConfig,
            **kwargs,
        )
        assert (
            not self.is_round_reducer_dp
        ), "To create a private round reducer, use PrivateSyncTrainer instead."
        self.reducer = instantiate(
            # pyre-fixme[16]: `SyncAggregator` has no attribute `cfg`.
            self.cfg.reducer,
            global_model=global_model,
            channel=channel,
            num_users_per_round=self.cfg.num_users_per_round,
            total_number_of_users=self.cfg.total_number_of_users,
        )
        self._global_model: IFLModel = global_model

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.reducer, "_target_"):
            cfg.reducer = RoundReducerConfig()

    @property
    def is_round_reducer_dp(self):
        # reducer can be a DictConfig (if constructed the normal way via constructor)
        # or a dataclass instance (if the param is set directly - not a recommended way).
        return issubclass(self.cfg.reducer.__class__, DPRoundReducerConfig) or (
            isinstance(self.cfg.reducer, DictConfig)
            and issubclass(OmegaConf.get_type(self.cfg.reducer), DPRoundReducerConfig)
        )

    @property
    def global_model(self) -> IFLModel:
        return self._global_model

    def collect_client_update(self, update: IFLModel, weight: float) -> None:
        """
        Collects update from one client and aggregates it internally.
        """
        self.reducer.collect_update(delta=update, weight=weight)

    def init_round(self, reducer: Optional[IFLRoundReducer] = None):
        """
        Just like an optimizer that requires zero_grad to be called
        beginning of each step, FL aggregator requires this function
        to be called beginning of each FL round.
        """
        if reducer is not None and reducer is not self.reducer:
            self.logger.warning("Changing the round reducer!")
            del self.reducer
            self.reducer = reducer
        self.reducer.reset(ref_model=self._global_model)

    @abc.abstractmethod
    def step(self) -> Optional[float]:
        pass


class FedAvgSyncAggregator(SyncAggregator):
    """
    Implements federated averaging
    """

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAvgSyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def step(self) -> Optional[float]:
        reduced_module, sum_weights = self.reducer.reduce()
        if not self.reducer.is_averaged:
            raise AttributeError(
                "Reduction type of a FedAvg reducer should be either "
                "AVERAGE or WEIGHTED_AVERAGE."
            )
        if sum_weights > 0:
            FLModelParamUtils.subtract_model(
                minuend=self._global_model.fl_get_module(),
                subtrahend=reduced_module,
                difference=self._global_model.fl_get_module(),
                only_federated_params=True,
            )


class SyncAggregatorWithOptimizer(SyncAggregator):
    """
    Base class for SyncAggregators that use a PyTorch optimizer underneath,
    like FedAvgWithLR and FedAdam
    """

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=SyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)
        self.optimizer = create_optimizer_for_sync_aggregator(
            # pyre-fixme[16]: `SyncAggregatorWithOptimizer` has no attribute `cfg`.
            self.cfg,
            global_model.fl_get_module(),
        )
        # creating two temporary models so we don't have to initialize them
        # every time step() is called
        self._reconstructed_grad = copy.deepcopy(self._global_model)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def init_round(self, reducer: Optional[IFLRoundReducer] = None):
        super().init_round(reducer)
        self.optimizer.zero_grad()

    def step(self) -> Optional[float]:
        """
        computes grad and takes an optimizer step:
        grad := (global_model - new_model)
        new_model := global_model - f(grad)
        f() depends on the optimizer:
        e.g. Adam uses first and second moments, LARS and LAMB
        normalize the gradient, etc.
        """
        reduced_module, sum_weights = self.reducer.reduce()
        if sum_weights > 0:
            FLModelParamUtils.set_gradient(
                model=self._global_model.fl_get_module(),
                reference_gradient=reduced_module,
            )
            self.optimizer.step()


class FedAvgWithLRSyncAggregator(SyncAggregatorWithOptimizer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAvgWithLRSyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass


class FedAdamSyncAggregator(SyncAggregatorWithOptimizer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAdamSyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass


class FedLARSSyncAggregator(SyncAggregatorWithOptimizer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedLARSSyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass


class FedLAMBSyncAggregator(SyncAggregatorWithOptimizer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedLAMBSyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass


def create_optimizer_for_sync_aggregator(config: SyncAggregatorConfig, model: Model):
    if config._target_ == FedAvgWithLRSyncAggregatorConfig._target_:
        return torch.optim.SGD(
            model.parameters(),
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `lr`.
            lr=config.lr,
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `momentum`.
            momentum=config.momentum,
        )
    elif config._target_ == FedAdamSyncAggregatorConfig._target_:
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `weight_decay`.
            weight_decay=config.weight_decay,
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `beta1`.
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `beta2`.
            betas=(config.beta1, config.beta2),
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `eps`.
            eps=config.eps,
        )
    elif config._target_ == FedLARSSyncAggregatorConfig._target_:
        return LARS(
            model.parameters(),
            lr=config.lr,
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `beta`.
            beta=config.beta,
            weight_decay=config.weight_decay,
        )

    elif config._target_ == FedLAMBSyncAggregatorConfig._target_:
        return LAMB(
            model.parameters(),
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
            eps=config.eps,
        )


@dataclass
class SyncAggregatorConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    reducer: IFLRoundReducerConfig = IFLRoundReducerConfig()
    num_users_per_round: int = 1
    total_number_of_users: int = 10000000000


@dataclass
class FedAvgSyncAggregatorConfig(SyncAggregatorConfig):
    _target_: str = fullclassname(FedAvgSyncAggregator)


@dataclass
class FedAvgWithLRSyncAggregatorConfig(SyncAggregatorConfig):
    _target_: str = fullclassname(FedAvgWithLRSyncAggregator)
    lr: float = 0.001
    momentum: float = 0.0


@dataclass
class FedAdamSyncAggregatorConfig(SyncAggregatorConfig):
    _target_: str = fullclassname(FedAdamSyncAggregator)
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class FedLARSSyncAggregatorConfig(SyncAggregatorConfig):
    _target_: str = fullclassname(FedLARSSyncAggregator)
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta: float = 0.9


@dataclass
class FedLAMBSyncAggregatorConfig(SyncAggregatorConfig):
    _target_: str = fullclassname(FedLAMBSyncAggregator)
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
