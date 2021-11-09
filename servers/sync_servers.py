#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import torch
import torch.nn as nn
from flsim.channels.base_channel import IFLChannel
from flsim.channels.message import SyncServerMessage
from flsim.active_user_selectors.simple_user_selector import (
    ActiveUserSelectorConfig,
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.layerwise_optimizers import LAMB, LARS
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class OptimizerType(Enum):
    fed_avg: str = "FedAvg"
    fed_avg_with_lr: str = "FedAvgWithLR"
    fed_adam: str = "FedAdam"
    fed_lamb: str = "FedLAMB"
    fed_lars: str = "FedLARS"
    fed_prox: str = "FedProx"

    @staticmethod
    def create_optimizer(
        model: nn.Module, config: OptimizerConfig
    ) -> torch.optim.Optimizer:
        if config.type == OptimizerType.fed_avg_with_lr:
            return torch.optim.SGD(
                model.parameters(),
                lr=config.lr,
                # pyre-ignore[16] Undefined attribute
                momentum=config.momentum,
            )
        elif config.type == OptimizerType.fed_adam:
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
        elif config.type == OptimizerType.fed_lars:
            # pyre-ignore[7]
            return LARS(
                model.parameters(),
                lr=config.lr,
                # pyre-ignore[16] Undefined attribute
                beta=config.beta,
                weight_decay=config.weight_decay,
            )

        elif config.type == OptimizerType.fed_lamb:
            # pyre-ignore[7]
            return LAMB(
                model.parameters(),
                lr=config.lr,
                beta1=config.beta1,
                beta2=config.beta2,
                weight_decay=config.weight_decay,
                eps=config.eps,
            )
        elif config.type == OptimizerType.fed_avg:
            return torch.optim.SGD(
                model.parameters(),
                lr=1.0,
                momentum=0,
            )
        else:
            raise ValueError(
                f"Optimizer type {config.type} not found. Please update OptimizerType.create_optimizer"
            )


class ISyncServer(abc.ABC):
    """
    Interface for Sync servers, all sync server should
    implement this interface.
    Responsiblities:
        Wrapper for aggregator and optimizer.
        Collects client updates and sends to aggregator.
        Changes the global model using aggregator and optimizer.
    """

    @abc.abstractmethod
    def init_round(self):
        """
        Clears the buffer and zero out grad in optimizer
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def receive_update_from_client(self, message: SyncServerMessage):
        """
        Receives new update from client
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self):
        """
        Update the global model
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def select_clients_for_training(
        self,
        num_total_users: int,
        users_per_round: int,
        data_provider: Optional[IFLDataProvider] = None,
        epoch: Optional[int] = None,
    ) -> List[int]:
        """
        Selects clients to participate in a round of training.

        The selection scheme depends on the underlying selector. This
        can include: random, sequential, high loss etc.

        Args:
            num_total_users ([int]): Number of total users (population size).
            users_per_round ([int]]): Number of users per round.
            data_provider (Optional[IFLDataProvider], optional): This is useful when the selection scheme
            is high loss. Defaults to None.
            epoch (Optional[int], optional): [description]. This is useful when the selection scheme
            is high loss. Defaults to None.

        Returns:
            List[int]: A list of client indicies
        """
        pass

    @property
    def global_model(self) -> IFLModel:
        """
        Returns the current global model
        """
        raise NotImplementedError()


class SyncServer(ISyncServer):
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
            config_class=SyncServerConfig,
            **kwargs,
        )
        self._optimizer: torch.optim.Optimizer = OptimizerType.create_optimizer(
            model=global_model.fl_get_module(),
            # pyre-fixme[16]: `SyncServer` has no attribute `cfg`.
            config=self.cfg.optimizer,
        )
        self._global_model: IFLModel = global_model
        self._aggregator: Aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._active_user_selector = instantiate(self.cfg.active_user_selector)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.active_user_selector, "_target_"):
            cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()

    @property
    def global_model(self):
        return self._global_model

    def select_clients_for_training(
        self,
        num_total_users,
        users_per_round,
        data_provider: Optional[IFLDataProvider] = None,
        epoch: Optional[int] = None,
    ):
        return self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_model=self.global_model,
            epoch=epoch,
        )

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()

    def receive_update_from_client(self, message: SyncServerMessage):
        self._aggregator.apply_weight_to_update(
            delta=message.delta, weight=message.weight
        )
        self._aggregator.add_update(delta=message.delta, weight=message.weight)

    def step(self):
        aggregated_model = self._aggregator.aggregate()
        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )
        self._optimizer.step()


@dataclass
class OptimizerConfig:
    type: OptimizerType = OptimizerType.fed_avg
    lr: float = 1.0


@dataclass
class FedAvgOptimizerConfig(OptimizerConfig):
    type: OptimizerType = OptimizerType.fed_avg


@dataclass
class FedAvgWithLROptimizerConfig(OptimizerConfig):
    type: OptimizerType = OptimizerType.fed_avg_with_lr
    lr: float = 0.001
    momentum: float = 0.0


@dataclass
class FedAdamOptimizerConfig(OptimizerConfig):
    type: OptimizerType = OptimizerType.fed_adam
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class FedLARSOptimizerConfig(OptimizerConfig):
    type: OptimizerType = OptimizerType.fed_lars
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta: float = 0.9


@dataclass
class FedLAMBOptimizerConfig(OptimizerConfig):
    type: OptimizerType = OptimizerType.fed_lamb
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class SyncServerConfig:
    _target_: str = fullclassname(SyncServer)
    _recursive_: bool = False
    only_federated_params: bool = True
    aggregation_type: AggregationType = AggregationType.WEIGHTED_AVERAGE
    optimizer: OptimizerConfig = FedAvgOptimizerConfig()
    active_user_selector: ActiveUserSelectorConfig = ActiveUserSelectorConfig()
