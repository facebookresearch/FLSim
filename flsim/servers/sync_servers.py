#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, List, Optional

from flsim.active_user_selectors.simple_user_selector import (
    ActiveUserSelectorConfig,
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import IdentityChannel, IFLChannel
from flsim.channels.message import Message
from flsim.clients.base_client import Client
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import (
    FedAvgOptimizerConfig,
    ServerOptimizerConfig,
)
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class ISyncServer(abc.ABC):
    """
    Interface for Sync servers, all sync server should implement this interface.
    Responsibilities:
        Wrapper for aggregator and optimizer.
        Collects client updates and sends to aggregator.
        Changes the global model using aggregator and optimizer.
    """

    @abc.abstractmethod
    def init_round(self):
        """Clears the buffer and zero out grad in optimizer.
        This function is called before each training round.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def receive_update_from_client(self, message: Message):
        """Receives new updates from each client and aggregates result.
        This includes calculating weights of each client update and summing them to get
        a final update for the global model.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self):
        """Apply the update the global model."""
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

        The selection scheme depends on the underlying selector. This can include:
        random, sequential, high loss etc.

        Args:
            num_total_users ([int]): Number of total users (population size).
            users_per_round ([int]): Number of users per round.
            data_provider (Optional[IFLDataProvider], optional): This is useful when the
                selection scheme is high loss. Defaults to None.
            epoch (Optional[int], optional): This is useful when the selection scheme is
                high loss. Defaults to None.

        Returns:
            List[int]: A list of client indices
        """
        pass

    def broadcast_message_to_clients(self, clients: Iterable[Client]) -> Message:
        """
        Create a message common for every client during generate_local_update.
        Message must include the global_model as it is the only way to send it to each client.
        A reference to the clients in the current round is always passed by sync_trainer.

        Args:
            clients Iterable[Client]: The list of clients.
            Need by SyncMimeServer

        Returns:
            Message: The message common for all clients. Pass the global model here.
            Trainer should pass this message while calling generate_local_update for each client.
        """
        return Message(model=self.global_model)

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
        self._optimizer = instantiate(
            # pyre-ignore[16]
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )
        self._global_model = global_model
        self._aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._active_user_selector = instantiate(self.cfg.active_user_selector)
        self._channel: IFLChannel = channel or IdentityChannel()

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        """Set default user selector and server optimizer."""
        if OmegaConf.is_missing(cfg.active_user_selector, "_target_"):
            cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()
        if OmegaConf.is_missing(cfg.server_optimizer, "_target_"):
            cfg.server_optimizer = FedAvgOptimizerConfig()

    @property
    def global_model(self) -> IFLModel:
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

    def receive_update_from_client(self, message: Message):
        message = self._channel.client_to_server(message)

        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )
        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

    def step(self):
        aggregated_model = self._aggregator.aggregate()
        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )
        self._optimizer.step()


@dataclass
class SyncServerConfig:
    _target_: str = fullclassname(SyncServer)
    _recursive_: bool = False
    only_federated_params: bool = True
    aggregation_type: AggregationType = AggregationType.WEIGHTED_AVERAGE
    server_optimizer: ServerOptimizerConfig = ServerOptimizerConfig()
    active_user_selector: ActiveUserSelectorConfig = ActiveUserSelectorConfig()
