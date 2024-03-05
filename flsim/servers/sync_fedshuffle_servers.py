# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from flsim.active_user_selectors.simple_user_selector import (
    ImportanceSamplingActiveUserSelector,
    UniformlyRandomActiveUserSelector,
)

from flsim.channels.base_channel import IdentityChannel, IFLChannel
from flsim.channels.message import Message
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import SyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg

from hydra.utils import instantiate


class SyncFedShuffleServer(SyncServer):
    """Implements the server configuration for FedShuffle
    Ref: https://arxiv.org/pdf/2204.13169.pdf

    Performs the following:
    1. Enables random batch shuffling by default
    2. For importance sampling:
        - Gather the number of train examples per client
        - Each client delta is weighed equally (i.e. 1)
    """

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IFLChannel] = None,
        **kwargs,
    ) -> None:
        # Same steps as in SyncServer
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SyncFedShuffleServerConfig,
            **kwargs,
        )
        self._global_model = global_model
        self._aggregator = Aggregator(
            module=global_model.fl_get_module(),
            # pyre-ignore[16]
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._active_user_selector = instantiate(self.cfg.active_user_selector)

        self._channel: IFLChannel = channel or IdentityChannel()

        self._optimizer = instantiate(
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )

        # FedShuffle specific changes
        # Only allow Uniform Random and Importance Sampling
        assert isinstance(
            self._active_user_selector, UniformlyRandomActiveUserSelector
        ) or isinstance(
            self._active_user_selector, ImportanceSamplingActiveUserSelector
        ), "Currently only Uniform and Importance Sampling user selectors are supported"

        self.samples_per_user = []

    def select_clients_for_training(
        self,
        num_total_users,
        users_per_round,
        data_provider: Optional[IFLDataProvider] = None,
        global_round_num: Optional[int] = None,
    ):
        assert (
            data_provider is not None
        ), "Data provider must be passed into FedShuffleServer"

        # samples_per_user is only needed for Importance Sampling
        if (
            isinstance(self._active_user_selector, ImportanceSamplingActiveUserSelector)
            and len(self.samples_per_user) == 0
        ):
            self.samples_per_user = [
                user.num_train_examples() for user in data_provider.train_users()
            ]

        selected_clients = self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            num_samples_per_user=self.samples_per_user,
            data_provider=data_provider,
            global_round_num=global_round_num,
        )

        return selected_clients

    def receive_update_from_client(self, message: Message):
        message = self._channel.client_to_server(message)

        # Weights used are different for importance sampling and uniform sampling
        if isinstance(self._active_user_selector, ImportanceSamplingActiveUserSelector):
            message.weight = 1.0

        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )


@dataclass
class SyncFedShuffleServerConfig(SyncServerConfig):
    """
    Contains configurations for a server with the FedShuffle configuration
    """

    _target_: str = fullclassname(SyncFedShuffleServer)
    aggregation_type: AggregationType = AggregationType.WEIGHTED_SUM
