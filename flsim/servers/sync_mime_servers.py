# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from __future__ import annotations

import copy

from dataclasses import dataclass
from typing import Iterable, Optional

from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
)

from flsim.channels.base_channel import IdentityChannel, IFLChannel
from flsim.channels.message import Message
from flsim.clients.base_client import Client
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import FedAvgWithLROptimizerConfig
from flsim.servers.aggregator import Aggregator
from flsim.servers.sync_servers import SyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg

from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class SyncMimeServer(SyncServer):
    """Implements the server configuration for MIME
    Ref: https://arxiv.org/pdf/2008.03606.pdf
    """

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IFLChannel] = None,
        **kwargs,
    ) -> None:
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SyncMimeServerConfig,
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

        # MIME Specific changes here
        # State optimizer only updates the state to be passed to all clients
        # It does not change the parameters
        self._state_optimizer = instantiate(
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )

        # MIME uses a vanilla SGD optimizer to update the parameters of the global model
        self._optimizer = instantiate(
            config=FedAvgWithLROptimizerConfig(lr=self.cfg.server_optimizer.lr),
            model=global_model.fl_get_module(),
        )
        self._grad_average = FLModelParamUtils.clone(global_model.fl_get_module())

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        """Set default user selector and server optimizer."""
        if OmegaConf.is_missing(cfg.active_user_selector, "_target_"):
            cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()
        if OmegaConf.is_missing(cfg.server_optimizer, "_target_"):
            cfg.server_optimizer = FedAvgWithLROptimizerConfig(lr=1.0)

    def broadcast_message_to_clients(
        self, clients: Iterable[Client], global_round_num: Optional[int] = 0
    ) -> Message:
        """
        Message has additional metadata apart from the model:
        Control Variate: Weighted average of the gradients across the clients
        Server Optimizer State: To be used during client training
        """
        self._grad_average.zero_grad()
        num_examples = 0.0
        for client in clients:
            grad, weight = client.full_dataset_gradient(self.global_model)
            FLModelParamUtils.multiply_gradient_by_weight(grad, weight, grad)
            FLModelParamUtils.add_gradients(
                self._grad_average, grad, self._grad_average
            )
            num_examples += weight
            del grad

        assert num_examples > 0, "All clients in the current round have zero data"
        FLModelParamUtils.multiply_gradient_by_weight(
            self._grad_average, 1.0 / num_examples, self._grad_average
        )

        return Message(
            model=self.global_model,
            server_opt_state=copy.deepcopy(self._state_optimizer.state_dict()["state"]),
            mime_control_variate=self._grad_average,
        )

    def step(self):
        # Update the optimizer state using the gradients in _grad_average
        # Copy the original model before
        model_state_dict = copy.deepcopy(
            self._global_model.fl_get_module().state_dict()
        )
        FLModelParamUtils.copy_gradients(
            self._grad_average, self._global_model.fl_get_module()
        )
        self._state_optimizer.step()

        # Reload original model parameters back
        self._global_model.fl_get_module().load_state_dict(model_state_dict)
        del model_state_dict

        # Simply perform updated_model = global_model - lr * average_delta
        aggregated_model = self._aggregator.aggregate()
        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )
        self._optimizer.step()


@dataclass
class SyncMimeServerConfig(SyncServerConfig):
    """
    Contains configurations for a server with the MIME configuration
    """

    _target_: str = fullclassname(SyncMimeServer)
