# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from flsim.channels.base_channel import IFLChannel
from flsim.channels.message import Message
from flsim.clients.base_client import Client
from flsim.interfaces.model import IFLModel
from flsim.servers.sync_servers import SyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname

from flsim.utils.fl.common import FLModelParamUtils


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
        super().__init__(global_model=global_model, channel=channel, **kwargs)
        self._grad_average = FLModelParamUtils.clone(global_model.fl_get_module())

    def broadcast_message_to_clients(self, clients: Iterable[Client]) -> Message:
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

        assert num_examples > 0, "All clients in the current round have zero data"
        FLModelParamUtils.multiply_gradient_by_weight(
            self._grad_average, 1.0 / num_examples, self._grad_average
        )

        return Message(
            model=self.global_model,
            server_opt_state=self._optimizer.state_dict()["state"],
            mime_control_variate=self._grad_average,
        )


@dataclass
class SyncMimeServerConfig(SyncServerConfig):
    """
    Contains configurations for a server with the MIME configuration
    """

    _target_: str = fullclassname(SyncMimeServer)
