# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from flsim.channels.message import Message
from flsim.clients.base_client import Client
from flsim.servers.sync_servers import SyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname


class SyncMimeLiteServer(SyncServer):
    """Implements the server configuration for MIMELite
    Ref: https://arxiv.org/pdf/2008.03606.pdf
    """

    def broadcast_message_to_clients(self, clients: Iterable[Client]) -> Message:
        """
        Message has additional metadata apart from the model:
        Server Optimizer State: To be used during client training
        """
        return Message(
            model=self.global_model,
            server_opt_state=self._optimizer.state_dict()["state"],
        )


@dataclass
class SyncMimeLiteServerConfig(SyncServerConfig):
    """
    Contains configurations for a server with the MIME configuration
    """

    _target_: str = fullclassname(SyncMimeLiteServer)
