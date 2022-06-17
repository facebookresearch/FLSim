# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import pytest

from flsim.channels.message import Message
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.pytest_helper import assertEmpty, assertEqual, assertIsInstance
from flsim.servers.sync_mimelite_servers import SyncMimeLiteServerConfig
from flsim.utils.test_utils import (
    create_model_with_value,
    SampleNet,
    verify_optimizer_state_dict_equal,
)
from hydra.utils import instantiate
from omegaconf import OmegaConf


class TestSyncMimeliteServers:
    def _fake_client(self):
        clnt = Client(dataset=None, **OmegaConf.structured(ClientConfig()))
        return clnt

    def _create_fake_clients(self, num_clients) -> List[Client]:
        return [self._fake_client() for _ in range(num_clients)]

    @pytest.mark.parametrize("num_clients", [10, 1])
    def test_broadcast_message(self, num_clients) -> None:
        """Check if server message contains the global model and optimizer state"""
        server_model = SampleNet(create_model_with_value(0))
        server = instantiate(
            SyncMimeLiteServerConfig(),
            global_model=server_model,
        )
        server.init_round()
        clients = self._create_fake_clients(num_clients)
        server_state_message = server.create_client_broadcast_message(clients)
        assertIsInstance(server_state_message, Message)
        assertEqual(server_model, server_state_message.model)
        error_msg = verify_optimizer_state_dict_equal(
            server._optimizer.state_dict()["state"],
            server_state_message.server_opt_state,
        )
        assertEmpty(error_msg, msg=error_msg)
