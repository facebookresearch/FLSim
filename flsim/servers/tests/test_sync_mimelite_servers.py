# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import flsim.configs  # noqa
import pytest
import torch
import torch.nn as nn

from flsim.channels.message import Message
from flsim.clients.sync_mimelite_client import MimeLiteClient, MimeLiteClientConfig
from flsim.common.pytest_helper import assertEmpty, assertEqual, assertIsInstance
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.optimizers.server_optimizers import FedAvgWithLROptimizerConfig
from flsim.servers.sync_mimelite_servers import SyncMimeLiteServerConfig
from flsim.utils.test_utils import (
    create_model_with_value,
    DatasetFromList,
    DummyUserData,
    SampleNet,
    verify_optimizer_state_dict_equal,
)

from hydra.utils import instantiate
from omegaconf import OmegaConf


class SampleFC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.fc = nn.Parameter(torch.tensor([4.0]))

    def forward(self, x):
        return x @ self.fc


class TestSyncMimeliteServers:
    def _fake_client(self, dataset=None):
        if dataset is None:
            dataset = [torch.rand(5, 2) for _ in range(3)]
            dataset = DatasetFromList(dataset)
            dataset = DummyUserData(dataset, SampleNet(SampleFC()))
        clnt = MimeLiteClient(
            dataset=dataset,
            **OmegaConf.structured(
                MimeLiteClientConfig(
                    optimizer=LocalOptimizerSGDConfig(lr=0.2, momentum=0.9)
                )
            ),
        )
        return clnt

    def test_mimelite_training(self):
        dataset1 = [torch.tensor([[0.6], [0.4]]), torch.tensor([[0.2]])]
        dataset2 = [torch.tensor([[0.1], [0.8]])]

        dataset1 = DatasetFromList(dataset1)
        dataset1 = DummyUserData(dataset1, SampleNet(SampleFC()))
        clnt1 = self._fake_client(dataset1)

        dataset2 = DatasetFromList(dataset2)
        dataset2 = DummyUserData(dataset2, SampleNet(SampleFC()))
        clnt2 = self._fake_client(dataset2)

        clients = [clnt1, clnt2]

        server_model = SampleNet(SampleFC())

        server = instantiate(
            SyncMimeLiteServerConfig(
                server_optimizer=FedAvgWithLROptimizerConfig(lr=1.0, momentum=0.9),
            ),
            global_model=server_model,
        )

        # Run 5 rounds of training and verify model weights
        for _ in range(5):
            server.init_round()
            broadcast_message = server.broadcast_message_to_clients(clients)
            for clnt in clients:
                delta, weight = clnt.generate_local_update(broadcast_message)
                server.receive_update_from_client(Message(delta, weight))
            server.step()

        assert torch.allclose(
            server_model.fl_get_module().fc, torch.tensor([2.30543])
        ), "Model parameter does not match after 5 rounds"

    def _create_fake_clients(self, num_clients) -> List[MimeLiteClient]:
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
        server_state_message = server.broadcast_message_to_clients(clients)
        assertIsInstance(server_state_message, Message)
        assertEqual(server_model, server_state_message.model)
        error_msg = verify_optimizer_state_dict_equal(
            server._optimizer.state_dict()["state"],
            server_state_message.server_opt_state,
        )
        assertEmpty(error_msg, msg=error_msg)
