#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from flsim.channels.message import Message
from flsim.clients.sync_mime_client import MimeClient, MimeClientConfig
from flsim.common.pytest_helper import (
    assertEmpty,
    assertEqual,
    assertIsInstance,
    assertNotEmpty,
)
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.optimizers.server_optimizers import FedAvgWithLROptimizerConfig
from flsim.servers.sync_mime_servers import SyncMimeServerConfig
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.test_utils import (
    create_model_with_value,
    DatasetFromList,
    DummyUserData,
    model_gradients_equal_to_value,
    SampleNet,
    verify_optimizer_state_dict_equal,
)
from hydra.utils import instantiate
from omegaconf import OmegaConf


class SampleFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Parameter(torch.tensor([0.4]))

    def forward(self, x):
        return x @ self.fc


class TestSyncMimeServers:
    def _fake_client(self, client_grad_value, weight):
        clnt = MimeClient(dataset=None, **OmegaConf.structured(MimeClientConfig()))

        def fill(module, *args):
            module = FLModelParamUtils.clone(module.fl_get_module())
            for m in module.parameters():
                if m.requires_grad:
                    m.grad = torch.ones_like(m.data) * client_grad_value
            return module, weight

        clnt.full_dataset_gradient = MagicMock(side_effect=fill)
        return clnt

    def _create_fake_clients(
        self, client_grad_values, client_weights
    ) -> List[MimeClient]:
        return [
            self._fake_client(client_grad, weight)
            for client_grad, weight in zip(client_grad_values, client_weights)
        ]

    def _create_client_updates(self, client_grad_values, client_weights) -> float:
        expected_value = float(np.average(client_grad_values, weights=client_weights))
        return expected_value

    @pytest.mark.parametrize(
        "num_clients",
        [10, 1],
    )
    @pytest.mark.parametrize(
        "num_rounds",
        [10, 1],
    )
    def test_broadcast_message(self, num_clients, num_rounds) -> None:
        """
        SyncMIMEServer: test gradient averaging algorithm in broadcast_message_to_client
        across multiple clients and rounds
        """
        server_model = SampleNet(create_model_with_value(0))
        server = instantiate(
            SyncMimeServerConfig(),
            global_model=server_model,
        )
        for _ in range(num_rounds):
            server.init_round()

            weights = [i + 1 for i in range(num_clients)]
            grads = [i + 1 for i in range(num_clients)]

            clients = self._create_fake_clients(grads, weights)
            server_state_message = server.broadcast_message_to_clients(clients)
            expected_mime_variate = self._create_client_updates(grads, weights)

            assertIsInstance(server_state_message, Message)
            assertEqual(server_model, server_state_message.model)
            error_msg = verify_optimizer_state_dict_equal(
                server._optimizer.state_dict()["state"],
                server_state_message.server_opt_state,
            )
            assertEmpty(error_msg, msg=error_msg)
            error_msg = model_gradients_equal_to_value(
                server_state_message.mime_control_variate,
                expected_mime_variate,
            )
            assertEmpty(error_msg, msg=error_msg)

            error_msg = model_gradients_equal_to_value(
                server_state_message.mime_control_variate,
                float(0.0),
            )
            assertNotEmpty(error_msg, msg=error_msg)

        # Verify error message if dicts are different
        # test different keys
        original_state_dict = {"key1": torch.Tensor([1.0])}
        message_state_dict = {"key2": torch.Tensor([1.0])}
        error_msg = verify_optimizer_state_dict_equal(
            original_state_dict,
            message_state_dict,
        )
        assertNotEmpty(error_msg, msg=error_msg)

        # test different types
        original_state_dict = {"key1": torch.Tensor([1.0])}
        message_state_dict = {"key1": 1.0}
        error_msg = verify_optimizer_state_dict_equal(
            original_state_dict,
            message_state_dict,
        )
        assertNotEmpty(error_msg, msg=error_msg)

        # test different tensor values
        original_state_dict = {"key1": torch.Tensor([1.0])}
        message_state_dict = {"key1": torch.Tensor([2.0])}
        error_msg = verify_optimizer_state_dict_equal(
            original_state_dict,
            message_state_dict,
        )
        assertNotEmpty(error_msg, msg=error_msg)

        # test nested and different values
        original_state_dict = {"key1": {"key2": 1.0}}
        message_state_dict = {"key1": {"key2": 2.0}}
        error_msg = verify_optimizer_state_dict_equal(
            original_state_dict,
            message_state_dict,
        )
        assertNotEmpty(error_msg, msg=error_msg)

    def test_empty_client_data(self):
        """
        Test if gradient average works if none of the clients have any data
        """
        num_clients = 5
        weights = [0 for i in range(num_clients)]
        grads = [i + 1 for i in range(num_clients)]

        clients = self._create_fake_clients(grads, weights)
        server_model = SampleNet(create_model_with_value(0))
        server = instantiate(
            SyncMimeServerConfig(),
            global_model=server_model,
        )
        try:
            server.broadcast_message_to_clients(clients)
        except AssertionError:
            pass
        else:
            assert "broadcast_message_to_clients must throw an assertion error\
             if all clients has no training data"

    def _mime_client(self, dataset=None):
        clnt = MimeClient(
            dataset=dataset,
            **OmegaConf.structured(
                MimeClientConfig(
                    optimizer=LocalOptimizerSGDConfig(lr=0.2, momentum=0.9)
                )
            ),
        )
        return clnt

    def test_mime_training(self):
        """
        Test if MIME Training algorithm produces correct model values after training on multiple rounds
        """
        dataset = [torch.tensor([[0.6], [0.4]]), torch.tensor([[0.2]])]
        dataset = DatasetFromList(dataset)
        dataset = DummyUserData(dataset, SampleNet(SampleFC()))
        clnt1 = self._mime_client(dataset)

        dataset = [torch.tensor([[0.1], [0.8]])]
        dataset = DatasetFromList(dataset)
        dataset = DummyUserData(dataset, SampleNet(SampleFC()))
        clnt2 = self._mime_client(dataset)

        clients = [clnt1, clnt2]

        server_model = SampleNet(SampleFC())

        server = instantiate(
            SyncMimeServerConfig(
                server_optimizer=FedAvgWithLROptimizerConfig(lr=1.0, momentum=0.9)
            ),
            global_model=server_model,
        )

        for _ in range(4):
            server.init_round()
            broadcast_message = server.broadcast_message_to_clients(clients)
            for clnt in clients:
                delta, weight = clnt.generate_local_update(broadcast_message)
                server.receive_update_from_client(Message(delta, weight))
            server.step()

        assert torch.allclose(
            server._global_model.sample_nn.fc, torch.tensor([-0.81619])
        ), "MIME parameters not matching"
