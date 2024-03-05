# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional

import flsim.configs  # noqa
import torch
import torch.nn as nn

from flsim.active_user_selectors.simple_user_selector import (
    ImportanceSamplingActiveUserSelectorConfig,
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.message import Message
from flsim.clients.sync_fedshuffle_client import (
    FedShuffleClient,
    FedShuffleClientConfig,
)
from flsim.common.pytest_helper import assertEmpty
from flsim.data.data_provider import FLDataProviderFromList
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.optimizers.server_optimizers import FedAvgWithLROptimizerConfig
from flsim.servers.aggregator import AggregationType
from flsim.servers.sync_fedshuffle_servers import SyncFedShuffleServerConfig
from flsim.utils import test_utils as utils

from hydra.utils import instantiate
from omegaconf import OmegaConf


class TestSyncFedShuffleServers:
    def _fake_data(
        self, num_batches=3, batch_size=2, rng: Optional[torch.Generator] = None
    ):
        dataset = [torch.rand(batch_size, 2, generator=rng) for _ in range(num_batches)]
        dataset = utils.DatasetFromList(dataset)
        return utils.DummyUserData(dataset, utils.SampleNet(utils.TwoFC()))

    def _fake_client(self, dataset, client_lr):
        optim_config = LocalOptimizerSGDConfig(lr=client_lr)
        dataset = dataset or self._fake_data()
        clnt = FedShuffleClient(
            dataset=dataset,
            **OmegaConf.structured(
                FedShuffleClientConfig(
                    optimizer=optim_config, shuffle_batch_order=False, epochs=2
                )
            ),
        )
        return clnt

    def _perform_fedshuffle_training(self, server, expected_model, client_lr):
        rng = torch.Generator().manual_seed(1234)

        clients = []
        train_dataset = []
        num_batches = [3, 5, 4, 6]

        for batches in num_batches:
            dataset = self._fake_data(batches, 2, rng)
            train_dataset.append(dataset.data.ds)
            clients.append(self._fake_client(dataset, client_lr))

        data_provider = FLDataProviderFromList(
            train_dataset, train_dataset, train_dataset, server.global_model
        )

        # Run 5 rounds of training and verify model weights
        for _ in range(5):
            server.init_round()
            selected_clients = server.select_clients_for_training(
                len(num_batches), 2, data_provider
            )
            broadcast_message = server.broadcast_message_to_clients(selected_clients)
            for clnt in selected_clients:
                clnt = clients[clnt]
                delta, weight = clnt.generate_local_update(broadcast_message)
                server.receive_update_from_client(Message(delta, weight))
            server.step()

        error_msg = utils.verify_models_equivalent_after_training(
            server.global_model, expected_model
        )
        assertEmpty(error_msg, error_msg)

    def test_fedshuffle_uniform_sampling_weighted_average_training(self):
        # FedShuffle + Uniform Sampling + Weighted Average aggregation
        server_model = utils.SampleNet(utils.linear_model(4.0))

        server = instantiate(
            SyncFedShuffleServerConfig(
                server_optimizer=FedAvgWithLROptimizerConfig(lr=2.0, momentum=0.9),
                active_user_selector=UniformlyRandomActiveUserSelectorConfig(
                    user_selector_seed=34
                ),
                aggregation_type=AggregationType.WEIGHTED_AVERAGE,
            ),
            global_model=server_model,
        )

        # Value obtained on FedShuffle's official implementation for same client data
        expected_model = utils.linear_model(0.0)
        expected_model.fc1.weight = nn.Parameter(
            torch.tensor([[3.59065829, 3.5928464]])
        )
        expected_model.fc1.bias = nn.Parameter(torch.tensor([3.21135659]))

        self._perform_fedshuffle_training(server, expected_model, client_lr=0.03)

    def test_fedshuffle_uniform_sampling_weighted_sum_training(self):
        # FedShuffle + Uniform Sampling + Weighted Sum aggregation
        server_model = utils.SampleNet(utils.linear_model(4.0))
        server = instantiate(
            SyncFedShuffleServerConfig(
                server_optimizer=FedAvgWithLROptimizerConfig(lr=0.2, momentum=0.9),
                active_user_selector=UniformlyRandomActiveUserSelectorConfig(
                    user_selector_seed=34
                ),
                aggregation_type=AggregationType.WEIGHTED_SUM,
            ),
            global_model=server_model,
        )

        # Value obtained on FedShuffle's official implementation for same client data
        expected_model = utils.linear_model(0.0)
        expected_model.fc1.weight = nn.Parameter(
            torch.tensor([[3.16115176, 3.17184993]])
        )
        expected_model.fc1.bias = nn.Parameter(torch.tensor([2.3984]))

        self._perform_fedshuffle_training(server, expected_model, client_lr=0.03)

    def test_fedshuffle_importance_sampling_weighted_average_training(self):
        # FedShuffle + Importance Sampling + Weighted Average aggregation
        server_model = utils.SampleNet(utils.linear_model(4.0))

        server = instantiate(
            SyncFedShuffleServerConfig(
                server_optimizer=FedAvgWithLROptimizerConfig(lr=0.2, momentum=0.9),
                active_user_selector=ImportanceSamplingActiveUserSelectorConfig(
                    user_selector_seed=34
                ),
                aggregation_type=AggregationType.WEIGHTED_AVERAGE,
            ),
            global_model=server_model,
        )

        # Value obtained on FedShuffle's official implementation for same client data
        expected_model = utils.linear_model(0.0)
        expected_model.fc1.weight = nn.Parameter(
            torch.tensor([[3.58047322, 3.61246808]])
        )
        expected_model.fc1.bias = nn.Parameter(torch.tensor([3.21135413]))

        self._perform_fedshuffle_training(server, expected_model, client_lr=0.3)

    def test_fedshuffle_importance_sampling_weighted_sum_training(self):
        # FedShuffle + Importance Sampling + Weighted Sum aggregation
        server_model = utils.SampleNet(utils.linear_model(4.0))
        server = instantiate(
            SyncFedShuffleServerConfig(
                server_optimizer=FedAvgWithLROptimizerConfig(lr=0.2, momentum=0.9),
                active_user_selector=ImportanceSamplingActiveUserSelectorConfig(
                    user_selector_seed=34
                ),
                aggregation_type=AggregationType.WEIGHTED_SUM,
            ),
            global_model=server_model,
        )

        # Value obtained on FedShuffle's official implementation for same client data
        expected_model = utils.linear_model(0.0)
        expected_model.fc1.weight = nn.Parameter(
            torch.tensor([[2.94659583, 3.04550605]])
        )
        expected_model.fc1.bias = nn.Parameter(torch.tensor([2.042368]))

        self._perform_fedshuffle_training(server, expected_model, client_lr=0.3)
