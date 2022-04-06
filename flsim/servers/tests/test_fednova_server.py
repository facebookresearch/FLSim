#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from dataclasses import dataclass
from typing import List

import torch
from flsim.clients.fednova_client import FedNovaClientConfig
from flsim.common.pytest_helper import assertEmpty
from flsim.data.data_provider import FLDataProviderFromList
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.optimizers.server_optimizers import (
    FedAvgOptimizerConfig,
)
from flsim.servers.aggregator import AggregationType
from flsim.servers.fednova_server import FedNovaServerConfig
from flsim.tests import utils
from flsim.tests.utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
    SampleNet,
)
from hydra.utils import instantiate


@dataclass
class MockClientUpdate:
    deltas: List[float]
    weights: List[float]
    expected_value: float


class TestFedNovaServer:
    def fake_data(self, num_batches: List[int]):
        datasets = []
        for user_batches in num_batches:
            datasets.append([torch.ones(1, 2) for _ in range(user_batches)])
        return FLDataProviderFromList(
            datasets, datasets, datasets, SampleNet(utils.Linear())
        )

    def test_fednova_server_update(self):
        """
        Test FedNova with the following scenario

        1. Global linear model init at 1 where optimal value is at 0.
        2. Batch size = 1 and we have 3 users with 1,2, and 3 examples
        3. Each client delta = 1
        Then
        Aggregated deltas = sum(number of steps * sampling prob) = sum(1*0.33, 2*0.33, 3*0.33) = 2
        new_global = init - aggregated = 1 - 2 = -1
        """
        server_model = SampleNet(create_model_with_value(1))
        num_users = 3
        num_batches = [1, 2, 3]

        server = instantiate(
            FedNovaServerConfig(
                aggregation_type=AggregationType.WEIGHTED_SUM,
                server_optimizer=FedAvgOptimizerConfig(),
            ),
            global_model=server_model,
        )
        data_provider = self.fake_data(num_batches=num_batches)
        clients = [
            instantiate(
                FedNovaClientConfig(
                    optimizer=LocalOptimizerSGDConfig(lr=1.0),
                ),
                dataset=user_data,
            )
            for user_data in data_provider.train_users()
        ]

        selected_clients = [
            clients[i]
            for i in server.select_clients_for_training(
                num_total_users=num_users,
                users_per_round=num_users,
                data_provider=data_provider,
            )
        ]
        server.init_round()
        for client in selected_clients:
            message = client.generate_local_update(server_model)
            server.receive_update_from_client(message)

        server.step()
        # weighted model should be 2
        # client_1 = 1
        # client_2 = 2
        # client_3 = 3
        # weights = [0.33, 0.33, 0.33]
        # grad = weighted model = 2
        # new = init - grad = 1 - 2 = -1
        msg = model_parameters_equal_to_value(server_model.fl_get_module(), -1.0)
        assertEmpty(msg, msg=msg)
