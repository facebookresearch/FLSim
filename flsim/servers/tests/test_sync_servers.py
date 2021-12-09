#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest
from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import IdentityChannel
from flsim.channels.half_precision_channel import HalfPrecisionChannel
from flsim.channels.message import Message
from flsim.common.pytest_helper import assertEqual, assertEmpty
from flsim.optimizers.server_optimizers import (
    FedAvgOptimizerConfig,
    FedAdamOptimizerConfig,
    FedAvgWithLROptimizerConfig,
    FedLARSOptimizerConfig,
    FedLAMBOptimizerConfig,
    OptimizerType,
)
from flsim.servers.aggregator import AggregationType
from flsim.servers.sync_servers import SyncServerConfig
from flsim.tests.utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
    verify_models_equivalent_after_training,
    SampleNet,
)
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate


@dataclass
class MockClientUpdate:
    deltas: List[float]
    weights: List[float]
    expected_value: float


class TestSyncServer:
    def _create_client_updates(self, num_clients, aggregation_type):
        deltas = [i + 1 for i in range(num_clients)]
        weights = [i + 1 for i in range(num_clients)]

        if aggregation_type == AggregationType.WEIGHTED_AVERAGE:
            expected_value = float(np.average(deltas, weights=weights))
        elif aggregation_type == AggregationType.AVERAGE:
            expected_value = float(np.average(deltas))
        elif aggregation_type == AggregationType.WEIGHTED_SUM:
            expected_value = float(sum(d * w for d, w in zip(deltas, weights)))
        elif aggregation_type == AggregationType.SUM:
            expected_value = float(sum(deltas))
        return MockClientUpdate(deltas, weights, expected_value)

    def _run_one_round_comparison(
        self,
        optimizer,
        server,
        optim_model,
        server_model,
        client_updates,
    ):
        server.init_round()
        optimizer.zero_grad()
        for delta, weight in zip(client_updates.deltas, client_updates.weights):
            server.receive_update_from_client(
                Message(model=SampleNet(create_model_with_value(delta)), weight=weight)
            )

        FLModelParamUtils.set_gradient(
            model=optim_model,
            reference_gradient=create_model_with_value(client_updates.expected_value),
        )
        optimizer.step()
        server.step()
        return server_model, optim_model

    def _compare_optim_and_server(self, opt_config, num_rounds, num_clients, agg_type):
        server_model = SampleNet(create_model_with_value(0))
        optim_model = create_model_with_value(0)
        server = instantiate(
            SyncServerConfig(aggregation_type=agg_type, server_optimizer=opt_config),
            global_model=server_model,
        )

        optimizer = OptimizerType.create_optimizer(optim_model, opt_config)
        client_updates = self._create_client_updates(
            num_clients, aggregation_type=agg_type
        )

        for _ in range(num_rounds):
            server_model, optim_model = self._run_one_round_comparison(
                optimizer, server, optim_model, server_model, client_updates
            )
            error_msg = verify_models_equivalent_after_training(
                server_model, optim_model
            )
            assertEmpty(error_msg, msg=error_msg)

    @pytest.mark.parametrize(
        "aggregation_type",
        [
            AggregationType.AVERAGE,
            AggregationType.WEIGHTED_AVERAGE,
        ],
    )
    @pytest.mark.parametrize(
        "num_clients",
        [10, 1],
    )
    @pytest.mark.parametrize(
        "num_rounds",
        [10, 1],
    )
    def test_fed_avg_sync_server(
        self, aggregation_type, num_clients, num_rounds
    ) -> None:
        server_model = SampleNet(create_model_with_value(0))
        server = instantiate(
            SyncServerConfig(
                aggregation_type=aggregation_type,
                server_optimizer=FedAvgOptimizerConfig(),
            ),
            global_model=server_model,
        )

        client_updates = self._create_client_updates(
            num_clients, aggregation_type=aggregation_type
        )

        for round_num in range(num_rounds):
            server.init_round()
            for delta, weight in zip(client_updates.deltas, client_updates.weights):
                server.receive_update_from_client(
                    Message(
                        model=SampleNet(create_model_with_value(delta)), weight=weight
                    )
                )
            server.step()
            error_msg = model_parameters_equal_to_value(
                server_model, -client_updates.expected_value * (round_num + 1)
            )
            assertEmpty(error_msg, msg=error_msg)

    @pytest.mark.parametrize(
        "aggregation_type",
        [
            AggregationType.AVERAGE,
            AggregationType.WEIGHTED_AVERAGE,
            AggregationType.SUM,
            AggregationType.WEIGHTED_SUM,
        ],
    )
    @pytest.mark.parametrize(
        "num_clients",
        [10, 1],
    )
    @pytest.mark.parametrize(
        "num_rounds",
        [10, 1],
    )
    def test_fed_sgd_sync_server(
        self, aggregation_type, num_clients, num_rounds
    ) -> None:
        opt_config = FedAvgWithLROptimizerConfig(lr=0.1)
        self._compare_optim_and_server(
            opt_config,
            agg_type=aggregation_type,
            num_rounds=num_rounds,
            num_clients=num_clients,
        )

    @pytest.mark.parametrize(
        "aggregation_type",
        [
            AggregationType.AVERAGE,
            AggregationType.WEIGHTED_AVERAGE,
            AggregationType.SUM,
            AggregationType.WEIGHTED_SUM,
        ],
    )
    @pytest.mark.parametrize(
        "num_clients",
        [10, 1],
    )
    @pytest.mark.parametrize(
        "num_rounds",
        [10, 1],
    )
    def test_fed_adam_sync_server(
        self, aggregation_type, num_clients, num_rounds
    ) -> None:
        opt_config = FedAdamOptimizerConfig(lr=0.1)
        self._compare_optim_and_server(
            opt_config,
            agg_type=aggregation_type,
            num_rounds=num_rounds,
            num_clients=num_clients,
        )

    @pytest.mark.parametrize(
        "aggregation_type",
        [
            AggregationType.AVERAGE,
            AggregationType.WEIGHTED_AVERAGE,
            AggregationType.SUM,
            AggregationType.WEIGHTED_SUM,
        ],
    )
    @pytest.mark.parametrize(
        "num_clients",
        [10, 1],
    )
    @pytest.mark.parametrize(
        "num_rounds",
        [10, 1],
    )
    def test_fed_lars_sync_server(
        self, aggregation_type, num_clients, num_rounds
    ) -> None:
        opt_config = FedLARSOptimizerConfig(lr=0.001)
        self._compare_optim_and_server(
            opt_config,
            agg_type=aggregation_type,
            num_rounds=num_rounds,
            num_clients=num_clients,
        )

    @pytest.mark.parametrize(
        "aggregation_type",
        [
            AggregationType.AVERAGE,
            AggregationType.WEIGHTED_AVERAGE,
            AggregationType.SUM,
            AggregationType.WEIGHTED_SUM,
        ],
    )
    @pytest.mark.parametrize(
        "num_clients",
        [10, 1],
    )
    @pytest.mark.parametrize(
        "num_rounds",
        [10, 1],
    )
    def test_fed_lamb_sync_server(
        self, aggregation_type, num_clients, num_rounds
    ) -> None:
        opt_config = FedLAMBOptimizerConfig(lr=0.001)
        self._compare_optim_and_server(
            opt_config,
            agg_type=aggregation_type,
            num_rounds=num_rounds,
            num_clients=num_clients,
        )

    def test_select_clients_for_training(self):
        """
        Selects 10 clients out of 100. SyncServer with seed = 0 should
        return the same indicies as those of uniform random selector.
        """
        selector_config = UniformlyRandomActiveUserSelectorConfig(user_selector_seed=0)
        uniform_selector = instantiate(selector_config)

        server = instantiate(
            SyncServerConfig(active_user_selector=selector_config),
            global_model=SampleNet(create_model_with_value(0)),
        )

        server_selected_indices = server.select_clients_for_training(
            num_total_users=100, users_per_round=10
        )
        uniform_selector_indices = uniform_selector.get_user_indices(
            num_total_users=100, users_per_round=10
        )
        assertEqual(server_selected_indices, uniform_selector_indices)

    @pytest.mark.parametrize(
        "channel",
        [HalfPrecisionChannel(), IdentityChannel()],
    )
    def test_server_channel_integration(self, channel):
        """From Client to Server, the channel should quantize and then dequantize the message
        therefore there should be no change in the model
        """
        server = instantiate(
            SyncServerConfig(),
            global_model=SampleNet(create_model_with_value(0)),
            channel=channel,
        )

        delta = create_model_with_value(1)
        init = copy.deepcopy(delta)
        server.receive_update_from_client(Message(model=SampleNet(delta), weight=1.0))
        error_msg = verify_models_equivalent_after_training(delta, init)
        assertEmpty(error_msg, msg=error_msg)
