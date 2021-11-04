#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from dataclasses import dataclass
from typing import List

import numpy as np
from flsim.channels.message import SyncServerMessage
from flsim.servers.aggregator import AggregationType
from flsim.servers.sync_servers import (
    SyncServerConfig,
    FedAvgOptimizerConfig,
    FedAdamOptimizerConfig,
    FedAvgWithLROptimizerConfig,
    OptimizerType,
    FedLARSOptimizerConfig,
    FedLAMBOptimizerConfig,
)
from flsim.tests.utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
    verify_models_equivalent_after_training,
    SampleNet,
)
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from libfb.py import testutil


@dataclass
class MockClientUpdate:
    deltas: List[float]
    weights: List[float]
    expected_value: float


class SyncServerTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

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
                SyncServerMessage(delta=create_model_with_value(delta), weight=weight)
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
            SyncServerConfig(aggregation_type=agg_type, optimizer=opt_config),
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
            self.assertEmpty(error_msg, msg=error_msg)

    @testutil.data_provider(
        lambda: (
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
        )
    )
    def test_fed_avg_sync_server(
        self, aggregation_type, num_clients, num_rounds
    ) -> None:
        server_model = SampleNet(create_model_with_value(0))
        server = instantiate(
            SyncServerConfig(
                aggregation_type=aggregation_type, optimizer=FedAvgOptimizerConfig()
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
                    SyncServerMessage(
                        delta=create_model_with_value(delta), weight=weight
                    )
                )
            server.step()
            error_msg = model_parameters_equal_to_value(
                server_model, -client_updates.expected_value * (round_num + 1)
            )
            self.assertEmpty(error_msg, msg=error_msg)

    @testutil.data_provider(
        lambda: (
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.SUM,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_SUM,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.SUM,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_SUM,
                "num_clients": 1,
                "num_rounds": 1,
            },
        )
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

    @testutil.data_provider(
        lambda: (
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.SUM,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_SUM,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.SUM,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_SUM,
                "num_clients": 1,
                "num_rounds": 1,
            },
        )
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

    @testutil.data_provider(
        lambda: (
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.SUM,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_SUM,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.SUM,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_SUM,
                "num_clients": 1,
                "num_rounds": 1,
            },
        )
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

    @testutil.data_provider(
        lambda: (
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_AVERAGE,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.SUM,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_SUM,
                "num_clients": 10,
                "num_rounds": 10,
            },
            {
                "aggregation_type": AggregationType.SUM,
                "num_clients": 1,
                "num_rounds": 1,
            },
            {
                "aggregation_type": AggregationType.WEIGHTED_SUM,
                "num_clients": 1,
                "num_rounds": 1,
            },
        )
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
