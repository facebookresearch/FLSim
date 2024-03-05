#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

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
from flsim.channels.product_quantization_channel import ProductQuantizationChannel
from flsim.channels.scalar_quantization_channel import ScalarQuantizationChannel
from flsim.channels.sparse_mask_channel import SparseMaskChannel
from flsim.common.pytest_helper import (
    assertAlmostEqual,
    assertEmpty,
    assertEqual,
    assertTrue,
)
from flsim.optimizers.server_optimizers import (
    FedAdamOptimizerConfig,
    FedAvgOptimizerConfig,
    FedAvgWithLROptimizerConfig,
    FedLAMBOptimizerConfig,
    FedLARSOptimizerConfig,
)
from flsim.servers.aggregator import AggregationType
from flsim.servers.sync_servers import (
    SyncPQServerConfig,
    SyncServerConfig,
    SyncSharedSparseServerConfig,
    SyncSQServerConfig,
)
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.test_utils import (
    calc_model_sparsity,
    create_model_with_value,
    model_parameters_equal_to_value,
    SampleNet,
    verify_models_equivalent_after_training,
)
from hydra.utils import instantiate


@dataclass
class MockClientUpdate:
    deltas: List[float]
    weights: List[float]
    expected_value: float


class TestSyncServer:
    def _create_client_updates(self, num_clients, aggregation_type) -> MockClientUpdate:
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
        # pyre-fixme[6]: Expected `List[float]` for 1st param but got `List[int]`.
        # pyre-fixme[6]: Expected `List[float]` for 2nd param but got `List[int]`.
        # pyre-fixme[61]: `expected_value` is undefined, or not always defined.
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

    def _compare_optim_and_server(
        self, opt_config, num_rounds, num_clients, agg_type
    ) -> None:
        server_model = SampleNet(create_model_with_value(0))
        optim_model = create_model_with_value(0)
        server = instantiate(
            SyncServerConfig(aggregation_type=agg_type, server_optimizer=opt_config),
            global_model=server_model,
        )

        optimizer = instantiate(config=opt_config, model=optim_model)
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

    def test_select_clients_for_training(self) -> None:
        """
        Selects 10 clients out of 100. SyncServer with seed = 0 should
        return the same indices as those of uniform random selector.
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
    def test_server_channel_integration(self, channel) -> None:
        """From Client to Server, the channel should quantize and then dequantize the message
        therefore there should be no change in the model
        """
        server = instantiate(
            SyncServerConfig(),
            global_model=SampleNet(create_model_with_value(0)),
            channel=channel,
        )

        delta = create_model_with_value(1)
        init = FLModelParamUtils.clone(delta)
        server.receive_update_from_client(Message(model=SampleNet(delta), weight=1.0))
        error_msg = verify_models_equivalent_after_training(delta, init)
        assertEmpty(error_msg, msg=error_msg)


class TestSyncSQServer:
    def test_sync_sq_server_instantiation(self) -> None:
        "Test SyncSQServer instantiation"
        _ = instantiate(
            SyncSQServerConfig(),
            global_model=SampleNet(create_model_with_value(0)),
            channel=ScalarQuantizationChannel(),
        )
        # test failure with non SQ channel
        with pytest.raises(Exception):
            _ = instantiate(
                SyncSQServerConfig(),
                global_model=SampleNet(create_model_with_value(0)),
                channel=IdentityChannel(),
            )

    def test_sync_sq_server_global_qparams_update(self) -> None:
        sync_sq_server = instantiate(
            SyncSQServerConfig(),
            global_model=SampleNet(create_model_with_value(0)),
            channel=ScalarQuantizationChannel(qscheme="symmetric"),
        )
        assertEmpty(
            sync_sq_server.global_qparams,
            msg="Global QParams of SyncSQServer should be empty on instantiation",
        )
        agg_model = create_model_with_value(1)
        sync_sq_server.update_qparams(agg_model)
        assertTrue(sync_sq_server.global_qparams.keys(), agg_model.state_dict().keys())
        qparams = list(sync_sq_server.global_qparams.values())
        assertTrue(all(zp == 0 for _, zp in qparams))
        assertTrue(all(sf == qparams[0][0] for sf, _ in qparams))


class TestSyncPQServer:
    def test_sync_pq_server_instantiation(self) -> None:
        "Test SyncPQServer instantiation"
        _ = instantiate(
            SyncPQServerConfig(),
            global_model=SampleNet(create_model_with_value(0)),
            channel=ProductQuantizationChannel(),
        )
        # test failure with non SQ channel
        with pytest.raises(Exception):
            _ = instantiate(
                SyncPQServerConfig(),
                global_model=SampleNet(create_model_with_value(0)),
                channel=IdentityChannel(),
            )

    def test_sync_pq_server_global_centroids_update(self) -> None:
        sync_pq_server = instantiate(
            SyncPQServerConfig(),
            global_model=SampleNet(create_model_with_value(0)),
            channel=ProductQuantizationChannel(
                max_num_centroids=2, num_codebooks=1, max_block_size=1
            ),
        )
        assertEmpty(
            sync_pq_server.global_pq_centroids,
            msg="Global QParams of SyncSQServer should be empty on instantiation",
        )
        agg_model = create_model_with_value(1)
        sync_pq_server.update_seed_centroids(agg_model)

        # we only quantize the first layer, the second one does not have enough elements
        quantized_layers = set(sync_pq_server.global_pq_centroids.keys())
        all_layers = set(agg_model.state_dict().keys())
        assertTrue(len(quantized_layers) > 0)
        assertTrue(quantized_layers.issubset(all_layers))


class TestSyncSharedSparseServer:
    def test_sync_shared_sparse_server_instantiation(self):
        _ = instantiate(
            SyncSharedSparseServerConfig(),
            global_model=SampleNet(create_model_with_value(0)),
            channel=SparseMaskChannel(),
        )
        # test failure with non sparse mask channel
        with pytest.raises(Exception):
            _ = instantiate(
                SyncSharedSparseServerConfig(),
                global_model=SampleNet(create_model_with_value(0)),
                channel=IdentityChannel(),
            )
        # test failure with TopK sparsity and shared sparse masks
        with pytest.raises(Exception):
            _ = instantiate(
                SyncSharedSparseServerConfig(),
                global_model=SampleNet(create_model_with_value(0)),
                channel=SparseMaskChannel(sparsity_method="topk"),
            )

    def test_sync_shared_sparse_server_mask_update(self):
        sync_shared_sparse_server = instantiate(
            SyncSharedSparseServerConfig(),
            global_model=SampleNet(create_model_with_value(1)),
            channel=SparseMaskChannel(
                use_shared_masks=True,
                mask_params_refresh_freq=1,
                proportion_of_zero_weights=0.6,
            ),
        )
        # test that the sparsity of the shared mask is as expected
        agg_model = SampleNet(create_model_with_value(1)).fl_get_module()
        sync_shared_sparse_server.update_mask_params(agg_model, "random")
        assertTrue(
            sync_shared_sparse_server.global_mask_params.keys(),
            agg_model.state_dict().keys(),
        )
        assertAlmostEqual(
            calc_model_sparsity(sync_shared_sparse_server.global_mask_params),
            0.6,
            delta=0.1,
        )
