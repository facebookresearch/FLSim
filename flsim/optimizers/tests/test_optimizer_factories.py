#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from flsim.common.pytest_helper import assertEqual, assertTrue
from flsim.optimizers.async_aggregators import (
    AsyncAggregatorConfig,
    FedAdamAsyncAggregatorConfig,
    FedAvgWithLRAsyncAggregatorConfig,
    HybridAggregatorConfig,
    FedAdamHybridAggregatorConfig,
    FedAvgWithLRHybridAggregatorConfig,
)
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.optimizers.optimizer_test_utils import OptimizerTestUtil
from flsim.optimizers.sync_aggregators import (
    SyncAggregatorConfig,
    FedAdamSyncAggregatorConfig,
    FedAvgWithLRSyncAggregatorConfig,
    FedAvgSyncAggregatorConfig,
)
from flsim.utils.sample_model import TwoLayerNet
from hydra.utils import instantiate
from omegaconf import OmegaConf


@pytest.fixture(scope="class")
def prepare_optimizer_factory_test(request):
    request.cls.model = TwoLayerNet(10, 5, 1)


@pytest.mark.usefixtures("prepare_optimizer_factory_test")
class TestOptimizerFactory:
    @pytest.mark.parametrize(
        "type_str,config", OptimizerTestUtil.provide_sync_factory_creation_dataset()
    )
    def test_sync_optimizer_config_creation_through_hydra(
        self, type_str: str, config: SyncAggregatorConfig
    ) -> None:
        if type_str == "FedAvg":
            assertTrue(
                # pyre-fixme[16]: Optional type has no attribute `__name__`.
                OmegaConf.get_type(config).__name__,
                FedAvgSyncAggregatorConfig.__name__,
            )
        if type_str == "FedAvgWithLR":
            assertTrue(
                OmegaConf.get_type(config).__name__,
                FedAvgWithLRSyncAggregatorConfig.__name__,
            )
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `lr`.
            assertEqual(config.lr, 0.1)
        if type_str == "FedAdam":
            assertTrue(
                OmegaConf.get_type(config).__name__,
                FedAdamSyncAggregatorConfig.__name__,
            )
            assertEqual(config.lr, 0.1)
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `weight_decay`.
            assertEqual(config.weight_decay, 0.9)
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `eps`.
            assertEqual(config.eps, 1e-8)

    @pytest.mark.parametrize(
        "type_str,config", OptimizerTestUtil.provide_async_factory_creation_dataset()
    )
    def test_async_optimizer_config_creation_through_hydra(
        self, type_str: str, config: AsyncAggregatorConfig
    ) -> None:
        if type_str == "FedAvgWithLR":
            assertTrue(
                # pyre-fixme[16]: Optional type has no attribute `__name__`.
                OmegaConf.get_type(config).__name__,
                FedAvgWithLRAsyncAggregatorConfig.__name__,
            )
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `lr`.
            assertEqual(config.lr, 0.1)
        if type_str == "FedAdam":
            assertTrue(
                OmegaConf.get_type(config).__name__,
                FedAdamAsyncAggregatorConfig.__name__,
            )
            assertEqual(config.lr, 0.1)
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `weight_decay`.
            assertEqual(config.weight_decay, 0.9)
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `eps`.
            assertEqual(config.eps, 1e-8)

    @pytest.mark.parametrize(
        "type_str,config", OptimizerTestUtil.provide_hybrid_factory_creation_dataset()
    )
    def test_hybrid_optimizer_config_creation_through_hydra(
        self, type_str: str, config: HybridAggregatorConfig
    ) -> None:
        if type_str == "FedAvgWithLR":
            assertTrue(
                # pyre-fixme[16]: Optional type has no attribute `__name__`.
                OmegaConf.get_type(config).__name__,
                FedAvgWithLRHybridAggregatorConfig.__name__,
            )
            # pyre-fixme[16]: `HybridAggregatorConfig` has no attribute `lr`.
            assertEqual(config.lr, 0.1)
            assertEqual(config.buffer_size, 3)
        if type_str == "FedAdam":
            assertTrue(
                OmegaConf.get_type(config).__name__,
                FedAdamHybridAggregatorConfig.__name__,
            )
            assertEqual(config.lr, 0.1)
            # pyre-fixme[16]: `HybridAggregatorConfig` has no attribute `weight_decay`.
            assertEqual(config.weight_decay, 0.9)
            # pyre-fixme[16]: `HybridAggregatorConfig` has no attribute `eps`.
            assertEqual(config.eps, 1e-8)
            assertEqual(config.buffer_size, 3)

    def test_local_optimizer_creation(self) -> None:
        config = {
            "_target_": LocalOptimizerSGDConfig._target_,
            "lr": 1.0,
            "weight_decay": 0.1,
        }
        # pyre-ignore[16]: for pytest fixture
        local_optimizer = instantiate(config, model=self.model)
        assertEqual(
            OptimizerTestUtil.get_value_from_optimizer(local_optimizer, "lr"),
            config["lr"],
        )
        assertEqual(
            OptimizerTestUtil.get_value_from_optimizer(local_optimizer, "weight_decay"),
            config["weight_decay"],
        )
