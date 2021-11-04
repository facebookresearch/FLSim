#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from flsim.optimizers.async_aggregators import (
    AsyncAggregatorConfig,
    FedAdamAsyncAggregatorConfig,
    FedAvgWithLRAsyncAggregatorConfig,
    HybridAggregatorConfig,
    FedAdamHybridAggregatorConfig,
    FedAvgWithLRHybridAggregatorConfig,
)
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.optimizers.sync_aggregators import (
    SyncAggregatorConfig,
    FedAdamSyncAggregatorConfig,
    FedAvgWithLRSyncAggregatorConfig,
    FedAvgSyncAggregatorConfig,
)
from flsim.tests.optimizer_test_utils import OptimizerTestUtil
from flsim.utils.sample_model import TwoLayerNet
from hydra.utils import instantiate
from libfb.py import testutil
from omegaconf import OmegaConf


class OptimizerFactoryTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model = TwoLayerNet(10, 5, 1)

    @testutil.data_provider(OptimizerTestUtil.provide_sync_factory_creation_dataset)
    def test_sync_optimizer_config_creation_through_hydra(
        self, type_str: str, config: SyncAggregatorConfig
    ) -> None:
        if type_str == "FedAvg":
            self.assertTrue(
                # pyre-fixme[16]: Optional type has no attribute `__name__`.
                OmegaConf.get_type(config).__name__,
                FedAvgSyncAggregatorConfig.__name__,
            )
        if type_str == "FedAvgWithLR":
            self.assertTrue(
                OmegaConf.get_type(config).__name__,
                FedAvgWithLRSyncAggregatorConfig.__name__,
            )
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `lr`.
            self.assertEqual(config.lr, 0.1)
        if type_str == "FedAdam":
            self.assertTrue(
                OmegaConf.get_type(config).__name__,
                FedAdamSyncAggregatorConfig.__name__,
            )
            self.assertEqual(config.lr, 0.1)
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `weight_decay`.
            self.assertEqual(config.weight_decay, 0.9)
            # pyre-fixme[16]: `SyncAggregatorConfig` has no attribute `eps`.
            self.assertEqual(config.eps, 1e-8)

    @testutil.data_provider(OptimizerTestUtil.provide_async_factory_creation_dataset)
    def test_async_optimizer_config_creation_through_hydra(
        self, type_str: str, config: AsyncAggregatorConfig
    ) -> None:
        if type_str == "FedAvgWithLR":
            self.assertTrue(
                # pyre-fixme[16]: Optional type has no attribute `__name__`.
                OmegaConf.get_type(config).__name__,
                FedAvgWithLRAsyncAggregatorConfig.__name__,
            )
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `lr`.
            self.assertEqual(config.lr, 0.1)
        if type_str == "FedAdam":
            self.assertTrue(
                OmegaConf.get_type(config).__name__,
                FedAdamAsyncAggregatorConfig.__name__,
            )
            self.assertEqual(config.lr, 0.1)
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `weight_decay`.
            self.assertEqual(config.weight_decay, 0.9)
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `eps`.
            self.assertEqual(config.eps, 1e-8)

    @testutil.data_provider(OptimizerTestUtil.provide_hybrid_factory_creation_dataset)
    def test_hybrid_optimizer_config_creation_through_hydra(
        self, type_str: str, config: HybridAggregatorConfig
    ) -> None:
        if type_str == "FedAvgWithLR":
            self.assertTrue(
                # pyre-fixme[16]: Optional type has no attribute `__name__`.
                OmegaConf.get_type(config).__name__,
                FedAvgWithLRHybridAggregatorConfig.__name__,
            )
            # pyre-fixme[16]: `HybridAggregatorConfig` has no attribute `lr`.
            self.assertEqual(config.lr, 0.1)
            self.assertEqual(config.buffer_size, 3)
        if type_str == "FedAdam":
            self.assertTrue(
                OmegaConf.get_type(config).__name__,
                FedAdamHybridAggregatorConfig.__name__,
            )
            self.assertEqual(config.lr, 0.1)
            # pyre-fixme[16]: `HybridAggregatorConfig` has no attribute `weight_decay`.
            self.assertEqual(config.weight_decay, 0.9)
            # pyre-fixme[16]: `HybridAggregatorConfig` has no attribute `eps`.
            self.assertEqual(config.eps, 1e-8)
            self.assertEqual(config.buffer_size, 3)

    def test_local_optimizer_creation(self) -> None:
        config = {
            "_target_": LocalOptimizerSGDConfig._target_,
            "lr": 1.0,
            "weight_decay": 0.1,
        }
        local_optimizer = instantiate(config, model=self.model)
        self.assertEqual(
            OptimizerTestUtil.get_value_from_optimizer(local_optimizer, "lr"),
            config["lr"],
        )
        self.assertEqual(
            OptimizerTestUtil.get_value_from_optimizer(local_optimizer, "weight_decay"),
            config["weight_decay"],
        )
