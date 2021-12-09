#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Union

from flsim.optimizers.async_aggregators import (
    FedAdamAsyncAggregatorConfig,
    FedAvgWithLRAsyncAggregatorConfig,
    FedAdamHybridAggregatorConfig,
    FedAvgWithLRHybridAggregatorConfig,
    AsyncAggregator,
    FedAdamAsyncAggregator,
    FedAvgWithLRAsyncAggregator,
)
from flsim.optimizers.sync_aggregators import (
    FedAdamSyncAggregatorConfig,
    FedAvgWithLRSyncAggregatorConfig,
    FedAvgSyncAggregatorConfig,
    FedAdamSyncAggregator,
    FedAvgWithLRSyncAggregator,
    FedAvgSyncAggregator,
    SyncAggregator,
)
from flsim.trainers.async_trainer import AsyncTrainer
from flsim.trainers.sync_trainer import SyncTrainer
from omegaconf import OmegaConf
from torch.optim.optimizer import Optimizer


class OptimizerTestUtil:
    SYNC_AGGREGATOR_TEST_CONFIGS = {
        "avg": {"dict_config": {"_base_": "base_fed_avg_sync_aggregator"}},
        "sgd": {
            "dict_config": {
                "_base_": "base_fed_avg_with_lr_sync_aggregator",
                "lr": 0.1,
                "momentum": 0.9,
            }
        },
        "adam": {
            "dict_config": {
                "_base_": "base_fed_adam_sync_aggregator",
                "lr": 0.1,
                "weight_decay": 0.9,
                "eps": 1e-8,
            }
        },
    }

    ASYNC_AGGREGATOR_TEST_CONFIGS = {
        "sgd": {
            "dict_config": {
                "_base_": "base_fed_avg_with_lr_async_aggregator",
                "lr": 0.1,
                "momentum": 0.9,
            }
        },
        "adam": {
            "dict_config": {
                "_base_": "base_fed_adam_async_aggregator",
                "lr": 0.1,
                "weight_decay": 0.9,
                "eps": 1e-8,
            }
        },
    }

    @classmethod
    def provide_sync_factory_creation_dataset(cls):
        return [
            (
                "FedAvg",
                OmegaConf.structured(FedAvgSyncAggregatorConfig),
            ),
            (
                "FedAvgWithLR",
                OmegaConf.structured(
                    FedAvgWithLRSyncAggregatorConfig(lr=0.1, momentum=0.9)
                ),
            ),
            (
                "FedAdam",
                OmegaConf.structured(
                    FedAdamSyncAggregatorConfig(lr=0.1, weight_decay=0.9, eps=1e-8)
                ),
            ),
        ]

    @classmethod
    def provide_async_factory_creation_dataset(cls):
        return [
            (
                "FedAvgWithLR",
                OmegaConf.structured(
                    FedAvgWithLRAsyncAggregatorConfig(lr=0.1, momentum=0.9)
                ),
            ),
            (
                "FedAdam",
                OmegaConf.structured(
                    FedAdamAsyncAggregatorConfig(lr=0.1, weight_decay=0.9, eps=1e-8)
                ),
            ),
        ]

    @classmethod
    def provide_hybrid_factory_creation_dataset(cls):
        return [
            (
                "FedAvgWithLR",
                OmegaConf.structured(
                    FedAvgWithLRHybridAggregatorConfig(
                        lr=0.1, momentum=0.9, buffer_size=3
                    )
                ),
            ),
            (
                "FedAdam",
                OmegaConf.structured(
                    FedAdamHybridAggregatorConfig(
                        lr=0.1,
                        weight_decay=0.9,
                        eps=1e-8,
                        buffer_size=3,
                    )
                ),
            ),
        ]

    @classmethod
    def get_value_from_optimizer(cls, optimizer: Optimizer, attribute_name: str):
        for param_group in optimizer.param_groups:
            return param_group[attribute_name]
        raise TypeError(f"Optimizer does not have attribute: {attribute_name}")

    @classmethod
    def get_sync_aggregator_test_configs(cls):
        return list(cls.SYNC_AGGREGATOR_TEST_CONFIGS.values())

    @classmethod
    def get_async_aggregator_test_configs(cls):
        return list(cls.ASYNC_AGGREGATOR_TEST_CONFIGS.values())

    @classmethod
    def _verify_trainer_common_aggregators(
        cls,
        aggregator: Union[SyncAggregator, AsyncAggregator],
        dict_config: Dict[str, Any],
    ):
        if "fed_avg_with_lr" in dict_config["_base_"]:
            assert isinstance(aggregator, FedAvgWithLRSyncAggregator) or isinstance(
                aggregator, FedAvgWithLRAsyncAggregator
            )
            assert (
                OptimizerTestUtil.get_value_from_optimizer(aggregator.optimizer, "lr")
                == 0.1
            ), "lr for FedavgwithLr optimizer should be 0.1"
            assert (
                OptimizerTestUtil.get_value_from_optimizer(
                    aggregator.optimizer, "momentum"
                )
                == 0.9
            ), "momentum for FedavgwithLr optimizer should be 0.9"
        if "fed_adam" in dict_config["_base_"]:
            assert isinstance(aggregator, FedAdamSyncAggregator) or isinstance(
                aggregator, FedAdamAsyncAggregator
            )
            assert (
                OptimizerTestUtil.get_value_from_optimizer(aggregator.optimizer, "lr")
                == 0.1
            ), "lr for adam optimizer should be 0.1"
            assert (
                OptimizerTestUtil.get_value_from_optimizer(
                    aggregator.optimizer, "weight_decay"
                )
                == 0.9
            ), "weight_decay for adam optimizer should be 0.9"
            assert (
                OptimizerTestUtil.get_value_from_optimizer(aggregator.optimizer, "eps")
                == 1e-8
            ), "eps for adam optimizer should be 1e-8"

    @classmethod
    def verify_async_trainer_aggregator(
        cls, trainer: AsyncTrainer, dict_config: Dict[str, Any]
    ):
        aggregator = trainer.aggregator
        assert isinstance(
            aggregator, AsyncAggregator
        ), "aggregator should be an instance of AsyncAggregator for AsyncTrainer"
        cls._verify_trainer_common_aggregators(aggregator, dict_config)
