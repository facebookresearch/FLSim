#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore  #  @manual

from .async_aggregators import (
    FedAdamAsyncAggregatorConfig,
    FedAvgWithLRAsyncAggregatorConfig,
    FedAvgWithLRWithMomentumAsyncAggregatorConfig,
    FedAdamHybridAggregatorConfig,
    FedAvgWithLRHybridAggregatorConfig,
)
from .local_optimizers import LocalOptimizerSGDConfig, LocalOptimizerFedProxConfig
from .optimizer_scheduler import (
    ConstantLRSchedulerConfig,
    LRBatchSizeNormalizerSchedulerConfig,
    ArmijoLineSearchSchedulerConfig,
)
from .server_optimizers import (
    FedAdamOptimizerConfig,
    FedAvgOptimizerConfig,
    FedAvgWithLROptimizerConfig,
    FedLAMBOptimizerConfig,
    FedLARSOptimizerConfig,
)
from .sync_aggregators import (
    FedAvgSyncAggregatorConfig,
    FedAdamSyncAggregatorConfig,
    FedAvgWithLRSyncAggregatorConfig,
    FedLARSSyncAggregatorConfig,
    FedLAMBSyncAggregatorConfig,
)

ConfigStore.instance().store(
    name="base_optimizer_sgd",
    node=LocalOptimizerSGDConfig,
    group="optimizer",
)


ConfigStore.instance().store(
    name="base_optimizer_fedprox",
    node=LocalOptimizerFedProxConfig,
    group="optimizer",
)

ConfigStore.instance().store(
    name="base_constant_lr_scheduler",
    node=ConstantLRSchedulerConfig,
    group="lr_scheduler",
)


ConfigStore.instance().store(
    name="base_lr_batch_size_normalizer_scheduler",
    node=LRBatchSizeNormalizerSchedulerConfig,
    group="lr_scheduler",
)


ConfigStore.instance().store(
    name="base_armijo_line_search_lr_scheduer",
    node=ArmijoLineSearchSchedulerConfig,
    group="lr_scheduler",
)


ConfigStore.instance().store(
    name="base_fed_avg_sync_aggregator",
    node=FedAvgSyncAggregatorConfig,
    group="aggregator",
)


ConfigStore.instance().store(
    name="base_fed_avg_with_lr_sync_aggregator",
    node=FedAvgWithLRSyncAggregatorConfig,
    group="aggregator",
)


ConfigStore.instance().store(
    name="base_fed_adam_sync_aggregator",
    node=FedAdamSyncAggregatorConfig,
    group="aggregator",
)


ConfigStore.instance().store(
    name="base_fed_lars_sync_aggregator",
    node=FedLARSSyncAggregatorConfig,
    group="aggregator",
)


ConfigStore.instance().store(
    name="base_fed_lamb_sync_aggregator",
    node=FedLAMBSyncAggregatorConfig,
    group="aggregator",
)


ConfigStore.instance().store(
    name="base_fed_avg_with_lr_async_aggregator",
    node=FedAvgWithLRAsyncAggregatorConfig,
    group="aggregator",
)


ConfigStore.instance().store(
    name="base_fed_avg_with_lr_with_momentum_async_aggregator",
    node=FedAvgWithLRWithMomentumAsyncAggregatorConfig,
    group="aggregator",
)

ConfigStore.instance().store(
    name="base_fed_adam_async_aggregator",
    node=FedAdamAsyncAggregatorConfig,
    group="aggregator",
)


ConfigStore.instance().store(
    name="base_fed_avg_with_lr_hybrid_aggregator",
    node=FedAvgWithLRHybridAggregatorConfig,
    group="aggregator",
)

ConfigStore.instance().store(
    name="base_fed_adam_hybrid_aggregator",
    node=FedAdamHybridAggregatorConfig,
    group="aggregator",
)

ConfigStore.instance().store(
    name="base_fed_adam",
    node=FedAdamOptimizerConfig,
    group="server_optimizer",
)


ConfigStore.instance().store(
    name="base_fed_avg",
    node=FedAvgOptimizerConfig,
    group="server_optimizer",
)


ConfigStore.instance().store(
    name="base_fed_avg_with_lr",
    node=FedAvgWithLROptimizerConfig,
    group="server_optimizer",
)


ConfigStore.instance().store(
    name="base_fed_lars",
    node=FedLARSOptimizerConfig,
    group="server_optimizer",
)

ConfigStore.instance().store(
    name="base_fed_lamb",
    node=FedLAMBOptimizerConfig,
    group="server_optimizer",
)
