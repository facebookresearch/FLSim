#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from hydra.core.config_store import ConfigStore  #  @manual

from .async_example_weights import (
    Log10ExampleWeightConfig,
    SqrtExampleWeightConfig,
    LinearExampleWeightConfig,
    EqualExampleWeightConfig,
)
from .async_staleness_weights import (
    ConstantStalenessWeightConfig,
    ThresholdStalenessWeightConfig,
    PolynomialStalenessWeightConfig,
)
from .async_weights import AsyncWeightConfig
from .training_event_generator import (
    PoissonAsyncTrainingStartTimeDistrConfig,
    ConstantAsyncTrainingStartTimeDistrConfig,
    AsyncTrainingEventGeneratorFromListConfig,
    AsyncTrainingEventGeneratorConfig,
)


ConfigStore.instance().store(
    name="base_log10_example_weight",
    node=Log10ExampleWeightConfig,
    group="example_weight",
)


ConfigStore.instance().store(
    name="base_sqrt_example_weight",
    node=SqrtExampleWeightConfig,
    group="example_weight",
)


ConfigStore.instance().store(
    name="base_linear_example_weight",
    node=LinearExampleWeightConfig,
    group="example_weight",
)


ConfigStore.instance().store(
    name="base_equal_example_weight",
    node=EqualExampleWeightConfig,
    group="example_weight",
)


ConfigStore.instance().store(
    name="base_constant_staleness_weight",
    node=ConstantStalenessWeightConfig,
    group="staleness_weight",
)


ConfigStore.instance().store(
    name="base_threshold_staleness_weight",
    node=ThresholdStalenessWeightConfig,
    group="staleness_weight",
)


ConfigStore.instance().store(
    name="base_polynomial_staleness_weight",
    node=PolynomialStalenessWeightConfig,
    group="staleness_weight",
)


ConfigStore.instance().store(
    name="base_async_weight",
    node=AsyncWeightConfig,
    group="async_weight",
)


ConfigStore.instance().store(
    name="base_poisson_training_start_time_distribution",
    node=PoissonAsyncTrainingStartTimeDistrConfig,
    group="training_start_time_distribution",
)


ConfigStore.instance().store(
    name="base_constant_training_start_time_distribution",
    node=ConstantAsyncTrainingStartTimeDistrConfig,
    group="training_start_time_distribution",
)


ConfigStore.instance().store(
    name="base_async_training_event_generator_from_list",
    node=AsyncTrainingEventGeneratorFromListConfig,
    group="training_event_generator",
)


ConfigStore.instance().store(
    name="base_async_training_event_generator",
    node=AsyncTrainingEventGeneratorConfig,
    group="training_event_generator",
)
