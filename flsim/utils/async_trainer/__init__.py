#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
