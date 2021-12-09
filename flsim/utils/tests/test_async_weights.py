#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from flsim.common.pytest_helper import assertEqual
from flsim.utils.async_trainer.async_example_weights import (
    ExampleWeight,
    AsyncExampleWeightConfig,
)
from flsim.utils.async_trainer.async_staleness_weights import (
    AsyncStalenessWeightConfig,
    StalenessWeight,
    ConstantStalenessWeightConfig,
    ThresholdStalenessWeightConfig,
    PolynomialStalenessWeightConfig,
)
from flsim.utils.async_trainer.async_weights import AsyncWeightConfig
from flsim.utils.tests.helpers.async_weights_test_utils import (
    AsyncExampleWeightsTestUtils,
    AsyncStalenessWeightsTestUtils,
)
from hydra.utils import instantiate


class TestAsyncExampleWeights:

    # two parametrize together produce a cartesian product
    @pytest.mark.parametrize(
        "example_weight_config, example_weight_class",
        AsyncExampleWeightsTestUtils.EXAMPLE_WEIGHT_TEST_CONFIGS,
    )
    @pytest.mark.parametrize(
        "staleness_weight_config, staleness_weight_class",
        AsyncStalenessWeightsTestUtils.STALENESS_WEIGHT_TEST_CONFIGS,
    )
    def test_string_conversion(
        self,
        example_weight_config: AsyncExampleWeightConfig,
        example_weight_class: ExampleWeight,
        staleness_weight_config: AsyncStalenessWeightConfig,
        staleness_weight_class: StalenessWeight,
    ) -> None:
        """Check that strings are correctly converted to AsyncWeight"""
        obj = instantiate(
            AsyncWeightConfig(
                staleness_weight=staleness_weight_config,
                example_weight=example_weight_config,
            )
        )
        assertEqual(obj.example_weight.__class__, example_weight_class)
        assertEqual(obj.staleness_weight.__class__, staleness_weight_class)

    @pytest.mark.parametrize(
        "example_weight_config, example_weight_class",
        AsyncExampleWeightsTestUtils.EXAMPLE_WEIGHT_TEST_CONFIGS,
    )
    def test_weight_compute(
        self,
        example_weight_config: AsyncExampleWeightConfig,
        example_weight_class: ExampleWeight,
        avg_num_examples: int = 1,
        avg_staleness: int = 1,
    ):
        """Test that all weight computation works as expected"""
        max_num_examples = 10000
        max_staleness = 10000
        cutoff = 5000
        value_after_cutoff = 0.001
        exponent = 0.5
        # dict below tells us how to initialize weight object for different
        # staleness weight types
        staleness_weight_configs = [
            ConstantStalenessWeightConfig(),
            ThresholdStalenessWeightConfig(
                cutoff=cutoff, value_after_cutoff=value_after_cutoff
            ),
            PolynomialStalenessWeightConfig(exponent=exponent),
        ]
        for staleness_weight_config in staleness_weight_configs:
            staleness_weight_obj = instantiate(staleness_weight_config)
            # for 10 random integers
            for _ in range(10):
                num_examples = np.random.randint(1, max_num_examples)

                staleness = np.random.randint(1, max_staleness)
                staleness_weight = staleness_weight_obj.weight(staleness)

                example_weight_config.avg_num_examples = avg_num_examples
                example_weight_obj = instantiate(example_weight_config)
                example_weight = example_weight_obj.weight(num_examples)

                expected_combined_weight = example_weight * staleness_weight
                combined_weight_object = instantiate(
                    AsyncWeightConfig(
                        example_weight=example_weight_config,
                        staleness_weight=staleness_weight_config,
                    )
                )
                combined_weight = combined_weight_object.weight(
                    num_examples=num_examples, staleness=staleness
                )
                assertEqual(expected_combined_weight, combined_weight)
