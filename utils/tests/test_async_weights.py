#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import numpy as np
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
from libfb.py import testutil


class AsyncExampleWeightsTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    # two data_providers together produce a cartesian product
    @testutil.data_provider(AsyncExampleWeightsTestUtils.provide_example_weight_configs)
    @testutil.data_provider(
        AsyncStalenessWeightsTestUtils.provide_staleness_weight_configs
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
        self.assertEqual(obj.example_weight.__class__, example_weight_class)
        self.assertEqual(obj.staleness_weight.__class__, staleness_weight_class)

    @testutil.data_provider(AsyncExampleWeightsTestUtils.provide_example_weight_configs)
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
                self.assertEqual(expected_combined_weight, combined_weight)
