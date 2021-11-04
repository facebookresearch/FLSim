#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import numpy as np
from flsim.utils.async_trainer.async_example_weights import (
    AsyncExampleWeightConfig,
    ExampleWeight,
)
from flsim.utils.tests.helpers.async_weights_test_utils import (
    AsyncExampleWeightsTestUtils,
)
from hydra.utils import instantiate
from libfb.py import testutil


class AsyncExampleWeightsTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    @testutil.data_provider(AsyncExampleWeightsTestUtils.provide_example_weight_configs)
    def test_string_conversion(
        self,
        example_weight_config: AsyncExampleWeightConfig,
        example_weight_class: ExampleWeight,
    ):
        """Check that strings are correctly converted to ExampleWeight"""
        obj = instantiate(example_weight_config)
        self.assertEqual(obj.__class__, example_weight_class)

    @testutil.data_provider(AsyncExampleWeightsTestUtils.provide_example_weight_configs)
    @testutil.data_provider(AsyncExampleWeightsTestUtils.provide_avg_num_examples)
    def test_example_weight_compute(
        self,
        example_weight_config: AsyncExampleWeightConfig,
        example_weight_class: ExampleWeight,
        avg_num_examples=1,
    ):
        """Test that all weight computation works as expected"""
        # generate 10 random integers
        max_num_examples = 10000
        for _ in range(10):
            num_examples = np.random.randint(1, max_num_examples)
            example_weight_config.avg_num_examples = avg_num_examples
            obj = instantiate(example_weight_config)
            self.assertEqual(
                obj.weight(num_examples),
                AsyncExampleWeightsTestUtils.expected_weight(
                    avg_num_examples=avg_num_examples,
                    num_examples=num_examples,
                    example_weight_class=example_weight_class,
                ),
            )
