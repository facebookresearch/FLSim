#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import log10, sqrt

from flsim.utils.async_trainer.async_example_weights import (
    EqualExampleWeightConfig,
    EqualExampleWeight,
    ExampleWeight,
    LinearExampleWeightConfig,
    LinearExampleWeight,
    Log10ExampleWeightConfig,
    Log10ExampleWeight,
    SqrtExampleWeightConfig,
    SqrtExampleWeight,
)
from flsim.utils.async_trainer.async_staleness_weights import (
    ConstantStalenessWeightConfig,
    ConstantStalenessWeight,
    PolynomialStalenessWeightConfig,
    PolynomialStalenessWeight,
    ThresholdStalenessWeightConfig,
    ThresholdStalenessWeight,
)


class AsyncExampleWeightsTestUtils:
    EXAMPLE_WEIGHT_TEST_CONFIGS = [
        (EqualExampleWeightConfig(), EqualExampleWeight),
        (LinearExampleWeightConfig(), LinearExampleWeight),
        (SqrtExampleWeightConfig(), SqrtExampleWeight),
        (Log10ExampleWeightConfig(), Log10ExampleWeight),
    ]

    AVG_NUMBER_OF_EXAMPLES = [1, 10000]

    @classmethod
    def expected_weight(
        cls,
        avg_num_examples: int,
        num_examples: int,
        example_weight_class: ExampleWeight,
    ) -> float:
        if example_weight_class == EqualExampleWeight:
            return 1.0
        elif example_weight_class == LinearExampleWeight:
            return num_examples / avg_num_examples
        elif example_weight_class == SqrtExampleWeight:
            return sqrt(num_examples) / sqrt(avg_num_examples)
        elif example_weight_class == Log10ExampleWeight:
            return log10(1 + num_examples) / log10(1 + avg_num_examples)
        else:
            raise AssertionError(f"Unknown example_weight type:{example_weight_class}")


class AsyncStalenessWeightsTestUtils:
    STALENESS_WEIGHT_TEST_CONFIGS = [
        (ConstantStalenessWeightConfig(), ConstantStalenessWeight),
        (
            ThresholdStalenessWeightConfig(cutoff=1, value_after_cutoff=0.1),
            ThresholdStalenessWeight,
        ),
        (PolynomialStalenessWeightConfig(exponent=0.5), PolynomialStalenessWeight),
    ]
    AVG_TEST_STALENESS = [1, 10000]

    @classmethod
    def get_constant_wt(cls) -> float:
        return 1.0

    @classmethod
    def get_threshold_wt(
        cls, staleness: int, cutoff: int, value_after_cutoff: float
    ) -> float:
        return 1.0 if staleness <= cutoff else value_after_cutoff

    @classmethod
    def get_polynomial_wt(cls, staleness: int, exponent: float) -> float:
        return 1 / ((1 + staleness) ** exponent)
