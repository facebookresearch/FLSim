#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

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
    @classmethod
    def provide_example_weight_configs(cls):
        return (
            {
                "example_weight_config": EqualExampleWeightConfig(),
                "example_weight_class": EqualExampleWeight,
            },
            {
                "example_weight_config": LinearExampleWeightConfig(),
                "example_weight_class": LinearExampleWeight,
            },
            {
                "example_weight_config": SqrtExampleWeightConfig(),
                "example_weight_class": SqrtExampleWeight,
            },
            {
                "example_weight_config": Log10ExampleWeightConfig(),
                "example_weight_class": Log10ExampleWeight,
            },
        )

    @classmethod
    def provide_avg_num_examples(cls):
        return ({"avg_num_examples": 1}, {"avg_num_examples": 10000})

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
    @classmethod
    def provide_staleness_weight_configs(cls):
        return (
            {
                "staleness_weight_config": ConstantStalenessWeightConfig(),
                "staleness_weight_class": ConstantStalenessWeight,
            },
            {
                "staleness_weight_config": ThresholdStalenessWeightConfig(
                    cutoff=1, value_after_cutoff=0.1
                ),
                "staleness_weight_class": ThresholdStalenessWeight,
            },
            {
                "staleness_weight_config": PolynomialStalenessWeightConfig(
                    exponent=0.5
                ),
                "staleness_weight_class": PolynomialStalenessWeight,
            },
        )

    @classmethod
    def provide_avg_staleness_weights(cls):
        return ({"avg_staleness": 1}, {"avg_staleness": 10000})

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
