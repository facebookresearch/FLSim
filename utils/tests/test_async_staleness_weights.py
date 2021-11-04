#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
import numpy as np
from flsim.utils.async_trainer.async_staleness_weights import (
    AsyncStalenessWeightConfig,
    StalenessWeight,
    ConstantStalenessWeightConfig,
    ThresholdStalenessWeightConfig,
    PolynomialStalenessWeightConfig,
)
from flsim.utils.tests.helpers.async_weights_test_utils import (
    AsyncStalenessWeightsTestUtils,
)
from hydra.utils import instantiate
from libfb.py import testutil


class AsyncStalenessWeightsTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    @testutil.data_provider(
        AsyncStalenessWeightsTestUtils.provide_staleness_weight_configs
    )
    def test_string_conversion(
        self,
        staleness_weight_config: AsyncStalenessWeightConfig,
        staleness_weight_class: StalenessWeight,
    ):
        obj = instantiate(staleness_weight_config)
        self.assertEqual(obj.__class__, staleness_weight_class)

    @testutil.data_provider(
        AsyncStalenessWeightsTestUtils.provide_avg_staleness_weights
    )
    def test_constant_weight_compute(self, avg_staleness=1):
        """Test that all constant weight computation works as expected"""
        max_staleness = 10000
        obj = instantiate(ConstantStalenessWeightConfig(avg_staleness=avg_staleness))
        for _i in range(10):
            staleness = np.random.randint(1, max_staleness)
            numerator = AsyncStalenessWeightsTestUtils.get_constant_wt()
            denom = AsyncStalenessWeightsTestUtils.get_constant_wt()
            self.assertEqual(obj.weight(staleness), numerator / denom)

    @testutil.data_provider(
        AsyncStalenessWeightsTestUtils.provide_avg_staleness_weights
    )
    def test_threshold_weight_compute(self, avg_staleness=1):
        """Test that threshold weight computation works as expected"""
        max_staleness = 10000
        for _i in range(10):
            cutoff = np.random.randint(1, max_staleness)
            value_after_cutoff = np.random.uniform(low=0.0, high=1.0)
            obj = instantiate(
                ThresholdStalenessWeightConfig(
                    avg_staleness=avg_staleness,
                    cutoff=cutoff,
                    value_after_cutoff=value_after_cutoff,
                )
            )
            staleness = np.random.randint(1, max_staleness)
            numerator = AsyncStalenessWeightsTestUtils.get_threshold_wt(
                staleness=staleness,
                cutoff=cutoff,
                value_after_cutoff=value_after_cutoff,
            )
            denom = AsyncStalenessWeightsTestUtils.get_threshold_wt(
                staleness=avg_staleness,
                cutoff=cutoff,
                value_after_cutoff=value_after_cutoff,
            )
            self.assertEqual(obj.weight(staleness), numerator / denom)

    @testutil.data_provider(
        AsyncStalenessWeightsTestUtils.provide_avg_staleness_weights
    )
    def test_polynomial_weight_compute(self, avg_staleness=1):
        """Test that threshold weight computation works as expected"""
        max_staleness = 10000
        for _i in range(10):
            exponent = np.random.uniform(low=0.0, high=1.0)
            obj = instantiate(
                PolynomialStalenessWeightConfig(
                    avg_staleness=avg_staleness, exponent=exponent
                )
            )
            staleness = np.random.randint(1, max_staleness)
            numerator = AsyncStalenessWeightsTestUtils.get_polynomial_wt(
                staleness=staleness, exponent=exponent
            )
            denom = AsyncStalenessWeightsTestUtils.get_polynomial_wt(
                staleness=avg_staleness, exponent=exponent
            )
            self.assertEqual(obj.weight(staleness), numerator / denom)

    def test_polynomial_weight_zero_exponent(self):
        """For polynomial weight, if exponent is zero, wt=1 regardless of
        staleness or average staleness
        """
        max_staleness = 10000
        # test for 10 random values of staleness and avg staleness
        for _i in range(10):
            staleness = np.random.randint(1, max_staleness)
            avg_staleness = np.random.randint(1, max_staleness)
            obj = instantiate(
                PolynomialStalenessWeightConfig(
                    avg_staleness=avg_staleness, exponent=0.0
                )
            )
            self.assertEqual(obj.weight(staleness), 1.0)

    def test_polynomial_weight_bad_exponent(self):
        """For polynomial weight, exponent must be between 0 and 1, else error"""
        cfg = PolynomialStalenessWeightConfig(avg_staleness=0, exponent=-0.1)

        # negative exponent causes error
        with self.assertRaises(
            (
                AssertionError,  # with Hydra 1.1
                hydra.errors.HydraException,  # with Hydra 1.0
            ),
        ):
            cfg.exponent = -0.1
            instantiate(cfg)

        # exponent greater than 1.0 causes error
        with self.assertRaises(
            (
                AssertionError,  # with Hydra 1.1
                hydra.errors.HydraException,  # with Hydra 1.0
            ),
        ):
            cfg.exponent = 1.1
            instantiate(cfg)

        # exponent = 0.0 is fine
        cfg.exponent = 0.0
        instantiate(cfg)

        # exponent = 1.0 is fine
        cfg.exponent = 1.0
        instantiate(cfg)
