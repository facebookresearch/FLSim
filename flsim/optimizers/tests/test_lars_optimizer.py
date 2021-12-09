#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from flsim.common.pytest_helper import assertEqual, assertAlmostEqual
from flsim.optimizers.sync_aggregators import (
    FedLARSSyncAggregator,
    FedAvgWithLRSyncAggregator,
    FedLARSSyncAggregatorConfig,
    FedAvgWithLRSyncAggregatorConfig,
)
from flsim.tests.utils import MockQuadratic1DFL, Quadratic1D
from omegaconf import OmegaConf


@pytest.fixture(scope="class")
def prepare_lars_optimizer_test(request):
    request.cls.quadratic1D_lars = MockQuadratic1DFL(Quadratic1D())
    request.cls.quadratic1D_sgd = MockQuadratic1DFL(Quadratic1D())


@pytest.mark.usefixtures("prepare_lars_optimizer_test")
class TestLarsOptimizer:
    def test_lars_multiple_steps(self):
        """
            a toy optimization example:
                min f(x) = 100 x^2 - 1

        minima is x=0.0, x is initialized at 1.0.
        """
        # set up quadratic parabola objective and optimizer
        dict_config_lars = {"lr": 0.01, "weight_decay": 0}
        lars_aggregator = FedLARSSyncAggregator(
            **OmegaConf.structured(FedLARSSyncAggregatorConfig(**dict_config_lars)),
            global_model=self.quadratic1D_lars,
        )

        for i in range(10):
            lars_aggregator.optimizer.zero_grad()
            metrics = self.quadratic1D_lars.fl_forward()
            loss_lars = metrics.loss
            loss_lars.backward()

            original_param_value = (
                self.quadratic1D_lars.fl_get_module().state_dict()["x"].item()
            )
            weight_norm = abs(original_param_value)
            lars_aggregator.optimizer.step()
            updated_param_value_lars = (
                self.quadratic1D_lars.fl_get_module().state_dict()["x"].item()
            )

            lars_gradient = list(self.quadratic1D_lars.fl_get_module().parameters())[
                0
            ].grad.data.item()
            lars_gradient_norm = abs(lars_gradient)

            if i == 0:
                assertAlmostEqual(
                    abs((original_param_value - updated_param_value_lars)),
                    0.01,
                    delta=1e-4,
                )
                assertAlmostEqual(lars_gradient_norm, 200, delta=1e-4)

            equivalent_sgd_lr = (
                dict_config_lars["lr"] * weight_norm / lars_gradient_norm
            )
            dict_config_sgd = {"lr": equivalent_sgd_lr}
            sgd_aggregator = FedAvgWithLRSyncAggregator(
                **OmegaConf.structured(
                    FedAvgWithLRSyncAggregatorConfig(**dict_config_sgd)
                ),
                global_model=self.quadratic1D_sgd,
            )

            sgd_aggregator.optimizer.zero_grad()
            metrics = self.quadratic1D_sgd.fl_forward()
            loss_sgd = metrics.loss
            loss_sgd.backward()
            original_param_value = (
                self.quadratic1D_sgd.fl_get_module().state_dict()["x"].item()
            )
            sgd_aggregator.optimizer.step()
            updated_param_value_sgd = (
                self.quadratic1D_sgd.fl_get_module().state_dict()["x"].item()
            )

            assertEqual(updated_param_value_lars, updated_param_value_sgd)
