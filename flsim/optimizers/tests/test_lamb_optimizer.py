#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
from flsim.common.pytest_helper import assertEqual
from flsim.optimizers.sync_aggregators import (
    FedAdamSyncAggregator,
    FedLAMBSyncAggregator,
    FedAdamSyncAggregatorConfig,
    FedLAMBSyncAggregatorConfig,
)
from flsim.tests.utils import MockQuadratic1DFL, Quadratic1D
from omegaconf import OmegaConf


def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _test_lamb_multiple_steps(test_case, weight_decay=0):
    """
        a toy optimization example:
            min f(x) = 100 x^2 - 1

    minima is x=0.0, x is initialized at 1.0.
    """
    # set up quadratic parabola objective and optimizer
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    dict_config_lamb = {
        "lr": 0.01,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "weight_decay": weight_decay,
    }
    lamb_aggregator = FedLAMBSyncAggregator(
        **OmegaConf.structured(FedLAMBSyncAggregatorConfig(**dict_config_lamb)),
        global_model=test_case.quadratic1D_lamb,
    )

    dict_config_adam = {"lr": 0.01, "weight_decay": weight_decay, "eps": eps}
    adam_aggregator = FedAdamSyncAggregator(
        **OmegaConf.structured(FedAdamSyncAggregatorConfig(**dict_config_adam)),
        global_model=test_case.quadratic1D_adam,
    )

    m_t = 0.0
    v_t = 0.0

    for i in range(1, 11):
        lamb_aggregator.optimizer.zero_grad()
        metrics = test_case.quadratic1D_lamb.fl_forward()
        loss_lamb = metrics.loss
        loss_lamb.backward()

        original_param_value = (
            test_case.quadratic1D_lamb.fl_get_module().state_dict()["x"].item()
        )
        weight_norm = abs(original_param_value)
        lamb_aggregator.optimizer.step()
        updated_param_value_lamb = (
            test_case.quadratic1D_lamb.fl_get_module().state_dict()["x"].item()
        )

        g_t = list(test_case.quadratic1D_lamb.fl_get_module().parameters())[
            0
        ].grad.data.item()

        bias_correction1 = 1.0 - beta1 ** i
        bias_correction2 = 1.0 - beta2 ** i

        m_t = beta1 * m_t + (1.0 - beta1) * g_t
        v_t = beta2 * v_t + (1.0 - beta2) * g_t ** 2

        lamb_update_unnormalized = (m_t / bias_correction1) / (
            math.sqrt(v_t / bias_correction2) + eps
        )
        lamb_update_norm = abs(lamb_update_unnormalized)

        equivalent_adam_lr = dict_config_lamb["lr"] * weight_norm / lamb_update_norm

        # we can't initialize a new adam each time because it would lose its momentum
        adjust_learning_rate(adam_aggregator.optimizer, equivalent_adam_lr)

        adam_aggregator.optimizer.zero_grad()
        metrics = test_case.quadratic1D_adam.fl_forward()
        loss_adam = metrics.loss
        loss_adam.backward()
        original_param_value = (
            test_case.quadratic1D_adam.fl_get_module().state_dict()["x"].item()
        )
        adam_aggregator.optimizer.step()
        updated_param_value_adam = (
            test_case.quadratic1D_adam.fl_get_module().state_dict()["x"].item()
        )

        assertEqual(updated_param_value_lamb, updated_param_value_adam)


@pytest.fixture(scope="class")
def prepare_lamb_optimizer_test(request):
    request.cls.quadratic1D_lamb = MockQuadratic1DFL(Quadratic1D())
    request.cls.quadratic1D_adam = MockQuadratic1DFL(Quadratic1D())


@pytest.mark.usefixtures("prepare_lamb_optimizer_test")
class TestLambOptimizer:
    def test_lamb_no_weight_decay(self):
        _test_lamb_multiple_steps(self, weight_decay=0.0)

    def test_lamb_weight_decay(self):
        _test_lamb_multiple_steps(self, weight_decay=0.1)
