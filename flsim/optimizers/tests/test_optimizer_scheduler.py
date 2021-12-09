#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from flsim.common.pytest_helper import assertEqual, assertTrue
from flsim.optimizers.optimizer_scheduler import (
    ArmijoLineSearch,
    ArmijoLineSearchSchedulerConfig,
)
from flsim.tests.utils import MockQuadratic1DFL, Quadratic1D
from omegaconf import OmegaConf


class TestOptimizerScheduler:
    def test_armijo_line_search_on_parabola(self):
        """
            a toy optimization example:
                min f(x) = 100 x^2 - 1

        minima is x=0.0, x is initialized at 1.0.
        Gradient descent with constant step-size 0.01 will never
        converge, and in fact, jump between -1 and +1 interleavingly.
        In contrast, Armijo line-search reduces step-sizes to avoid
        "over-shooting" and converges to 0.
        """
        # set up quadratic parabola objective and optimizer
        quadratic1D = MockQuadratic1DFL(Quadratic1D())
        optimizer = torch.optim.SGD(
            quadratic1D.fl_get_module().parameters(), lr=0.01, momentum=0.0
        )
        # run (deterministic) GD for 10 steps with step-size = 0.01,
        # with constant step-size = 0.01, and x0 = 1.0, the iteration
        # never converge and will jump between -1.0 and 1.0 interleavingly
        for i in range(10):
            optimizer.zero_grad()
            metrics = quadratic1D.fl_forward()
            quadratic_func_val = metrics.loss
            quadratic_func_val.backward()
            optimizer.step()
            obj_val = quadratic1D.fl_get_module().state_dict()["x"].item()
            assertEqual(obj_val, (-1.0) ** (i + 1))

        # set up (again) quadratic parabola objective and optimizer
        quadratic1D = MockQuadratic1DFL(Quadratic1D())
        optimizer = torch.optim.SGD(
            quadratic1D.fl_get_module().parameters(), lr=0.01, momentum=0.0
        )
        # use Armijo line-search for optimizer step-size selection
        # same initial step-size
        config = ArmijoLineSearchSchedulerConfig()
        config.base_lr = 0.01
        config.reset = True
        armijo_line_search_scheduler = ArmijoLineSearch(
            optimizer=optimizer, **OmegaConf.structured(config)
        )
        # run for 10 steps
        for t in range(10):
            optimizer.zero_grad()
            metrics = quadratic1D.fl_forward()
            quadratic_func_val = metrics.loss
            quadratic_func_val.backward()
            armijo_line_search_scheduler.step(metrics, quadratic1D, None, t)
            optimizer.step()
        # check converging to 0 (true answer)
        assertTrue(quadratic1D.fl_get_module().state_dict()["x"].item() <= 1e-7)
