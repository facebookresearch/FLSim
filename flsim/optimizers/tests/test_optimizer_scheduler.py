#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from flsim.common.pytest_helper import assertAlmostEqual, assertEqual, assertTrue
from flsim.optimizers.optimizer_scheduler import (
    ArmijoLineSearch,
    ArmijoLineSearchSchedulerConfig,
    MultiStepLRScheduler,
    MultiStepLRSchedulerConfig,
    StepLRScheduler,
    StepLRSchedulerConfig,
)
from flsim.utils.test_utils import MockQuadratic1DFL, Quadratic1D
from omegaconf import OmegaConf


class TestOptimizerScheduler:
    def test_armijo_line_search_on_parabola(self) -> None:
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
            assertAlmostEqual(obj_val, (-1.0) ** (i + 1), places=6)

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


class TestMultiStepLRScheduler:
    def test_decay_lr_correct(self):
        quadratic1D = MockQuadratic1DFL(Quadratic1D())
        lr = 10.0
        optimizer = torch.optim.SGD(
            quadratic1D.fl_get_module().parameters(), lr=lr, momentum=0.0
        )
        config = MultiStepLRSchedulerConfig(base_lr=lr, gamma=0.1, milestones=[2, 6])
        scheduler = MultiStepLRScheduler(
            optimizer=optimizer, **OmegaConf.structured(config)
        )
        lrs = []
        for t in range(6):
            scheduler.step(global_round_num=t)
            lrs.append(scheduler.get_lr()[0])

        assertEqual(
            [
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            lrs,
        )


class TestStepLRScheduler:
    def test_step_lr_correct(self):
        quadratic1D = MockQuadratic1DFL(Quadratic1D())
        lr = 10.0
        optimizer = torch.optim.SGD(
            quadratic1D.fl_get_module().parameters(), lr=lr, momentum=0.0
        )
        config = StepLRSchedulerConfig(base_lr=lr, gamma=0.1, step_size=2)
        scheduler = StepLRScheduler(optimizer=optimizer, **OmegaConf.structured(config))
        lrs = []
        for t in range(6):
            scheduler.step(global_round_num=t)
            lrs.append(scheduler.get_lr()[0])

        assertEqual(
            [10.0, 10.0, 1.0, 1.0, 0.10000000000000002, 0.10000000000000002],
            lrs,
        )
