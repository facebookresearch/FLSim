#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from flsim.common.pytest_helper import (
    assertTrue,
    assertIsInstance,
    assertAlmostEqual,
    assertEqual,
)
from flsim.common.timeout_simulator import (
    GaussianTimeOutSimulator,
    GaussianTimeOutSimulatorConfig,
)
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
)
from omegaconf import OmegaConf


class TestTrainingTimeOutSimulator:
    def test_online_stat_computation_correct(self):
        timeout_simulator = GaussianTimeOutSimulator(
            **OmegaConf.structured(
                GaussianTimeOutSimulatorConfig(
                    timeout_wall_per_round=3.0,
                    fl_stopping_time=99999.0,
                    duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                        training_duration_mean=1.0, training_duration_sd=1.0
                    ),
                ),
            )
        )
        assertTrue(isinstance(timeout_simulator, GaussianTimeOutSimulator))
        num_users = 1000
        max_sample_per_user = 20
        num_samples_per_user = [
            (user % max_sample_per_user) + 1 for user in range(num_users)
        ]
        training_time_all_users = []
        for i in range(num_users):
            sim_device_perf = timeout_simulator.simulate_per_example_training_time()
            sim_train_time = timeout_simulator.simulate_training_time(
                sim_device_perf, num_samples_per_user[i]
            )
            timeout_simulator.track_training_time_distribution(sim_train_time)
            training_time_all_users.append(sim_train_time)

        # using np.allclose to compare floats
        assertTrue(
            np.allclose(
                [timeout_simulator.sample_mean_per_user],
                [np.mean(training_time_all_users)],
            )
        )
        assertTrue(
            np.allclose(
                [timeout_simulator.sample_var_per_user],
                [np.var(training_time_all_users, ddof=1)],
            )
        )

    def test_fl_stops_small_stopping_time(self):
        """
        create a dummy "training loop" (loop through users and rounds
        without actual training) and stops the training loop by explicitly
        comparing against fl_stopping_time, versus stopping the training
        loop via internally tracked variablesin timeout_simulator and
        timeout_simulator.stop_fl() method.
        """

        timeout_simulator = GaussianTimeOutSimulator(
            **OmegaConf.structured(
                GaussianTimeOutSimulatorConfig(
                    timeout_wall_per_round=7.0,
                    fl_stopping_time=10.0,
                    duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                        training_duration_mean=5.0, training_duration_sd=1.0
                    ),
                ),
            )
        )
        assertIsInstance(timeout_simulator, GaussianTimeOutSimulator)
        num_users = 1000
        max_sample_per_user = 20
        num_rounds = 10
        num_users_per_round = int(num_users / num_rounds)
        num_samples_per_user = [
            (user % max_sample_per_user) + 1 for user in range(num_users)
        ]
        elapsed_time_each_round = []
        torch.manual_seed(1)
        # ground-truth: maually calling private functions in timeout_simulator and
        # record all elapsed time into list for book-keeping
        for r in range(num_rounds):
            training_time_users_in_round = []
            for i in range(num_users_per_round):
                sim_device_perf = timeout_simulator.simulate_per_example_training_time()
                sim_train_time = timeout_simulator.simulate_training_time(
                    sim_device_perf, num_samples_per_user[r * num_users_per_round + i]
                )
                training_time_users_in_round.append(sim_train_time)
            elapsed_time_each_round.append(max(training_time_users_in_round))
            if sum(elapsed_time_each_round) >= timeout_simulator._fl_stopping_time:
                fl_stopping_round_ground_truth = r
                break

        torch.manual_seed(1)
        # using timeout_simulator.track_fl_elapsed_time and timeout_simulator.stop_fl()
        # to determine stopping condition
        for r in range(num_rounds):
            training_time_users_in_round = []
            for i in range(num_users_per_round):
                sim_device_perf = timeout_simulator.simulate_per_example_training_time()
                sim_train_time = timeout_simulator.simulate_training_time(
                    sim_device_perf, num_samples_per_user[r * num_users_per_round + i]
                )
                training_time_users_in_round.append(sim_train_time)

            timeout_simulator.track_fl_elapsed_time(training_time_users_in_round)
            if timeout_simulator.stop_fl():
                fl_stopping_round_simulator = r
                break

        assertAlmostEqual(
            sum(elapsed_time_each_round),
            timeout_simulator._fl_total_elapse_time,
            delta=1e-6,
        )
        assertEqual(fl_stopping_round_ground_truth, fl_stopping_round_simulator)

    def test_fl_stops_small_stopping_time_2(self):
        r"""
        training time should not be negative
        """
        timeout_simulator = GaussianTimeOutSimulator(
            **OmegaConf.structured(
                GaussianTimeOutSimulatorConfig(
                    timeout_wall_per_round=9999,
                    fl_stopping_time=99999,
                    duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                        training_duration_mean=1.0,
                        training_duration_sd=10.0,
                        training_duration_min=0.00001,
                    ),
                ),
            )
        )
        for _ in range(1000):
            assert timeout_simulator.simulate_per_example_training_time() > 0.0
