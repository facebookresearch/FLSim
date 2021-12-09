#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type

import numpy as np
from flsim.utils.timing.training_duration_distribution import (
    DurationDistributionConfig,
    PerExampleGaussianDurationDistributionConfig,
    PerUserGaussianDurationDistributionConfig,
)


class ConstantEventGenTestConfig:
    """This class specifies expected behavior for a training_simulator
    that uses an event generator where:
    a) Time between events is always 1 sec (or 1 unit of time)
    b) Training duration per example is 1 sec. 4 examples per user
       Thus, overall training duration is 4 secs
    c) There are 4 users, trained over 1 epoch
    This class is used both in a unit-test below, and in an integration
    test for ASyncTrainer

    In this case, training proceeds as follows:
    t=0, user1 STARTS training. model_seqnum=0
    t=1, user2 STARTS training. model_seqnum=0
    t=2, user3 STARTS training. model_seqnum=0
    t=3, user4 STARTS training. model_seqnum=0
    t=4, user1 FINISHES trng. Jobs pending=4. seqnum=1. SeqnumDiff=0
    t=5, user2 FINISHES trng. Jobs pending=3. seqnum=2. SeqnumDiff=1
    t=6, user3 FINISHES trng. Jobs pending=2. seqnum=3. SeqnumDiff=2
    t=7, user4 FINISHES trng. Jobs pending=1. seqnum=4. SeqnumDiff=3

    So, average number of pending jobs: mean([4, 3, 2, 1]) = 10/4 = 2.5
    SeqnumDiffs: [0,1,2,3]. Mean=6/4=1.5, SD = 1.18
    """

    num_examples_per_user = 4
    training_duration_distribution_config: Type[
        DurationDistributionConfig
    ] = PerExampleGaussianDurationDistributionConfig

    training_rate = 1
    training_duration_mean = 1
    training_duration_sd = 0
    num_users = 4

    pending_jobs = training_rate * num_users * training_rate
    seqnum_diffs = [0, 1, 2, 3]
    mean_pending_jobs = np.mean(pending_jobs)
    mean_seqnum_diff = np.mean(seqnum_diffs)
    sd_seqnum_diff = np.std(seqnum_diffs)


class ConstantEventGenTestConfigPerUserGaussian(ConstantEventGenTestConfig):
    """Same as ConstantEventGenTestConfig, but assumes that training duration
    distribution is a Per-User gaussian
    """

    # in parent class, training_duration_distrib was PerExampleGaussian
    # thus, parent training time per user = #examples-per-user * training_duration_mean
    # however, in this class, training_duration_distrb = PerUserGaussian
    # thus, training time per user = training_duration_mean
    # so we multiply training_duration_mean by #examples-per-user to keep
    # training duration constant
    num_examples_per_user = 1
    training_duration_mean = 4
    training_duration_distribution_config: Type[
        DurationDistributionConfig
    ] = PerUserGaussianDurationDistributionConfig


class PoissonEventGenTestConfig:
    """This class specifies expected behavior for a training_simulator
    that uses an event generator where:
    a) Time between events is 1 sec, Poisson distributed
    b) Training duration per example is 1/4 sec. 4 examples per user
       Thus, overall training duration is 1 sec
    c) There are 4 users,
    This class is used both in a unit-test below, and in an integration
    test for AsyncTrainer
    """

    num_examples_per_user = 4
    training_duration_distribution_config: Type[
        DurationDistributionConfig
    ] = PerExampleGaussianDurationDistributionConfig
    training_rate = 1
    training_duration_mean = 1 / 4
    training_duration_sd = 0
    num_users = 4

    mean_pending_jobs = 1.25
    mean_seqnum_diff = 0.25
    sd_seqnum_diff = 0.433


class PoissonEventGenTestConfigPerUserGaussian(PoissonEventGenTestConfig):
    """Same as PoissonEventGenTestConfig, but assumes that training duration
    distribution is a Per-User gaussian
    """

    num_examples_per_user = 1
    training_duration_distribution_config: Type[
        DurationDistributionConfig
    ] = PerUserGaussianDurationDistributionConfig
    training_duration_mean = 1
