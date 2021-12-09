#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple, Type

import numpy as np
import pytest
import torch
from flsim.clients.base_client import ClientConfig
from flsim.common.pytest_helper import (
    assertAlmostEqual,
    assertEqual,
    assertTrue,
    assertLessEqual,
    assertGreaterEqual,
)
from flsim.common.training_event_handler import TestAsyncTrainingEventHandler
from flsim.common.training_simulator import AsyncTrainingSimulator
from flsim.data.data_provider import FLDataProviderFromList, IFLDataProvider
from flsim.tests.utils import check_inherit_logging_level
from flsim.utils.async_trainer.async_user_selector import (
    AsyncUserSelector,
    RandomAsyncUserSelector,
    RoundRobinAsyncUserSelector,
)
from flsim.utils.async_trainer.device_state import DeviceState, TrainingState
from flsim.utils.async_trainer.training_event_generator import (
    AsyncTrainingEventGenerator,
    AsyncTrainingEventGeneratorConfig,
    AsyncTrainingEventGeneratorFromList,
    AsyncTrainingEventGeneratorFromListConfig,
    ConstantAsyncTrainingStartTimeDistrConfig,
    EventTimingInfo,
    PoissonAsyncTrainingStartTimeDistrConfig,
)
from flsim.utils.sample_model import MockFLModel
from flsim.utils.tests.helpers.test_training_simulator_utils import (  # noqa: B950 line is too long
    ConstantEventGenTestConfig,
    ConstantEventGenTestConfigPerUserGaussian,
    PoissonEventGenTestConfig,
    PoissonEventGenTestConfigPerUserGaussian,
)
from flsim.utils.timing.training_duration_distribution import (
    DurationDistributionConfig,
    PerExampleGaussianDurationDistributionConfig,
    PerUserGaussianDurationDistributionConfig,
    PerUserHalfNormalDurationDistributionConfig,
    PerUserUniformDurationDistributionConfig,
    PerUserExponentialDurationDistributionConfig,
)
from omegaconf import OmegaConf


class Globals:
    DEVICE_STATE_IDX = 0
    TRAINING_STATE_IDX = 1


@pytest.fixture(scope="class")
def prepare_training_simulator_test_utils(request):
    torch.random.manual_seed(0)
    request.cls.shared_client_config = ClientConfig(
        epochs=1,
        max_clip_norm_normalized=0,
        only_federated_params=True,
        random_seed=1,
        store_models_and_optimizers=False,
    )


@pytest.mark.usefixtures("prepare_training_simulator_test_utils")
class TestTrainingSimulatorUtils:
    def random_user_selector(self, data_provider: IFLDataProvider) -> AsyncUserSelector:
        return RandomAsyncUserSelector(data_provider=data_provider)

    def round_robin_user_selector(
        self, data_provider: IFLDataProvider
    ) -> AsyncUserSelector:
        return RoundRobinAsyncUserSelector(data_provider=data_provider)

    def _create_data_provider(self, num_users: int, examples_per_user: int):
        # one_user_data has 1 batch of len = examples_per_user
        one_user_data = [list(range(examples_per_user))]
        data = [one_user_data] * num_users

        return FLDataProviderFromList(
            train_user_list=data,
            eval_user_list=data,
            test_user_list=data,
            model=MockFLModel(num_examples_per_user=examples_per_user),
        )

    def _verify_event(
        self,
        training_event: Tuple[DeviceState, TrainingState],
        exp_start_time: int,
        exp_end_time: int,
        exp_training_state: TrainingState,
    ):
        assertEqual(
            # pyre-fixme[16]: `TrainingState` has no attribute `training_start_time`.
            training_event[Globals.DEVICE_STATE_IDX].training_schedule.start_time,
            exp_start_time,
        )
        assertEqual(
            training_event[Globals.DEVICE_STATE_IDX].training_schedule.end_time,
            exp_end_time,
        )
        assertEqual(training_event[Globals.TRAINING_STATE_IDX], exp_training_state)

    def test_training_simulator(self):
        """Check that the priority queue in the training simulator works as
        expected by creating a training simulator with a known list of
        event start times and event durations.
        """
        event_list = [
            EventTimingInfo(prev_event_start_to_current_start=1, duration=3),
            EventTimingInfo(prev_event_start_to_current_start=2, duration=5),
            EventTimingInfo(prev_event_start_to_current_start=2, duration=1),
            EventTimingInfo(prev_event_start_to_current_start=10, duration=10),
        ]
        start_times_gaps = [val.prev_event_start_to_current_start for val in event_list]
        start_times = [
            sum(start_times_gaps[0 : (x + 1)]) for x in range(0, len(start_times_gaps))
        ]
        durations = [d.duration for d in event_list]
        end_times = [t[0] + t[1] for t in zip(start_times, durations)]
        distr = AsyncTrainingEventGeneratorFromList(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorFromListConfig(training_events=event_list)
            )
        )
        # time-sorted event list:
        # event0 starts at time  1
        # event1 starts at time  3
        # event0 ends at time    4
        # event2 starts at time  5
        # event2 ends at time    6
        # event1 ends at time    8
        num_users = 50
        job_scheduler = TestAsyncTrainingEventHandler()
        data_provider = self._create_data_provider(
            num_users=num_users, examples_per_user=1
        )
        training_sim = AsyncTrainingSimulator(
            job_scheduler=job_scheduler,
            user_selector=self.random_user_selector(data_provider=data_provider),
            event_generator=distr,
            shared_client_config=self.shared_client_config,
            num_train_end_events_per_epoch=len(event_list) - 1,
        )
        training_sim.run_one_epoch()
        # events are stored in increasing order of ending time
        seq_index = 0
        event_index = 0
        # 0: event0 starts at time 1
        self._verify_event(
            job_scheduler.training_events[seq_index],
            start_times[event_index],
            end_times[event_index],
            TrainingState.TRAINING,
        )
        # 1: event1 starts at time 3
        seq_index += 1
        event_index = 1
        self._verify_event(
            job_scheduler.training_events[seq_index],
            start_times[event_index],
            end_times[event_index],
            TrainingState.TRAINING,
        )
        # 2: event0 ends at time 4
        seq_index += 1
        event_index = 0
        self._verify_event(
            job_scheduler.training_events[seq_index],
            start_times[event_index],
            end_times[event_index],
            TrainingState.TRAINING_FINISHED,
        )
        # 3: event2 starts time 5
        seq_index += 1
        event_index = 2
        self._verify_event(
            job_scheduler.training_events[seq_index],
            start_times[event_index],
            end_times[event_index],
            TrainingState.TRAINING,
        )
        # 4: event2 ends time 6
        seq_index += 1
        event_index = 2
        self._verify_event(
            job_scheduler.training_events[seq_index],
            start_times[event_index],
            end_times[event_index],
            TrainingState.TRAINING_FINISHED,
        )
        # 5: event1 ends time 8
        seq_index += 1
        event_index = 1
        self._verify_event(
            job_scheduler.training_events[seq_index],
            start_times[event_index],
            end_times[event_index],
            TrainingState.TRAINING_FINISHED,
        )

        assertEqual(job_scheduler.current_seqnum, 3)
        # check stats for seqnum
        # global model updates are as below: {user_seqnum, global_seqnum}
        # {1, 1} event1 ends
        # {2, 2} event3 ends
        # {1, 3} event2 ends
        # seqnum_diffs = [0, 0, 2], mean = 2/3, sd = sqrt(24/27)
        assertAlmostEqual(job_scheduler.seqnum_diff_mean(), 2 / 3)
        assertAlmostEqual(job_scheduler.seqnum_std(), math.sqrt(24 / 27))

    def test_async_stats(self):
        """Check that the priority JobQueueStats functionality in the training
        simulator works as expected by creating a training simulator with a
        known list of event start times and event durations.
        """
        event_list = [
            EventTimingInfo(prev_event_start_to_current_start=1, duration=3),
            EventTimingInfo(prev_event_start_to_current_start=2, duration=5),
            EventTimingInfo(prev_event_start_to_current_start=2, duration=1),
            EventTimingInfo(prev_event_start_to_current_start=10, duration=10),
        ]
        distr = AsyncTrainingEventGeneratorFromList(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorFromListConfig(training_events=event_list)
            )
        )
        # time-sorted event list:
        # event1 starts at time  1
        # event2_starts at time  3
        # event1 ends at time    4, 2 pending jobs
        # event3 starts at time  5
        # event3 ends at time    6, 2 pending job
        # event2 ends at time    8, 1 pending jobs
        num_users = 50
        data_provider = self._create_data_provider(
            num_users=num_users, examples_per_user=1
        )
        job_scheduler = TestAsyncTrainingEventHandler()
        training_sim = AsyncTrainingSimulator(
            job_scheduler=job_scheduler,
            user_selector=self.random_user_selector(data_provider=data_provider),
            event_generator=distr,
            shared_client_config=self.shared_client_config,
            num_train_end_events_per_epoch=len(event_list) - 1,
        )
        training_sim.run_one_epoch()
        avg_pending_jobs = training_sim.avg_pending_jobs()
        assertTrue(avg_pending_jobs == 5 / 3)

    def test_sequential_training(self) -> None:
        """Check that in sequential training (where mean and SD of training
        time is zero), jobs are truly trained sequentialy: i.e, if job A
        starts training, job A ends training before any other jobs start
        training
        """
        num_users = 50
        training_start_time_distr = PoissonAsyncTrainingStartTimeDistrConfig(
            training_rate=10
        )
        duration_distr = PerExampleGaussianDurationDistributionConfig(
            training_duration_mean=0, training_duration_sd=0
        )
        distr = AsyncTrainingEventGenerator(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorConfig(
                    training_start_time_distribution=training_start_time_distr,
                    duration_distribution_generator=duration_distr,
                )
            )
        )
        job_scheduler = TestAsyncTrainingEventHandler()
        data_provider = self._create_data_provider(
            num_users=num_users, examples_per_user=1
        )
        training_sim = AsyncTrainingSimulator(
            job_scheduler=job_scheduler,
            user_selector=self.random_user_selector(data_provider=data_provider),
            event_generator=distr,
            # pyre-ignore[16]: for pytest fixture
            shared_client_config=self.shared_client_config,
            num_train_end_events_per_epoch=num_users,
        )
        training_sim.run_one_epoch()
        # two 'events' are generated for each job,
        # and stored in job_scheduler.training_events
        # one TRAINING event and one TRAINING_FINISHED event
        assertEqual(len(job_scheduler.training_events), num_users * 2)
        for user_num in range(num_users):
            # two events for each user
            user_index = 2 * user_num
            assertEqual(
                job_scheduler.training_events[user_index][Globals.TRAINING_STATE_IDX],
                TrainingState.TRAINING,
            )
            assertEqual(
                job_scheduler.training_events[user_index + 1][
                    Globals.TRAINING_STATE_IDX
                ],
                TrainingState.TRAINING_FINISHED,
            )
            # verify that both events are from the same user
            assertEqual(
                job_scheduler.training_events[user_index][Globals.DEVICE_STATE_IDX],
                job_scheduler.training_events[user_index + 1][Globals.DEVICE_STATE_IDX],
            )

    def test_uniform_event_gen_stats(self) -> None:
        """Test that when using an EventGenerator with uniformly distributed
        time between events, and constant duration of training events,
        stats about #pending_jobs and model_seqnum_diff are correct
        """
        for test_config in [
            ConstantEventGenTestConfig,
            ConstantEventGenTestConfigPerUserGaussian,
        ]:
            num_users = test_config.num_users
            num_examples_per_user = test_config.num_examples_per_user
            training_start_time_distr = ConstantAsyncTrainingStartTimeDistrConfig(
                training_rate=test_config.training_rate
            )
            training_duration_mean = test_config.training_duration_mean
            training_duration_sd = test_config.training_duration_sd
            duration_distr = test_config.training_duration_distribution_config(
                training_duration_mean=training_duration_mean,
                training_duration_sd=training_duration_sd,
            )
            distr = AsyncTrainingEventGenerator(
                **OmegaConf.structured(
                    AsyncTrainingEventGeneratorConfig(
                        training_start_time_distribution=training_start_time_distr,
                        duration_distribution_generator=duration_distr,
                    )
                )
            )
            job_scheduler = TestAsyncTrainingEventHandler()
            data_provider = self._create_data_provider(
                num_users=num_users, examples_per_user=num_examples_per_user
            )
            training_sim = AsyncTrainingSimulator(
                job_scheduler=job_scheduler,
                user_selector=self.random_user_selector(data_provider=data_provider),
                event_generator=distr,
                # pyre-ignore[16]: for pytest fixture
                shared_client_config=self.shared_client_config,
                num_train_end_events_per_epoch=num_users,
            )
            training_sim.run_one_epoch()
            assertEqual(
                training_sim.queue_stats.avg_pending_jobs(),
                test_config.mean_pending_jobs,
            )
            assertEqual(job_scheduler.seqnum_diff_mean(), test_config.mean_seqnum_diff)
            assertEqual(job_scheduler.seqnum_std(), test_config.sd_seqnum_diff)

    def test_poisson_event_gen_stats(self) -> None:
        """Test that when using an EventGenerator with poisson distributed
        time between events, and constant duration of training events,
        stats about #pending_jobs and model_seqnum_diff are correct
        """
        for test_config in [
            PoissonEventGenTestConfig,
            PoissonEventGenTestConfigPerUserGaussian,
        ]:
            np.random.seed(1)
            torch.manual_seed(1)
            num_users = test_config.num_users
            num_examples_per_user = test_config.num_examples_per_user
            training_start_time_distr = PoissonAsyncTrainingStartTimeDistrConfig(
                training_rate=test_config.training_rate
            )
            duration_distr = PerExampleGaussianDurationDistributionConfig(
                training_duration_mean=test_config.training_duration_mean,
                training_duration_sd=test_config.training_duration_sd,
            )
            distr = AsyncTrainingEventGenerator(
                **OmegaConf.structured(
                    AsyncTrainingEventGeneratorConfig(
                        training_start_time_distribution=training_start_time_distr,
                        duration_distribution_generator=duration_distr,
                    )
                )
            )
            job_scheduler = TestAsyncTrainingEventHandler()
            data_provider = self._create_data_provider(
                num_users=num_users, examples_per_user=num_examples_per_user
            )
            training_sim = AsyncTrainingSimulator(
                job_scheduler=job_scheduler,
                user_selector=self.random_user_selector(data_provider=data_provider),
                event_generator=distr,
                # pyre-ignore[16]: for pytest fixture
                shared_client_config=self.shared_client_config,
                num_train_end_events_per_epoch=num_users,
            )
            training_sim.run_one_epoch()
            assertAlmostEqual(
                training_sim.queue_stats.avg_pending_jobs(),
                test_config.mean_pending_jobs,
                delta=0.50,
            )
            assertEqual(job_scheduler.seqnum_diff_mean(), test_config.mean_seqnum_diff)
            assertAlmostEqual(
                job_scheduler.seqnum_std(), test_config.sd_seqnum_diff, delta=0.001
            )

    @staticmethod
    def get_user_finishing_seq(
        total_num_users: int,
        training_duration_distr_config: Type[DurationDistributionConfig],
    ) -> List[int]:
        per_user = PerUserGaussianDurationDistributionConfig
        per_example = PerExampleGaussianDurationDistributionConfig
        if training_duration_distr_config == per_user:
            # sequential finishing order
            return list(range(total_num_users))
        elif training_duration_distr_config == per_example:
            # reverse finishing order
            return list(reversed(range(total_num_users)))
        else:
            raise AssertionError(
                f"Unknown type string:{training_duration_distr_config}"
            )

    def verify_num_examples_multiplier(
        self, duration_distr_config: Type[DurationDistributionConfig]
    ) -> None:
        """if duration_distr_type == per-example-gaussian
           training_duration=num_examples*training_duration_per_example
        else
           training_duration=training_duration_per_example

        Use a config where training time is completely determined by
        number of examples. Eg:
            - GaussianDuration,PoissonStartTime
            - training_rate: very high
            - mean training time: very high
        num_examples_per_user = [n, n-1, n-2, ..., 3, 2, 1]

        If per-example-gaussian
            The last user should finish first.
            So training finish time should be: [user n, user n-1, ..., user2, user1]
        else if per-user-gaussian
            First user should finish first
            So training finish time should be: [user 1, user 2, user 3, .... user n]
        """
        num_users = 50
        training_start_time_distr = PoissonAsyncTrainingStartTimeDistrConfig(
            training_rate=1000
        )
        duration_distr = duration_distr_config(
            training_duration_mean=1000,
            training_duration_sd=0,
        )
        distr = AsyncTrainingEventGenerator(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorConfig(
                    training_start_time_distribution=training_start_time_distr,
                    duration_distribution_generator=duration_distr,
                )
            )
        )
        num_examples_per_user = list(reversed(range(1, num_users + 1)))
        data = [
            [1] * num_example
            for num_example, _ in zip(num_examples_per_user, range(num_users))
        ]
        data_provider = FLDataProviderFromList(
            train_user_list=data,
            eval_user_list=data,
            test_user_list=data,
            model=MockFLModel(),
        )

        job_scheduler = TestAsyncTrainingEventHandler()
        training_sim = AsyncTrainingSimulator(
            job_scheduler=job_scheduler,
            user_selector=self.round_robin_user_selector(data_provider),
            event_generator=distr,
            # pyre-ignore[16]: for pytest fixture
            shared_client_config=self.shared_client_config,
            num_train_end_events_per_epoch=num_users,
        )
        training_sim.run_one_epoch()
        ts_idx = Globals.TRAINING_STATE_IDX
        ds_idx = Globals.DEVICE_STATE_IDX
        # two 'events' are generated for each job,
        # and stored in job_scheduler.training_events
        # one TRAINING event and one TRAINING_FINISHED event
        assertEqual(len(job_scheduler.training_events), num_users * 2)
        first_n_events = job_scheduler.training_events[:num_users]
        # First num_user events should be TRAINING events
        assertEqual(
            [i[ts_idx] for i in first_n_events], [TrainingState.TRAINING] * num_users
        )
        # First num_user events should have user_index in sequence
        assertEqual(
            [i[ds_idx].user_info.user_index for i in first_n_events],
            list(range(num_users)),
        )

        last_n_events = job_scheduler.training_events[num_users:]
        # Next num_user events should be TRAINING_FINISHED events:
        assertEqual(
            [i[ts_idx] for i in last_n_events],
            [TrainingState.TRAINING_FINISHED] * num_users,
        )
        # Users should finish in seq if type = PerUserGaussian,
        # in reverse seq if type = PerExampleGaussian
        user_finishing_seq = self.get_user_finishing_seq(
            num_users, duration_distr_config
        )
        assertEqual(
            [i[ds_idx].user_info.user_index for i in last_n_events], user_finishing_seq
        )

    def test_training_duration_per_example_gaussian(self):
        pass

    def test_training_duration_per_user_gaussian(self):
        pass

    def test_training_end_training_start_relative_priority(self):
        r"""
        Test that when training end and training start events have the same time,
        training end event happens earlier (has lower priority)
        """

    def test_logging_level(self):
        num_users = 50
        training_start_time_distr = PoissonAsyncTrainingStartTimeDistrConfig(
            training_rate=10
        )
        duration_distr = PerExampleGaussianDurationDistributionConfig(
            training_duration_mean=0, training_duration_sd=0
        )
        distr = AsyncTrainingEventGenerator(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorConfig(
                    training_start_time_distribution=training_start_time_distr,
                    duration_distribution_generator=duration_distr,
                )
            )
        )
        job_scheduler = TestAsyncTrainingEventHandler()
        data_provider = self._create_data_provider(
            num_users=num_users, examples_per_user=1
        )
        training_sim = AsyncTrainingSimulator(
            job_scheduler=job_scheduler,
            user_selector=self.random_user_selector(data_provider=data_provider),
            event_generator=distr,
            shared_client_config=self.shared_client_config,
            num_train_end_events_per_epoch=num_users,
        )
        assertTrue(check_inherit_logging_level(training_sim, 50))
        assertTrue(check_inherit_logging_level(training_sim, 10))

    def _training_staleness_distribution(
        self,
        duration_distr,
        training_rate=1000,
        num_users=10000,
        examples_per_user=1,
    ):
        data_provider = self._create_data_provider(
            num_users=num_users, examples_per_user=examples_per_user
        )
        training_start_time_distr = PoissonAsyncTrainingStartTimeDistrConfig(
            training_rate=training_rate
        )
        distr = AsyncTrainingEventGenerator(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorConfig(
                    training_start_time_distribution=training_start_time_distr,
                    duration_distribution_generator=duration_distr,
                )
            )
        )
        handler = TestAsyncTrainingEventHandler()

        training_sim = AsyncTrainingSimulator(
            job_scheduler=handler,
            user_selector=RandomAsyncUserSelector(data_provider=data_provider),
            event_generator=distr,
            shared_client_config=self.shared_client_config,
            num_train_end_events_per_epoch=num_users,
        )
        training_sim.run_one_epoch()
        return np.array(handler.num_unseen_global_model_updates)

    def test_per_user_half_normal_training_duration(self):
        """
        Test for staleness with half normal

        Condition 1:
            Check right skewed to make sure that the distribution is
            heavy tailed and there is no concentration of users with low
            staleness more than usual
        Condition 2:
            Check that the distribution has a long tail that approximate a guassian tail
            All values should be within 6 std of the mean
        """
        num_users = 1000
        training_rate = 10
        training_min = 0.00
        training_std = 1.5
        duration_distr_config = PerUserHalfNormalDurationDistributionConfig(
            training_duration_sd=training_std, training_duration_min=training_min
        )
        staleness = self._training_staleness_distribution(
            duration_distr_config, training_rate=training_rate, num_users=num_users
        )
        mean = np.mean(staleness)
        sd = np.std(staleness)

        assertLessEqual(np.median(staleness), mean)
        # check that not all values are within 2 sd
        assertGreaterEqual(max(staleness), mean + 2 * sd)
        # check all values are within 7 sd of the mean
        # 2e-11 = e^(−k^2/2) = exp(-7^2/2)
        assertLessEqual(max(staleness), mean + 7 * sd)

    def test_per_user_uniform_training_duration(self):
        num_users = 2000
        training_rate = 10
        training_min = 0.0
        training_mean = 1.0

        duration_distr_config = PerUserUniformDurationDistributionConfig(
            training_duration_mean=training_mean, training_duration_min=training_min
        )
        staleness = self._training_staleness_distribution(
            duration_distr_config, training_rate=training_rate, num_users=num_users
        )
        mean = np.mean(staleness)
        sd = np.std(staleness)

        # check that under uniform mean and median close to the same
        assertAlmostEqual(np.median(staleness), mean, delta=1.5)

        # check that all values are within 4 sd from the mean
        assertLessEqual(max(staleness), mean + 4 * sd)
        assertGreaterEqual(min(staleness), mean - 4 * sd)

    def test_per_user_exponential_training_duration(self):
        for training_mean in [0.5, 1, 2]:
            num_users = 2000
            training_rate = 10

            duration_distr_config = PerUserExponentialDurationDistributionConfig(
                training_duration_mean=training_mean
            )
            staleness = self._training_staleness_distribution(
                duration_distr_config, training_rate=training_rate, num_users=num_users
            )
            mean = np.mean(staleness)
            median = np.median(staleness)

            expected_mean = training_rate * training_mean
            assertAlmostEqual(expected_mean, mean, delta=1.5)
            assertLessEqual(median, mean)
