#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Type

import pytest
from flsim.clients.async_client import AsyncClientDevice, AsyncClientFactory
from flsim.clients.base_client import ClientConfig
from flsim.common.pytest_helper import assertEqual, assertIsNotNone
from flsim.data.data_provider import FLDataProviderFromList
from flsim.utils.async_trainer.async_user_selector import (
    RandomAsyncUserSelector,
    RoundRobinAsyncUserSelector,
)
from flsim.utils.async_trainer.device_state import TrainingState
from flsim.utils.async_trainer.training_event_generator import (
    AsyncTrainingEventGenerator,
    AsyncTrainingEventGeneratorConfig,
    AsyncTrainingEventGeneratorFromList,
    AsyncTrainingEventGeneratorFromListConfig,
    EventTimingInfo,
    PoissonAsyncTrainingStartTimeDistrConfig,
)
from flsim.utils.data.fake_data_utils import create_mock_data_provider
from flsim.utils.sample_model import DummyAlphabetFLModel, MockFLModel
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
    PerUserGaussianDurationDistributionConfig,
    DurationDistributionConfig,
)
from omegaconf import OmegaConf


@pytest.fixture(scope="class")
def prepare_shared_client_config(request):
    request.cls.shared_client_config = OmegaConf.structured(
        ClientConfig(
            epochs=1,
            max_clip_norm_normalized=0,
            only_federated_params=True,
            random_seed=1,
            store_models_and_optimizers=False,
        )
    )


@pytest.mark.usefixtures("prepare_shared_client_config")
class TestAsyncClientDeviceGeneration:
    def _verify_event(
        self,
        client: AsyncClientDevice,
        expected_start_time: int,
        expected_end_time: int,
    ):
        assertEqual(client.training_schedule.start_time, expected_start_time)
        assertEqual(client.training_schedule.end_time, expected_end_time)

    def test_provide_client_event_generation(self) -> None:
        r"""
        Check if client provider returns the client with the correct
        start time and end time
        """
        # (start time, duration)
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
        event_generator = AsyncTrainingEventGeneratorFromList(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorFromListConfig(training_events=event_list)
            )
        )
        num_users = len(event_list)
        data_provider = create_mock_data_provider(
            num_users=num_users, examples_per_user=1
        )
        user_selector = RandomAsyncUserSelector(data_provider=data_provider)

        current_time = 0
        for start, end in zip(start_times, end_times):
            client = AsyncClientFactory.create(
                current_time=current_time,
                event_generator=event_generator,
                user_selector=user_selector,
                # pyre-ignore [16]: for pytest
                client_config=self.shared_client_config,
            )
            self._verify_event(client, expected_start_time=start, expected_end_time=end)
            # how we move forward in time in async is by setting the current time
            # to start time of the client on top of the heap
            current_time = client.next_event_time()

    def test_sequential_client_training_schedule(self) -> None:
        r"""
        Check that training event generator produces TrainingSchedule sequentially
        (where mean and SD of training time is 0), clients are truly produced sequentialy:
        i.e, if client starts training, client A ends training before any other client start
        """
        num_users = 100
        training_start_time_distr = PoissonAsyncTrainingStartTimeDistrConfig(
            training_rate=10
        )
        duration_distr = PerExampleGaussianDurationDistributionConfig(
            training_duration_mean=0, training_duration_sd=0
        )
        event_generator = AsyncTrainingEventGenerator(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorConfig(
                    training_start_time_distribution=training_start_time_distr,
                    duration_distribution_generator=duration_distr,
                )
            )
        )
        examples_per_user = 1
        data_provider = create_mock_data_provider(
            num_users=num_users, examples_per_user=examples_per_user
        )
        user_selector = RandomAsyncUserSelector(data_provider=data_provider)

        current_time = 0
        clients = []
        for _ in range(num_users):
            client = AsyncClientFactory.create(
                current_time=current_time,
                event_generator=event_generator,
                user_selector=user_selector,
                # pyre-ignore [16]: for pytest
                client_config=self.shared_client_config,
            )
            assertEqual(
                client.training_schedule.start_time, client.training_schedule.end_time
            )
            current_time = client.next_event_time()
            clients.append(client)

        # verify that clients were produced and hence trained sequentially
        for client_1, client_2 in zip(clients, clients[1:]):
            # check that client_1 should end trainign before  client_2 start training
            assert (
                client_1.training_schedule.end_time
                <= client_2.training_schedule.start_time
            )

            # check that start time is strictly monotonic increasing
            assert (
                client_1.training_schedule.start_time
                < client_2.training_schedule.start_time
            )

    def _build_clients_training_duration_dist(
        self, duration_distr_config: Type[DurationDistributionConfig], num_users: int
    ) -> List[AsyncClientDevice]:
        r"""
        Per-Example-Gaussian:
            training_duration = num_examples * training_duration_per_example
        Per-User-Gaussian:
            training_duration = training_duration_per_example

        Use a config where training time is completely determined by
        number of examples. Eg:
            - GaussianDuration,PoissonStartTime
            - training_rate: very high
            - mean training time: very high
        num_examples_per_user = [n, n-1, n-2, ..., 3, 2, 1]
        """
        training_start_time_distr = PoissonAsyncTrainingStartTimeDistrConfig(
            training_rate=1000
        )
        duration_distr = duration_distr_config(
            training_duration_mean=1000,
            training_duration_sd=0,
        )
        event_generator = AsyncTrainingEventGenerator(
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
        user_selector = RoundRobinAsyncUserSelector(data_provider=data_provider)

        clients = []
        current_time = 0
        for _ in range(num_users):
            client = AsyncClientFactory.create(
                current_time=current_time,
                event_generator=event_generator,
                user_selector=user_selector,
                # pyre-ignore [16]: for pytest
                client_config=self.shared_client_config,
            )
            current_time = client.next_event_time()
            clients.append(client)
        return clients

    def test_training_duration_per_example_gaussian(self):
        r"""
        Per-Example-Gaussian:
            The last user should finish first.
            So training finish time should be: [user n, user n-1, ..., user2, user1]
        """
        num_users = 50
        clients = self._build_clients_training_duration_dist(
            duration_distr_config=PerExampleGaussianDurationDistributionConfig,
            num_users=num_users,
        )
        # check that end time is strictly monotonic decreasing
        for client_1, client_2 in zip(clients, clients[1:]):
            assert (
                client_1.training_schedule.end_time
                > client_2.training_schedule.end_time
            )

    def test_training_duration_per_user_gaussian(self):
        r"""
        Per-User-Gaussian:
            First user should finish first
            So training finish time should be: [user 1, user 2, user 3, .... user n]
        """
        num_users = 50
        clients = self._build_clients_training_duration_dist(
            duration_distr_config=PerUserGaussianDurationDistributionConfig,
            num_users=num_users,
        )
        # check that end time is strictly monotonic increasing
        for client_1, client_2 in zip(clients, clients[1:]):
            assert (
                client_1.training_schedule.end_time
                < client_2.training_schedule.end_time
            )


@pytest.fixture(scope="class")
def prepare_async_client_device(request):
    request.cls.shared_client_config = OmegaConf.structured(
        ClientConfig(
            epochs=1,
            max_clip_norm_normalized=0,
            only_federated_params=True,
            random_seed=1,
            store_models_and_optimizers=False,
        )
    )
    request.cls.event_list = [
        EventTimingInfo(prev_event_start_to_current_start=1, duration=3),
        EventTimingInfo(prev_event_start_to_current_start=2, duration=5),
        EventTimingInfo(prev_event_start_to_current_start=2, duration=1),
        EventTimingInfo(prev_event_start_to_current_start=10, duration=10),
    ]
    request.cls.event_generator = AsyncTrainingEventGeneratorFromList(
        **OmegaConf.structured(
            AsyncTrainingEventGeneratorFromListConfig(
                training_events=request.cls.event_list
            )
        )
    )


@pytest.mark.usefixtures("prepare_async_client_device")
class TestAsyncClientDevice:
    def _build_data_provider(
        self, num_examples, examples_per_user, user_batch_size, global_model
    ) -> FLDataProviderFromList:
        dummy_dataset = DummyAlphabetDataset(num_examples)
        data_provider, _ = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, examples_per_user, user_batch_size, global_model
        )
        return data_provider

    def test_async_client_start_training(self) -> None:
        global_model = DummyAlphabetFLModel()

        examples_per_user = 10
        data_provider = self._build_data_provider(
            num_examples=100,
            examples_per_user=examples_per_user,
            user_batch_size=2,
            global_model=global_model,
        )

        user_selector = RandomAsyncUserSelector(data_provider=data_provider)

        client = AsyncClientFactory.create(
            current_time=0,
            # pyre-ignore [16]: for pytest
            event_generator=self.event_generator,
            user_selector=user_selector,
            # pyre-ignore [16]: for pytest
            client_config=self.shared_client_config,
        )
        current_seqnum = 1
        # verify that client is waiting to start
        assert client.is_waiting_to_start()

        client.training_started(model_seqnum=current_seqnum, init_model=global_model)
        # verify that that we saved a copy of the global model
        assertIsNotNone(client.local_model)
        # verify that client has the correct seq num
        assertEqual(client.model_seqnum, current_seqnum)
        # verify that client state is training
        assert not client.is_waiting_to_start()
        assertEqual(client.training_state, TrainingState.TRAINING)

    def test_async_client_training(self) -> None:
        num_examples = 10
        examples_per_user = 10
        user_batch_size = 2
        training_start_time = 1
        training_duration = 3
        training_end_time = training_start_time + training_duration

        global_model = DummyAlphabetFLModel()
        data_provider = self._build_data_provider(
            num_examples=num_examples,
            examples_per_user=examples_per_user,
            user_batch_size=user_batch_size,
            global_model=global_model,
        )

        event_list = [
            EventTimingInfo(
                prev_event_start_to_current_start=training_start_time,
                duration=training_duration,
            )
        ]
        event_generator = AsyncTrainingEventGeneratorFromList(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorFromListConfig(training_events=event_list)
            )
        )

        user_selector = RandomAsyncUserSelector(data_provider=data_provider)
        client = AsyncClientFactory.create(
            current_time=0,
            event_generator=event_generator,
            user_selector=user_selector,
            # pyre-ignore [16]: for pytest
            client_config=self.shared_client_config,
        )

        current_seqnum = 1
        assertEqual(client.next_event_time(), training_start_time)
        client.training_started(model_seqnum=current_seqnum, init_model=global_model)
        (
            client_delta,
            final_local_model,
            num_examples_trained,
        ) = client.train_local_model(metric_reporter=None)
        client.training_ended()

        assertEqual(num_examples_trained, examples_per_user)
        assertEqual(client.training_state, TrainingState.TRAINING_FINISHED)
        assertEqual(client.next_event_time(), training_end_time)
        assertEqual(client.model_seqnum, current_seqnum)

    def test_async_client_less_than(self) -> None:
        num_examples = 10
        examples_per_user = 10
        user_batch_size = 2

        global_model = DummyAlphabetFLModel()
        data_provider = self._build_data_provider(
            num_examples=num_examples,
            examples_per_user=examples_per_user,
            user_batch_size=user_batch_size,
            global_model=global_model,
        )
        user_selector = RandomAsyncUserSelector(data_provider=data_provider)
        # two clients
        # client 1 starts training at 1
        # client 2 starts training at 2
        # verify that client 1 will be less than client 2
        event_list = [
            EventTimingInfo(prev_event_start_to_current_start=1, duration=1),
            EventTimingInfo(prev_event_start_to_current_start=2, duration=1),
        ]
        event_generator = AsyncTrainingEventGeneratorFromList(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorFromListConfig(training_events=event_list)
            )
        )
        client_1 = AsyncClientFactory.create(
            current_time=0,
            event_generator=event_generator,
            user_selector=user_selector,
            # pyre-ignore [16]: for pytest
            client_config=self.shared_client_config,
        )
        client_2 = AsyncClientFactory.create(
            current_time=0,
            event_generator=event_generator,
            user_selector=user_selector,
            client_config=self.shared_client_config,
        )
        assert client_1 < client_2

        # two clients currently training (training_state=TRAINING_STARTED)
        # client a ends training at 2
        # client b ends training at 3
        # verify that client a will be less than client b
        event_list = [
            EventTimingInfo(prev_event_start_to_current_start=1, duration=1),
            EventTimingInfo(prev_event_start_to_current_start=1, duration=2),
        ]
        event_generator = AsyncTrainingEventGeneratorFromList(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorFromListConfig(training_events=event_list)
            )
        )
        client_a = AsyncClientFactory.create(
            current_time=0,
            event_generator=event_generator,
            user_selector=user_selector,
            client_config=self.shared_client_config,
        )
        client_b = AsyncClientFactory.create(
            current_time=0,
            event_generator=event_generator,
            user_selector=user_selector,
            client_config=self.shared_client_config,
        )
        client_b.training_started(1, global_model)
        assert client_a < client_b
