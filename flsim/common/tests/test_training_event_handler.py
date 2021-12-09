#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import pytest
from flsim.clients.async_client import AsyncClientDevice
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.pytest_helper import assertEqual
from flsim.common.training_event_handler import TestAsyncTrainingEventHandler
from flsim.data.data_provider import FLDataProviderFromList, FLUserDataFromList
from flsim.utils.async_trainer.async_user_selector import AsyncUserSelectorInfo
from flsim.utils.async_trainer.device_state import TrainingSchedule
from flsim.utils.sample_model import MockFLModel
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from omegaconf import OmegaConf


@pytest.fixture(scope="class")
def prepare_job_scheduler_util_test(request):
    request.cls.shared_client_config = ClientConfig(
        epochs=1,
        max_clip_norm_normalized=0,
        only_federated_params=True,
        random_seed=1,
        store_models_and_optimizers=False,
    )


@pytest.mark.usefixtures("prepare_job_scheduler_util_test")
class TestJobSchedulerUtil:
    def _build_data_provider(
        self, num_examples, examples_per_user, user_batch_size, global_model
    ) -> FLDataProviderFromList:
        dummy_dataset = DummyAlphabetDataset(num_examples)
        data_provider, _ = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, examples_per_user, user_batch_size, global_model
        )
        return data_provider

    def _create_mock_base_client_info(self, examples_per_user, user_index):
        user_data = FLUserDataFromList(
            [1] * examples_per_user,
            MockFLModel(num_examples_per_user=examples_per_user),
        )
        user_info = AsyncUserSelectorInfo(user_data=user_data, user_index=user_index)
        base_client = Client(
            **OmegaConf.structured(self.shared_client_config),
            dataset=user_info.user_data,
            name=f"client_{user_info.user_index}",
        )
        return user_info, base_client

    def test_test_async_job_scheduler(self) -> None:
        num_users = 100
        model_trainer = TestAsyncTrainingEventHandler()
        assertEqual(len(model_trainer.num_unseen_global_model_updates), 0)
        assertEqual(len(model_trainer.training_events), 0)

        for user_index in range(num_users):
            training_schedule = TrainingSchedule(
                creation_time=0,
                start_time=user_index,
                end_time=num_users + (num_users - user_index - 1),
            )
            user_info, base_client = self._create_mock_base_client_info(
                examples_per_user=1, user_index=user_index
            )

            client = AsyncClientDevice(
                training_schedule=training_schedule,
                client=base_client,
                user_info=user_info,
            )
            model_trainer.on_training_start(client)
            assertEqual(client.model_seqnum, model_trainer.current_seqnum)
            assertEqual(client.model_seqnum, user_index)
            model_trainer.on_training_end(client)
        # each user generates two events: one TrainingFinished, and one TrainingStart
        assertEqual(len(model_trainer.training_events), 2 * num_users)
        # training proceeds sequentially, so num_unseen_global_model_updates should
        # be a list of zeroes
        assertEqual(model_trainer.num_unseen_global_model_updates, [0] * num_users)

    def test_staleness_compute_in_job_scheduler(self) -> None:
        # train in LIFO order: the first user to start training is the last
        # one to finish training
        # in this case, assume n users, observed staleness should be:
        # 0 (user n), 1 (user n-1), 2 (user n-2)... n (user 1)
        # timeline:
        # USERS START TRAINING
        # t=0, user 0 starts training
        # t=1, user 1 starts training
        # ...
        # t=n-1, user n-1 starts training
        # USERS END TRAINING
        # t=n, user n-1 ends training
        # t=n+1, user n-2 ends training
        # ...
        # t=n+(n-1), user 0 ends training

        num_users = 100
        model_trainer = TestAsyncTrainingEventHandler()
        clients: List[AsyncClientDevice] = []
        for user_index in range(num_users):
            # user i starts training at time i, ends training at time n + (n-i-1)
            training_schedule = TrainingSchedule(
                creation_time=0,
                start_time=user_index,
                end_time=num_users + (num_users - user_index - 1),
            )
            user_info, base_client = self._create_mock_base_client_info(
                examples_per_user=1, user_index=user_index
            )

            client = AsyncClientDevice(
                training_schedule=training_schedule,
                client=base_client,
                user_info=user_info,
            )
            model_trainer.on_training_start(client)
            clients.append(client)
        while len(clients):
            client = clients.pop()
            model_trainer.on_training_end(client)
        # observed staleness should be [0, 1, 2, ....num_users-1]
        assertEqual(
            model_trainer.num_unseen_global_model_updates, list(range(num_users))
        )
