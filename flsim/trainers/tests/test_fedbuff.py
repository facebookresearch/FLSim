#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from enum import Enum
from typing import Union

import numpy as np
import torch
from flsim.common.pytest_helper import assertEqual, assertEmpty
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.async_aggregators import (
    AsyncAggregatorConfig,
    FedAdamAsyncAggregatorConfig,
    FedAvgWithLRAsyncAggregatorConfig,
    FedAdamHybridAggregatorConfig,
    FedAvgWithLRHybridAggregatorConfig,
    create_optimizer_for_async_aggregator,
)
from flsim.optimizers.server_optimizers import (
    FedAdamOptimizerConfig,
    FedAvgWithLROptimizerConfig,
    OptimizerType,
)
from flsim.servers.sync_servers import (
    SyncServerConfig,
)
from flsim.tests.utils import (
    MetricsReporterWithMockedChannels,
    verify_models_equivalent_after_training,
)
from flsim.trainers.async_trainer import AsyncTrainerConfig
from flsim.trainers.sync_trainer import SyncTrainerConfig
from flsim.utils.config_utils import is_target
from flsim.utils.sample_model import DummyAlphabetFLModel
from flsim.utils.tests.helpers.async_trainer_test_utils import (
    create_async_trainer,
    create_event_generator_config,
    get_safe_global_lr,
    run_fl_training,
)
from flsim.utils.tests.helpers.sync_trainer_test_utils import create_sync_trainer
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from flsim.utils.tests.helpers.test_utils import FLTestUtils


class TrainerType(Enum):
    SYNC: str = SyncTrainerConfig._target_.split(".")[-1]
    ASYNC: str = AsyncTrainerConfig._target_.split(".")[-1]
    NONFL: str = "NonFL"


class HybridFLTestUtils:
    @staticmethod
    def get_data_provider(
        num_examples: int,
        num_fl_users: int,
        examples_per_user: int,
        batch_size: int,
        model: IFLModel,
    ) -> IFLDataProvider:
        dummy_dataset = DummyAlphabetDataset(num_examples)
        (
            data_provider,
            fl_data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, examples_per_user, batch_size, model
        )
        assert fl_data_loader.num_total_users == num_fl_users
        return data_provider

    @staticmethod
    def train_comparable_model(
        trainer_to_compare_hybrid_fl_with,
        data_provider,
        global_model,
        server_config,
        local_lr,
        epochs,
        training_rate,
        training_duration_mean,
        training_duration_sd,
    ) -> IFLModel:
        metric_reporter = MetricsReporterWithMockedChannels()
        if trainer_to_compare_hybrid_fl_with == TrainerType.SYNC:
            trainer = create_sync_trainer(
                model=global_model,
                local_lr=local_lr,
                epochs=epochs,
                users_per_round=training_rate,
                server_config=server_config,
            )
            model_to_compare, _ = trainer.train(
                data_provider=data_provider,
                metric_reporter=metric_reporter,
                num_total_users=data_provider.num_users(),
                distributed_world_size=1,
            )
        elif trainer_to_compare_hybrid_fl_with == TrainerType.ASYNC:
            trainer = create_async_trainer(
                model=global_model,
                local_lr=local_lr,
                epochs=epochs,
                aggregator_config=server_config,
                event_generator_config=create_event_generator_config(
                    training_rate=training_rate,
                    training_duration_mean=training_duration_mean,
                    training_duration_sd=training_duration_sd,
                ),
            )
            model_to_compare, _ = trainer.train(
                data_provider=data_provider,
                metric_reporter=metric_reporter,
                num_total_users=data_provider.num_users(),
                distributed_world_size=1,
            )
        elif trainer_to_compare_hybrid_fl_with == TrainerType.NONFL:
            # create an optimizer from aggregator_config.
            # for tests in this file, optimizer will either be
            # torch.optim.SGD or torch.optim.Adam
            if isinstance(server_config, AsyncAggregatorConfig):
                optimizer = create_optimizer_for_async_aggregator(
                    config=server_config,
                    model=global_model.fl_get_module(),
                )
            elif isinstance(server_config, SyncServerConfig):
                optimizer = OptimizerType.create_optimizer(
                    model=global_model.fl_get_module(),
                    config=server_config.server_optimizer,
                )
            else:
                raise AssertionError(f"Incompatible server config:{server_config}")
            model_to_compare, _ = FLTestUtils.train_non_fl(
                data_provider=data_provider,
                global_model=global_model,
                optimizer=optimizer,
                metrics_reporter=metric_reporter,
                epochs=epochs,
            )
        # pyre-fixme[61]: `model_to_compare` may not be initialized here.
        return model_to_compare

    @staticmethod
    def get_hybrid_aggregator(
        aggregator_config, buffer_size, hybrid_lr
    ) -> Union[FedAdamHybridAggregatorConfig, FedAvgWithLRHybridAggregatorConfig]:
        if isinstance(aggregator_config, AsyncAggregatorConfig):
            if "FedAdam" in aggregator_config._target_:
                hybrid_aggregator = FedAdamHybridAggregatorConfig(
                    lr=hybrid_lr,
                    # pyre-ignore[16]
                    weight_decay=aggregator_config.weight_decay,
                    # pyre-ignore[16]
                    eps=aggregator_config.eps,
                    buffer_size=buffer_size,
                )
            else:  # "FedAvgWithLR" in aggregator_config._target_:
                hybrid_aggregator = FedAvgWithLRHybridAggregatorConfig(
                    lr=hybrid_lr,
                    # pyre-ignore[16]
                    momentum=aggregator_config.momentum,
                    buffer_size=buffer_size,
                )
            return hybrid_aggregator
        elif isinstance(aggregator_config, SyncServerConfig):
            if is_target(aggregator_config.server_optimizer, FedAdamOptimizerConfig):
                hybrid_aggregator = FedAdamHybridAggregatorConfig(
                    lr=hybrid_lr,
                    # pyre-ignore[16]
                    weight_decay=aggregator_config.server_optimizer.weight_decay,
                    # pyre-ignore[16]
                    eps=aggregator_config.server_optimizer.eps,
                    buffer_size=buffer_size,
                )
            else:
                hybrid_aggregator = FedAvgWithLRHybridAggregatorConfig(
                    lr=hybrid_lr,
                    # pyre-ignore[16]
                    momentum=aggregator_config.server_optimizer.momentum,
                    buffer_size=buffer_size,
                )
            return hybrid_aggregator
        else:
            raise ValueError("Invalid config", aggregator_config)

    @staticmethod
    def compare_hybrid_fl_same(
        trainer_to_compare_hybrid_fl_with,
        trainer_to_compare_aggregator_config,
        hybrid_aggregator_config,
        base_local_lr,
        hybrid_local_lr,
        epochs,
        num_examples,
        num_fl_users,
        batch_size,
        examples_per_user,
        buffer_size,
        training_rate,
        training_duration_mean,
        training_duration_sd,
    ) -> str:
        # we need to make three copies:
        # to train the model we want to compare with
        global_model = DummyAlphabetFLModel()
        # to train hybrid model
        global_model_hybrid_copy = copy.deepcopy(global_model)
        # to verify training indeed took place
        global_model_init_copy = copy.deepcopy(global_model)

        data_provider = HybridFLTestUtils.get_data_provider(
            num_examples=num_examples,
            num_fl_users=num_fl_users,
            examples_per_user=examples_per_user,
            batch_size=batch_size,
            model=global_model,
        )

        def get_base_trained_model():
            model_to_compare = HybridFLTestUtils.train_comparable_model(
                trainer_to_compare_hybrid_fl_with=trainer_to_compare_hybrid_fl_with,
                data_provider=data_provider,
                global_model=global_model,
                server_config=trainer_to_compare_aggregator_config,
                epochs=epochs,
                local_lr=base_local_lr,
                training_rate=training_rate,
                training_duration_mean=training_duration_mean,
                training_duration_sd=training_duration_sd,
            )
            return model_to_compare

        def get_hybrid_trained_model():
            hybrid_fl_trainer = create_async_trainer(
                model=global_model_hybrid_copy,
                local_lr=hybrid_local_lr,
                epochs=epochs,
                event_generator_config=create_event_generator_config(
                    training_rate=training_rate,
                    training_duration_mean=training_duration_mean,
                    training_duration_sd=training_duration_sd,
                ),
                aggregator_config=hybrid_aggregator_config,
            )

            hybrid_model, _ = hybrid_fl_trainer.train(
                data_provider=data_provider,
                metric_reporter=MetricsReporterWithMockedChannels(),
                num_total_users=data_provider.num_users(),
                distributed_world_size=1,
            )
            return hybrid_model

        hybrid_trained_model = get_hybrid_trained_model()
        base_model = get_base_trained_model()

        error_msg = verify_models_equivalent_after_training(
            base_model,
            hybrid_trained_model,
            global_model_init_copy,
            rel_epsilon=1e-4,
            abs_epsilon=1e-6,
        )
        return error_msg

    @staticmethod
    def get_data_params(
        min_num_users, max_num_users, min_examples_per_user, max_examples_per_user
    ):
        r"""
        Generate data parameters for FL training with
        num users in range [min_num_users, max_num_users)
        """
        num_fl_users = np.random.randint(min_num_users, max_num_users)
        examples_per_user = np.random.randint(
            min_examples_per_user, max_examples_per_user
        )
        num_examples = examples_per_user * num_fl_users
        # num_fl_users + 1 because randint upper bound is exclusive
        training_rate = np.random.randint(min_num_users, num_fl_users + 1)
        return num_fl_users, examples_per_user, num_examples, training_rate

    @staticmethod
    def compare_nonfl_hybrid_uneven_data_split(
        total_examples,
        num_fl_users,
        buffer_size,
        non_fl_lr,
        hybrid_global_lr,
        examples_per_user_hybrid,
        examples_per_user_nonfl,
        batch_size_hybrid,
        batch_size_nonfl,
        epochs,
        local_lr=1.0,
    ):
        # to verify training indeed took place
        reference_untrained_model = DummyAlphabetFLModel()
        # to train hybrid model
        hybrid_model = copy.deepcopy(reference_untrained_model)
        # to train nonfl
        nonfl_model = copy.deepcopy(reference_untrained_model)

        nonfl_data_provider = HybridFLTestUtils.get_data_provider(
            num_examples=total_examples,
            num_fl_users=1,
            examples_per_user=examples_per_user_nonfl,
            batch_size=batch_size_nonfl,
            model=nonfl_model,
        )

        hybrid_data_provider = HybridFLTestUtils.get_data_provider(
            num_examples=total_examples,
            num_fl_users=num_fl_users,
            examples_per_user=examples_per_user_hybrid,
            batch_size=batch_size_hybrid,
            model=hybrid_model,
        )

        hybrid_trained_model, _ = run_fl_training(
            fl_model=hybrid_model,
            fl_data_provider=hybrid_data_provider,
            epochs=epochs,
            local_lr=local_lr,
            aggregator_config=FedAvgWithLRHybridAggregatorConfig(
                lr=hybrid_global_lr, buffer_size=buffer_size
            ),
            # sequential training training_rate >> training_duration
            training_rate=1,
            training_duration_mean=0,
        )

        optimizer = torch.optim.SGD(
            nonfl_model.fl_get_module().parameters(), lr=non_fl_lr
        )
        nonfl_trained_model, _ = FLTestUtils.train_non_fl(
            data_provider=nonfl_data_provider,
            global_model=nonfl_model,
            optimizer=optimizer,
            epochs=epochs,
        )
        print(f"Local LR {local_lr} Non LR {non_fl_lr} Hybrid LR {hybrid_global_lr}")
        return verify_models_equivalent_after_training(
            nonfl_trained_model,
            hybrid_trained_model,
            reference_untrained_model,
            rel_epsilon=1e-4,
            abs_epsilon=1e-6,
        )

    @staticmethod
    def compare_async_hybrid_uneven_data_split(
        total_examples,
        num_fl_users,
        hybrid_num_fl_users,
        async_global_lr,
        hybrid_global_lr,
        batch_size_hybrid,
        batch_size_async,
        epochs,
        training_rate,
        training_duration_mean,
        local_lr=1.0,
    ):
        # to verify training indeed took place
        reference_untrained_model = DummyAlphabetFLModel()
        # to train hybrid model
        hybrid_model = copy.deepcopy(reference_untrained_model)
        # to train async
        async_model = copy.deepcopy(reference_untrained_model)

        async_data_provider = HybridFLTestUtils.get_data_provider(
            num_examples=total_examples,
            num_fl_users=1,
            examples_per_user=total_examples,
            batch_size=batch_size_async,
            model=async_model,
        )
        async_trained_model, _ = run_fl_training(
            fl_model=async_model,
            fl_data_provider=async_data_provider,
            epochs=epochs,
            local_lr=local_lr,
            aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=async_global_lr),
            training_rate=training_rate,
            training_duration_mean=training_duration_mean,
        )

        hybrid_data_provider = HybridFLTestUtils.get_data_provider(
            num_examples=total_examples,
            num_fl_users=num_fl_users,
            examples_per_user=total_examples // num_fl_users,
            batch_size=batch_size_hybrid,
            model=hybrid_model,
        )
        hybrid_trained_model, _ = run_fl_training(
            fl_model=hybrid_model,
            fl_data_provider=hybrid_data_provider,
            epochs=epochs,
            local_lr=local_lr,
            aggregator_config=FedAvgWithLRHybridAggregatorConfig(
                lr=hybrid_global_lr, buffer_size=hybrid_num_fl_users
            ),
            training_rate=training_rate,
            training_duration_mean=training_duration_mean,
        )
        print(
            f"Local LR {local_lr} Async LR {async_global_lr} Hybrid LR {hybrid_global_lr}"
        )
        return verify_models_equivalent_after_training(
            async_trained_model,
            hybrid_trained_model,
            reference_untrained_model,
            rel_epsilon=1e-4,
            abs_epsilon=1e-6,
        )


class TestFedBuff:
    @classmethod
    def setup_class(cls):
        np.random.seed(0)

    def test_async_hybrid_same_multiple_clients_to_sync(self):
        r"""
        Test when hybrid has multiple clients to sync
        1. Training in parallel. Meaning, both mechanisms start from the same global model (training_time >> training rate)
        2. Total examples is the same for both and local lr is 1.0
        3. Pure Async 1 user with 10 examples should be the same as hybrid 10 users each with 1 example
        """
        # training in parallel training_duration >> training_rate
        training_rate = 1
        training_duration_mean = training_rate * 100

        total_examples = 10
        num_fl_users = 10
        batch_size_hybrid = 1
        batch_size_async = total_examples

        buffer_size = num_fl_users
        local_lr = np.random.sample()
        async_global_lr = np.random.sample()
        hybrid_global_lr = async_global_lr / buffer_size
        error_msg = HybridFLTestUtils.compare_async_hybrid_uneven_data_split(
            total_examples=total_examples,
            num_fl_users=num_fl_users,
            hybrid_num_fl_users=buffer_size,
            async_global_lr=async_global_lr,
            hybrid_global_lr=hybrid_global_lr,
            batch_size_hybrid=batch_size_hybrid,
            batch_size_async=batch_size_async,
            epochs=5,
            training_rate=training_rate,
            training_duration_mean=training_duration_mean,
            local_lr=local_lr,
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_non_fl_hybrid_same_multiple_clients_to_sync(self):
        """
        Test nonFL and hybrid same with multiple clients to sync

        Both mechanisms train on the number of examples yield the same result if
        1. hybrid is trained sequentially (training rate >> training duration)
        2. NonFL has 1 user with 10 examples and hybrid has 10 users each with 1 example trained for 1 epoch
        3. Local_lr = some random value, global_lr = some random value
        4. Hybrid_global_lr = global_lr / (buffer_size * local_lr)
        """
        total_examples = 10
        num_fl_users = 10
        examples_per_user_hybrid = 1
        batch_size_hybrid = 1
        batch_size_nonfl = total_examples

        buffer_size = 10
        nonfl_lr = np.random.sample()
        hybrid_local_lr = np.random.sample()
        hybrid_global_lr = nonfl_lr / (buffer_size * hybrid_local_lr)

        error_msg = HybridFLTestUtils.compare_nonfl_hybrid_uneven_data_split(
            total_examples=total_examples,
            num_fl_users=num_fl_users,
            examples_per_user_hybrid=examples_per_user_hybrid,
            examples_per_user_nonfl=total_examples,
            batch_size_hybrid=batch_size_hybrid,
            batch_size_nonfl=batch_size_nonfl,
            buffer_size=buffer_size,
            non_fl_lr=nonfl_lr,
            hybrid_global_lr=hybrid_global_lr,
            local_lr=hybrid_local_lr,
            epochs=1,
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_async_hybrid_same_sync_every_client(self):
        r"""
        Hybrid and Async should yield the same model:
        1. Async and Hybrid training duration distribution are the same
        training_duration_sd should be small
        2. Hybrid update global model every 1 client
        """
        # random learning rate between 0 and 10
        local_lr = np.random.sample()
        global_lr = np.random.sample() * 10
        buffer_size = 1
        for base_aggregator_config in [
            FedAvgWithLRAsyncAggregatorConfig(lr=global_lr),
            FedAdamAsyncAggregatorConfig(lr=global_lr, eps=1e-2),
        ]:
            for batch_size in [4, 16, 32]:
                (
                    num_fl_users,
                    examples_per_user,
                    num_examples,
                    training_rate,
                ) = HybridFLTestUtils.get_data_params(
                    min_num_users=2,
                    max_num_users=10,
                    min_examples_per_user=1,
                    max_examples_per_user=10,
                )
                error_msg = HybridFLTestUtils.compare_hybrid_fl_same(
                    trainer_to_compare_hybrid_fl_with=TrainerType.ASYNC,
                    trainer_to_compare_aggregator_config=base_aggregator_config,
                    hybrid_aggregator_config=HybridFLTestUtils.get_hybrid_aggregator(
                        aggregator_config=base_aggregator_config,
                        buffer_size=buffer_size,
                        hybrid_lr=global_lr,
                    ),
                    hybrid_local_lr=local_lr,
                    base_local_lr=local_lr,
                    epochs=5,
                    num_examples=num_examples,
                    num_fl_users=num_fl_users,
                    batch_size=batch_size,
                    examples_per_user=examples_per_user,
                    buffer_size=buffer_size,
                    training_rate=training_rate,
                    training_duration_mean=1,
                    training_duration_sd=0,
                )
                assertEqual(error_msg, "")

    def test_nonfl_hybrid_same_sgd(self):
        r"""
        Hybrid and NonFL should yield the same model:
        1. Hybrid's training rate = 1, training duration ~ N(0, 0)
        2. Round robin user selector
        3. Hybrid takes global step  every 1 clients
        """
        (
            num_fl_users,
            examples_per_user,
            num_examples,
            _,
        ) = HybridFLTestUtils.get_data_params(
            min_num_users=1,
            max_num_users=10,
            min_examples_per_user=1,
            max_examples_per_user=10,
        )
        local_lr = 1.0
        buffer_size = 1
        for batch_size in [4, 16, 32]:
            global_lr = get_safe_global_lr(batch_size, examples_per_user)
            base_aggregator_config = SyncServerConfig(
                server_optimizer=FedAvgWithLROptimizerConfig(lr=global_lr, momentum=0.0)
            )
            error = HybridFLTestUtils.compare_hybrid_fl_same(
                trainer_to_compare_hybrid_fl_with=TrainerType.NONFL,
                trainer_to_compare_aggregator_config=base_aggregator_config,
                hybrid_aggregator_config=HybridFLTestUtils.get_hybrid_aggregator(
                    aggregator_config=base_aggregator_config,
                    buffer_size=buffer_size,
                    hybrid_lr=global_lr,
                ),
                hybrid_local_lr=local_lr,
                base_local_lr=local_lr,
                epochs=1,
                num_examples=num_examples,
                num_fl_users=num_fl_users,
                batch_size=batch_size,
                examples_per_user=examples_per_user,
                buffer_size=1,
                training_rate=1,
                training_duration_mean=0,
                training_duration_sd=0,
            )
            assertEqual(error, "")

    def test_nonfl_hybrid_same_adam(self):
        r"""
        Hybrid and NonFL should yield the same model:
        1. Hybrid's training rate = 1, training duration ~ N(0, 0)
        2. Round robin user selector
        3. Hybrid takes global step  every 1 clients
        """
        (
            num_fl_users,
            examples_per_user,
            num_examples,
            _,
        ) = HybridFLTestUtils.get_data_params(
            min_num_users=1,
            max_num_users=10,
            min_examples_per_user=1,
            max_examples_per_user=10,
        )
        local_lr = 1.0
        global_lr = np.random.sample() * 0.01
        base_aggregator_config = SyncServerConfig(
            server_optimizer=FedAdamOptimizerConfig(lr=global_lr, eps=1e-2)
        )
        buffer_size = 1
        error = HybridFLTestUtils.compare_hybrid_fl_same(
            trainer_to_compare_hybrid_fl_with=TrainerType.NONFL,
            trainer_to_compare_aggregator_config=base_aggregator_config,
            hybrid_aggregator_config=HybridFLTestUtils.get_hybrid_aggregator(
                aggregator_config=base_aggregator_config,
                buffer_size=buffer_size,
                hybrid_lr=global_lr,
            ),
            hybrid_local_lr=local_lr,
            base_local_lr=local_lr,
            epochs=5,
            num_examples=num_examples,
            num_fl_users=num_fl_users,
            batch_size=examples_per_user,
            examples_per_user=examples_per_user,
            buffer_size=1,
            training_rate=1,
            training_duration_mean=0,
            training_duration_sd=0,
        )
        assertEqual(error, "")

    def test_sync_hybrid_same_sgd(self):
        r"""
        Hybrid and Sync should yield the same model:
        1. For simplification, assume training_rate = total_users.
        Without this constraint, Hybrid==Sync only for 1 round of sync.
        2. `(Training_rate*num_users) << training_duration_mean`.
        E.g, `#users=10, training rate = 1, training_duration_mean > 10`.
        This is needed so that all users train in parallel.
        In particular, in a single epoch, every user starts with the same initial model
        3. Round robin user selector
        4. Set hybrid_global_lr = sync_global_lr / num_clients_sync
        """
        (
            num_fl_users,
            examples_per_user,
            num_examples,
            _,
        ) = HybridFLTestUtils.get_data_params(
            min_num_users=1,
            max_num_users=5,
            min_examples_per_user=1,
            max_examples_per_user=5,
        )
        local_lr = 1.0
        global_lr = 1.0
        base_aggregator_config = SyncServerConfig(
            server_optimizer=FedAvgWithLROptimizerConfig(lr=global_lr, momentum=0.0)
        )
        buffer_size = num_fl_users
        for batch_size in [4, 16, 32]:
            print(f"{num_fl_users} {examples_per_user} {local_lr} {global_lr}")
            error = HybridFLTestUtils.compare_hybrid_fl_same(
                trainer_to_compare_hybrid_fl_with=TrainerType.SYNC,
                trainer_to_compare_aggregator_config=base_aggregator_config,
                hybrid_aggregator_config=HybridFLTestUtils.get_hybrid_aggregator(
                    aggregator_config=base_aggregator_config,
                    buffer_size=buffer_size,
                    hybrid_lr=global_lr / buffer_size,
                ),
                hybrid_local_lr=local_lr,
                base_local_lr=local_lr,
                epochs=1,
                num_examples=num_examples,
                num_fl_users=num_fl_users,
                batch_size=batch_size,
                examples_per_user=examples_per_user,
                buffer_size=buffer_size,
                training_rate=buffer_size,
                training_duration_mean=buffer_size * 2,
                training_duration_sd=0,
            )
            assertEqual(error, "")

    def test_sync_hybrid_same_adam(self):
        r"""
        For sync == hybrid adam,
        `hybrid_local_lr = sync_local_lr / buffer_size`
        and hybrid and sync global lr's should be the same.

        Sync and hybrid compute different "delta"
        hybrid_delta = sync_delta * buffer_size
        to fix this difference, for SGD, we set `global_lr_hybrid = global_lr_sync/buffer_size`
        however, for Adam, this normalization doesn't work:
        Since Adam stores first and second moments (mean and variance) of deltas.
        In particular, the following are not equivalent:
        delta=d, lr=l
        delta=d*k, lr=l/k
        instead, we have to set `local_lr_hybrid = local_lr_sync / buffer_size`
        """
        (
            num_fl_users,
            examples_per_user,
            num_examples,
            _,
        ) = HybridFLTestUtils.get_data_params(
            min_num_users=1,
            max_num_users=10,
            min_examples_per_user=1,
            max_examples_per_user=10,
        )

        local_lr = 1.0
        global_lr = np.random.sample() * 0.01
        base_aggregator_config = SyncServerConfig(
            server_optimizer=FedAdamOptimizerConfig(lr=global_lr, eps=1e-2)
        )
        buffer_size = num_fl_users
        error = HybridFLTestUtils.compare_hybrid_fl_same(
            trainer_to_compare_hybrid_fl_with=TrainerType.SYNC,
            trainer_to_compare_aggregator_config=base_aggregator_config,
            hybrid_aggregator_config=HybridFLTestUtils.get_hybrid_aggregator(
                aggregator_config=base_aggregator_config,
                buffer_size=buffer_size,
                hybrid_lr=global_lr,
            ),
            hybrid_local_lr=local_lr / num_fl_users,
            base_local_lr=local_lr,
            epochs=1,
            num_examples=num_examples,
            num_fl_users=num_fl_users,
            batch_size=examples_per_user,
            examples_per_user=examples_per_user,
            buffer_size=buffer_size,
            training_rate=buffer_size,
            training_duration_mean=buffer_size * 100,
            training_duration_sd=0,
        )
        print(f"{num_fl_users} {examples_per_user} {local_lr} {global_lr}")
        assertEqual(error, "")

    def test_partial_model_update(self):
        r"""
        Test for partial update

        Assume we have 2 users (user1 and user2), buffer_size = 2, and both nonfl and fl training have
        exactly one batch training sequentially (training_rate >> training_duration), and denote initial global model as g0
        with the following timeline

        user1 starts training (receives g0)
        user1 ends training
        user2 starts training (note: user2 should receive g0)
        user2 ends training

        this sequence should produce the same model as non-fl training on user1 + user2 datasets
        """
        num_fl_users = 2
        total_examples = 20
        examples_per_user_hybrid = total_examples // num_fl_users
        buffer_size = num_fl_users

        nonfl_lr = np.random.sample()
        hybrid_local_lr = np.random.sample()
        # We need to normalize global_lr to account for the local training
        # hence we need to divide by the (buffer_size * hybrid_local_lr)
        hybrid_global_lr = nonfl_lr / (buffer_size * hybrid_local_lr)

        error_msg = HybridFLTestUtils.compare_nonfl_hybrid_uneven_data_split(
            total_examples=total_examples,
            num_fl_users=num_fl_users,
            buffer_size=buffer_size,
            non_fl_lr=nonfl_lr,
            hybrid_global_lr=hybrid_global_lr,
            local_lr=hybrid_local_lr,
            examples_per_user_hybrid=examples_per_user_hybrid,
            examples_per_user_nonfl=total_examples,
            batch_size_hybrid=examples_per_user_hybrid,
            batch_size_nonfl=total_examples,
            epochs=1,
        )
        print(
            f"NonFL LR {nonfl_lr} Hybrid Local LR {hybrid_local_lr}  Hybrid Global LR {hybrid_global_lr}"
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_remaining_clients_to_sync(self):
        """
        An aggregator can have unaggregated clients before
        an end of the epoch

        This case can happen in two scenario
        1. clients_to_sync > total_users
        2. total_users is not divisible by clients_to_sync (num_fl_users % clients_to_sync != 0)
        """
        # training in parallel training_duration >> training_rate
        training_rate = 1
        training_duration_mean = training_rate * 100

        total_examples = 10
        num_fl_users = 10
        batch_size_hybrid = 1
        batch_size_async = total_examples

        # test for scenario 1
        buffer_size = num_fl_users * 10
        local_lr = np.random.sample()
        async_global_lr = np.random.sample()

        hybrid_global_lr = async_global_lr / num_fl_users
        error_msg = HybridFLTestUtils.compare_async_hybrid_uneven_data_split(
            total_examples=total_examples,
            num_fl_users=num_fl_users,
            hybrid_num_fl_users=buffer_size,
            async_global_lr=async_global_lr,
            hybrid_global_lr=hybrid_global_lr,
            batch_size_hybrid=batch_size_hybrid,
            batch_size_async=batch_size_async,
            epochs=1,
            training_rate=training_rate,
            training_duration_mean=training_duration_mean,
            local_lr=local_lr,
        )
        assertEmpty(error_msg, msg=error_msg)

        # test for scenario 2
        buffer_size = 4
        hybrid_global_lr = async_global_lr / num_fl_users
        error_msg = HybridFLTestUtils.compare_async_hybrid_uneven_data_split(
            total_examples=total_examples,
            num_fl_users=num_fl_users,
            hybrid_num_fl_users=buffer_size,
            async_global_lr=async_global_lr,
            hybrid_global_lr=hybrid_global_lr,
            batch_size_hybrid=batch_size_hybrid,
            batch_size_async=batch_size_async,
            epochs=1,
            training_rate=training_rate,
            training_duration_mean=training_duration_mean,
            local_lr=local_lr,
        )
        assertEmpty(error_msg, msg=error_msg)
