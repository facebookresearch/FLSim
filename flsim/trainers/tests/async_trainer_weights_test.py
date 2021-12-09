#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Tuple

import numpy as np
import torch
from flsim.common.pytest_helper import (
    assertEqual,
    assertEmpty,
)
from flsim.data.data_provider import FLDataProviderFromList, IFLDataProvider
from flsim.data.data_sharder import RoundRobinSharder
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from flsim.interfaces.model import IFLModel
from flsim.optimizers.async_aggregators import (
    FedAvgWithLRAsyncAggregatorConfig,
    FedAvgWithLRHybridAggregatorConfig,
)
from flsim.tests.utils import verify_models_equivalent_after_training
from flsim.utils.async_trainer.async_example_weights import (
    AsyncExampleWeightConfig,
    ExampleWeight,
    EqualExampleWeightConfig,
    LinearExampleWeightConfig,
    SqrtExampleWeightConfig,
    Log10ExampleWeightConfig,
)
from flsim.utils.async_trainer.async_staleness_weights import (
    ConstantStalenessWeightConfig,
    PolynomialStalenessWeightConfig,
    ThresholdStalenessWeightConfig,
)
from flsim.utils.async_trainer.training_event_generator import (
    AsyncTrainingEventGeneratorFromListConfig,
    EventTimingInfo,
)
from flsim.utils.sample_model import DummyAlphabetFLModel
from flsim.utils.tests.helpers.async_trainer_test_utils import (
    async_train_one_user,
    get_equal_split_data,
    get_fl_data_provider,
    run_fl_nonfl_training,
    run_fl_training,
    run_fl_training_with_event_generator,
)
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from hydra.utils import instantiate


class TestAsyncTrainerWeights:
    def _get_fl_local_lr_and_example_weights_config(
        self,
        example_weight_config: AsyncExampleWeightConfig,
        num_examples_per_user: int,
        normalize_wts: bool,
    ) -> Tuple[float, AsyncExampleWeightConfig]:
        """Compute the fl_local_learning_rate and example_weight_config
        that will cause FL and Non-FL training to produce the same results.
        Normalize weights: when this option is true, example weight is divided
        by the average example weight
        """
        return_config = copy.deepcopy(example_weight_config)
        if normalize_wts:
            # effectively equal to not having any example weight, since all users
            # in the test have the same number of examples
            return_config.avg_num_examples = num_examples_per_user
            return (1.0, return_config)
        else:
            # create un-normalized example weight object
            return_config.avg_num_examples = 1
            example_weight_obj: ExampleWeight = instantiate(return_config)
            example_weight = example_weight_obj.weight(num_examples_per_user)
            fl_local_lr = 1.0 / example_weight
            return (fl_local_lr, return_config)

    def compare_num_examples_wt(
        self,
        base_learning_rate: float,
        example_weight_config: AsyncExampleWeightConfig,
        rel_epsilon: float,
        normalize_wts: bool,
    ):
        """Run training for two tasks:
        data1: 2 training examples, same for data2 and data3
        example_weight_type: defines how number of examples affect weight
          Example_weight_str is used to chose a function ExWt(x)
          When example_weight_str = 'linear', ExWt(x) = x
          When example_weight_str = 'sqrt', ExWt(x) = sqrt(x)
          When example_weight_str = 'log10', ExWt(x) = log10(x)
        Task1: Non-FL task, data = {data1, data2, data3}, LR=base_lr
        Task2: FL async task, training_duration = 0,
                LR=base_lr/ExWt(2), example_weight_type=LINEAR
                data = {user1: data1, user2: data2, user3: data3}
                Use  user selector,
                so we train in order user1-->user2-->user3
                user_epochs_per_round=1
        Train Task1, Task2. Show that you end up with the same model
        """
        num_training_examples_per_user = 2
        num_users = 3
        num_examples = 6

        torch.manual_seed(1)

        global_model = DummyAlphabetFLModel()
        # will be used later to verify training indeed took place
        global_model_init_copy = copy.deepcopy(global_model)
        nonfl_model = copy.deepcopy(global_model_init_copy)

        (
            fl_local_learning_rate,
            example_weight_config,
        ) = self._get_fl_local_lr_and_example_weights_config(
            example_weight_config=example_weight_config,
            num_examples_per_user=num_training_examples_per_user,
            normalize_wts=normalize_wts,
        )
        fl_aggregator_config = FedAvgWithLRAsyncAggregatorConfig(lr=base_learning_rate)
        nonfl_optimizer = torch.optim.SGD(
            nonfl_model.fl_get_module().parameters(), lr=base_learning_rate
        )
        num_epochs = 1
        fl_data_provider, nonfl_data_loader = get_equal_split_data(
            num_examples=num_examples,
            num_fl_users=num_users,
            fl_batch_size=num_training_examples_per_user,
            model=global_model,
            one_batch_per_user_only=False,
        )
        fl_trained_model, nonfl_trained_model = run_fl_nonfl_training(
            fl_model=global_model,
            nonfl_model=nonfl_model,
            nonfl_optimizer=nonfl_optimizer,
            fl_data_provider=fl_data_provider,
            nonfl_data_loader=nonfl_data_loader,
            epochs=num_epochs,
            fl_local_lr=fl_local_learning_rate,
            fl_aggregator_config=fl_aggregator_config,
            training_rate=1,
            training_duration_mean=0,
            training_duration_sd=0,
            example_weight_config=example_weight_config,
        )
        assertEqual(
            verify_models_equivalent_after_training(
                fl_trained_model,
                nonfl_trained_model,
                global_model_init_copy,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def test_linear_example_wts_sgd(self):
        for example_weight_config in (
            EqualExampleWeightConfig(),
            LinearExampleWeightConfig(),
            SqrtExampleWeightConfig(),
            Log10ExampleWeightConfig(),
        ):
            for normalize_wts in [True, False]:
                # choose a random learning rate between 0 and 1
                learning_rate = np.random.random_sample()
                # adding a printf to debug flaky test failure in Sandcastle
                print(f"Learning rate: {learning_rate}")
                self.compare_num_examples_wt(
                    base_learning_rate=learning_rate,
                    rel_epsilon=1e-6,
                    example_weight_config=example_weight_config,
                    normalize_wts=normalize_wts,
                )

    # TODO: add test_linear_example_wts_adam once we know how to scale
    # momentum and adaptive learning rate appropriately with weights

    def _get_fl_data_round_robin_sharding(
        self, num_examples: int, num_fl_users: int, fl_batch_size: int, model: IFLModel
    ) -> IFLDataProvider:
        dummy_dataset = DummyAlphabetDataset(num_examples)
        fl_data_sharder = RoundRobinSharder(num_shards=num_fl_users)
        fl_data_loader = FLDatasetDataLoaderWithBatch(
            dummy_dataset,
            dummy_dataset,
            dummy_dataset,
            fl_data_sharder,
            fl_batch_size,
            fl_batch_size,
            fl_batch_size,
        )
        fl_data_provider = FLDataProviderFromList(
            fl_data_loader.fl_train_set(),
            fl_data_loader.fl_eval_set(),
            fl_data_loader.fl_test_set(),
            model,
        )
        assert fl_data_loader.num_total_users == num_fl_users, "Error in data sharding"
        return fl_data_provider

    def test_default_example_weights(self):
        """Create an FL Async task with default values for example weight (equal example weight)
        Note: In async, weight of a user update = example_weight * staleness_weight
        Equal example weight => example_weight = 1.0 irrespective of #examples
        Verify that weight doesn't depend on number of examples. Train 2 FL tasks:
        Task1: {user1:data1}, {user2:data2}
        Task2: {user1:data1,data1}, {user2:data2,data2} (each user gets duplicate copy of data)
        Both tasks are trained in "sequential" order
        Task1 and Task2 produce the same model
        """
        num_epochs = 1
        num_users = 2
        torch.manual_seed(1)
        # create dummy FL model on alphabet
        trained_models = []
        initial_model = DummyAlphabetFLModel()
        local_lr = np.random.random_sample() * 10
        global_lr = np.random.random_sample() * 10
        dataset_size = 26  # number of examples in DummyAlphabetDataset
        for replication_factor in [1, 2]:
            num_examples = replication_factor * dataset_size  # either 26 or 52
            fl_model = copy.deepcopy(initial_model)
            fl_data_provider = self._get_fl_data_round_robin_sharding(
                num_examples=num_examples,
                num_fl_users=num_users,
                fl_batch_size=dataset_size,
                model=fl_model,
            )
            fl_trained_model, _ = run_fl_training(
                fl_model=fl_model,
                fl_data_provider=fl_data_provider,
                epochs=num_epochs,
                local_lr=local_lr,
                aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=global_lr),
                training_rate=1,
                training_duration_mean=0,
                training_duration_sd=0,
            )
            trained_models.append(fl_trained_model)
        assertEqual(len(trained_models), 2, "Expected to train two models")
        assertEqual(
            verify_models_equivalent_after_training(
                trained_models[0],
                trained_models[1],
                initial_model,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def test_default_staleness_weights(self):
        r"""Create an FL Async task with default values for staleness weight (no staleness weight)
        Verify that weight doesn't depend on staleness. Train on two users, where training events
        are as follows. Assume initial model=M
        T0: UserA starts training. Gets initial model M.
        T1: UserB starts training. Gets initial model M.
        T2: UserA finishes training. Local training: M->M_A. Global model updated to M_A
        T3: User B finishes training. Local training: M->M_B.
            Global model update: M_A->M_Final
            M_Final = M_A + (M_B - M)*staleness. **Since staleness=1**
            M_Final = M_A + M_B - M
        """
        num_epochs = 1
        torch.manual_seed(1)
        # create dummy FL model on alphabet
        initial_model = DummyAlphabetFLModel()
        local_lr = np.random.random_sample() * 10
        global_lr = 1.0
        fl_model = copy.deepcopy(initial_model)
        num_examples = 26
        num_fl_users = 2
        fl_data_provider, nonfl_data_loader = get_equal_split_data(
            num_examples=num_examples,
            num_fl_users=num_fl_users,
            fl_batch_size=num_examples // num_fl_users,  # each user has exactly 1 batch
            model=initial_model,
            one_batch_per_user_only=False,
        )
        fl_trained_model, _ = run_fl_training(
            fl_model=fl_model,
            fl_data_provider=fl_data_provider,
            epochs=num_epochs,
            local_lr=local_lr,
            aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=global_lr),
            training_rate=1,
            training_duration_mean=100,
            training_duration_sd=0,
        )
        # Simulate FL training: train on user1, update global model. Then train
        # on user 2, update global model.
        data_loader_iter = iter(nonfl_data_loader)
        user1_batch = next(data_loader_iter)
        user2_batch = next(data_loader_iter)
        simulated_global_model = copy.deepcopy(initial_model)
        # train user1
        simulated_global_model = async_train_one_user(
            global_model_at_training_start=simulated_global_model,
            global_model_at_training_end=simulated_global_model,
            batches=[user1_batch],
            local_lr=local_lr,
        )
        # train user2
        # key difference from training user1: global_model_at_training_start
        # is different from global_model_at_training_end
        simulated_global_model = async_train_one_user(
            global_model_at_training_start=initial_model,
            global_model_at_training_end=simulated_global_model,
            batches=[user2_batch],
            local_lr=local_lr,
        )
        error_msg = verify_models_equivalent_after_training(
            fl_trained_model,
            simulated_global_model,
            initial_model,
            rel_epsilon=1e-4,
            abs_epsilon=1e-6,
        )
        assertEqual(error_msg, "")

    def test_threshold_staleness_weights(self):
        r"""
        Run training for two tasks:
        Task1: FL async task, training_duration = 100, training_rate=1
                data = {user1: data1, user2: data2, user3: data3}
                staleness_weight = {threshold. if staleness>0, weight=0}
                timeline is:
                t=0, user 1 starts training
                t=1, user 2 strats training
                t=3, user 3 starts training
                t=100, user 1 finishes training. staleness=0. staleness_wt=1.0
                t=101, user 2 finishes training. staleness=1. staleness_wt=0
                t=102, user 3 finishes training. staleness=2. stalenes_wt=0
                Because users 2 and 3 have non-zero staleness, their model updates
                get no weight at all, so it's as if they didn't even exist
        Task2: {user1: data1}
        Train Task1, Task2. Show that we end up with the same model
        """
        num_epochs = 1
        num_users = 3
        num_examples_per_user = 8
        # training config when 3 users, with staleness
        staleness_training_rate = 1
        staleness_training_duration_mean = 100
        staleness_cutoff = 0
        staleness_wt_after_cutoff = 0.0

        # training config that forces sequential training
        sequential_training_rate = 1
        sequential_training_duration_mean = 0

        local_lr = 1.0
        aggregator_config = FedAvgWithLRAsyncAggregatorConfig(lr=1.0)

        # set seed to 1 before model init
        torch.manual_seed(1)
        initial_model = DummyAlphabetFLModel()

        fl_data_provider_with_staleness, _ = get_equal_split_data(
            num_examples=num_users * num_examples_per_user,
            num_fl_users=num_users,
            fl_batch_size=num_examples_per_user,
            model=initial_model,
            one_batch_per_user_only=False,
        )
        fl_trained_model_with_staleness, _ = run_fl_training(
            fl_model=copy.deepcopy(initial_model),
            fl_data_provider=fl_data_provider_with_staleness,
            epochs=num_epochs,
            local_lr=local_lr,
            aggregator_config=aggregator_config,
            training_rate=staleness_training_rate,
            training_duration_mean=staleness_training_duration_mean,
            training_duration_sd=0,
            staleness_weight_config=ThresholdStalenessWeightConfig(
                avg_staleness=0,
                cutoff=staleness_cutoff,
                value_after_cutoff=staleness_wt_after_cutoff,
            ),
        )

        fl_data_provider_sequential, _ = get_equal_split_data(
            num_examples=num_examples_per_user,
            num_fl_users=1,
            fl_batch_size=num_examples_per_user,
            model=initial_model,
            one_batch_per_user_only=False,
        )
        fl_trained_model_sequential, _ = run_fl_training(
            fl_model=copy.deepcopy(initial_model),
            fl_data_provider=fl_data_provider_sequential,
            epochs=num_epochs,
            local_lr=local_lr,
            aggregator_config=aggregator_config,
            training_rate=sequential_training_rate,
            training_duration_mean=sequential_training_duration_mean,
            training_duration_sd=0,
            staleness_weight_config=ConstantStalenessWeightConfig(),
        )
        assertEqual(
            verify_models_equivalent_after_training(
                fl_trained_model_with_staleness,
                fl_trained_model_sequential,
                initial_model,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def test_polynomial_staleness_weights(self):
        r"""
        Integration test for polynomial staleness weight
        Train task where polynomial staleness weight cancels out example_weight,
        resulting in total_wt=1.0
        training_data: {
            data1 = 26 examples
            user1:{data1, data1},
            user2:{data1,}
        Task1: FL async task, training_duration = 100, training_rate=1
                staleness_weight = {
                    polynomial, exponent=1.0 => wt = 1/(staleness+1)
                    staleness=0, wt=1.
                    staleness=1, wt=1/2,
                }
                example_weight = linear, avg_num_examples=26
                timeline is:
                t=1, user 1 starts training
                t=2, user 2 starts training
                t=3, user 2 finishes training
                       staleness=0. staleness_wt=1.0, example_wt=26/26, total_wt=1
                t=101, user 1 finishes training.
                       staleness=1. staleness_wt=0.5, example_wt=52/26, total_wt=1
        Task2: FL async task, training_duration = 100, training_rate=1
                staleness_weight = constant
                example_weight = constant
                all users get total_weight = 1.0
        Train Task1, Task2. Show that we end up with the same model
        """
        # common configs
        num_epochs = 1
        # set seed to 1 before model init
        torch.manual_seed(1)
        # train fl fask

        local_lr = 1.0
        aggregator_config = FedAvgWithLRAsyncAggregatorConfig(lr=1.0)

        num_users = 2
        # user1 has dataset_size examples, user2 has 2*dataset_size examples
        dataset_size = 26
        max_num_examples_per_user = dataset_size * num_users
        total_num_examples = dataset_size * 3

        # create config such that training events happen in the following order
        # t=1, user 1 starts training
        # t=2, user 2 starts training
        # t=3, user 2 finishes training
        # t=101, user 1 finishes training.
        user_1_start_time_delta = 1
        user_2_start_time_delta = 1
        user_1_training_duration = 100
        user_2_training_duration = 1
        user1_training_events = EventTimingInfo(
            prev_event_start_to_current_start=user_1_start_time_delta,
            duration=user_1_training_duration,
        )
        user2_training_events = EventTimingInfo(
            prev_event_start_to_current_start=user_2_start_time_delta,
            duration=user_2_training_duration,
        )
        event_generator_config = AsyncTrainingEventGeneratorFromListConfig(
            training_events=[user1_training_events, user2_training_events]
        )

        # set seed to 1 before model init
        torch.manual_seed(1)
        initial_model = DummyAlphabetFLModel()

        # split data such that
        # user1: {a,b,c...z,a,b,c...z}, user2:{a,b,c...z}
        data_provider = get_fl_data_provider(
            num_examples=total_num_examples,
            num_fl_users=2,
            examples_per_user=max_num_examples_per_user,
            batch_size=max_num_examples_per_user,
            model=initial_model,
        )

        poly_staleness_wt_model = run_fl_training_with_event_generator(
            fl_model=copy.deepcopy(initial_model),
            fl_data_provider=data_provider,
            epochs=num_epochs,
            local_lr=local_lr,
            aggregator_config=aggregator_config,
            training_event_generator_config=event_generator_config,
            staleness_weight_config=PolynomialStalenessWeightConfig(
                avg_staleness=0, exponent=1.0
            ),
            example_weight_config=LinearExampleWeightConfig(
                avg_num_examples=dataset_size
            ),
        )

        equal_staleness_wt_model = run_fl_training_with_event_generator(
            fl_model=copy.deepcopy(initial_model),
            fl_data_provider=data_provider,
            epochs=num_epochs,
            local_lr=local_lr,
            aggregator_config=aggregator_config,
            training_event_generator_config=event_generator_config,
            staleness_weight_config=ConstantStalenessWeightConfig(avg_staleness=0),
            example_weight_config=LinearExampleWeightConfig(
                avg_num_examples=dataset_size
            ),
        )

        assertEqual(
            verify_models_equivalent_after_training(
                poly_staleness_wt_model,
                equal_staleness_wt_model,
                initial_model,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def test_example_wts_and_staleness_wts_lr_scale(self):
        """
        Test for same number of examples and a fixed global learning rate.
        Async with no example weight should be the same as linear example weight where
        ex_local_lr= 1, ex_global_lr = 1
        no_ex_local_lr = ex_local_lr, no_ex_global_lr = ex_global_lr / max_num_examples
        """
        total_num_examples = 100
        num_users = 10
        examples_per_user = total_num_examples // num_users
        batch_size = 3

        init_model = DummyAlphabetFLModel()
        data_provider = get_fl_data_provider(
            num_examples=total_num_examples,
            num_fl_users=num_users,
            examples_per_user=examples_per_user,
            batch_size=batch_size,
            model=init_model,
        )

        local_lr = 1.0
        epochs = 5
        training_mean = 1
        training_std = 1
        buffer_size = 5

        example_wt_off_model = copy.deepcopy(init_model)
        example_wt_off_global_lr = 1.0
        example_wt_off_model, _ = run_fl_training(
            fl_model=example_wt_off_model,
            fl_data_provider=data_provider,
            epochs=epochs,
            local_lr=local_lr,
            aggregator_config=FedAvgWithLRHybridAggregatorConfig(
                lr=example_wt_off_global_lr, buffer_size=buffer_size
            ),
            training_rate=1,
            training_duration_mean=training_mean,
            training_duration_sd=training_std,
            staleness_weight_config=PolynomialStalenessWeightConfig(
                avg_staleness=0, exponent=0.5
            ),
            example_weight_config=EqualExampleWeightConfig(avg_num_examples=1),
        )

        example_wt_on_model = copy.deepcopy(init_model)
        example_wt_on_global_lr = example_wt_off_global_lr / examples_per_user
        example_wt_on_model, _ = run_fl_training(
            fl_model=example_wt_on_model,
            fl_data_provider=data_provider,
            epochs=epochs,
            local_lr=local_lr,
            aggregator_config=FedAvgWithLRHybridAggregatorConfig(
                lr=example_wt_on_global_lr, buffer_size=buffer_size
            ),
            training_rate=1,
            training_duration_mean=training_mean,
            training_duration_sd=training_std,
            staleness_weight_config=PolynomialStalenessWeightConfig(
                avg_staleness=0, exponent=0.5
            ),
            example_weight_config=LinearExampleWeightConfig(avg_num_examples=1),
        )

        error_msg = verify_models_equivalent_after_training(
            example_wt_on_model,
            example_wt_off_model,
            init_model,
            rel_epsilon=1e-4,
            abs_epsilon=1e-6,
        )
        assertEmpty(error_msg, msg=error_msg)
