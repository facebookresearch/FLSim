#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, List, Dict

import numpy as np
import pytest
import torch
from flsim.common.pytest_helper import (
    assertTrue,
    assertEqual,
    assertLess,
    assertAlmostEqual,
    assertListEqual,
)
from flsim.common.timeline import Timeline
from flsim.common.timeout_simulator import GaussianTimeOutSimulatorConfig
from flsim.interfaces.metrics_reporter import TrainingStage, Channel
from flsim.interfaces.model import IFLModel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.optimizers.async_aggregators import (
    AsyncAggregatorConfig,
    FedAdamAsyncAggregatorConfig,
    FedAvgWithLRAsyncAggregatorConfig,
    FedAvgWithLRHybridAggregatorConfig,
)
from flsim.optimizers.optimizer_scheduler import LRBatchSizeNormalizerSchedulerConfig
from flsim.tests.utils import (
    verify_models_equivalent_after_training,
    RandomEvalMetricsReporter,
    MetricsReporterWithMockedChannels,
)
from flsim.utils.async_trainer.async_example_weights import (
    EqualExampleWeightConfig,
    LinearExampleWeightConfig,
)
from flsim.utils.async_trainer.training_event_generator import (
    AsyncTrainingEventGeneratorConfig,
    ConstantAsyncTrainingStartTimeDistrConfig,
    AsyncTrainingEventGeneratorFromListConfig,
    EventTimingInfo,
)
from flsim.utils.sample_model import DummyAlphabetFLModel, ConstantGradientFLModel
from flsim.utils.tests.helpers.async_trainer_test_utils import (
    async_train_one_user,
    create_async_trainer,
    create_event_generator_config,
    get_data,
    get_equal_split_data,
    get_fl_data_provider,
    get_safe_global_lr,
    get_unequal_split_data,
    run_fl_training_with_event_generator,
    run_fl_training,
    assert_fl_nonfl_same,
    assert_fl_nonfl_same_equal_data_split,
)
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
)
from flsim.utils.timing.training_duration_distribution import (
    PerUserUniformDurationDistributionConfig,
)


class TestMetricsReporter(MetricsReporterWithMockedChannels):
    """Don't reset metrics when reported. Else, at the end of
    training, metrics have zeroes
    """

    def reset(self):
        pass


class ConcurrencyMetricsReporter(FLMetricsReporter):
    ACCURACY = "Accuracy"

    def __init__(self, channels: List[Channel]):
        self.concurrency_metrics = []
        self.eval_rounds = []
        super().__init__(channels)

    def compare_metrics(self, eval_metrics, best_metrics):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True
        return eval_metrics > best_metrics

    def compute_scores(self) -> Dict[str, Any]:
        # compute accuracy
        correct = torch.Tensor([0])
        for i in range(len(self.predictions_list)):
            all_preds = self.predictions_list[i]
            pred = all_preds.data.max(1, keepdim=True)[1]

            assert pred.device == self.targets_list[i].device, (
                f"Pred and targets moved to different devices: "
                f"pred >> {pred.device} vs. targets >> {self.targets_list[i].device}"
            )
            if i == 0:
                correct = correct.to(pred.device)

            correct += pred.eq(self.targets_list[i].data.view_as(pred)).sum()

        # total number of data
        total = sum(len(batch_targets) for batch_targets in self.targets_list)

        accuracy = 100.0 * correct.item() / total

        return {self.ACCURACY: accuracy}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        return scores[self.ACCURACY]

    def report_metrics(
        self,
        reset,
        stage,
        extra_metrics=None,
        **kwargs,
    ):

        if stage != TrainingStage.EVAL:
            assert (
                extra_metrics is not None
            ), "Async Trainer metrics reporting should have extra metrics"
            metrics = [m for m in extra_metrics if m.name == "Concurrency_Rate"]
            assert (
                len(metrics) == 1
            ), "Conrrency rate should be one of the extra metrics"
            concurrency_rate = metrics[0]
            self.concurrency_metrics.append(concurrency_rate.value)
        else:
            timeline: Timeline = kwargs.get("timeline", Timeline(global_round=1))
            self.eval_rounds.append(timeline.global_round)
        return super().report_metrics(reset, stage, extra_metrics, **kwargs)


class TestAsyncTrainer:

    # TODO: add test_one_user_sequential_user_same
    # TODO: add test where each user has unequal amount of data, both SGD and adam

    def _assert_fl_nonfl_same_unequal_data_split(
        self,
        fl_batch_size: int,
        num_examples: int,
        num_fl_users: int,
        max_examples_per_user: int,
        epochs: int,
        local_lr: float,
        aggregator_config: AsyncAggregatorConfig,
        training_rate=1,
        training_duration_mean=0,
        training_duration_sd=0,
    ):
        torch.manual_seed(1)
        # create dummy FL model on alphabet
        global_model = DummyAlphabetFLModel()
        # if aggregator is FedAdam, each user should have only one batch
        # otherwise non-FL and FL training won't be the same
        one_batch_per_user_only = "FedAdam" in aggregator_config._target_
        fl_data_provider, nonfl_data_loader = get_unequal_split_data(
            num_examples=num_examples,
            num_fl_users=num_fl_users,
            max_examples_per_user=max_examples_per_user,
            fl_batch_size=fl_batch_size,
            model=global_model,
            one_batch_per_user_only=one_batch_per_user_only,
        )
        assertEqual(
            assert_fl_nonfl_same(
                global_model=global_model,
                fl_data_provider=fl_data_provider,
                nonfl_data_loader=nonfl_data_loader,
                epochs=epochs,
                local_lr=local_lr,
                aggregator_config=aggregator_config,
                training_rate=training_rate,
                training_duration_mean=training_duration_mean,
                training_duration_sd=training_duration_sd,
            ),
            "",
        )

    def assert_fl_nonfl_same_one_user_sgd(self):
        """
        Given:
            data_for_fl={user1:batchA, batchB, batchC, ...}
            data_for_non_fl={batchA, batchB, batchC, ...}

        Check that the following produce the same trained model:
        1. FL training, global_opt=SGD, local_lr=x, global_lr=x
        2. Non-FL training, opt=SGD, lr=x
        """
        # try 3 different batch sizes:
        # 4=> user has multiple batches
        # 32=> user has exactly 1 batch, same as #examples
        # 128=> user has exactly 1 batch, but bigger than #examples
        num_examples = 32
        for fl_batch_size in [4, 32, 128]:
            # choose random learning rate between 0 and 10
            local_lr = np.random.random_sample() * 10
            global_lr = get_safe_global_lr(
                fl_batch_size=fl_batch_size, max_examples_per_user=num_examples
            )
            assertEqual(
                assert_fl_nonfl_same_equal_data_split(
                    num_examples=num_examples,
                    num_fl_users=1,
                    fl_batch_size=fl_batch_size,
                    training_rate=1,
                    training_duration_mean=0,
                    training_duration_sd=0,
                    epochs=5,
                    local_lr=local_lr,
                    aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=global_lr),
                ),
                "",
            )

    def assert_fl_nonfl_same_multiple_users_sgd(self):
        """
        Given:
            data_for_fl={user1:batch1A, batch1B..., user2: batch2A, batch2B,...}
            data_for_non_fl={batch1A, batch1B, ..., batch2A, batch2B, ...}

        Check that the following produce the same trained model:
        1. AsyncFL training, training_rate=1 (sequential), global_opt=SGD, local_lr=x, global_lr=x
        2. Non-FL training, opt=SGD, lr=x
        """
        num_fl_users = 4
        num_examples = 32
        # equal data split
        # try 3 different batch size
        # 4=> each user has multiple batches
        # 16=> each user has exactly 1 batch, same as #examples_per_user
        # 128=> each user has exactly 1 batch, but bigger than #examples_per_user
        equal_split_examples_per_user = num_examples / num_fl_users
        for fl_batch_size in [4, 16, 32]:
            # random learning rate between 0 and 10
            local_lr = np.random.random_sample() * 10
            global_lr = get_safe_global_lr(
                fl_batch_size=fl_batch_size,
                max_examples_per_user=equal_split_examples_per_user,
            )
            assertEqual(
                assert_fl_nonfl_same_equal_data_split(
                    num_examples=num_examples,
                    num_fl_users=num_fl_users,
                    fl_batch_size=fl_batch_size,
                    training_rate=1,
                    training_duration_mean=0,
                    training_duration_sd=0,
                    epochs=5,
                    local_lr=local_lr,
                    aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=global_lr),
                ),
                "",
            )

        # unequal data split
        # user1 = 24 examples, user2 = 8 examples
        # 4=> each user has multiple batches
        # 8=> user1 has multiple batches, user 2 has 1 batch
        # cannot try batch size larger than 8
        unequal_split_max_examples_per_user = 24
        for fl_batch_size in [4, 8]:
            # random learning rate between 0 and 10
            local_lr = np.random.random_sample() * 10
            global_lr = get_safe_global_lr(
                fl_batch_size=fl_batch_size,
                max_examples_per_user=unequal_split_max_examples_per_user,
            )
            self._assert_fl_nonfl_same_unequal_data_split(
                num_examples=num_examples,
                num_fl_users=2,
                max_examples_per_user=unequal_split_max_examples_per_user,
                fl_batch_size=fl_batch_size,
                training_rate=1,
                training_duration_mean=0,
                training_duration_sd=0,
                epochs=5,
                local_lr=local_lr,
                aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=global_lr),
            )

    def assert_fl_nonfl_same_one_user_adam(self):
        """
        Given:
            data_for_fl={user1:batchA, batchB, batchC, ...}
            data_for_non_fl={batchA, batchB, batchC, ...}

        Check that the following produce the same trained model:
        1. FL training, global_opt=Adam, local_lr=x, global_lr=x
        2. Non-FL training, opt=Adam, lr=x
        """
        # for Adam, for FL and non-FL to be equivalent,
        # each user must have maximum one batch of data,
        # because local optimizer on each user has to be SGD
        # 32=> user has exactly 1 batch, same as #examples_per_user
        # 128=> user has exactly 1 batch, but bigger than #examples_per_user
        for fl_batch_size in [32, 128]:
            # random learning rate between 0 and 0.01
            global_lr = np.random.random_sample() * 0.01
            assertEqual(
                assert_fl_nonfl_same_equal_data_split(
                    num_examples=32,
                    num_fl_users=1,
                    fl_batch_size=fl_batch_size,
                    training_rate=1,
                    training_duration_mean=0,
                    training_duration_sd=0,
                    epochs=5,
                    local_lr=1.0,
                    aggregator_config=FedAdamAsyncAggregatorConfig(
                        lr=global_lr, eps=1e-2
                    ),
                ),
                "",
            )

    def assert_fl_nonfl_same_multiple_users_adam(self):
        """
        Given:
            data_for_fl={user1:batch1A, batch1B..., user2: batch2A, batch2B,...}
            data_for_non_fl={batch1A, batch1B, ..., batch2A, batch2B, ...}

        Check that the following produce the same trained model:
        1. AsyncFL training, training_rate=1 (sequential), global_opt=Adam, local_lr=x, global_lr=x
        2. Non-FL training, opt=Adam, lr=x
        """
        # for Adam, for FL and non-FL to be equivalent,
        # each user must have maximum one batch of data,
        # because local optimizer on each user has to be SGD
        # 8=> each user has exactly 1 batch, same as #examples_per_user
        # 128=> each user has exactly 1 batch, but bigger than #examples_per_user
        for fl_batch_size in [8, 128]:
            # random learning rate between 0 and 0.001
            global_lr = np.random.random_sample() * 0.001
            assertEqual(
                assert_fl_nonfl_same_equal_data_split(
                    num_examples=32,
                    num_fl_users=4,
                    fl_batch_size=fl_batch_size,
                    training_rate=1,
                    training_duration_mean=0,
                    training_duration_sd=0,
                    epochs=5,
                    local_lr=1.0,
                    aggregator_config=FedAdamAsyncAggregatorConfig(
                        lr=global_lr, eps=1e-2
                    ),
                ),
                "",
            )

    def _test_local_lr_normalization(self, num_users: int):
        """Run training for two tasks:
            Assumption: batch_size > data on any user (so all users have incomplete batches)
            AsyncFLTask1: data = {user1:data1, user2:data2},
                local_lr_normalization = False, weighting scheme = linear, lr = base_lr
            AsyncFLTask2: data = {user1:data1, user2:data2},
                local_lr_normalization = True, weighting scheme = equal, lr = base_lr
            Task1, Step 1:
               LR = base_lr
               Grad = (grad_user1*base_lr)
               Wt = linear = num_ex_user1
               Model_new = Model_old - (grad_user1*base_lr) * num_ex_user1
            Task2, Step 1:
               LR = base_lr * batch_size
               Grad = grad_user1 *(num_ex_user_1/batch_size) * (base_lr*batch_size)
               Wt = equal = 1
               Model_new = Model_old - (grad_user1*num_ex_user1*base_lr)
            So both models should take the same step
        Verify that the trained models have the same parameters
        """
        fl_batch_size = 128
        base_lr = 0.1
        # local_lr_normalization scales learning rate by num_examples_in_batch
        # we upweight the learning rate so that after local_lr_normalization, the
        # fl-task1 has the same learning rate as fl-task2
        upweighted_lr = base_lr * fl_batch_size

        num_total_examples = 32
        epochs = 5
        # create dummy FL model on alphabet
        torch.manual_seed(1)
        init_model = DummyAlphabetFLModel()

        fl_data_provider, _ = get_equal_split_data(
            num_examples=num_total_examples,
            num_fl_users=num_users,
            fl_batch_size=fl_batch_size,
            model=init_model,
        )

        def get_base_optimizer_and_trained_model():
            init_model_local = copy.deepcopy(init_model)
            async_trainer = create_async_trainer(
                model=init_model_local,
                local_lr=base_lr,
                epochs=epochs,
                aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=1.0),
                example_weight_config=LinearExampleWeightConfig(),
                event_generator_config=create_event_generator_config(
                    training_rate=1.0, training_duration_mean=0, training_duration_sd=0
                ),
            )
            trained_fl_model, _ = async_trainer.train(
                data_provider=fl_data_provider,
                metric_reporter=MetricsReporterWithMockedChannels(),
                num_total_users=num_users,
                distributed_world_size=1,
            )
            return trained_fl_model

        def get_lrnorm_optimizer_and_trained_model():
            init_model_local = copy.deepcopy(init_model)
            async_trainer = create_async_trainer(
                model=init_model_local,
                local_lr=upweighted_lr,
                epochs=epochs,
                aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=1.0),
                example_weight_config=EqualExampleWeightConfig(),
                local_lr_scheduler_config=LRBatchSizeNormalizerSchedulerConfig(
                    base_lr=upweighted_lr, local_lr_normalizer=fl_batch_size
                ),
                event_generator_config=create_event_generator_config(
                    training_rate=1.0, training_duration_mean=0, training_duration_sd=0
                ),
            )
            trained_fl_model, _ = async_trainer.train(
                data_provider=fl_data_provider,
                metric_reporter=MetricsReporterWithMockedChannels(),
                num_total_users=num_users,
                distributed_world_size=1,
            )
            return trained_fl_model

        base_trained_model = get_base_optimizer_and_trained_model()
        lrnorm_trained_model = get_lrnorm_optimizer_and_trained_model()
        assertEqual(
            verify_models_equivalent_after_training(
                base_trained_model,
                lrnorm_trained_model,
                init_model,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def test_local_lr_normalization_single_user(self):
        self._test_local_lr_normalization(num_users=1)

    def test_local_lr_normalization_multiple_users(self):
        self._test_local_lr_normalization(num_users=4)

    def test_async_training_metrics_reporting(self):
        """Verify that in async training, metrics are only reported
        training_end event fires, not when training_start event fires.
        We do this by:
        a) Creating data where user1=8 examples, user2=1 example
        b) Creating training event order as following:
             t1: user1 starts training
             t2: user2 starts training
             t3: user2 finishes training (should cause metric reporting with #examples=1)
             t11: user1 finishes trainig (should cause metric reporting with #examples=8)
             We verify the order in which metrics were reported.
             metric_reporter.num_examples_list should be = [1, 8]
        b) Creating training event order as following:
             t1: user1 starts training
             t2: user2 starts training
             t3: user1 finishes training (should cause metric reporting with #examples=8)
             t10: user2 finishes trainig (should cause metric reporting with #examples=1)
             We verify the order in which metrics were reported.
             metric_reporter.num_examples_list should be = [8, 1]
        """
        # first entry in tuple: time gap between training start and previous training start
        # second entry in tuple: training duration
        num_examples = 9
        num_fl_users = 2
        max_examples_per_user = 8
        fl_batch_size = 8
        num_epochs = 2
        global_model = DummyAlphabetFLModel()

        fl_data_provider = get_fl_data_provider(
            num_examples,
            num_fl_users,
            max_examples_per_user,
            fl_batch_size,
            global_model,
        )

        def _verify_training(
            user_1_start_time_delta: int,
            user1_training_duration: int,
            user2_start_time_delta: int,
            user2_training_duration: int,
            expected_num_examples: List[int],
        ):
            """Given start time and duration for user1 and user2, verify that that
            num_examples in training has order expected_num_examples
            """
            user1_training_events = EventTimingInfo(
                prev_event_start_to_current_start=user_1_start_time_delta,
                duration=user1_training_duration,
            )
            user2_training_events = EventTimingInfo(
                prev_event_start_to_current_start=user2_start_time_delta,
                duration=user2_training_duration,
            )
            async_trainer = create_async_trainer(
                model=global_model,
                local_lr=1.0,
                epochs=num_epochs,
                aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=1.0),
                event_generator_config=AsyncTrainingEventGeneratorFromListConfig(
                    training_events=[user1_training_events, user2_training_events]
                ),
            )

            metric_reporter = TestMetricsReporter()
            fl_model, _ = async_trainer.train(
                data_provider=fl_data_provider,
                metric_reporter=metric_reporter,
                num_total_users=fl_data_provider.num_users(),
                distributed_world_size=1,
            )
            assertEqual(
                metric_reporter.num_examples_list, expected_num_examples * num_epochs
            )

        # user2 completes training before user1
        _verify_training(
            user_1_start_time_delta=1,
            user1_training_duration=3,
            user2_start_time_delta=1,
            user2_training_duration=1,
            expected_num_examples=[1, 8],
        )

        # user1 completes training before user2
        _verify_training(
            user_1_start_time_delta=1,
            user1_training_duration=1,
            user2_start_time_delta=10,
            user2_training_duration=2,
            expected_num_examples=[8, 1],
        )

    def _run_sequential_training_two_users(
        self,
        initial_model: IFLModel,
        local_lr: int,
        first_user_batches: List[Any],
        second_user_batches: List[Any],
    ) -> IFLModel:
        """Sequential training.
        user1 = user with data: first_user_batches
        user2 = user with data: second_user_batches

        order of events:
        user1 starts training
        user1 finishes training. Global model update
        user2 starts training
        user2 finishes training. Global model update
        """
        global_model = copy.deepcopy(initial_model)
        # sequential training. user1 finishes first. user2 takes user1 trained model, and trains
        for batches in [first_user_batches, second_user_batches]:
            updated_global_model = async_train_one_user(
                global_model_at_training_start=global_model,
                global_model_at_training_end=global_model,
                batches=batches,
                local_lr=local_lr,
            )
            global_model = copy.deepcopy(updated_global_model)
        return global_model

    def _run_parallel_training_two_users(
        self,
        initial_model: IFLModel,
        local_lr: int,
        first_user_batches: List[Any],
        second_user_batches: List[Any],
    ) -> IFLModel:
        """Parallel training.
        user1 = user with data: first_user_batches
        user2 = user with data: second_user_batches

        order of events:
        user1 starts training
        user2 start training
        user1 finishes training. Global model update
        user2 finishes training. Global model update
        NB: this function and _run_sequential_training_two_users share a lot of code, but
            they are much easier to read as separate functions. Trading off code duplication
            for readability.
        """
        global_model = copy.deepcopy(initial_model)
        for batches in [first_user_batches, second_user_batches]:
            # both users start training with the same initial model
            # loop iteration 1 will handle first_user_batches, which will change
            # global_model. so global_model in loop iteration 2 will already include
            # changes from loop iteration 1
            updated_global_model = async_train_one_user(
                global_model_at_training_start=initial_model,
                global_model_at_training_end=global_model,
                batches=batches,
                local_lr=local_lr,
            )
            global_model = copy.deepcopy(updated_global_model)
        return global_model

    def test_num_examples_computation(self):
        r"""
        Test that num_examples for each user affects training duration. Test this for different batch sizes
        Two users: U1 and U2. U2 always has 2 examples. U1 may have 2, 4, or 6 examples
        Training duration = number of examples
        Timing:
          U1 has 2 examples. Sequential training
            t=0 = U1 starts training.
            t=2 = U1 ends training
            t=3 = U2 starts training
            t=5 = U2 ends training
          U1 has 4 examples. U1 and U2 train in parallel. U1 ends first
            t=0 = U1 starts training.
            t=3 = U2 starts training
            t=4 = U1 ends training
            t=5 = U2 ends training
          U1 has 6 examples. U1 and U2 train in parallel. U2 ends first
            t=0 = U1 starts training.
            t=3 = U2 starts training
            t=5 = U2 ends training
            t=6 = U1 ends training
        Verify by comparing with non-FL training that the above timing is correct.
        Do this for batch_size=1 and batch_size=2
        """
        num_epochs = 1
        torch.manual_seed(1)
        initial_model = DummyAlphabetFLModel()
        local_lr = 1.0
        global_lr = 1.0
        num_examples_user2 = 2
        batch_size = 2
        for num_examples_user1 in [2, 4, 6]:
            fl_model = copy.deepcopy(initial_model)
            fl_data_provider, nonfl_data_loader = get_data(
                num_examples=num_examples_user1 + num_examples_user2,
                num_fl_users=2,
                examples_per_user=num_examples_user1,
                fl_batch_size=batch_size,
                nonfl_batch_size=batch_size,
                model=initial_model,
            )
            fl_trained_model = run_fl_training_with_event_generator(
                fl_model=fl_model,
                fl_data_provider=fl_data_provider,
                epochs=num_epochs,
                local_lr=local_lr,
                aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=global_lr),
                training_event_generator_config=create_event_generator_config(
                    training_rate=0.33, training_duration_mean=1, training_duration_sd=0
                ),
            )
            # first num_examples_user1/batch_size batches are for user1. Rest for user2
            all_batches = list(nonfl_data_loader)
            num_user1_batches = num_examples_user1 // batch_size
            user1_batches = all_batches[:num_user1_batches]
            user2_batches = all_batches[num_user1_batches:]
            if num_examples_user1 == 2:
                # U1 training takes 2 units of time
                # sequence:
                # t=0 U1 starts training
                # t=2 U1 finishes training
                # t=3 U2 starts training
                # t=5 U2 finishes training
                simulated_global_model = self._run_sequential_training_two_users(
                    first_user_batches=user1_batches,
                    second_user_batches=user2_batches,
                    initial_model=initial_model,
                    local_lr=local_lr,
                )
            elif num_examples_user1 == 4:
                # U1 training takes 4 units of time
                # sequence:
                # t=0 U1 starts training
                # t=3 U2 starts training
                # t=4 U1 finishes training
                # t=5 U2 finishes training
                simulated_global_model = self._run_parallel_training_two_users(
                    first_user_batches=user1_batches,
                    second_user_batches=user2_batches,
                    initial_model=initial_model,
                    local_lr=local_lr,
                )
            elif num_examples_user1 == 6:
                # U1 training takes 6 units of time
                # sequence:
                # t=0 U1 starts training
                # t=3 U2 starts training
                # t=5 U2 finishes training
                # t=6 U1 finishes training
                simulated_global_model = self._run_parallel_training_two_users(
                    first_user_batches=user2_batches,
                    second_user_batches=user1_batches,
                    initial_model=initial_model,
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

    def test_async_training_with_timeout(self):
        """
        Test async training with timeout on.

        Under the condition when mean_per_example=1, std_per_example=0
        The total number of examples trained should be less than the timeout limit

        The test does the following:
        1. Create a data provider with 10 users and each user with 10 examples
        2. Set batch size = 1, hence the metrics reporter losses = num_trained_examples
        3. Check if num_trained_examples <= timeout_limit * num_users
        """
        local_lr = 1.0
        global_lr = 1.0
        epochs = 1
        init_model_local = DummyAlphabetFLModel()
        metrics_reporter = TestMetricsReporter()

        num_users = 10
        num_examples = 100

        timeout_limit = np.random.randint(1, 10)
        async_trainer = create_async_trainer(
            model=init_model_local,
            local_lr=local_lr,
            epochs=epochs,
            aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=global_lr),
            event_generator_config=create_event_generator_config(
                training_rate=1.0, training_duration_mean=0.0, training_duration_sd=0.0
            ),
            timeout_simulator_config=GaussianTimeOutSimulatorConfig(
                timeout_wall_per_round=timeout_limit,
                fl_stopping_time=1e4,
                duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                    training_duration_mean=1.0, training_duration_sd=0.0
                ),
            ),
        )

        data_provider = get_fl_data_provider(
            num_examples=num_examples,
            num_fl_users=num_users,
            examples_per_user=10,
            batch_size=1,
            model=init_model_local,
        )

        final_model, _ = async_trainer.train(
            data_provider=data_provider,
            metric_reporter=metrics_reporter,
            num_total_users=num_users,
            distributed_world_size=1,
        )
        total_examples_trained = len(metrics_reporter.losses)
        total_time = timeout_limit * num_users
        assertTrue(total_examples_trained <= total_time)

    def test_max_staleness_cutoff(self):
        """
        Test for max staleness cut off

        There are two scenario:
        1. Users are trained in parallel with training_mean = 1,
        and max_staleness is some small number then not all users will participate in training
        2. Users are trained sequentially, which means there is no staleness hence all users can participate
        """
        global_model = DummyAlphabetFLModel()
        metrics_reporter = TestMetricsReporter()
        num_users = 10
        num_examples = num_users
        training_rate = num_users
        max_staleness = np.random.randint(1, 5)

        # training in parallel so not all clients can participate
        async_trainer = create_async_trainer(
            model=global_model,
            local_lr=1.0,
            epochs=1,
            aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=1.0),
            event_generator_config=create_event_generator_config(
                training_rate=training_rate,
                training_duration_mean=1.0,
                training_duration_sd=0,
            ),
            max_staleness=max_staleness,
        )

        data_provider = get_fl_data_provider(
            num_examples=num_examples,
            num_fl_users=num_users,
            examples_per_user=1,
            batch_size=1,
            model=global_model,
        )

        final_model, _ = async_trainer.train(
            data_provider=data_provider,
            metric_reporter=metrics_reporter,
            num_total_users=num_users,
            distributed_world_size=1,
        )
        assertLess(async_trainer.global_round, num_users)

        # train sequentially then all clients can participate
        async_trainer = create_async_trainer(
            model=global_model,
            local_lr=1.0,
            epochs=1,
            aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=1.0),
            event_generator_config=create_event_generator_config(
                training_rate=training_rate,
                training_duration_mean=0,
                training_duration_sd=0,
            ),
            max_staleness=0,
        )

        data_provider = get_fl_data_provider(
            num_examples=num_examples,
            num_fl_users=num_users,
            examples_per_user=1,
            batch_size=1,
            model=global_model,
        )

        final_model, _ = async_trainer.train(
            data_provider=data_provider,
            metric_reporter=metrics_reporter,
            num_total_users=num_users,
            distributed_world_size=1,
        )
        assertTrue(async_trainer.global_round - 1 == num_users)

    def test_number_of_steps(self):
        """This test checks that async training takes the same number of optimizer.step()
        as #epochs * #users * #batches_per_user
        """
        num_epochs = 5
        torch.manual_seed(1)
        batch_size = 10
        for num_fl_users in [1, 2]:
            for batches_per_user in [1, 2]:
                fl_model = ConstantGradientFLModel()
                num_total_examples = num_fl_users * batch_size * batches_per_user
                examples_per_user = batch_size * batches_per_user
                fl_data_provider = get_fl_data_provider(
                    num_examples=num_total_examples,
                    num_fl_users=num_fl_users,
                    examples_per_user=examples_per_user,
                    batch_size=batch_size,
                    model=fl_model,
                )
                run_fl_training(
                    fl_model=fl_model,
                    fl_data_provider=fl_data_provider,
                    epochs=num_epochs,
                    local_lr=1.0,
                    aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=1.0),
                    training_rate=1,
                    training_duration_mean=0,
                    training_duration_sd=0,
                )
                # Async FL training is sequential since mean_training_time is 0
                num_optimizer_steps = num_epochs * num_fl_users * batches_per_user
                # ConstantGradientFLModel has a property that its bias term = #of times optimizer.step() is called
                assertTrue(
                    np.isclose(
                        fl_model.fl_get_module().bias.detach().item(),
                        num_optimizer_steps,
                    ),
                    f"Expected: {num_optimizer_steps}, Found: {fl_model.fl_get_module().bias.detach().item()}",
                )

    def test_best_eval_model_is_kept(self):
        """This test checks that AsyncTrainer retains the model with the best eval performance
        To check for this, we use a special MetricsReporter that tracks which model produced
        the best eval results
        """
        num_epochs = 5
        torch.manual_seed(1)
        np.random.seed(1)
        # ConstantGradientFLModel has a property that its bias term = #of times optimizer.step() is called
        batch_size = 10
        num_fl_users = 10
        batches_per_user = 1
        fl_model = ConstantGradientFLModel()
        num_total_examples = num_fl_users * batch_size * batches_per_user
        examples_per_user = batch_size * batches_per_user
        for buffer_size in [1, 2, 3]:
            fl_data_provider = get_fl_data_provider(
                num_examples=num_total_examples,
                num_fl_users=num_fl_users,
                examples_per_user=examples_per_user,
                batch_size=batch_size,
                model=fl_model,
            )
            metric_reporter = RandomEvalMetricsReporter()
            # run_fl_training returns test results. However,
            # RandomEvalMetricsReporter() fakes it so test results are always
            # the best_eval_results
            best_model, best_eval_results = run_fl_training(
                fl_model=fl_model,
                fl_data_provider=fl_data_provider,
                epochs=num_epochs,
                local_lr=1.0,
                aggregator_config=FedAvgWithLRHybridAggregatorConfig(
                    lr=1.0, buffer_size=buffer_size
                ),
                metrics_reporter=metric_reporter,
                do_eval=True,
                report_train_metrics_after_aggregation=True,
                eval_epoch_frequency=0.001,  # report every global model update
            )
            assertEqual(best_eval_results, metric_reporter.best_eval_result)
            # TODO: also check that best_model matches metric_reporter.best_eval_model
            # after fixing code

    @pytest.mark.parametrize(
        "num_users,training_rate,num_epochs", [(100, 10, 2), (50, 10, 2)]
    )
    def test_constant_concurrency(self, num_users, training_rate, num_epochs):
        """
        Test for constant concurrency from one epoch to another
        We expect training_rate #users to be training simultaneously

        In this test, we expect these invariants to hold
        1. Concurrency rate ~ training rate from epoch 1 to epoch n during convergence
        2. Mean converging concurrency for all of the epochs ~ training rate
        3. Async trainer will stop after each epoch to run the eval loop therefore
            we will evaluate every num_users rounds
        """
        torch.manual_seed(2)

        global_model = DummyAlphabetFLModel()
        num_examples = num_users
        event_generator_config = AsyncTrainingEventGeneratorConfig(
            training_start_time_distribution=ConstantAsyncTrainingStartTimeDistrConfig(
                training_rate=training_rate
            ),
            duration_distribution_generator=PerUserUniformDurationDistributionConfig(
                training_duration_mean=1.0,
                training_duration_min=0.01,
            ),
        )
        metrics_reporter = ConcurrencyMetricsReporter([Channel.STDOUT])
        # training in parallel so not all clients can participate
        async_trainer = create_async_trainer(
            model=global_model,
            local_lr=1.0,
            epochs=num_epochs,
            aggregator_config=FedAvgWithLRAsyncAggregatorConfig(lr=1.0),
            event_generator_config=event_generator_config,
            do_eval=True,
            train_metrics_reported_per_epoch=num_users,
        )

        data_provider = get_fl_data_provider(
            num_examples=num_examples,
            num_fl_users=num_users,
            examples_per_user=1,
            batch_size=1,
            model=global_model,
        )

        final_model, _ = async_trainer.train(
            data_provider=data_provider,
            metric_reporter=metrics_reporter,
            num_total_users=num_users,
        )
        # invariant 1, after check that concurrency eventually reaches
        # steady state and is similar to training rate
        steady_state_seqnum = int(0.9 * len(metrics_reporter.concurrency_metrics))
        for c in metrics_reporter.concurrency_metrics[steady_state_seqnum:]:
            assertAlmostEqual(c, training_rate, delta=1.0)
        # invariant 2
        assertAlmostEqual(
            np.mean(metrics_reporter.concurrency_metrics[steady_state_seqnum:]),
            training_rate,
            delta=1.0,
        )
        # invariant 3
        assertEqual(len(metrics_reporter.eval_rounds), num_epochs)
        assertListEqual(
            metrics_reporter.eval_rounds,
            [(i * num_users) for i in range(1, num_epochs + 1)],
        )
