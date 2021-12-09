#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest
import torch
from flsim.common.pytest_helper import assertTrue, assertEqual, assertEmpty
from flsim.interfaces.model import IFLModel
from flsim.optimizers.async_aggregators import (
    AsyncAggregationType,
    AsyncAggregatorConfig,
    FedAvgWithLRAsyncAggregatorConfig,
    FedAvgWithLRWithMomentumAsyncAggregatorConfig,
    FedAvgWithLRHybridAggregatorConfig,
)
from flsim.tests.utils import MockQuadratic1DFL, Quadratic1D, SampleNet, TwoFC
from flsim.tests.utils import (
    verify_models_equivalent_after_training,
)
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate


@dataclass
class MockClientModel:
    delta: SampleNet
    after_train: SampleNet
    weight: float


class TestAsyncAggregator:
    def _test_one_step(
        self,
        param_after_local_training: float,
        param_after_global_training: float,
        weight: float,
        config: AsyncAggregatorConfig,
    ):
        """
        Test async aggregator by:
        1. Create AsyncAggregator of given type (fed_buff_aggregation or fed_async_aggregation)
        2. Set local param after training to param_after_local_training
        3. Set global param (simulate update from another user) to
           param_after_global_training
        4. Verify that async_optimizer.step(weight=weight) sets global param to correct value
        """
        # initial parameters: x=1.0, y=1.0 (x=param, y=constant)
        init_val = 1
        global_model = MockQuadratic1DFL(Quadratic1D())
        async_aggregator = instantiate(config, global_model=global_model)
        local_model = copy.deepcopy(global_model)
        delta = copy.deepcopy(global_model)
        # local training, x becomes param_after_local_training
        local_model.fl_get_module().x.data = torch.Tensor([param_after_local_training])
        # global update from another user, x becomes param_after_global_training
        global_model.fl_get_module().x.data = torch.Tensor(
            [param_after_global_training]
        )
        # client delta is init_val - param_after_local_training
        delta.fl_get_module().x.data = torch.Tensor(
            [init_val - param_after_local_training]
        )

        async_aggregator.on_client_training_end(
            client_delta=delta,
            final_local_model=local_model,
            weight=weight,
        )
        if config.aggregation_type == AsyncAggregationType.fed_buff_aggregation:
            # model delta = final_local_model - local_model_deltaing = param_after_local_training  - 1
            # global_model = param_after_global_training + weight*(param_after_local_training - 1)
            global_model_expected = param_after_global_training + weight * (
                param_after_local_training - init_val
            )
        else:
            # global_model = (1- weight)*param_after_global_training + weight*param_after_local_training
            global_model_expected = (
                1 - weight
            ) * param_after_global_training + weight * param_after_local_training

        assertTrue(
            torch.allclose(
                global_model.fl_get_module().x.data,
                torch.Tensor([global_model_expected]),
                atol=1e-7,
            )
        )

    def test_fed_buff_aggregation(self):
        """
        Test that AsyncAggregator with 'fed_buff_aggregation' works as expected
        1. Create global_model, some_param=1
        2. Copy global_model into local_model
        3. Simulate local training by local_model changing some_param to a (some_param=a, delta=a-1)
        4. Simulate global model updata by other user, some_param being set to b (some_param=b)
        5. Call AsyncOptimizer.step(). global_model.some_param should be b + (a-1) (some_param == b + a -1)
        """

        num_random_values = 5
        num_random_weights = 5
        for config in [
            FedAvgWithLRAsyncAggregatorConfig(
                aggregation_type=AsyncAggregationType.fed_buff_aggregation, lr=1.0
            ),
            FedAvgWithLRHybridAggregatorConfig(
                aggregation_type=AsyncAggregationType.fed_buff_aggregation,
                lr=1.0,
                buffer_size=1,
            ),
        ]:
            for _ in range(num_random_values):
                for _ in range(num_random_weights):
                    self._test_one_step(
                        param_after_local_training=np.random.random_sample(),
                        param_after_global_training=np.random.random_sample(),
                        weight=np.random.random_sample(),
                        config=config,
                    )

    def test_fed_async_aggregation(self):
        """
        Test that AsyncAggregator with 'fed_async_aggregation' works as expected
        1. Create global_model, some_param=1
        2. Copy global_model into local_model
        3. Simulate local training by local_model changing some_param to a (some_param=a)
        4. Simulate global model updata by other user, some_param being set to b (some_param=b)
        5. Call AsyncOptimizer.step(weight=1). global_model.some_param should be
            w*a + (1-w)*b (some_param == w*a + (1-w)*b)
        """
        num_random_values = 5
        num_random_weights = 5
        for _ in range(num_random_values):
            for _ in range(num_random_weights):
                self._test_one_step(
                    param_after_local_training=np.random.random_sample(),
                    param_after_global_training=np.random.random_sample(),
                    weight=np.random.random_sample(),
                    config=FedAvgWithLRAsyncAggregatorConfig(
                        aggregation_type=AsyncAggregationType.fed_async_aggregation,
                        lr=1.0,
                    ),
                )

    def _create_n_clients(self, num_clients):
        return [
            MockClientModel(
                delta=SampleNet(TwoFC()),
                after_train=SampleNet(TwoFC()),
                weight=np.random.random_sample(),
            )
            for _ in range(num_clients)
        ]

    def _symmetry_test(self, num_users, hybrid_config):

        hybrid_global_model_1 = SampleNet(TwoFC())
        hybrid_global_model_2 = copy.deepcopy(hybrid_global_model_1)

        hybrid_aggregator_1 = instantiate(
            hybrid_config, global_model=hybrid_global_model_1
        )

        hybrid_aggregator_2 = instantiate(
            hybrid_config, global_model=hybrid_global_model_2
        )

        client_models = self._create_n_clients(num_users)

        for client_model in client_models:
            hybrid_aggregator_1.zero_grad()
            hybrid_aggregator_1.on_client_training_end(
                client_model.delta,
                client_model.after_train,
                weight=client_model.weight,
            )

        random.shuffle(client_models)
        for client_model in client_models:
            hybrid_aggregator_2.zero_grad()
            hybrid_aggregator_2.on_client_training_end(
                client_model.delta,
                client_model.after_train,
                weight=client_model.weight,
            )

        return FLModelParamUtils.get_mismatched_param(
            models=[
                hybrid_global_model_1.fl_get_module(),
                hybrid_global_model_2.fl_get_module(),
            ],
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )

    def _equivalence_test(self, num_users, hybrid_config, async_config):
        async_global_model = SampleNet(TwoFC())
        hybrid_global_model = copy.deepcopy(async_global_model)

        async_aggregator = instantiate(async_config, global_model=async_global_model)

        hybrid_aggregator = instantiate(hybrid_config, global_model=hybrid_global_model)

        client_models = self._create_n_clients(num_users)

        for client_model in client_models:
            async_aggregator.zero_grad()
            async_aggregator.on_client_training_end(
                client_model.delta,
                client_model.after_train,
                weight=client_model.weight,
            )

        for client_model in client_models:
            hybrid_aggregator.zero_grad()
            hybrid_aggregator.on_client_training_end(
                client_model.delta,
                client_model.after_train,
                weight=client_model.weight,
            )

        return FLModelParamUtils.get_mismatched_param(
            models=[
                async_global_model.fl_get_module(),
                hybrid_global_model.fl_get_module(),
            ],
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )

    def test_hybrid_async_symmetry(self):
        """
        Test for symmetry:
        To satisfy symmetry, a hybrid async aggregation algorithm should be invariant to the order of user updates
        f(userA, userB) = f(userB, userA) where f() is aggregation mechanism

        1. Create async and hybrid aggregators with same global model
        2. Create a list of N clients
        3. Run hybrid_aggregator
        4. Shuffle client list
        5. Run async_aggregator
        6. Both should reach the same final global model
        """
        num_users = 10
        global_lr = 1.0
        hybrid_config = FedAvgWithLRHybridAggregatorConfig(lr=global_lr, buffer_size=1)

        error_msg = self._symmetry_test(
            num_users=num_users, hybrid_config=hybrid_config
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_hybrid_async_equivalence(self):
        """
        To satisfy equivalence,
        1. Assume both mechanisms have the same starting point
        2. Denote N = number of users
        3. Assume buffer_size is a factor of N
        4. Pure async and hybrid-async would reach the same final global model

        For simplicity, we assume buffer_size = N
        """
        num_users = 10
        global_lr = 1.00

        async_config = FedAvgWithLRAsyncAggregatorConfig(lr=global_lr)
        hybrid_config = FedAvgWithLRHybridAggregatorConfig(lr=global_lr, buffer_size=10)

        error_msg = self._equivalence_test(
            num_users=num_users, hybrid_config=hybrid_config, async_config=async_config
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_global_update(self):
        """
        Test the aggregator only updates global model if
        threshold is reached
        """
        num_epochs = 5
        for _ in range(num_epochs):
            num_total_users = np.random.randint(1, 20)
            buffer_size = np.random.randint(1, num_total_users + 1)

            hybrid_config = FedAvgWithLRHybridAggregatorConfig(
                lr=1.0, buffer_size=buffer_size
            )
            global_model = SampleNet(TwoFC())
            hybrid_aggregator = instantiate(hybrid_config, global_model=global_model)
            client_models = self._create_n_clients(num_total_users)
            for client_num, client in enumerate(client_models):
                is_global_model_updated = hybrid_aggregator.on_client_training_end(
                    client.delta, client.after_train, weight=1
                )
                # client_num is 0th index hence we need the + 1
                should_update_global_model = (client_num + 1) % buffer_size == 0
                assertEqual(is_global_model_updated, should_update_global_model)

    def train_async_with_zero_weight(
        self,
        initial_model: IFLModel,
        client_models: List[MockClientModel],
        num_epochs: int,
        num_total_users: int,
        momentum: float,
        train_with_zero_weight_in_middle: bool,
    ) -> IFLModel:
        """'Train' initial model by applying randomly generated client model updates to
        it, by repeatedly calling aggregator.on_client_training_end
        We do it thrice:
        a) Train for num_epochs/2
        b) If train_with_zero_weight_in_middle, train for num_epochs with zero weight
        c) Train for num_epochs/2
        Return final model
        """
        assert num_epochs % 2 == 0, "Training must be over even number of epochs"
        # config = AsyncAggregatorFedSGDConfig(lr=1.0, momentum=momentum)
        config = FedAvgWithLRWithMomentumAsyncAggregatorConfig(
            lr=1.0, momentum=momentum
        )
        aggregator = instantiate(config, global_model=initial_model)
        half_epochs = int(num_epochs / 2)

        def print_debug(prefix: str):
            for key, value in aggregator.optimizer.state.items():
                print(f"{prefix}: {key}:{value}")
                break

        for _ in range(half_epochs):
            for client in client_models:
                aggregator.on_client_training_end(
                    client.delta, client.after_train, weight=1
                )
        print_debug("After first loop")
        if train_with_zero_weight_in_middle:
            # training with zero weight should change neither the model, nor
            # the velocity computation inside the optimizer
            for _ in range(half_epochs):
                for client in client_models:
                    aggregator.on_client_training_end(
                        client.delta, client.after_train, weight=0
                    )
        print_debug("After second loop")
        for _ in range(half_epochs):
            for client in client_models:
                aggregator.on_client_training_end(
                    client.delta, client.after_train, weight=1
                )
        print_debug("After third loop")
        return aggregator.global_model

    @pytest.mark.parametrize(
        "num_total_users,num_epochs, momentum",
        [(1, 2, 0.5), (10, 10, 0.5), (10, 10, 0)],
    )
    def test_momentum_implementation_zero_weight(
        self, num_total_users, num_epochs, momentum
    ):
        """In FedAsyncAggregatorWithMomentum.on_client_training_end, when weight=0,
        neither velocity nor model should be updated
        We test this by comparing two training runs:
        RUN 1
            a) Running a few FL rounds in FedSyncAggregatorWithMomentum
            b) Calling on_client_training_end with weight=0
            c) Running some more FL rounds with FedSyncAggregatorWithMomentum
        RUN 2
            Same as RUN 1, except no (b)
        RUN 1 and RUN 2 should produce the same model
        """

        # function starts here
        initial_model = SampleNet(TwoFC())
        client_models = self._create_n_clients(num_total_users)
        torch.manual_seed(1)
        np.random.seed(1)
        global_model_trained1 = self.train_async_with_zero_weight(
            initial_model=copy.deepcopy(initial_model),
            client_models=client_models,
            num_epochs=num_epochs,
            num_total_users=num_total_users,
            momentum=momentum,
            train_with_zero_weight_in_middle=False,
        )
        torch.manual_seed(1)
        np.random.seed(1)
        global_model_trained2 = self.train_async_with_zero_weight(
            initial_model=copy.deepcopy(initial_model),
            client_models=client_models,
            num_epochs=num_epochs,
            num_total_users=num_total_users,
            momentum=momentum,
            train_with_zero_weight_in_middle=True,
        )

        error_msg = verify_models_equivalent_after_training(
            global_model_trained1,
            global_model_trained2,
            initial_model,
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )
        assertEqual(error_msg, "")

    @pytest.mark.parametrize(
        "num_total_users,num_epochs, momentum, lr",
        [(1, 2, 0.5, 10), (10, 10, 0.5, 10), (10, 10, 0, 10)],
    )
    def test_momentum_implementation_one_weight(
        self, num_total_users, num_epochs, momentum, lr
    ):
        """FedAsyncAggregatorWithMomentum.on_client_training_end should behave
        exactly like SGD with momentum when weight = 1
        We test this by
        a) Running SGD with momentum
        b) Running AsyncFL in sequential model with momentum
        Showing that (a) and (b) produce the same results
        """
        momentum = 0.5
        num_epochs = 10
        num_total_users = 10
        lr = 1.0

        initial_model = SampleNet(TwoFC())
        client_models = self._create_n_clients(num_total_users)

        # run async training
        torch.manual_seed(1)
        np.random.seed(1)
        config = FedAvgWithLRWithMomentumAsyncAggregatorConfig(lr=lr, momentum=momentum)
        aggregator = instantiate(config, global_model=copy.deepcopy(initial_model))
        for _ in range(num_epochs):
            for client in client_models:
                aggregator.on_client_training_end(
                    client.delta, client.after_train, weight=1
                )

        # run SGD training
        torch.manual_seed(1)
        np.random.seed(1)
        sgd_model = copy.deepcopy(initial_model)
        sgd_optimizer = torch.optim.SGD(
            sgd_model.fl_get_module().parameters(), lr=lr, momentum=momentum
        )

        for _ in range(num_epochs):
            for client in client_models:
                FLModelParamUtils.set_gradient(
                    model=sgd_model.fl_get_module(),
                    reference_gradient=client.delta.fl_get_module(),
                )
                sgd_optimizer.step()

        error_msg = verify_models_equivalent_after_training(
            aggregator.global_model,
            sgd_model,
            initial_model,
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )
        assertEqual(error_msg, "")
