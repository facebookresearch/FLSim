#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Optional, Tuple

import numpy as np
import torch
from flsim.clients.base_client import ClientConfig
from flsim.common.timeout_simulator import (
    NeverTimeOutSimulatorConfig,
    TimeOutSimulatorConfig,
)
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.optimizers.async_aggregators import AsyncAggregatorConfig
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.optimizers.optimizer_scheduler import (
    OptimizerSchedulerConfig,
    ConstantLRSchedulerConfig,
)
from flsim.tests.utils import (
    FakeMetricReporter,
    verify_models_equivalent_after_training,
)
from flsim.trainers.async_trainer import AsyncTrainer, AsyncTrainerConfig
from flsim.utils.async_trainer.async_example_weights import (
    EqualExampleWeightConfig,
    AsyncExampleWeightConfig,
)
from flsim.utils.async_trainer.async_staleness_weights import (
    ConstantStalenessWeightConfig,
    AsyncStalenessWeightConfig,
)
from flsim.utils.async_trainer.async_user_selector import AsyncUserSelectorType
from flsim.utils.async_trainer.async_weights import AsyncWeightConfig
from flsim.utils.async_trainer.training_event_generator import (
    AsyncTrainingEventGeneratorConfig,
    ConstantAsyncTrainingStartTimeDistrConfig,
    EventGeneratorConfig,
)
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.sample_model import DummyAlphabetFLModel
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from flsim.utils.tests.helpers.test_utils import FLTestUtils
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
)
from omegaconf import OmegaConf

CONSTANT_LR_SCHEDULER_CONFIG = ConstantLRSchedulerConfig()
EQUAL_EXAMPLE_WEIGHT_CONFIG = EqualExampleWeightConfig()
CONSTANT_STALENESS_WEIGHT_CONFIG = ConstantStalenessWeightConfig()


def create_event_generator_config(
    training_rate: int, training_duration_mean: float, training_duration_sd: float
) -> EventGeneratorConfig:
    return AsyncTrainingEventGeneratorConfig(
        training_start_time_distribution=ConstantAsyncTrainingStartTimeDistrConfig(
            training_rate=training_rate
        ),
        duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
            training_duration_mean=training_duration_mean,
            training_duration_sd=training_duration_sd,
        ),
    )


def create_async_trainer(
    model: IFLModel,
    local_lr: float,
    epochs: int,
    event_generator_config: EventGeneratorConfig,
    aggregator_config: AsyncAggregatorConfig,
    example_weight_config: AsyncExampleWeightConfig = EQUAL_EXAMPLE_WEIGHT_CONFIG,
    local_lr_scheduler_config: OptimizerSchedulerConfig = CONSTANT_LR_SCHEDULER_CONFIG,
    staleness_weight_config: AsyncStalenessWeightConfig = CONSTANT_STALENESS_WEIGHT_CONFIG,
    timeout_simulator_config: Optional[TimeOutSimulatorConfig] = None,
    max_staleness: float = 1e10,
    report_train_metrics_after_aggregation: bool = False,
    eval_epoch_frequency: float = 1.0,
    always_keep_trained_model: bool = False,
    do_eval: bool = False,
    train_metrics_reported_per_epoch: int = 1,
):
    async_trainer = AsyncTrainer(
        model=model,
        cuda_enabled=False,
        **OmegaConf.structured(
            AsyncTrainerConfig(
                aggregator=aggregator_config,
                client=ClientConfig(
                    epochs=1,
                    optimizer=LocalOptimizerSGDConfig(
                        lr=local_lr,
                    ),
                    lr_scheduler=local_lr_scheduler_config,
                ),
                epochs=epochs,
                training_event_generator=event_generator_config,
                async_user_selector_type=AsyncUserSelectorType.ROUND_ROBIN,
                async_weight=AsyncWeightConfig(
                    example_weight=example_weight_config,
                    staleness_weight=staleness_weight_config,
                ),
                timeout_simulator=timeout_simulator_config
                or NeverTimeOutSimulatorConfig(),
                max_staleness=max_staleness,
                do_eval=do_eval,
                report_train_metrics_after_aggregation=report_train_metrics_after_aggregation,
                eval_epoch_frequency=eval_epoch_frequency,
                always_keep_trained_model=always_keep_trained_model,
                train_metrics_reported_per_epoch=train_metrics_reported_per_epoch,
            )
        ),
    )
    return async_trainer


def get_nonfl_optimizer(
    nonfl_model: IFLModel,
    fl_local_lr: float,
    fl_aggregator_config: AsyncAggregatorConfig,
) -> torch.optim.Optimizer:
    """Given FL trainer settings (local_lr, aggregator config),
    return a non-fl Optimizer that will produce equivalent behavior
    """
    agg_type = fl_aggregator_config._target_
    # if FL global optimizer was Adam, then FL local lr should be 1.0,
    # otherwise we cannot produce same results between FL and non-FL
    assert ("FedAdam" not in agg_type) or (
        fl_local_lr == 1.0
    ), f"When using FedAdam, fl_local lr should be 1. Instead, its {fl_local_lr}"
    # non_fl lr = fl_global_optimizer_lr*fl_local_optimizer_lr
    # Example: fl_local_optimizer lr=1.0, then non_fl LR = fl_global_optimizer_lr
    # Example: fl_local_optimizer lr=0.1, then non_fl LR = 0.1*fl_global_optimizer_lr
    # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `lr`.
    nonfl_lr = fl_aggregator_config.lr * fl_local_lr

    if "FedAvgWithLR" in agg_type:
        optimizer = torch.optim.SGD(
            nonfl_model.fl_get_module().parameters(),
            lr=nonfl_lr,
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `momentum`.
            momentum=fl_aggregator_config.momentum,
        )
    elif "FedAdam" in agg_type:
        optimizer = torch.optim.Adam(
            nonfl_model.fl_get_module().parameters(),
            lr=nonfl_lr,
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `weight_decay`.
            weight_decay=fl_aggregator_config.weight_decay,
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `beta1`.
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `beta2`.
            betas=(fl_aggregator_config.beta1, fl_aggregator_config.beta2),
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `eps`.
            eps=fl_aggregator_config.eps,
        )
    else:
        raise AssertionError(f"Unknown aggregator {agg_type}")
    return optimizer


def assert_fl_nonfl_same(
    global_model: IFLModel,
    fl_data_provider: IFLDataProvider,
    nonfl_data_loader: torch.utils.data.DataLoader,
    epochs: int,
    local_lr: float,
    aggregator_config: AsyncAggregatorConfig,
    training_rate=1,
    training_duration_mean=0,
    training_duration_sd=0,
) -> str:
    """
    Given:
        data_for_fl={user1:batch1, user2:batch2}
        data_for_non_fl={batch1, batch2}
    Check that the following produce the same trained model:
    1. FL training, 1 user per round. global_opt=SGD, local_lr=x, global_lr=x
    2. Non-FL training, opt=SGD, lr=x
    Return value: model parameters that don't match between fl and non-fl training
    """
    # will be used later to verify training indeed took place
    reference_untrained_model = copy.deepcopy(global_model)
    nonfl_model = copy.deepcopy(reference_untrained_model)
    nonfl_optimizer = get_nonfl_optimizer(
        nonfl_model=nonfl_model,
        fl_local_lr=local_lr,
        fl_aggregator_config=aggregator_config,
    )
    trained_fl_model, trained_nonfl_model = run_fl_nonfl_training(
        fl_model=global_model,
        nonfl_model=nonfl_model,
        nonfl_optimizer=nonfl_optimizer,
        fl_data_provider=fl_data_provider,
        nonfl_data_loader=nonfl_data_loader,
        epochs=epochs,
        fl_local_lr=local_lr,
        fl_aggregator_config=aggregator_config,
        training_rate=training_rate,
        training_duration_mean=training_duration_mean,
        training_duration_sd=training_duration_sd,
    )
    return verify_models_equivalent_after_training(
        trained_fl_model,
        trained_nonfl_model,
        reference_untrained_model,
        rel_epsilon=1e-4,
        abs_epsilon=1e-4,
    )


def run_fl_nonfl_training(
    fl_model: IFLModel,
    nonfl_model: IFLModel,
    nonfl_optimizer: torch.optim.Optimizer,
    fl_data_provider: IFLDataProvider,
    nonfl_data_loader: torch.utils.data.DataLoader,
    epochs: int,
    fl_local_lr: float,
    fl_aggregator_config: AsyncAggregatorConfig,
    training_rate=1,
    training_duration_mean=0,
    training_duration_sd=0,
    example_weight_config: AsyncExampleWeightConfig = EQUAL_EXAMPLE_WEIGHT_CONFIG,
    staleness_weight_config: AsyncStalenessWeightConfig = CONSTANT_STALENESS_WEIGHT_CONFIG,
) -> Tuple[IFLModel, IFLModel]:
    """Run the following training
    1. FL training: train fl_model with fl_data_provider, fl_aggregator_config and fl_local_lr,
    2. Non-FL training: train nonfl_model with nonfl_data_loader and nonfl_optmizer
    """
    fl_trained_model, _ = run_fl_training(
        fl_model=fl_model,
        fl_data_provider=fl_data_provider,
        epochs=epochs,
        local_lr=fl_local_lr,
        aggregator_config=fl_aggregator_config,
        training_rate=training_rate,
        training_duration_mean=training_duration_mean,
        training_duration_sd=training_duration_sd,
        example_weight_config=example_weight_config,
        staleness_weight_config=staleness_weight_config,
        always_keep_trained_model=True,
    )
    nonfl_trained_model = FLTestUtils.run_nonfl_training(
        model=nonfl_model,
        optimizer=nonfl_optimizer,
        data_loader=nonfl_data_loader,
        epochs=epochs,
    )
    return fl_trained_model, nonfl_trained_model


def run_fl_training(
    fl_model: IFLModel,
    fl_data_provider: IFLDataProvider,
    epochs: int,
    local_lr: float,
    aggregator_config: AsyncAggregatorConfig,
    training_rate=1,
    training_duration_mean=0,
    training_duration_sd=0,
    example_weight_config: AsyncExampleWeightConfig = EQUAL_EXAMPLE_WEIGHT_CONFIG,
    staleness_weight_config: AsyncStalenessWeightConfig = CONSTANT_STALENESS_WEIGHT_CONFIG,
    metrics_reporter: Optional[IFLMetricsReporter] = None,
    do_eval: bool = False,
    report_train_metrics_after_aggregation: bool = False,
    eval_epoch_frequency: float = 1.0,
    always_keep_trained_model: float = False,
) -> Tuple[IFLModel, Any]:
    torch.manual_seed(1)
    async_trainer = create_async_trainer(
        model=fl_model,
        local_lr=local_lr,
        epochs=epochs,
        aggregator_config=aggregator_config,
        event_generator_config=create_event_generator_config(
            training_rate=training_rate,
            training_duration_mean=training_duration_mean,
            training_duration_sd=training_duration_sd,
        ),
        example_weight_config=example_weight_config,
        staleness_weight_config=staleness_weight_config,
        do_eval=do_eval,
        report_train_metrics_after_aggregation=report_train_metrics_after_aggregation,
        eval_epoch_frequency=eval_epoch_frequency,
    )
    if metrics_reporter is None:
        metrics_reporter = FakeMetricReporter()
    fl_trained_model, test_metrics = async_trainer.train(
        data_provider=fl_data_provider,
        metric_reporter=metrics_reporter,
        num_total_users=fl_data_provider.num_users(),
        distributed_world_size=1,
    )
    return fl_trained_model, test_metrics


def get_data(
    num_examples: int,
    num_fl_users: int,
    examples_per_user: int,
    fl_batch_size: int,
    nonfl_batch_size: int,
    model: IFLModel,
) -> Tuple[IFLDataProvider, torch.utils.data.DataLoader]:
    fl_data_provider = get_fl_data_provider(
        num_examples=num_examples,
        num_fl_users=num_fl_users,
        examples_per_user=examples_per_user,
        batch_size=fl_batch_size,
        model=model,
    )
    # non-FL data
    dummy_dataset = DummyAlphabetDataset(num_examples)
    nonfl_data_loader = torch.utils.data.DataLoader(
        dummy_dataset, batch_size=nonfl_batch_size, shuffle=False
    )
    return fl_data_provider, nonfl_data_loader


def get_fl_data_provider(
    num_examples: int,
    num_fl_users: int,
    examples_per_user: int,
    batch_size: int,
    model: IFLModel,
) -> IFLDataProvider:
    dummy_dataset = DummyAlphabetDataset(num_examples)
    data_provider, data_loader = DummyAlphabetDataset.create_data_provider_and_loader(
        dummy_dataset, examples_per_user, batch_size, model
    )
    return data_provider
    assert data_loader.num_total_users == num_fl_users, "Error in data sharding"


def get_nonfl_batch_size(fl_batch_size: int, min_examples_per_user: int):
    # how to chose batch_size for non-fl training?
    # if fl_batch_size is bigger than min_examples_per_user, use fl_batch_size
    # however, if fl_batch_size is *smaller* than min_examples_per_user, must
    # use min_examples_per_user as the batch size.
    # example: fl_batch_size=128. 2 users, each with 4 data points
    # non-fl training must run 2 batches of training to match fl-training.
    # so batch size must be 4
    # example: fl_batch_size=8. min_examples_per_user = 1
    # nonfl_batch_size = 8
    return min(fl_batch_size, min_examples_per_user)


def _equal_data_split_params(
    num_examples: int,
    num_fl_users: int,
    fl_batch_size: int,
    one_batch_per_user_only: bool,
) -> Tuple[int, int]:
    """Assume FL data is equally split among users
    Find examples_per_user and nonfl_batch_size
    """
    assert (
        num_examples % num_fl_users == 0
    ), f"Expect num_examples({num_examples}) to be multiple of num_fl_users({num_fl_users})"
    examples_per_user = num_examples // num_fl_users
    if one_batch_per_user_only:
        assert (
            fl_batch_size >= examples_per_user
        ), f"Expected each user to have max 1 batch. Batch_size:{fl_batch_size}, num_examples_per_user:{examples_per_user}"
    nonfl_batch_size = get_nonfl_batch_size(
        fl_batch_size=fl_batch_size, min_examples_per_user=examples_per_user
    )
    assert nonfl_batch_size == min(
        fl_batch_size, examples_per_user
    ), f"fl_batch_size:{fl_batch_size}, nonfl_batch_size:{nonfl_batch_size} "
    f"examples_per_user:{examples_per_user}. To ensure FL and non-FL take the same number of local SGD steps,"
    "nonfl_batch_size should be the same as the lower of (examples_per_user, fl_batch_size)"

    return examples_per_user, nonfl_batch_size


def _unequal_data_split_params(
    num_examples: int,
    num_fl_users: int,
    max_examples_per_user: int,
    fl_batch_size: int,
    one_batch_per_user_only: bool,
) -> Tuple[int, int]:
    r"""
    FL data may be unequally split among users with sequential sharding
    E.g: max_num_examples_per_user = 8. total examples = 12. num_users=2
    user1: 8 examples. user2: 4 examples
    """
    # if equal split. i.e, if every user can get max_examples_per_user
    if num_examples / num_fl_users == max_examples_per_user:
        return _equal_data_split_params(
            num_examples=num_examples,
            num_fl_users=num_fl_users,
            fl_batch_size=fl_batch_size,
            one_batch_per_user_only=one_batch_per_user_only,
        )
    # we must have enough examples such that at least one user gets max_examples_per_user
    assert max_examples_per_user > num_fl_users
    # last user gets leftover examples
    examples_with_last_user = num_examples % max_examples_per_user
    if one_batch_per_user_only:
        assert fl_batch_size >= max_examples_per_user, (
            f"Expected each user to have max 1 batch. Batch_size:{fl_batch_size},"
            f" num_examples_per_user:{max_examples_per_user}"
        )
    nonfl_batch_size = get_nonfl_batch_size(
        fl_batch_size=fl_batch_size, min_examples_per_user=examples_with_last_user
    )
    assert nonfl_batch_size == min(fl_batch_size, max_examples_per_user), (
        f"fl_batch_size:{fl_batch_size}, nonfl_batch_size:{nonfl_batch_size} "
        f"max_examples_per_user:{max_examples_per_user}. "
        f"To ensure FL and non-FL take the same number of local SGD steps,"
        f"nonfl_batch_size should be the same as the lower of (max_examples_per_user, fl_batch_size)",
    )
    return max_examples_per_user, nonfl_batch_size


def get_unequal_split_data(
    num_examples: int,
    num_fl_users: int,
    max_examples_per_user: int,
    fl_batch_size: int,
    model: IFLModel,
    one_batch_per_user_only: bool = False,
):
    r"""
    If FL data is unequally split among users with sequential sharding
    E.g: max_num_examples_per_user = 8. total examples = 12. num_users=2
    user1: 8 examples. user2: 4 examples
    Return FLDataProvider and non-FL DataLoader
    """
    examples_per_user, nonfl_batch_size = _unequal_data_split_params(
        num_examples=num_examples,
        num_fl_users=num_fl_users,
        max_examples_per_user=max_examples_per_user,
        fl_batch_size=fl_batch_size,
        one_batch_per_user_only=one_batch_per_user_only,
    )
    return get_data(
        num_examples=num_examples,
        num_fl_users=num_fl_users,
        examples_per_user=examples_per_user,
        fl_batch_size=fl_batch_size,
        nonfl_batch_size=nonfl_batch_size,
        model=model,
    )


def get_equal_split_data(
    num_examples: int,
    num_fl_users: int,
    fl_batch_size: int,
    model: IFLModel,
    one_batch_per_user_only: bool = False,
) -> Tuple[IFLDataProvider, torch.utils.data.DataLoader]:
    """Assume FL data is equally split among users
    Return FLDataProvider and non-fl DataLoader
    """
    examples_per_user, nonfl_batch_size = _equal_data_split_params(
        num_examples=num_examples,
        num_fl_users=num_fl_users,
        fl_batch_size=fl_batch_size,
        one_batch_per_user_only=one_batch_per_user_only,
    )
    return get_data(
        num_examples=num_examples,
        num_fl_users=num_fl_users,
        examples_per_user=examples_per_user,
        fl_batch_size=fl_batch_size,
        nonfl_batch_size=nonfl_batch_size,
        model=model,
    )


def assert_fl_nonfl_same_equal_data_split(
    fl_batch_size: int,
    num_examples: int,
    num_fl_users: int,
    epochs: int,
    local_lr: float,
    aggregator_config: AsyncAggregatorConfig,
    training_rate=1,
    training_duration_mean=0,
    training_duration_sd=0,
) -> str:
    # TODO: can this test share common code (eg, nonFL training) with
    # test_trainer._test_fl_nonfl_same_equal_data_split?
    torch.manual_seed(1)
    # create dummy FL model on alphabet
    global_model = DummyAlphabetFLModel()
    # if aggregator is FedAdam, each user should have only one batch
    # otherwise non-FL and FL training won't be the same
    one_batch_per_user_only = "FedAdam" in aggregator_config._target_
    fl_data_provider, nonfl_data_loader = get_equal_split_data(
        num_examples=num_examples,
        num_fl_users=num_fl_users,
        fl_batch_size=fl_batch_size,
        model=global_model,
        one_batch_per_user_only=one_batch_per_user_only,
    )
    return assert_fl_nonfl_same(
        global_model=global_model,
        fl_data_provider=fl_data_provider,
        nonfl_data_loader=nonfl_data_loader,
        epochs=epochs,
        local_lr=local_lr,
        aggregator_config=aggregator_config,
        training_rate=training_rate,
        training_duration_mean=training_duration_mean,
        training_duration_sd=training_duration_sd,
    )


def async_train_one_user(
    global_model_at_training_start: IFLModel,
    global_model_at_training_end: IFLModel,
    batches,
    local_lr: float,
) -> IFLModel:
    local_model = copy.deepcopy(global_model_at_training_start)
    local_optimizer = torch.optim.SGD(
        local_model.fl_get_module().parameters(), lr=local_lr
    )
    for batch in batches:
        FLTestUtils.run_nonfl_training_one_batch(
            model=local_model, optimizer=local_optimizer, training_batch=batch
        )
    simulate_async_global_model_update(
        global_model=global_model_at_training_end,
        local_model_before_training=global_model_at_training_start,
        local_model_after_training=local_model,
    )
    return global_model_at_training_end


def simulate_async_global_model_update(
    global_model: IFLModel,
    local_model_before_training: IFLModel,
    local_model_after_training: IFLModel,
):
    # TODO: use AsyncAggregator._update_global_model, after John's refactoring
    reconstructed_grad = copy.deepcopy(global_model)
    FLModelParamUtils.reconstruct_gradient(
        old_model=local_model_before_training.fl_get_module(),
        new_model=local_model_after_training.fl_get_module(),
        grads=reconstructed_grad.fl_get_module(),
    )
    FLModelParamUtils.set_gradient(
        model=global_model.fl_get_module(),
        reference_gradient=reconstructed_grad.fl_get_module(),
    )
    optimizer = torch.optim.SGD(global_model.fl_get_module().parameters(), lr=1.0)
    optimizer.step()


def run_fl_training_with_event_generator(
    fl_model: IFLModel,
    fl_data_provider: IFLDataProvider,
    epochs: int,
    local_lr: float,
    aggregator_config: AsyncAggregatorConfig,
    training_event_generator_config: EventGeneratorConfig,
    example_weight_config: AsyncExampleWeightConfig = EQUAL_EXAMPLE_WEIGHT_CONFIG,
    staleness_weight_config: AsyncStalenessWeightConfig = CONSTANT_STALENESS_WEIGHT_CONFIG,
) -> IFLModel:
    torch.manual_seed(1)
    async_trainer = create_async_trainer(
        model=fl_model,
        local_lr=local_lr,
        epochs=epochs,
        aggregator_config=aggregator_config,
        event_generator_config=training_event_generator_config,
        example_weight_config=example_weight_config,
        staleness_weight_config=staleness_weight_config,
    )
    fl_trained_model, _ = async_trainer.train(
        data_provider=fl_data_provider,
        metric_reporter=FakeMetricReporter(),
        num_total_users=fl_data_provider.num_users(),
        distributed_world_size=1,
    )
    return fl_trained_model


def get_safe_global_lr(fl_batch_size: int, max_examples_per_user: int) -> float:
    """Return a global_lr to use in FL, that can produce equivalent training
    results as non-fl training.
    Return value: either 1.0, or a random float between 0 and 10
    global_lr can be anything if each user has exactly 1 batch.
    otherwise, it must be 1.0
    why? because FL will take exactly one global step for each user
    while non-FL will take num_examples/batch_size global steps
    """
    if fl_batch_size >= max_examples_per_user:
        return np.random.random_sample() * 10
    else:
        return 1.0
