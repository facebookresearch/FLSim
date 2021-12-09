#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from flsim.active_user_selectors.simple_user_selector import (
    SequentialActiveUserSelectorConfig,
)
from flsim.clients.base_client import ClientConfig
from flsim.clients.dp_client import DPClientConfig
from flsim.common.pytest_helper import assertEqual, assertEmpty, assertNotEmpty
from flsim.optimizers.async_aggregators import FedAvgWithLRHybridAggregatorConfig
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.privacy.common import PrivacySetting
from flsim.reducers.base_round_reducer import ReductionType
from flsim.reducers.weighted_dp_round_reducer import WeightedDPRoundReducerConfig
from flsim.servers.aggregator import AggregationType
from flsim.servers.sync_dp_servers import SyncDPSGDServerConfig
from flsim.servers.sync_servers import SyncServerConfig
from flsim.tests.utils import (
    MetricsReporterWithMockedChannels,
    FakeMetricReporter,
    verify_models_equivalent_after_training,
)
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from flsim.utils.async_trainer.async_staleness_weights import (
    PolynomialStalenessWeightConfig,
)
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.sample_model import DummyAlphabetFLModel
from flsim.utils.tests.helpers.async_trainer_test_utils import (
    get_fl_data_provider,
    run_fl_training,
)
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from omegaconf import OmegaConf
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer


class TestDifferentialPrivacyIntegration:
    def _get_weighted_dp_reducer_config(
        self,
        noise,
        clipping,
        reduction_type=ReductionType.WEIGHTED_SUM,
        min_weight=1e-6,
        max_weight=1.0,
    ):
        return WeightedDPRoundReducerConfig(
            reduction_type=reduction_type,
            min_weight=min_weight,
            max_weight=max_weight,
            privacy_setting=PrivacySetting(
                noise_multiplier=noise, clipping_value=clipping
            ),
        )

    def _create_optimizer(
        self,
        model: nn.Module,
        data_provider,
        lr: float,
        momentum: float,
        sample_level_dp: bool,
        dp_config: Dict[str, Any],
        train_batch_size: int,
    ):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        optimizer.zero_grad()
        if sample_level_dp:
            model = GradSampleModule(model)

            optimizer = DPOptimizer(
                optimizer=optimizer,
                noise_multiplier=dp_config["sample_dp_noise_multiplier"],
                max_grad_norm=dp_config["sample_dp_clipping_value"],
                expected_batch_size=train_batch_size,
            )

        return optimizer

    def _load_data(self, one_user: bool, data_size: int = 26):
        """
        Loads the data, which is a Dummy alphabet, either for 1 user with
        `data_size` samples, or for N (`data_size`) users, each with 1 sample.
        """
        if one_user:  # the single user gets the whole shard (which is the data)
            shard_size = data_size
            local_batch_size = data_size
        else:  # will have N (`data_size`) users, each with one sample
            shard_size = 1
            local_batch_size = 1
        dummy_dataset = DummyAlphabetDataset(num_rows=data_size)
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, DummyAlphabetFLModel()
        )
        assertEqual(data_loader.num_total_users, data_size / shard_size)
        assertEqual(data_loader.num_total_users, data_provider.num_users())
        self.data_size = data_size  # pyre-ignore
        return data_provider, data_loader.train_batch_size

    def _train_vanilla_pytorch_dp_model(
        self,
        lr: float,
        momentum: float,
        sample_level_dp: bool,
        dp_config: Dict[str, Any],
    ):
        """
        Trains a vanilla Pytorch DP model, without FL (there is 1 user)
        """
        model_wrapper = DummyAlphabetFLModel()
        data_provider, train_batch_size = self._load_data(one_user=True)

        optimizer = self._create_optimizer(
            model_wrapper.model,
            data_provider,
            lr,
            momentum,
            sample_level_dp,
            dp_config,
            train_batch_size,
        )
        model_wrapper.model.train()
        for one_user_data in data_provider.train_data():
            for batch in one_user_data:
                batch_metrics = model_wrapper.fl_forward(batch)
                loss = batch_metrics.loss
                loss.backward()
                optimizer.step()

        return model_wrapper

    def _train_fl_model(
        self,
        lr: float,
        momentum: float,
        one_user: bool,
        dp_config: Optional[Dict[str, Any]] = None,
        noise_func_seed: Optional[int] = None,
        data_size: int = 26,
        **kwargs,
    ):
        """
        Trains an FL model, with or without DP
        """
        # create dummy FL model on alphabet
        global_fl_model = DummyAlphabetFLModel()
        data_provider, _ = kwargs.pop(
            "data_provider", self._load_data(one_user, data_size)
        )
        world_size = 1

        metrics_reporter = FakeMetricReporter()
        if one_user:
            users_per_round = 1
        else:  # otherwise, we have `data_size` users
            users_per_round = data_size
        epochs = kwargs.pop("epoch", 1)
        users_per_round = kwargs.pop("users_per_round", users_per_round)
        metrics_reporter = kwargs.pop("metrics_reporter", metrics_reporter)
        eval_epoch_frequency = kwargs.pop("eval_epoch_frequency", 1.0)
        train_metrics_reported_per_epoch = kwargs.pop(
            "train_metrics_reported_per_epoch", 1
        )
        aggregation_type = kwargs.pop("aggregation_type", AggregationType.AVERAGE)

        sync_trainer = SyncTrainer(
            model=global_fl_model,
            cuda_enabled=False,
            **OmegaConf.structured(
                SyncTrainerConfig(
                    users_per_round=users_per_round,
                    epochs=epochs,
                    always_keep_trained_model=False,
                    train_metrics_reported_per_epoch=train_metrics_reported_per_epoch,
                    report_train_metrics=True,
                    eval_epoch_frequency=eval_epoch_frequency,
                    do_eval=True,
                    report_train_metrics_after_aggregation=True,
                    client=DPClientConfig(
                        epochs=1,
                        optimizer=LocalOptimizerSGDConfig(lr=lr, momentum=momentum),
                        privacy_setting=PrivacySetting(
                            alphas=dp_config["alphas"],
                            noise_multiplier=dp_config["sample_dp_noise_multiplier"],
                            clipping_value=dp_config["sample_dp_clipping_value"],
                            target_delta=dp_config["delta"],
                            noise_seed=noise_func_seed,
                        ),
                    )
                    if dp_config is not None
                    else ClientConfig(
                        epochs=1,
                        optimizer=LocalOptimizerSGDConfig(lr=lr, momentum=momentum),
                    ),
                    server=SyncDPSGDServerConfig(
                        active_user_selector=SequentialActiveUserSelectorConfig(),
                        privacy_setting=PrivacySetting(
                            alphas=dp_config["alphas"],
                            noise_multiplier=dp_config["user_dp_noise_multiplier"],
                            clipping_value=dp_config["user_dp_clipping_value"],
                            target_delta=dp_config["delta"],
                            noise_seed=noise_func_seed,
                        ),
                        aggregation_type=aggregation_type,
                    )
                    if dp_config is not None
                    else SyncServerConfig(),
                )
            ),
        )

        global_fl_model, _eval_metric = sync_trainer.train(
            data_provider,
            metrics_reporter,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )

        return global_fl_model

    def test_dp_turned_off_by_params(self) -> None:
        """
        Tests DP and no-DP produce the same exact model, when DP parameters are off.
        Basically, tests the equivalence of calling SyncServer and SyncDPServer with noise = 0 and clip = inf
        """
        lr = 0.1
        momentum = 0.0

        # first, call SyncTrainer (DP is off)
        torch.manual_seed(1)
        fl_model_with_vanilla_server = self._train_fl_model(
            lr=lr, momentum=momentum, one_user=False, dp_config=None
        )
        # set DP parameters off.
        off_dp_config = {
            "alphas": [10, 100],
            "sample_dp_noise_multiplier": 0.0,
            "sample_dp_clipping_value": float("inf"),
            "user_dp_noise_multiplier": 0.0,
            "user_dp_clipping_value": float("inf"),
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        fl_model_with_dp_server = self._train_fl_model(
            lr=lr, momentum=momentum, one_user=False, dp_config=off_dp_config
        )
        error_msg = verify_models_equivalent_after_training(
            fl_model_with_vanilla_server,
            fl_model_with_dp_server,
            model_init=None,
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_dp_ineffective(self) -> None:
        """
        Tests DP and no-DP produce the same exact model, when DP parameters are ineffective.
        Basically, tests the equivalence of the following 2 scenarios, with N users:

        1. Calling SyncTrainer (this is "no DP")
        2. Calling PrivateSyncTrainer, and DP (this is "DP") when parameters are ineffective

        Note:
            To make the dp_config ineffective, we need to set noise = 0 and clipping value to
            a large number. This is different from test test_dp_turned_off_by_params() that sets
            clipping value to inf, as this test actually applies DP, but because of parameter
            values dp is ineffective, the results would be identical to when DP is OFF.
        """
        lr = 0.1
        momentum = 0.0

        torch.manual_seed(1)
        # Call vanilla sync server
        fl_model_with_vanilla_server = self._train_fl_model(
            lr=lr, momentum=momentum, one_user=False, dp_config=None
        )

        # Next call set DP parameters ineffective.
        ineffective_dp_config = {
            "alphas": [10, 100],
            "sample_dp_noise_multiplier": 0.0,
            "sample_dp_clipping_value": 9999.9,
            "user_dp_noise_multiplier": 0.0,
            "user_dp_clipping_value": 9999.9,
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        fl_model_with_dp_server = self._train_fl_model(
            lr=lr,
            momentum=momentum,
            one_user=False,
            dp_config=ineffective_dp_config,
        )

        error_msg = verify_models_equivalent_after_training(
            fl_model_with_vanilla_server,
            fl_model_with_dp_server,
            model_init=None,
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_frameworks_one_client_sample_dp_off(self) -> None:
        """
        Tests if Pytorch-DP and FL simulator, generate the same resutls, when
        sample-level DP is off. Essentially, tests equivalence of the following
        scenarios, when there is 1 user:

        1. Using FL simulator, when sample-level DP is off
        2. Using Pytorch DP, when sample-level DP is off

        Note:
            For this test, user-level DP is off, since it is not relavant to Pytorch-DP's
            provided functionality, which is sample-level DP.
        """
        lr = 1e-2
        momentum = 0.9

        torch.manual_seed(1)
        fl_model = self._train_fl_model(lr=lr, momentum=momentum, one_user=True)
        torch.manual_seed(1)
        vanilla_dp_model = self._train_vanilla_pytorch_dp_model(
            lr=lr, momentum=momentum, sample_level_dp=False, dp_config={}
        )
        assertEqual(
            FLModelParamUtils.get_mismatched_param(
                [fl_model.fl_get_module(), vanilla_dp_model.fl_get_module()], 1e-6
            ),
            "",
        )

    def test_frameworks_one_client_clipping_only(self) -> None:
        """
        Tests if Pytorch-DP and FL simulator, generate the same resutls, when there
        is only 1 user and when sample-level DP is on (with noising off).
        Essentially, tests equivalence of the following scenarios:

        1. Using FL simulator, when there is one user and DP is on (with noise=0),
        2. Using Pytorch DP (one user), when DP is on,
        (DP parameters are the same for the 2 scenarios.)

        Note:
            For this test, user-level DP is off, since it is not relavant to Pytorch-DP's
            provided functionality, which is sample-level DP.
        """
        lr = 0.25
        momentum = 0.0
        alphas = [1 + x / 10.0 for x in range(1, 100)] + [
            float(y) for y in list(range(12, 64))
        ]
        dp_config = {
            "alphas": alphas,
            "sample_dp_noise_multiplier": 0.0,
            "sample_dp_clipping_value": 1.0,
            "user_dp_noise_multiplier": 0.0,
            "user_dp_clipping_value": float("inf"),
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        fl_model = self._train_fl_model(
            lr=lr, momentum=momentum, one_user=True, dp_config=dp_config
        )
        torch.manual_seed(1)
        vanilla_dp_model = self._train_vanilla_pytorch_dp_model(
            lr=lr, momentum=momentum, sample_level_dp=True, dp_config=dp_config
        )
        assertEqual(
            FLModelParamUtils.get_mismatched_param(
                [fl_model.fl_get_module(), vanilla_dp_model.fl_get_module()], 1e-6
            ),
            "",
        )

    def test_user_dp_equivalent_sample_dp_when_one_client_one_example(
        self,
    ) -> None:
        """
        Tests if user-level DP is equivalent to sample-level DP, when we
        have one user, and that user has one example (one epoch)

        """
        lr = 1.0
        momentum = 0.0

        alphas = [1 + x / 10.0 for x in range(1, 100)] + [
            float(y) for y in list(range(12, 64))
        ]

        dp_config_user_dp_off = {
            "alphas": alphas,
            "sample_dp_noise_multiplier": 0.7,
            "sample_dp_clipping_value": 1.0,
            "user_dp_noise_multiplier": 0.0,
            "user_dp_clipping_value": float("inf"),
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        dp_model_sample_level_dp_on = self._train_fl_model(
            lr=lr,
            momentum=momentum,
            one_user=True,
            dp_config=dp_config_user_dp_off,
            data_size=1,
            noise_func_seed=1234,
        )

        dp_config_sample_dp_off = {
            "alphas": alphas,
            "sample_dp_noise_multiplier": 0,
            "sample_dp_clipping_value": float("inf"),
            "user_dp_noise_multiplier": 0.7,
            "user_dp_clipping_value": 1.0,
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        dp_model_user_level_dp_on = self._train_fl_model(
            lr=lr,
            momentum=momentum,
            one_user=True,
            dp_config=dp_config_sample_dp_off,
            data_size=1,
            noise_func_seed=1234,
        )

        error_msg = verify_models_equivalent_after_training(
            dp_model_sample_level_dp_on,
            dp_model_user_level_dp_on,
            model_init=None,
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_user_dp_equivalent_sample_dp(self) -> None:
        """
        Tests if user-level DP is equivalent to sample-level DP under a certain
        degenerate condition. It tests the equivalence of the following:

        1. User-level DP, with N users, 1 example each (can be N different examples),
        2. Sample-level DP, with 1 user, N examples (can be N different examples),

        under these Conditions:

        Condition 1. [Just as a sanity check] when DP is off (user-level and sample-level)
        Condition 2. When DP is on, noise_multiplier for both cases is 0 (i.e. clipping only)
        Condition 2. When DP is on, noise_multiplier for both cases is 1, manual seed is set

        Note:
            For both cases, we set lr = 1.0 and momentum = 0.0

        """
        lr = 1.0
        momentum = 0.0

        # Condition 1
        torch.manual_seed(1)
        no_dp_model_one_user = self._train_fl_model(
            lr=lr, momentum=momentum, one_user=True
        )
        torch.manual_seed(1)
        no_dp_model = self._train_fl_model(lr=lr, momentum=momentum, one_user=False)

        error_msg = verify_models_equivalent_after_training(
            no_dp_model_one_user,
            no_dp_model,
            model_init=None,
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )
        assertEmpty(error_msg, msg=error_msg)

        # Condition 2
        alphas = [1 + x / 10.0 for x in range(1, 100)] + [
            float(y) for y in list(range(12, 64))
        ]
        dp_config_user_dp_off = {
            "alphas": alphas,
            "sample_dp_noise_multiplier": 0.0,
            "sample_dp_clipping_value": 0.8,
            "user_dp_noise_multiplier": 0,
            "user_dp_clipping_value": 0.8,
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        dp_model_one_user_sample_dp = self._train_fl_model(
            lr=lr, momentum=momentum, one_user=True, dp_config=dp_config_user_dp_off
        )
        dp_config_sample_dp_off = {
            "alphas": alphas,
            "sample_dp_noise_multiplier": 0.0,
            "sample_dp_clipping_value": 0.8,
            "user_dp_noise_multiplier": 0.0,
            "user_dp_clipping_value": 0.8,
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        dp_model_user_dp = self._train_fl_model(
            lr=lr,
            momentum=momentum,
            one_user=False,
            dp_config=dp_config_sample_dp_off,
        )

        error_msg = verify_models_equivalent_after_training(
            dp_model_one_user_sample_dp,
            dp_model_user_dp,
            model_init=None,
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )
        assertEmpty(error_msg, msg=error_msg)

        # Condition 3
        dp_config_user_dp_off = {
            "alphas": alphas,
            "sample_dp_noise_multiplier": 1,
            "sample_dp_clipping_value": 0.5,
            "user_dp_noise_multiplier": 0.0,
            "user_dp_clipping_value": 0.5,
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        dp_model_one_user_sample_dp = self._train_fl_model(
            lr=lr,
            momentum=momentum,
            one_user=True,
            dp_config=dp_config_user_dp_off,
            noise_func_seed=1000,
        )
        dp_config_sample_dp_off = {
            "alphas": alphas,
            "sample_dp_noise_multiplier": 0.0,
            "sample_dp_clipping_value": 0.5,
            "user_dp_noise_multiplier": 1,
            "user_dp_clipping_value": 0.5,
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        dp_model_user_dp = self._train_fl_model(
            lr=lr,
            momentum=momentum,
            one_user=False,
            dp_config=dp_config_sample_dp_off,
            noise_func_seed=1000,
        )
        error_msg = verify_models_equivalent_after_training(
            dp_model_one_user_sample_dp,
            dp_model_user_dp,
            None,
            rel_epsilon=1e-6,
            abs_epsilon=1e-6,
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_private_trainer_reporting(self) -> None:
        lr = 1.0
        momentum = 0.0
        alphas = [1 + x / 10.0 for x in range(1, 100)] + [
            float(y) for y in list(range(12, 64))
        ]
        dp_config = {
            "alphas": alphas,
            "sample_dp_noise_multiplier": 0.0,
            "sample_dp_clipping_value": 2.0,
            "user_dp_noise_multiplier": 1.0,
            "user_dp_clipping_value": 2.0,
            "delta": 0.00001,
        }
        torch.manual_seed(1)
        metrics_reporter = MetricsReporterWithMockedChannels()
        self._train_fl_model(
            lr=lr,
            momentum=momentum,
            one_user=False,
            dp_config=dp_config,
            users_per_round=4,
            epochs=5,
            metrics_reporter=metrics_reporter,
            train_metrics_reported_per_epoch=10,
            eval_epoch_frequency=0.1,
        )
        # we have 26 users, 4 users per round, that makes 7 rounds per epcoh
        # we are reporting train, aggregation, and eval metrics all
        # at 10 rounds per epoch, so we should get a total of 21 reports

        def count_word(result, word):
            return str(result).count(word)

        # check for existance of sample_dp and user_dp in reports.
        # they are logged only in Aggrefation and we aggregated 7 times.

        # print to std our prints sample_dp wich is a dict of four values once
        # hence we should get 7 occurrences.
        assertEqual(
            count_word(metrics_reporter.stdout_results, "sample level dp"),
            7,
            metrics_reporter.stdout_results,
        )

        # summary writer breaks dict to 4 plots, hence we get 28 occurrences.
        assertEqual(
            count_word(metrics_reporter.tensorboard_results, "sample level dp"),
            28,
            metrics_reporter.tensorboard_results,
        )

        # for user_dp we only log one value per round.
        assertEqual(
            count_word(metrics_reporter.stdout_results, "user level dp"),
            7,
            metrics_reporter.stdout_results,
        )

        assertEqual(
            count_word(metrics_reporter.tensorboard_results, "user level dp"),
            7,
            metrics_reporter.tensorboard_results,
        )

    def _test_dp_no_dp_same_weighted_async(
        self,
        noise,
        clip_norm,
        data_provider,
        buffer_size,
        epochs=1,
    ):
        local_lr = np.random.sample()
        global_lr = np.random.sample()

        dp_model = DummyAlphabetFLModel()
        nondp_model = copy.deepcopy(dp_model)
        staleness_config = PolynomialStalenessWeightConfig(
            exponent=0.5, avg_staleness=0
        )

        aggregator_config_nondp = FedAvgWithLRHybridAggregatorConfig(
            lr=global_lr, buffer_size=buffer_size
        )
        aggregator_config_dp = FedAvgWithLRHybridAggregatorConfig(
            lr=global_lr, buffer_size=buffer_size
        )

        for reduction_type in [
            ReductionType.WEIGHTED_SUM,
            ReductionType.WEIGHTED_AVERAGE,
        ]:
            nondp_fl_trained_model, _ = run_fl_training(
                fl_model=nondp_model,
                fl_data_provider=data_provider,
                epochs=epochs,
                local_lr=local_lr,
                aggregator_config=aggregator_config_nondp,
                training_duration_mean=1,
                staleness_weight_config=staleness_config,
            )

            reducer_config = self._get_weighted_dp_reducer_config(
                noise=noise,
                clipping=clip_norm,
                reduction_type=reduction_type,
            )
            aggregator_config_dp.reducer = reducer_config
            dp_fl_trained_model, _ = run_fl_training(
                fl_model=dp_model,
                fl_data_provider=data_provider,
                epochs=epochs,
                local_lr=local_lr,
                aggregator_config=aggregator_config_dp,
                training_duration_mean=1,
                staleness_weight_config=staleness_config,
            )

            return verify_models_equivalent_after_training(
                nondp_fl_trained_model,
                dp_fl_trained_model,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            )

    def test_user_dp_variable_weights_same(self) -> None:
        """
        There are two cases when DP and NonDP should equal

        1) privacy is on, noise = 0, and clip norm is a large value
        2) privacy is on, noise * clip_norm << abs_epsilon

        Note:
            The current implementation has privacy on when noise >= 0 or clip_norm < inf
        """

        data_provider = get_fl_data_provider(
            num_examples=100,
            num_fl_users=10,
            examples_per_user=10,
            batch_size=5,
            model=DummyAlphabetFLModel(),
        )

        buffer_size = np.random.randint(1, 10)

        # sanity check DP == NonDP when privacy off
        error_msg = self._test_dp_no_dp_same_weighted_async(
            noise=-1,
            clip_norm=float("inf"),
            data_provider=data_provider,
            buffer_size=buffer_size,
        )
        assertEmpty(error_msg, msg=error_msg)
        # scenario 1
        error_msg = self._test_dp_no_dp_same_weighted_async(
            noise=0,
            clip_norm=1e6,
            data_provider=data_provider,
            buffer_size=buffer_size,
        )
        assertEmpty(error_msg, msg=error_msg)
        # scenario 2
        error_msg = self._test_dp_no_dp_same_weighted_async(
            noise=1e-14,
            clip_norm=1e6,
            data_provider=data_provider,
            buffer_size=buffer_size,
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_user_dp_variable_weights_different(self) -> None:
        """
        Test when noise is some trivial number then
        DP model should not equal to NonDP model
        """
        buffer_size = np.random.randint(1, 10)
        data_provider = get_fl_data_provider(
            num_examples=10,
            num_fl_users=10,
            examples_per_user=1,
            batch_size=1,
            model=DummyAlphabetFLModel(),
        )
        is_different_msg = self._test_dp_no_dp_same_weighted_async(
            # noise between 0.1 and 1.0
            noise=max(0.1, np.random.sample()),
            clip_norm=10,
            data_provider=data_provider,
            buffer_size=buffer_size,
        )
        assertNotEmpty(is_different_msg, msg=is_different_msg)
