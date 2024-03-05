#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from flsim.common.pytest_helper import assertEqual, assertNotEqual
from flsim.secure_aggregation.secure_aggregator import FixedPointConfig
from flsim.servers.aggregator import AggregationType
from flsim.servers.sync_secagg_servers import SyncSecAggServerConfig
from flsim.servers.sync_servers import SyncServerConfig
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.sample_model import DummyAlphabetFLModel
from flsim.utils.test_utils import FakeMetricReporter, MetricsReporterWithMockedChannels
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from flsim.utils.tests.helpers.test_sync_trainer_utils import create_sync_trainer


class TestSecureAggregationIntegration:
    def _load_data(self, num_users: int = 26):
        """
        Loads the data, which is a Dummy alphabet for N (`num_users`) users,
        each with 1 sample.
        """
        shard_size = 1
        local_batch_size = 1
        dummy_dataset = DummyAlphabetDataset(num_rows=num_users)
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, DummyAlphabetFLModel()
        )
        assertEqual(data_loader.num_total_users, num_users / shard_size)
        assertEqual(data_loader.num_total_users, data_provider.num_train_users())
        return data_provider, data_loader.train_batch_size

    def _train_fl_model(
        self,
        sec_agg_enable: bool = False,
        fixedpoint=None,
        num_users: int = 26,
        users_per_round: int = 26,
        epochs: int = 1,
        metrics_reporter=None,
        report_train_metrics: bool = False,
        report_train_metrics_after_aggregation: bool = False,
        train_metrics_reported_per_epoch: int = 1,
    ):
        """
        Trains an FL model, with or without Secure Aggregation
        """
        # create dummy FL model on alphabet
        global_fl_model = DummyAlphabetFLModel()
        data_provider, _ = self._load_data(num_users)
        world_size = 1
        sync_trainer = create_sync_trainer(
            model=global_fl_model,
            local_lr=0.1,
            users_per_round=users_per_round,
            epochs=epochs,
            user_epochs_per_round=1,
            do_eval=True,
            server_config=(
                SyncSecAggServerConfig(
                    aggregation_type=AggregationType.AVERAGE, fixedpoint=fixedpoint
                )
                if sec_agg_enable
                else SyncServerConfig(
                    aggregation_type=AggregationType.AVERAGE,
                )
            ),
        )
        sync_trainer.cfg.train_metrics_reported_per_epoch = (
            train_metrics_reported_per_epoch
        )
        sync_trainer.cfg.report_train_metrics = report_train_metrics
        sync_trainer.cfg.report_train_metrics_after_aggregation = (
            report_train_metrics_after_aggregation
        )
        if metrics_reporter is None:
            metrics_reporter = FakeMetricReporter()
        global_fl_model, _eval_metric = sync_trainer.train(
            data_provider,
            metrics_reporter,
            num_total_users=data_provider.num_train_users(),
            distributed_world_size=world_size,
        )

        return global_fl_model

    def test_secagg_not_equivalent_no_secagg(self) -> None:
        """
        Tests that training with secure aggregation will produce a different
        model than training without secure aggregation
        """
        # First, call secure trainer
        fixedpoint = FixedPointConfig(num_bytes=1, scaling_factor=1000)
        torch.manual_seed(1)
        fl_model_with_secure_trainer = self._train_fl_model(
            sec_agg_enable=True,
            fixedpoint=fixedpoint,
        )
        # Next, call sync trainer
        torch.manual_seed(1)
        fl_model_with_trainer = self._train_fl_model()
        assertNotEqual(
            FLModelParamUtils.get_mismatched_param(
                [
                    fl_model_with_trainer.fl_get_module(),
                    fl_model_with_secure_trainer.fl_get_module(),
                ],
                1e-6,
            ),
            "",
        )

    def test_secagg_not_equivalent_no_secagg_large_range(self) -> None:
        """
        Tests that training with secure aggregation will produce a different
        model than training without secure aggregation, even when the
        range of fixed point number is very large (and the scaling factor is 1).

        We get a different model because we round during conversion, even in a big
        fixed point range. For example, 127.1 (float) becomes 127 (in fixed point),
        regardless of the size of the range.
        """
        # First, call secure trainer
        fixedpoint = FixedPointConfig(num_bytes=7, scaling_factor=1)
        torch.manual_seed(1)
        fl_model_with_secure_trainer = self._train_fl_model(
            sec_agg_enable=True,
            fixedpoint=fixedpoint,
        )
        # Next, call sync trainer
        torch.manual_seed(1)
        fl_model_with_trainer = self._train_fl_model()
        assertNotEqual(
            FLModelParamUtils.get_mismatched_param(
                [
                    fl_model_with_trainer.fl_get_module(),
                    fl_model_with_secure_trainer.fl_get_module(),
                ],
                1e-6,
            ),
            "",
        )

    def test_overflow_reporting(self) -> None:
        """
        Tests whether the overflow parameters are reported enough
        """
        fixedpoint = FixedPointConfig(num_bytes=1, scaling_factor=100)

        metrics_reporter = MetricsReporterWithMockedChannels()
        self._train_fl_model(
            sec_agg_enable=True,
            fixedpoint=fixedpoint,
            users_per_round=2,
            epochs=3,
            metrics_reporter=metrics_reporter,
            report_train_metrics=True,
            report_train_metrics_after_aggregation=True,
            train_metrics_reported_per_epoch=26,
        )

        def count_word(result, word):
            return str(result).count(word)

        # We have 26 users, 2 users_per_round, which makes 13 rounds per epoch.
        # We also have 3 epochs. So we should get 39 reports for overflow.
        # (train_metrics_reported_per_epoch is large so we don't miss a report)
        assertEqual(
            count_word(metrics_reporter.stdout_results, "overflow per round"),
            39,
            metrics_reporter.stdout_results,
        )

        # for tensorboard results, we write 39*2 results related to overflow,
        # as we report each {covert, aggregate} overflow once
        assertEqual(
            count_word(metrics_reporter.tensorboard_results, "overflow per round"),
            78,
            metrics_reporter.tensorboard_results,
        )
