#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from flsim.common.pytest_helper import assertEqual, assertNotEqual
from flsim.secure_aggregation.secure_aggregator import FixedPointConfig
from flsim.servers.aggregator import AggregationType
from flsim.servers.sync_secagg_servers import SyncSecAggServerConfig
from flsim.servers.sync_servers import SyncServerConfig
from flsim.tests.utils import (
    FakeMetricReporter,
)
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.sample_model import DummyAlphabetFLModel
from flsim.utils.tests.helpers.sync_trainer_test_utils import (
    create_sync_trainer,
)
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset


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
        assertEqual(data_loader.num_total_users, data_provider.num_users())
        return data_provider, data_loader.train_batch_size

    def _train_fl_model(
        self,
        sec_agg_enable: bool = False,
        fixedpoint=None,
        num_users: int = 26,
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
            users_per_round=num_users,
            epochs=1,
            user_epochs_per_round=1,
            do_eval=True,
            server_config=SyncSecAggServerConfig(
                aggregation_type=AggregationType.AVERAGE, fixedpoint=fixedpoint
            )
            if sec_agg_enable
            else SyncServerConfig(
                aggregation_type=AggregationType.AVERAGE,
            ),
        )
        metrics_reporter = FakeMetricReporter()
        global_fl_model, _eval_metric = sync_trainer.train(
            data_provider,
            metrics_reporter,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )

        return global_fl_model

    def test_secagg_not_equivalent_no_secagg(self) -> None:
        """
        Tests training with secure aggregation will produce a different
        model than training without secure aggregation
        """
        # First, call SyncTrainer with SecureRoundReducer
        fixedpoint = FixedPointConfig(num_bytes=1, scaling_factor=1000)
        torch.manual_seed(1)
        fl_model_with_secure_round_reducer = self._train_fl_model(
            sec_agg_enable=True,
            fixedpoint=fixedpoint,
        )
        # Next, call SyncTrainer (with RoundReducer)
        torch.manual_seed(1)
        fl_model_with_round_reducer = self._train_fl_model()
        assertNotEqual(
            FLModelParamUtils.get_mismatched_param(
                [
                    fl_model_with_round_reducer.fl_get_module(),
                    fl_model_with_secure_round_reducer.fl_get_module(),
                ],
                1e-6,
            ),
            "",
        )

    def test_secagg_not_equivalent_no_secagg_large_range(self) -> None:
        """
        Tests training with secure aggregation will produce a different
        model than training without secure aggregation, even when the
        range of fixedpoint number is very large (and scaling factro is 1).

        The reason that we get a different model is because, even in a big
        fixedpoint range, we still do rounding when we convert, E.g., 127.1
        (float) becomes 127 (in fixedpoint), no matter how big the range is.
        """
        # First, call SyncTrainer with SecureRoundReducer
        fixedpoint = FixedPointConfig(num_bytes=8, scaling_factor=1)
        torch.manual_seed(1)
        fl_model_with_secure_round_reducer = self._train_fl_model(
            sec_agg_enable=True,
            fixedpoint=fixedpoint,
        )
        # Next, call SyncTrainer (with RoundReducer)
        torch.manual_seed(1)
        fl_model_with_round_reducer = self._train_fl_model()
        assertNotEqual(
            FLModelParamUtils.get_mismatched_param(
                [
                    fl_model_with_round_reducer.fl_get_module(),
                    fl_model_with_secure_round_reducer.fl_get_module(),
                ],
                1e-6,
            ),
            "",
        )
