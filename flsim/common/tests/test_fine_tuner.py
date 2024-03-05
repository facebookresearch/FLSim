#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from flsim.clients.base_client import ClientConfig
from flsim.common.fine_tuner import FineTuner
from flsim.common.pytest_helper import assertEmpty, assertEqual
from flsim.common.timeline import Timeline
from flsim.interfaces.metrics_reporter import TrainingStage
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.utils import test_utils as utils
from flsim.utils.cuda import DEFAULT_CUDA_MANAGER


class TestFineTuner:
    def _fake_data(self, num_batches, batch_size):
        dataset = [torch.ones(batch_size, 2) for _ in range(num_batches)]
        dataset = utils.DatasetFromList(dataset)
        return utils.DummyUserData(dataset, utils.SampleNet(utils.Linear()))

    def _get_model(self, value):
        model = utils.SampleNet(utils.Linear())
        model.fl_get_module().fill_all(value)
        return model

    def test_fine_tune_model(self) -> None:
        """
        Global model starts at 1, grad = 1, then model after
        1 batch is going be 0
        """
        metrics_reporter = utils.SimpleMetricReporter()
        global_model = self._get_model(1)
        data = self._fake_data(num_batches=1, batch_size=1)
        client_config = ClientConfig(optimizer=LocalOptimizerSGDConfig(lr=1.0))
        fine_tuned_model = FineTuner.fine_tune_model(
            global_model,
            data,
            client_config,
            metrics_reporter,
            # pyre-fixme[6]: Expected `CudaTransferMinimizer` for 5th param but got
            #  `ICudaStateManager`.
            DEFAULT_CUDA_MANAGER,
            epochs=1,
        )
        error_msg = utils.model_parameters_equal_to_value(fine_tuned_model, 0.0)
        assertEmpty(error_msg, error_msg)

    def test_fine_tune_and_evaluate(self) -> None:
        global_model = self._get_model(1)
        num_clients = 10
        num_batches = 10
        batch_size = 1

        data = [
            self._fake_data(num_batches=num_batches, batch_size=batch_size)
            for _ in range(num_clients)
        ]
        client_config = ClientConfig()
        metrics_reporter = utils.SimpleMetricReporter()
        FineTuner.fine_tune_and_evaluate(
            data,
            global_model,
            client_config,
            metrics_reporter,
            # pyre-fixme[6]: Expected `CudaTransferMinimizer` for 5th param but got
            #  `ICudaStateManager`.
            DEFAULT_CUDA_MANAGER,
            TrainingStage.PERSONALIZED_TEST,
            Timeline(epoch=1, round=2, global_round=2, rounds_per_epoch=10),
            epochs=1,
        )
        assertEqual(len(metrics_reporter.batch_metrics), num_batches * num_clients)
        assertEqual(
            sum([p.num_examples for p in metrics_reporter.batch_metrics]),
            num_clients * batch_size * num_batches,
        )
