#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from flsim.clients.fedsgd_client import FedSGDClient, FedSGDClientConfig
from flsim.clients.base_client import Client, ClientConfig
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.common.pytest_helper import (
    assertEmpty,
    assertEqual,
)
from flsim.tests.utils import DatasetFromList, DummyUserData, SampleNet, verify_models_equivalent_after_training
from flsim.utils.sample_model import SimpleLinearNet
from omegaconf import OmegaConf


class TestFedSGDClient:
    def _fake_data(self, num_batches, batch_size):
        torch.manual_seed(0)
        dataset = [torch.rand(batch_size, 2) for _ in range(num_batches)]
        dataset = DatasetFromList(dataset)
        return DummyUserData(dataset, SampleNet(SimpleLinearNet(2, 1)))

    def _get_fedsgd_client(self, num_batches, batch_size) -> FedSGDClient:
        config = FedSGDClientConfig(optimizer=LocalOptimizerSGDConfig(lr=0.1))
        return FedSGDClient(
            **OmegaConf.structured(config),
            dataset=self._fake_data(num_batches=num_batches, batch_size=batch_size)
        )

    def _get_fedavg_client(self, num_batches, batch_size) -> Client:
        config = ClientConfig(optimizer=LocalOptimizerSGDConfig(lr=1.0))
        return Client(
            **OmegaConf.structured(config),
            dataset=self._fake_data(num_batches=num_batches, batch_size=batch_size)
        )

    def test_fedsgd_fedavg_same(self):
        fedsgd_client = self._get_fedsgd_client(num_batches=10, batch_size=32)
        fedavg_client = self._get_fedavg_client(num_batches=1, batch_size=320)
        
        global_model = SampleNet(SimpleLinearNet(2, 1))

        fedsgd_delta, fedsgd_weight = fedsgd_client.generate_local_update(global_model)
        fedavg_delta, fedavg_weight = fedavg_client.generate_local_update(global_model)
        error_msg = verify_models_equivalent_after_training(fedsgd_delta, fedavg_delta)
        assertEqual(fedavg_weight, fedsgd_weight)
        assertEmpty(error_msg, msg=error_msg)