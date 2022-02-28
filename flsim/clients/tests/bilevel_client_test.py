#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from flsim.clients.bilevel_client import BiLevelClient, BiLevelClientConfig
from flsim.optimizers.local_optimizers import LocalOptimizerBiLevelConfig
from flsim.tests import utils
from omegaconf import OmegaConf


class Quadratic1D(nn.Module):
    """
    a toy optimization example:
        min f(x) = 100 x^2 - 1

    minima is x=0.0
    """

    def __init__(self, x):
        super(Quadratic1D, self).__init__()
        self.x = nn.Parameter(
            torch.tensor([x], requires_grad=True, dtype=torch.float64)
        )
        self.y = torch.tensor([1.0])

    def forward(self, batch):
        return 100 * torch.square(self.x) - self.y


class TestBilevelClient:
    def fake_data(self, num_batches=None, batch_size=None):
        num_batches = num_batches or self.num_batches
        batch_size = batch_size or self.batch_size
        torch.manual_seed(0)
        dataset = [
            ([None] * batch_size, torch.rand(batch_size, 2)) for _ in range(num_batches)
        ]
        dataset = utils.DatasetFromList(dataset)
        return utils.DummyUserData(dataset, utils.SampleNet(utils.TwoFC()))

    def create_quadratic_model(self, init):
        model = utils.SampleNet(Quadratic1D(x=init))
        return model

    def test_fixed_global_model(self):
        config = BiLevelClientConfig(
            max_local_steps=10,
            target_local_acc=0.1,
            optimizer=LocalOptimizerBiLevelConfig(lr=0.1, lambda_=1.0),
        )
        client = BiLevelClient(
            **OmegaConf.structured(config),
            dataset=self.fake_data(num_batches=1, batch_size=1)
        )
        global_model = self.create_quadratic_model(init=10)
        client_update, _ = client.generate_local_update(global_model)
        print("Client model", next(client_update.fl_get_module().parameters()))
        print("Global model ", next(global_model.fl_get_module().parameters()))
