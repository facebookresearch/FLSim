#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest
from flsim.channels.message import Message
from flsim.common.pytest_helper import assertEmpty, assertEqual
from flsim.optimizers.server_optimizers import ServerFTRLOptimizerConfig
from flsim.privacy.common import ClippingSetting, PrivacySetting
from flsim.privacy.privacy_engine import CummuNoiseEffTorch, CummuNoiseTorch
from flsim.servers.sync_ftrl_servers import SyncFTRLServerConfig
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.test_utils import (
    linear_model,
    SampleNet,
    verify_models_equivalent_after_training,
)
from hydra.utils import instantiate


@dataclass
class MockClientUpdate:
    deltas: List[float]
    weights: List[float]
    average: float


class TestSyncServer:
    def _create_client_updates(self, num_clients) -> MockClientUpdate:
        deltas = [float(i + 1) for i in range(num_clients)]
        weights = [float(i + 1) for i in range(num_clients)]
        average = float(np.average(deltas))
        return MockClientUpdate(deltas, weights, average)

    def _setup(
        self,
        efficient,
        noise_multiplier,
        clip_value,
        total_users,
        tree_completion,
        restart_rounds,
    ):
        fl_model = SampleNet(linear_model(0))
        nonfl_model = SampleNet(linear_model(0))

        optimizer = instantiate(
            config=ServerFTRLOptimizerConfig(lr=1.0),
            model=nonfl_model.fl_get_module(),
            record_last_noise=True,
        )

        if efficient:
            noise_gen = CummuNoiseEffTorch(
                std=noise_multiplier * clip_value,
                shapes=[p.shape for p in nonfl_model.fl_get_module().parameters()],
                device="cpu",
                seed=0,
            )
        else:
            noise_gen = CummuNoiseTorch(
                std=noise_multiplier * clip_value,
                shapes=[p.shape for p in nonfl_model.fl_get_module().parameters()],
                device="cpu",
                seed=0,
            )

        server = instantiate(
            SyncFTRLServerConfig(
                server_optimizer=ServerFTRLOptimizerConfig(lr=1.0),
                privacy_setting=PrivacySetting(
                    noise_multiplier=noise_multiplier,
                    clipping=ClippingSetting(clipping_value=clip_value),
                    noise_seed=0,
                ),
                tree_completion=tree_completion,
                efficient=efficient,
                restart_rounds=restart_rounds,
            ),
            global_model=fl_model,
        )

        client_updates = self._create_client_updates(total_users)
        return server, nonfl_model, fl_model, client_updates, noise_gen, optimizer

    @pytest.mark.parametrize(
        "efficient",
        [
            True,
            False,
        ],
    )
    def test_ftrl_same_nonfl_server(self, efficient) -> None:
        noise_multiplier = 0
        clip_value = 1000
        total_users = 1
        users_per_round = 1

        (
            server,
            nonfl_model,
            fl_model,
            client_updates,
            noise_gen,
            optimizer,
        ) = self._setup(
            efficient,
            noise_multiplier,
            clip_value,
            total_users,
            tree_completion=False,
            restart_rounds=1000,
        )

        for _ in range(10):
            server.init_round()
            optimizer.zero_grad()
            server.select_clients_for_training(total_users, users_per_round)
            for delta, weight in zip(client_updates.deltas, client_updates.weights):
                server.receive_update_from_client(
                    Message(model=SampleNet(linear_model(delta)), weight=weight)
                )

            FLModelParamUtils.set_gradient(
                model=nonfl_model.fl_get_module(),
                reference_gradient=linear_model(client_updates.average),
            )
            noise = noise_gen()
            # nonfl_model = init - lr * (grad + noise)
            # = 0 - (1 + noise)
            # = -(1 + noise)
            optimizer.step(noise)
            fl_noise = server.step()[0]

            assertEqual(fl_noise.value, sum([p.sum() for p in noise]))
            error_msg = verify_models_equivalent_after_training(fl_model, nonfl_model)
            assertEmpty(error_msg, msg=error_msg)

    @pytest.mark.parametrize(
        "efficient",
        [
            True,
            False,
        ],
    )
    def test_ftrl_same_nonfl_server_with_restart(self, efficient):
        noise_multiplier = 0
        clip_value = 1000
        total_users = 1
        users_per_round = 1

        (
            server,
            nonfl_model,
            fl_model,
            client_updates,
            noise_gen,
            optimizer,
        ) = self._setup(
            efficient,
            noise_multiplier,
            clip_value,
            total_users,
            tree_completion=True,
            restart_rounds=1,
        )

        for _ in range(10):
            server.init_round()
            optimizer.zero_grad()
            server.select_clients_for_training(total_users, users_per_round)
            for delta, weight in zip(client_updates.deltas, client_updates.weights):
                server.receive_update_from_client(
                    Message(model=SampleNet(linear_model(delta)), weight=weight)
                )

            FLModelParamUtils.set_gradient(
                model=nonfl_model.fl_get_module(),
                reference_gradient=linear_model(client_updates.average),
            )
            noise = noise_gen()
            # nonfl_model = init - lr * (grad + noise)
            # = 0 - (1 + noise)
            # = -(1 + noise)
            optimizer.step(noise)
            fl_noise = server.step()[0]

            assertEqual(fl_noise.value, sum([p.sum() for p in noise]))
            error_msg = verify_models_equivalent_after_training(fl_model, nonfl_model)
            assertEmpty(error_msg, msg=error_msg)
