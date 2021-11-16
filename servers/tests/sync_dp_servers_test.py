#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import math
from unittest.mock import MagicMock

import numpy as np
import torch
from flsim.channels.message import Message
from flsim.privacy.common import PrivacySetting
from flsim.privacy.privacy_engine import GaussianPrivacyEngine
from flsim.servers.aggregator import AggregationType
from flsim.servers.sync_dp_servers import SyncDPSGDServerConfig
from flsim.servers.sync_servers import FedAvgOptimizerConfig, SyncServerConfig
from flsim.tests.utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
    verify_models_equivalent_after_training,
    SampleNet,
)
from hydra.utils import instantiate
from libfb.py import testutil


class SyncDPSGDServerTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def _get_num_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def _create_server(
        self,
        server_model,
        num_rounds,
        num_clients,
        clipping_value,
        noise_multiplier,
    ):
        server = instantiate(
            SyncDPSGDServerConfig(
                aggregation_type=AggregationType.AVERAGE,
                optimizer=FedAvgOptimizerConfig(lr=1.0),
                privacy_setting=PrivacySetting(
                    clipping_value=clipping_value,
                    noise_multiplier=noise_multiplier,
                    noise_seed=0,
                ),
            ),
            global_model=server_model,
        )
        server.select_clients_for_training(
            num_total_users=num_rounds * num_clients, users_per_round=num_clients
        )
        return server

    @testutil.data_provider(
        lambda: (
            {"clipping_value": 1, "num_clients": 10},
            {"clipping_value": 1, "num_clients": 1},
            {"clipping_value": 1e10, "num_clients": 10},
            {"clipping_value": 1e10, "num_clients": 1},
        )
    )
    def test_no_noise_with_clip(self, clipping_value, num_clients) -> None:
        """
        Test DP-SGD with no noise and with user norm clipping.
        """
        server_model = SampleNet(create_model_with_value(0))
        num_rounds = 10
        delta_param = 1.0
        num_params = self._get_num_params(server_model.fl_get_module())

        clipped_delta = math.sqrt(clipping_value ** 2 / num_params)
        expected_value = -float(min(np.average(clipped_delta), delta_param))
        server = self._create_server(
            server_model, num_rounds, num_clients, clipping_value, noise_multiplier=0
        )

        for round_num in range(num_rounds):
            server.init_round()
            for _ in range(num_clients):
                delta = SampleNet(create_model_with_value(delta_param))
                server.receive_update_from_client(Message(model=delta, weight=1.0))
            server.step()
            error_msg = model_parameters_equal_to_value(
                server_model, expected_value * (round_num + 1)
            )
            self.assertEmpty(error_msg, msg=error_msg)

    @testutil.data_provider(
        lambda: (
            {"clipping_value": 1, "noise": 1, "num_clients": 10},
            {"clipping_value": 1, "noise": 1, "num_clients": 1},
        )
    )
    def test_noise_and_clip(self, clipping_value, noise, num_clients) -> None:
        """
        Test user-level DP-SGD.
        We assume the following:
            1. Server model init  at 0
            2. Trains for 10 rounds and take the simple average of the client updates.
            3. The learning rate = 1.0.
            4. The norm of each user delta is greater than the clipping value

        The DP-SGD update rule is: w_t = w_t-1 - lr * (avg(grad) + sgd_noise)
        With the above assumptions, w_t = 0 - 1.0 * (avg(grad) + sgd_noise) = -(avg(grad) + sgd_noise)
        """
        server_model = SampleNet(create_model_with_value(0))
        num_rounds = 10

        num_params = self._get_num_params(server_model.fl_get_module())
        clipped_delta = math.sqrt(clipping_value ** 2 / num_params)

        server = self._create_server(
            server_model,
            num_rounds,
            num_clients,
            clipping_value,
            noise_multiplier=noise,
        )

        GaussianPrivacyEngine._generate_noise = MagicMock(
            side_effect=lambda size, sensitivity: torch.ones(size) * sensitivity
        )
        for round_num in range(num_rounds):
            server.init_round()
            for _ in range(num_clients):
                delta = SampleNet(create_model_with_value(1.0))
                server.receive_update_from_client(Message(model=delta, weight=1.0))
            server.step()

            expected_value = float(
                -np.average(clipped_delta) - (noise * clipping_value / num_clients)
            )
            error_msg = model_parameters_equal_to_value(
                server_model, expected_value * (round_num + 1)
            )
            self.assertEmpty(error_msg, msg=error_msg)

    def test_no_noise_no_clip(self):
        """
        Test that DP-SGD server with no clipping and no noise is the same as vanilla SyncServer
        """
        global_value = 0
        client_value = 1.0
        dp_model = SampleNet(create_model_with_value(global_value))
        no_dp_model = SampleNet(create_model_with_value(global_value))
        num_rounds = 10
        num_clients = 10

        dp_server = self._create_server(
            dp_model,
            num_rounds,
            num_clients=num_clients,
            clipping_value=1e10,
            noise_multiplier=0,
        )
        no_dp_server = instantiate(
            SyncServerConfig(
                aggregation_type=AggregationType.AVERAGE,
                optimizer=FedAvgOptimizerConfig(lr=1.0),
            ),
            global_model=no_dp_model,
        )

        for _ in range(num_rounds):
            no_dp_server.init_round()
            dp_server.init_round()
            for _ in range(num_clients):
                dp_server.receive_update_from_client(
                    Message(
                        model=SampleNet(
                            create_model_with_value(global_value - client_value)
                        ),
                        weight=1.0,
                    )
                )
                no_dp_server.receive_update_from_client(
                    Message(
                        model=SampleNet(
                            create_model_with_value(global_value - client_value)
                        ),
                        weight=1.0,
                    )
                )
            dp_server.step()
            no_dp_server.step()

            error_msg = verify_models_equivalent_after_training(dp_model, no_dp_model)
            self.assertEmpty(error_msg, msg=error_msg)

    def test_noise_added_correctly(self):
        """
        Test where noise is a fixed value, 0.8
        update = global (all 0) - local (all 2.0) = all 2.0
        update norm = sqrt(num_params*delta^2)=sqrt(21*2^2)=sqrt(84)= 9.165
        and this will be clipped to clipping_value of 7, which
        means that the parameters of the clipped update will be all equal to sqrt(49/21)= 1.52
        w_t = w_t-1 - lr * (avg(grad) + sgd_noise)
        w_t = 0 - 1.0 * (avg(grad) + sgd_noise) = -(avg(grad) + sgd_noise) = -(1.52 + 0.8) = -2.32

        Same test as https://fburl.com/code/mfaini2k
        """
        num_clients = 10
        clipping_value = 7.0
        noise = 0.8
        global_value = 0.0
        client_value = -2.0

        server_model = SampleNet(create_model_with_value(global_value))

        num_params = self._get_num_params(server_model.fl_get_module())
        clipped_delta = math.sqrt(clipping_value ** 2 / num_params)
        expected_value = float(-np.average(clipped_delta) - noise)

        server = self._create_server(
            server_model,
            num_rounds=1,
            num_clients=num_clients,
            clipping_value=clipping_value,
            noise_multiplier=noise,
        )

        GaussianPrivacyEngine._generate_noise = MagicMock(return_value=noise)
        server.init_round()
        for _ in range(num_clients):
            delta = create_model_with_value(global_value - client_value)
            server.receive_update_from_client(Message(model=SampleNet(delta), weight=1))
        server.step()

        error_msg = model_parameters_equal_to_value(server_model, expected_value)
        self.assertEmpty(error_msg, msg=error_msg)
