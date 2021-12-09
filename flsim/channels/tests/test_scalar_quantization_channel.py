#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Type

import pytest
from flsim.channels.communication_stats import (
    ChannelDirection,
)
from flsim.channels.message import Message
from flsim.channels.scalar_quantization_channel import (
    ScalarQuantizationChannelConfig,
    ScalarQuantizationChannel,
)
from flsim.common.pytest_helper import assertEqual, assertIsInstance
from flsim.tests import utils
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate


class TestScalarQuantizationChannel:
    @pytest.mark.parametrize(
        "config",
        [
            ScalarQuantizationChannelConfig(
                n_bits=8,
                quantize_per_tensor=True,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ScalarQuantizationChannel],
    )
    def test_int8_instantiation(self, config: Type, expected_type: Type) -> None:
        """
        Tests instantiation of the scalar quantization channel.
        """

        # test instantiation
        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

    @pytest.mark.parametrize(
        "config",
        [
            ScalarQuantizationChannelConfig(n_bits=8, quantize_per_tensor=True),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ScalarQuantizationChannel],
    )
    def test_int8_server_to_client(self, config: Type, expected_type: Type) -> None:
        """
        Tests server to client transmission of the message. Models
        before and after transmission should be identical since we
        only scalar quantize in the client to server direction.
        """

        # instantiation
        channel = instantiate(config)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        download_model = deepcopy(base_model)

        # test server -> client, models should be strictly identical
        message = Message(download_model)
        message = channel.server_to_client(message)

        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), download_model.fl_get_module()]
        )
        assertEqual(mismatched, "", mismatched)

    @pytest.mark.parametrize(
        "config",
        [
            ScalarQuantizationChannelConfig(n_bits=8, quantize_per_tensor=True),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ScalarQuantizationChannel],
    )
    def test_int8_client_to_server(self, config: Type, expected_type: Type) -> None:
        """
        Tests client to server transmission of the message. Models
        before and after transmission should be almost identical
        due to int8 emulation (n_bits = 8 here).
        """

        # instantiation
        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # test client -> server, models should be almost equal due to int8 emulation
        message = message = Message(upload_model)
        message = channel.client_to_server(message)

        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), upload_model.fl_get_module()],
            rel_epsilon=1e-8,
            abs_epsilon=1e-2,
        )
        assertEqual(mismatched, "", mismatched)

    @pytest.mark.parametrize(
        "config",
        [
            ScalarQuantizationChannelConfig(
                n_bits=8, quantize_per_tensor=True, report_communication_metrics=True
            ),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ScalarQuantizationChannel],
    )
    def test_int8_stats(self, config: Type, expected_type: Type) -> None:
        """
        Tests stats measurement. We manually compute bytes sent from client
        to server for the two weight matrices (remember that we do not quantize
        the biases), and check that it's a quarter of the bytes sent from
        server to client (subtracting scales and zero_points).
        """

        # instantiation
        channel = instantiate(config)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        download_model = deepcopy(base_model)
        upload_model = deepcopy(base_model)

        # server -> client
        message = Message(download_model)
        message = channel.server_to_client(message)

        # client -> server
        message = Message(upload_model)
        message = channel.client_to_server(message)

        # test communication stats measurements (here per_tensor int8 quantization)
        stats = channel.stats_collector.get_channel_stats()
        client_to_server_bytes = stats[ChannelDirection.CLIENT_TO_SERVER].mean()
        server_to_client_bytes = stats[ChannelDirection.SERVER_TO_CLIENT].mean()

        channels_first_layer = 5
        channels_second_layer = 1
        n_tot_channels = channels_first_layer + channels_second_layer
        bias_size_bytes = n_tot_channels * channel.BYTES_PER_FP32

        n_layers = 2
        scale_zero_point_bytes = n_layers * (
            channel.BYTES_PER_FP64 + channel.BYTES_PER_FP32
        )
        int8_weight_bytes = (
            client_to_server_bytes - bias_size_bytes - scale_zero_point_bytes
        )
        float_weights_bytes = server_to_client_bytes - bias_size_bytes
        assertEqual(4 * int8_weight_bytes, float_weights_bytes)

    @pytest.mark.parametrize(
        "config",
        [
            ScalarQuantizationChannelConfig(n_bits=4, quantize_per_tensor=False),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ScalarQuantizationChannel],
    )
    def test_quantize_constant(self, config: Type, expected_type: Type) -> None:
        """
        Tests client to server transmission of the message. Models
        before and after transmission should be identical since we
        quantize a model filled with weights and biases equal to 1.
        """

        # instantiation
        channel = instantiate(config)

        # create dummy model with constant weights
        two_fc = utils.TwoFC()
        two_fc.fill_all(1.0)
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # test client -> server, all weights are equal so quantization error is
        # zero or every acceptable value of n_bits (here tested with n_bits = 4)
        message = Message(upload_model)
        message = channel.client_to_server(message)

        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), upload_model.fl_get_module()]
        )
        assertEqual(mismatched, "", mismatched)
