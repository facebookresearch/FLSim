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
from flsim.channels.half_precision_channel import (
    HalfPrecisionChannelConfig,
    HalfPrecisionChannel,
)
from flsim.channels.message import Message
from flsim.common.pytest_helper import assertEqual, assertIsInstance
from flsim.tests import utils
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate


class TestHalfPrecisionChannel:
    @pytest.mark.parametrize(
        "config",
        [HalfPrecisionChannelConfig()],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [HalfPrecisionChannel],
    )
    def test_fp16_instantiation(self, config: Type, expected_type: Type) -> None:
        """
        Tests instantiation of the `fp16` channel.
        """

        # instantiation
        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

    @pytest.mark.parametrize(
        "config",
        [HalfPrecisionChannelConfig()],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [HalfPrecisionChannel],
    )
    def test_fp16_server_to_client(self, config: Type, expected_type: Type) -> None:
        """
        Tests server to client transmission of the message. Models
        before and after transmission should be identical since
        we emulate `fp16` only from client to server.
        """

        # test instantiation
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
        [HalfPrecisionChannelConfig()],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [HalfPrecisionChannel],
    )
    def test_fp16_client_to_server(self, config: Type, expected_type: Type) -> None:
        """
        Tests client to server transmission of the message. Models
        before and after transmission should be almost identical
        due to fp16 emulation.
        """

        # instantiation
        channel = instantiate(config)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # test client -> server, models should be almost equal due to fp16 emulation
        message = Message(upload_model)
        message = channel.client_to_server(message)

        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), upload_model.fl_get_module()],
            rel_epsilon=1e-8,
            abs_epsilon=1e-3,
        )
        assertEqual(mismatched, "", mismatched)

    @pytest.mark.parametrize(
        "config",
        [HalfPrecisionChannelConfig(report_communication_metrics=True)],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [HalfPrecisionChannel],
    )
    def test_fp16_stats(self, config: Type, expected_type: Type) -> None:
        """
        Tests stats measurement. Bytes send from server to client should be
        twice as big as bytes sent from client to server (`fp16` emulation).
        """

        # instantiation
        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

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

        # test communiction stats measurements
        stats = channel.stats_collector.get_channel_stats()
        client_to_server_bytes = stats[ChannelDirection.CLIENT_TO_SERVER].mean()
        server_to_client_bytes = stats[ChannelDirection.SERVER_TO_CLIENT].mean()
        assertEqual(2 * client_to_server_bytes, server_to_client_bytes)
