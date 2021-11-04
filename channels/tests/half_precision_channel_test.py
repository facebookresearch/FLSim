#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from copy import deepcopy
from typing import Type

from flsim.channels.communication_stats import (
    ChannelStatsCollector,
    ChannelDirection,
)
from flsim.channels.half_precision_channel import (
    HalfPrecisionChannelConfig,
    HalfPrecisionChannel,
)
from flsim.channels.message import Message
from flsim.tests import utils
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from libfb.py import testutil


class HalfPrecisionChannelTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    @testutil.data_provider(
        lambda: (
            {
                "config": HalfPrecisionChannelConfig(),
                "expected_type": HalfPrecisionChannel,
            },
        )
    )
    def test_fp16_instantiation(self, config: Type, expected_type: Type) -> None:
        """
        Tests instantiation of the `fp16` channel.
        """

        # instantiation
        channel = instantiate(config)
        self.assertIsInstance(channel, expected_type)

    @testutil.data_provider(
        lambda: (
            {
                "config": HalfPrecisionChannelConfig(),
                "expected_type": HalfPrecisionChannel,
            },
        )
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
        message = channel.create_channel_message(download_model)
        message = channel.server_to_client(message)
        message.update_model_(download_model)
        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), download_model.fl_get_module()]
        )
        self.assertEqual(mismatched, "", mismatched)

    @testutil.data_provider(
        lambda: (
            {
                "config": HalfPrecisionChannelConfig(),
                "expected_type": HalfPrecisionChannel,
            },
        )
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
        message = channel.create_channel_message(upload_model)
        message = channel.client_to_server(message)
        message.update_model_(upload_model)
        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), upload_model.fl_get_module()],
            rel_epsilon=1e-8,
            abs_epsilon=1e-3,
        )
        self.assertEqual(mismatched, "", mismatched)

    @testutil.data_provider(
        lambda: (
            {
                "config": HalfPrecisionChannelConfig(),
                "expected_type": HalfPrecisionChannel,
            },
        )
    )
    def test_fp16_stats(self, config: Type, expected_type: Type) -> None:
        """
        Tests stats measurement. Bytes send from server to client should be
        twice as big as bytes sent from client to server (`fp16` emulation).
        """

        # instantiation
        channel = instantiate(config)
        self.assertIsInstance(channel, expected_type)

        # attach stats collector
        stats_collector = ChannelStatsCollector()
        channel.attach_stats_collector(stats_collector)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        download_model = deepcopy(base_model)
        upload_model = deepcopy(base_model)

        # server -> client
        message = channel.create_channel_message(download_model)
        message = channel.server_to_client(message)
        message.update_model_(download_model)

        # client -> server
        message = channel.create_channel_message(upload_model)
        message = channel.client_to_server(message)
        message.update_model_(upload_model)

        # test communiction stats measurements
        stats = stats_collector.get_channel_stats()
        client_to_server_bytes = stats[ChannelDirection.CLIENT_TO_SERVER].mean()
        server_to_client_bytes = stats[ChannelDirection.SERVER_TO_CLIENT].mean()
        self.assertEqual(2 * client_to_server_bytes, server_to_client_bytes)
