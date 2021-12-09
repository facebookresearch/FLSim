#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Type

import pytest
from flsim.channels.base_channel import FLChannelConfig, IdentityChannel
from flsim.channels.communication_stats import (
    ChannelDirection,
)
from flsim.channels.message import Message
from flsim.common.pytest_helper import assertEqual, assertIsInstance
from flsim.tests import utils
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate


class TestIdentityChannel:
    @pytest.mark.parametrize(
        "config",
        [FLChannelConfig()],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [IdentityChannel],
    )
    def test_identity_instantiation(self, config: Type, expected_type: Type) -> None:
        """
        Tests instantiation of the identity channel.
        """

        # test instantiation
        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

    @pytest.mark.parametrize(
        "config",
        [FLChannelConfig()],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [IdentityChannel],
    )
    def test_identity_server_to_client(self, config: Type, expected_type: Type) -> None:
        """
        Tests server to client transmission of the message. Models
        before and after transmission should be identical.
        """

        # instantiate channel
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
        [FLChannelConfig()],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [IdentityChannel],
    )
    def test_identity_client_to_server(self, config: Type, expected_type: Type) -> None:
        """
        Tests client to server transmission of the message. Models
        before and after transmission should be identical.
        """

        # instantiation
        channel = instantiate(config)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # test client -> server, models should be strictly identical
        message = Message(upload_model)
        message = channel.client_to_server(message)
        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), upload_model.fl_get_module()]
        )
        assertEqual(mismatched, "", mismatched)

    @pytest.mark.parametrize(
        "config",
        [FLChannelConfig(report_communication_metrics=True)],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [IdentityChannel],
    )
    def test_identity_stats(self, config: Type, expected_type: Type) -> None:
        """
        Tests stats measurement. Bytes sent from client to server
        and from server to client should be identical.
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

        # test communication stats measurements
        stats = channel.stats_collector.get_channel_stats()
        client_to_server_bytes = stats[ChannelDirection.CLIENT_TO_SERVER].mean()
        server_to_client_bytes = stats[ChannelDirection.SERVER_TO_CLIENT].mean()
        assertEqual(client_to_server_bytes, server_to_client_bytes)
