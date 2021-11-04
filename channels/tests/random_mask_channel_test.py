#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from copy import deepcopy
from typing import Type

from flsim.channels.communication_stats import (
    ChannelStatsCollector,
    ChannelDirection,
)
from flsim.channels.message import Message
from flsim.channels.random_mask_channel import (
    RandomMaskChannel,
    RandomMaskChannelConfig,
)
from flsim.tests import utils
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.model_size_utils import calc_model_sparsity
from hydra.utils import instantiate
from libfb.py import testutil


class RandomMaskChannelTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    @testutil.data_provider(
        lambda: (
            {
                "config": RandomMaskChannelConfig(proportion_of_zero_weights=0.6),
                "expected_type": RandomMaskChannel,
            },
        )
    )
    def test_random_mask_instantiation(self, config: Type, expected_type: Type) -> None:
        """
        Tests instantiation of the random mask channel.
        """

        # test instantiation
        channel = instantiate(config)
        self.assertIsInstance(channel, expected_type)

    @testutil.data_provider(
        lambda: (
            {
                "config": RandomMaskChannelConfig(proportion_of_zero_weights=0.6),
                "expected_type": RandomMaskChannel,
            },
        )
    )
    def test_random_mask_server_to_client(
        self, config: Type, expected_type: Type
    ) -> None:
        """
        Tests server to client transmission of the message. Models
        before and after transmission should be identical since
        we random mask only on the client to server direction.
        """

        # instantiation
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
                "config": RandomMaskChannelConfig(proportion_of_zero_weights=0.6),
                "expected_type": RandomMaskChannel,
            },
        )
    )
    def test_random_mask_client_to_server(
        self, config: Type, expected_type: Type
    ) -> None:
        """
        Tests client to server transmission of the message. Model
        after transmission should have the right sparsity ratio.
        """

        # instantiation
        channel = instantiate(config)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # test client -> server, check for sparsity ratio
        message = channel.create_channel_message(upload_model)
        message = channel.client_to_server(message)
        message.update_model_(upload_model)

        sparsity = calc_model_sparsity(upload_model.fl_get_module().state_dict())

        # sparsity ratio should be approximately proportion_of_zero_weights
        # approximately since we round the number of parameters to prune
        # to an integer, see random_mask_channel.py
        self.assertAlmostEqual(
            channel.cfg.proportion_of_zero_weights, sparsity, delta=0.05
        )

    @testutil.data_provider(
        lambda: (
            {
                "config": RandomMaskChannelConfig(proportion_of_zero_weights=0.6),
                "expected_type": RandomMaskChannel,
            },
        )
    )
    def test_random_mask_stats(self, config: Type, expected_type: Type) -> None:
        """
        Tests stats measurement. We assume that the sparse tensor
        is stored in COO format and manually compute the number
        of bytes sent from client to server to check that it
        matches that the channel computes.
        """

        # instantiation
        channel = instantiate(config)

        # attach stats collector
        stats_collector = ChannelStatsCollector()
        channel.attach_stats_collector(stats_collector)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # client -> server
        message = channel.create_channel_message(upload_model)
        message = channel.client_to_server(message)
        message.update_model_(upload_model)

        # test communiction stats measurements
        stats = stats_collector.get_channel_stats()
        client_to_server_bytes = stats[ChannelDirection.CLIENT_TO_SERVER].mean()

        # compute sizes
        n_weights = sum([p.numel() for p in two_fc.parameters() if p.ndim == 2])
        n_biases = sum([p.numel() for p in two_fc.parameters() if p.ndim == 1])
        non_zero_weights = n_weights - int(
            n_weights * channel.cfg.proportion_of_zero_weights
        )
        non_zero_biases = n_biases - int(
            n_biases * channel.cfg.proportion_of_zero_weights
        )
        n_dim_weights = 2
        n_dim_biases = 1
        true_size_bytes_weights = (
            # size of the index
            non_zero_weights * RandomMaskChannel.BYTES_PER_INT64 * n_dim_weights
            # size of values
            + non_zero_weights * RandomMaskChannel.BYTES_PER_FP32
        )
        true_size_bytes_biases = (
            # size of the index
            non_zero_biases * RandomMaskChannel.BYTES_PER_INT64 * n_dim_biases
            # size of values
            + non_zero_biases * RandomMaskChannel.BYTES_PER_FP32
        )

        # size of the values
        true_size_bytes = true_size_bytes_weights + true_size_bytes_biases

        self.assertEqual(client_to_server_bytes, true_size_bytes)
