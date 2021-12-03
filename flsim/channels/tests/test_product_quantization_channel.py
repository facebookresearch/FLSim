#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from copy import deepcopy
from typing import Type

import pytest
from flsim.channels.communication_stats import (
    ChannelDirection,
)
from flsim.channels.message import Message
from flsim.channels.product_quantization_channel import (
    ProductQuantizationChannelConfig,
    ProductQuantizationChannel,
)
from flsim.common.pytest_helper import assertEqual, assertIsInstance
from flsim.tests import utils
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate


class TestProductQuantizationChannel:
    @pytest.mark.parametrize(
        "config",
        [
            {
                "_target_": ProductQuantizationChannelConfig._target_,
                "max_num_centroids": 256,
                "num_codebooks": 1,
                "max_block_size": 9,
            }
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ProductQuantizationChannel],
    )
    def test_pq_instantiation(self, config: Type, expected_type: Type) -> None:
        """
        Tests instantiation of the scalar quantization channel.
        """

        # test instantiation
        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

    @pytest.mark.parametrize(
        "config",
        [
            {
                "_target_": ProductQuantizationChannelConfig._target_,
                "max_num_centroids": 4,
                "num_codebooks": 1,
                "max_block_size": 2,
            },
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ProductQuantizationChannel],
    )
    def test_pq_server_to_client(self, config: Type, expected_type: Type) -> None:
        """
        Tests server to client transmission of the message. Models
        before and after transmission should be identical since we
        only product quantize in the client to server direction.
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
            {
                "_target_": ProductQuantizationChannelConfig._target_,
                "max_num_centroids": 2,
                "num_codebooks": 1,
                "max_block_size": 1,
            },
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ProductQuantizationChannel],
    )
    def test_pq_client_to_server(self, config: Type, expected_type: Type) -> None:
        """
        Tests client to server transmission of the message. Models
        before and after transmission should be identical since we quantize
        with 2 codebooks of 1 centroid each with constnt weights.

        Notes:
            - Product Quantization itself is tested in `test_product_quantization`.
        """

        # instantiation
        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

        # create dummy model
        two_fc = utils.TwoFC()
        two_fc.fc1.weight.data[:, 0] = 1
        two_fc.fc1.weight.data[:, 1] = 2
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # test client -> server, models should be almost equal due to int8 emulation
        message = Message(upload_model)
        message = channel.client_to_server(message)

        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), upload_model.fl_get_module()]
        )
        assertEqual(mismatched, "", mismatched)

    @pytest.mark.parametrize(
        "config",
        [
            {
                "_target_": ProductQuantizationChannelConfig._target_,
                "max_num_centroids": 2,
                "num_codebooks": 1,
                "max_block_size": 1,
                "min_numel_to_quantize": 10,
                "report_communication_metrics": True,
            },
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [ProductQuantizationChannel],
    )
    def test_pq_stats(self, config: Type, expected_type: Type) -> None:
        """
        Tests stats measurement. We manually compute bytes sent from client
        to the server and verify that it matches the bytes computed by the
        channel. Here, we quantize with 1 bit per weight of assignment and
        need to store one centroids of one byte per layer.
        """

        # instantiation
        channel = instantiate(config)

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # client -> server
        message = Message(upload_model)
        message = channel.client_to_server(message)

        # test communication stats measurements (here per_tensor int8 quantization)
        stats = channel.stats_collector.get_channel_stats()
        client_to_server_bytes = stats[ChannelDirection.CLIENT_TO_SERVER].mean()

        # biases are not quantized
        n_biases = 6
        bias_size_bytes = n_biases * channel.BYTES_PER_FP32

        # layer 2 with 5 elements is not quantized since min_numel_to_quantize = 10
        n_weights_layer_1 = 10
        n_weights_layer_2 = 5

        # weights of layer 1 are quantized with 1 bit per weight of assignments
        weight_size_bytes = n_weights_layer_1 / 8.0

        # weights of layer 2 are not quantized and stored in `fp32`
        weight_size_bytes += n_weights_layer_2 * channel.BYTES_PER_FP32

        # storing codebook, two `fp32` centroid for the first layer
        centroids_bytes = 2 * channel.BYTES_PER_FP32

        compressed_bytes = bias_size_bytes + weight_size_bytes + centroids_bytes

        assertEqual(client_to_server_bytes, compressed_bytes)
