#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from copy import deepcopy
from typing import Type

import pytest
import torch
from flsim.channels.communication_stats import ChannelDirection
from flsim.channels.message import Message
from flsim.channels.sketch_channel import (
    SketchChannelConfig,
    SketchChannel,
)
from flsim.common.pytest_helper import (
    assertAlmostEqual,
    assertEqual,
    assertIsInstance,
    assertRaises,
    assertTrue,
)
from flsim.tests import utils
from flsim.utils.count_sketch import CountSketch
from hydra.utils import instantiate


class TestCountSketch:
    def test_count_sketch_creation(self) -> None:
        """
        Tests instantiation of the count sketch channel.
        """

        good_hash = torch.randint(0, 10, (10, 2))
        bad_hash = torch.randint(0, 10, (2, 1))

        cs = CountSketch(width=10, depth=10, h=good_hash, g=good_hash)
        assertIsInstance(cs, CountSketch)

        with assertRaises(AssertionError):
            _ = CountSketch(width=10, depth=5, h=bad_hash)

        with assertRaises(AssertionError):
            _ = CountSketch(width=10, depth=5, g=bad_hash)

    def test_count_sketch_update_query(self) -> None:
        """
        Tests if it is possible to sketch and unsketch one number using CountSketch.
        Also check that unsketching an id that were never sketched would return 0.
        """

        cs = CountSketch(width=10001, depth=11, prime=2 ** 31 - 1, independence=4)
        # test simple update and query
        cs.update(torch.tensor([0]), torch.tensor([2.0]))
        assertTrue(torch.isclose(cs.query(torch.tensor([0])), torch.tensor((2.0))))
        assertTrue(torch.isclose(cs.query(torch.tensor([1])), torch.tensor((0.0))))
        assertEqual(torch.sum(torch.abs(cs.buckets)).item(), 2 * 11)

    def test_count_sketch_model(self) -> None:
        """
        Tests if CountSketch stores the correct parameter names and tensor sizes
        after sketching a model and the unsketched model has approximately the same
        weights.
        """

        cs = CountSketch(width=10001, depth=11, prime=2 ** 31 - 1, independence=4)
        two_fc = utils.TwoFC()
        two_fc.fill_all(0.2)
        # should have 21 parameters
        model = utils.SampleNet(two_fc)

        cs.sketch_model(model=model)

        for (cs_name, cs_param), (model_name, model_param) in zip(
            list(cs.param_sizes.items()), list(model.fl_get_module().named_parameters())
        ):
            assertEqual(cs_name, model_name)
            assertEqual(cs_param, model_param.size())

        for (_, cs_param) in cs.unsketch_model().items():
            assertTrue(
                False
                not in torch.isclose(
                    cs_param, torch.full_like(cs_param, fill_value=0.2)
                )
            )

    def test_count_sketch_size(self) -> None:
        width = 1000
        depth = 10
        cs = CountSketch(width=width, depth=depth)
        size = cs.get_size_in_bytes()
        assertAlmostEqual(size, width * depth * 4, delta=1)


class TestSketchChannelTest:
    @pytest.mark.parametrize(
        "config",
        [SketchChannelConfig(num_col=1000, num_hash=7, prime=2 ** 13 - 1)],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [SketchChannel],
    )
    def test_sketch_channel(self, config: Type, expected_type: Type) -> None:
        """
        Tests client to server direction. Models should be almost
        identical since we use a very large sketch wrt model size.
        """

        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

        two_fc = utils.TwoFC()
        two_fc.fill_all(0.2)
        model = utils.SampleNet(two_fc)

        message = Message(model)
        message = channel.client_to_server(message)
        cs = message.count_sketch
        assertIsInstance(cs, CountSketch)

        # same test as the count_sketch test
        for (cs_name, cs_param), (model_name, model_param) in zip(
            list(cs.param_sizes.items()), list(model.fl_get_module().named_parameters())
        ):
            assertEqual(cs_name, model_name)
            assertEqual(cs_param, model_param.size())

        for (_, cs_param) in cs.unsketch_model().items():
            assertTrue(
                False
                not in torch.isclose(
                    cs_param, torch.full_like(cs_param, fill_value=0.2)
                )
            )

    @pytest.mark.parametrize(
        "config",
        [
            SketchChannelConfig(
                num_col=1000,
                num_hash=7,
                prime=2 ** 13 - 1,
                report_communication_metrics=True,
            )
        ],
    )
    def test_sketch_stats(self, config: Type) -> None:
        """
        Tests stats measurement. We manually compute bytes sent from client
        to the server and verify that it matches the bytes computed by the
        channel.
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

        # test communication stats measurements
        stats = channel.stats_collector.get_channel_stats()
        client_to_server_bytes = stats[ChannelDirection.CLIENT_TO_SERVER].mean()

        # the sketch has `num_col` elements per hash. Assuming fp32 elements.
        computed_sketch_size_bytes = (
            config.num_col * config.num_hash * channel.BYTES_PER_FP32
        )

        assertEqual(client_to_server_bytes, computed_sketch_size_bytes)
