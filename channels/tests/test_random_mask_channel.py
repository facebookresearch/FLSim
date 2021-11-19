#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from copy import deepcopy
from typing import OrderedDict
from typing import Type

import pytest
import torch
from flsim.channels.communication_stats import (
    ChannelDirection,
)
from flsim.channels.message import Message
from flsim.channels.random_mask_channel import (
    RandomMaskChannel,
    RandomMaskChannelConfig,
)
from flsim.common.pytest_helper import assertEqual, assertIsInstance, assertAlmostEqual
from flsim.tests import utils
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.tests.helpers.test_models import FCModel
from hydra.utils import instantiate
from torch.nn.utils import prune


class TestRandomMaskChannel:
    @classmethod
    def calc_model_sparsity(cls, state_dict: OrderedDict):
        """
        Calculates model sparsity (fraction of zeroed weights in state_dict).
        """
        non_zero = 0
        tot = 1e-6
        for _, param in state_dict.items():
            non_zero += torch.count_nonzero(param).item()
            tot += float(param.numel())
        return 1.0 - non_zero / (tot + 1e-6)

    def test_sparse_model_size(self) -> None:
        model = FCModel()
        # Prune model to a quarter of its size
        params_to_prune = [
            (model.fc1, "weight"),
            (model.fc1, "bias"),
            (model.fc2, "weight"),
            (model.fc2, "bias"),
            (model.fc3, "weight"),
            (model.fc3, "bias"),
        ]
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.75,
        )
        for module, name in params_to_prune:
            prune.remove(module, name)
        sparsity = self.calc_model_sparsity(model.state_dict())
        assertAlmostEqual(
            0.75,
            sparsity,
            delta=0.02,  # Accounts for 2 percentage points difference
        )

    @pytest.mark.parametrize(
        "config",
        [
            RandomMaskChannelConfig(
                proportion_of_zero_weights=0.6,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [RandomMaskChannel],
    )
    def test_random_mask_instantiation(self, config: Type, expected_type: Type) -> None:
        """
        Tests instantiation of the random mask channel.
        """

        # test instantiation
        channel = instantiate(config)
        assertIsInstance(channel, expected_type)

    @pytest.mark.parametrize(
        "config",
        [
            RandomMaskChannelConfig(
                proportion_of_zero_weights=0.6,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [RandomMaskChannel],
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
        message = Message(download_model)
        message = channel.server_to_client(message)

        mismatched = FLModelParamUtils.get_mismatched_param(
            [base_model.fl_get_module(), download_model.fl_get_module()]
        )
        assertEqual(mismatched, "", mismatched)

    @pytest.mark.parametrize(
        "config",
        [
            RandomMaskChannelConfig(
                proportion_of_zero_weights=0.6,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [RandomMaskChannel],
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
        message = Message(upload_model)
        message = channel.client_to_server(message)

        sparsity = self.calc_model_sparsity(upload_model.fl_get_module().state_dict())

        # sparsity ratio should be approximately proportion_of_zero_weights
        # approximately since we round the number of parameters to prune
        # to an integer, see random_mask_channel.py
        assertAlmostEqual(channel.cfg.proportion_of_zero_weights, sparsity, delta=0.05)

    @pytest.mark.parametrize(
        "config",
        [
            RandomMaskChannelConfig(
                proportion_of_zero_weights=0.6,
                report_communication_metrics=True,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "expected_type",
        [RandomMaskChannel],
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

        # create dummy model
        two_fc = utils.TwoFC()
        base_model = utils.SampleNet(two_fc)
        upload_model = deepcopy(base_model)

        # client -> server
        message = Message(upload_model)
        message = channel.client_to_server(message)

        # test communiction stats measurements
        stats = channel.stats_collector.get_channel_stats()
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

        assertEqual(client_to_server_bytes, true_size_bytes)
