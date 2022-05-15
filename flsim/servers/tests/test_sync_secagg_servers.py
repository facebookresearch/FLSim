#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from flsim.channels.message import Message
from flsim.common.pytest_helper import assertEqual
from flsim.secure_aggregation.secure_aggregator import FixedPointConfig
from flsim.servers.sync_secagg_servers import SyncSecAggServerConfig
from flsim.utils.test_utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
    SampleNet,
    TwoFC,
)
from hydra.utils import instantiate


class TestSyncSecAggServer:
    def _create_server(self, model, fixedpoint, channel=None):
        return instantiate(
            SyncSecAggServerConfig(fixedpoint=fixedpoint),
            global_model=model,
            channel=channel,
        )

    def test_sync_secagg_server_init(self) -> None:
        """
        Tests whether secure aggregator object is initiated
        """
        model = SampleNet(TwoFC())
        # test secure aggregation with flat FP config
        fixedpoint = FixedPointConfig(num_bytes=2, scaling_factor=100)
        server = self._create_server(model, fixedpoint=fixedpoint)
        assertEqual(len(server._secure_aggregator.converters), 4)
        assertEqual(  # verify an arbitrary layer of the model
            server._secure_aggregator.converters["fc2.bias"].scaling_factor, 100
        )

    def test_secure_aggregator_receive_update_from_client(self) -> None:
        """
        Tests whether secure aggregator operations work correctly
        when a model update is received and server model is updated
        """
        scaling_factor = 100
        fixedpoint = FixedPointConfig(num_bytes=2, scaling_factor=scaling_factor)
        server = self._create_server(
            SampleNet(create_model_with_value(0)), fixedpoint=fixedpoint
        )
        server.init_round()

        m1_param = 7.2345
        m1_w = 3.0
        model1 = create_model_with_value(m1_param)
        server.receive_update_from_client(Message(SampleNet(model1), weight=m1_w))

        m2_param = -3.45612
        m2_w = 7.0
        model2 = create_model_with_value(m2_param)
        server.receive_update_from_client(Message(SampleNet(model2), weight=m2_w))

        expected_param = float(
            round(m1_param * scaling_factor * m1_w + m2_param * scaling_factor * m2_w)
        )

        server.step()
        mismatched = model_parameters_equal_to_value(
            server.global_model.fl_get_module(),
            -(expected_param / scaling_factor) / (m1_w + m2_w),
        )
        assertEqual(mismatched, "", mismatched)
