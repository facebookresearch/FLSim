#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from flsim.channels.message import Message
from flsim.channels.scalar_quantization_channel import ScalarQuantizationChannel
from flsim.common.pytest_helper import assertEmpty, assertEqual, assertTrue
from flsim.secure_aggregation.secure_aggregator import FixedPointConfig
from flsim.servers.sync_secagg_servers import (
    SyncSecAggServerConfig,
    SyncSecAggSQServerConfig,
)
from flsim.servers.sync_servers import SyncSQServerConfig
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.test_utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
    SampleNet,
    TwoFC,
)
from hydra.utils import instantiate


class TestSyncSecAggServer:
    def test_sync_secagg_server_init(self) -> None:
        """
        Tests whether secure aggregator object is initiated
        """
        model = SampleNet(TwoFC())
        # test secure aggregation with flat FP config
        fixed_point_config = FixedPointConfig(num_bytes=2, scaling_factor=100)
        server = instantiate(
            SyncSecAggServerConfig(fixedpoint=fixed_point_config), global_model=model
        )
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
        fixed_point_config = FixedPointConfig(
            num_bytes=2, scaling_factor=scaling_factor
        )
        server = instantiate(
            SyncSecAggServerConfig(fixedpoint=fixed_point_config),
            global_model=SampleNet(create_model_with_value(0)),
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


class TestSyncSecAggSQServer:
    def test_sync_sq_with_secagg_same_as_sync_sq(self) -> None:
        "Test that global model after update on sync secgg SQ server is same as that"
        " of sync SQ server when the overflow room is large."
        fixed_point_config = FixedPointConfig(num_bytes=4, scaling_factor=100)
        sync_sq_server = instantiate(
            SyncSQServerConfig(),
            global_model=SampleNet(create_model_with_value(0.05)),
            channel=ScalarQuantizationChannel(use_shared_qparams=True),
        )
        sync_secagg_sq_server = instantiate(
            SyncSecAggSQServerConfig(fixedpoint=fixed_point_config),
            global_model=SampleNet(create_model_with_value(0.05)),
            channel=ScalarQuantizationChannel(sec_agg_mode=True),
        )
        assertTrue(sync_secagg_sq_server._channel.use_shared_qparams)

        client1 = SampleNet(create_model_with_value(0.01))
        client2 = SampleNet(create_model_with_value(0.02))

        sync_sq_server.update_qparams(create_model_with_value(0.005))
        sync_sq_server.receive_update_from_client(
            Message(model=client1, weight=1.0, qparams=sync_sq_server.global_qparams)
        )
        sync_sq_server.receive_update_from_client(
            Message(model=client2, weight=1.0, qparams=sync_sq_server.global_qparams)
        )
        sync_sq_server.step()

        sync_secagg_sq_server.update_qparams(create_model_with_value(0.005))
        sync_secagg_sq_server.receive_update_from_client(
            Message(
                model=client1,
                weight=1.0,
                qparams=sync_secagg_sq_server.global_qparams,
            )
        )
        sync_secagg_sq_server.receive_update_from_client(
            Message(
                model=client2,
                weight=1.0,
                qparams=sync_secagg_sq_server.global_qparams,
            )
        )
        sync_secagg_sq_server.step()

        mismatched_param_names = set(
            FLModelParamUtils.get_mismatched_param(
                [
                    sync_sq_server.global_model.fl_get_module(),
                    sync_secagg_sq_server.global_model.fl_get_module(),
                ],
                abs_epsilon=1e-3,
            )
        )
        quantized_param_names = {
            n for n, p in create_model_with_value(0).named_parameters() if p.ndim > 1
        }
        assertEmpty(quantized_param_names.intersection(mismatched_param_names))
