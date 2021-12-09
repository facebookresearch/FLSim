#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from dataclasses import dataclass
from tempfile import mkstemp
from typing import List, Type
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.pytest_helper import (
    assertEqual,
    assertAlmostEqual,
    assertNotEqual,
    assertFalse,
    assertTrue,
    assertIsInstance,
)
from flsim.privacy.common import PrivacySetting
from flsim.reducers.base_round_reducer import (
    ReductionType,
    RoundReducer,
    RoundReducerConfig,
)
from flsim.reducers.dp_round_reducer import DPRoundReducer, DPRoundReducerConfig
from flsim.reducers.secure_round_reducer import (
    SecureRoundReducer,
    SecureRoundReducerConfig,
)
from flsim.reducers.weighted_dp_round_reducer import (
    EstimatorType,
    WeightedDPRoundReducer,
    WeightedDPRoundReducerConfig,
)
from flsim.secure_aggregation.secure_aggregator import FixedPointConfig
from flsim.tests import utils
from flsim.utils.async_trainer.async_example_weights import EqualExampleWeightConfig
from flsim.utils.async_trainer.async_staleness_weights import (
    PolynomialStalenessWeightConfig,
)
from flsim.utils.async_trainer.async_weights import AsyncWeight, AsyncWeightConfig
from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


def create_ref_model(ref_model_param_value):
    ref_model = utils.SampleNet(utils.TwoFC())
    ref_model.fl_get_module().fill_all(ref_model_param_value)
    return ref_model


def get_dp_round_reducer(
    ref_model=None,
    clipping_value=99999.99,
    reduction_type=ReductionType.AVERAGE,
    noise_multiplier=0,
    num_users_per_round: int = 1,
    total_number_of_users: int = 1,
    reset=True,
):
    ref_model = ref_model or utils.SampleNet(utils.TwoFC())
    privacy_setting = PrivacySetting(
        noise_multiplier=noise_multiplier, clipping_value=clipping_value
    )
    dp_rr = DPRoundReducer(
        **OmegaConf.structured(
            DPRoundReducerConfig(
                reduction_type=reduction_type, privacy_setting=privacy_setting
            )
        ),
        global_model=ref_model,
        num_users_per_round=num_users_per_round,
        total_number_of_users=total_number_of_users,
    )
    if reset:
        dp_rr.reset(ref_model)
    return dp_rr


def init_process(rank, world_size, reducer, models, file_loc, pipe, backend="gloo"):
    """
    Initialize the distributed environment for multi-process testing, and
    runs a simple reduction task and returns the mean of all the parameters
    int reduced module.
    """
    dist.init_process_group(
        backend, init_method=f"file://{file_loc}", rank=rank, world_size=world_size
    )
    reducer.reset(models[0])
    for i, m in enumerate(models):
        if i % world_size == rank:
            reducer.collect_update(m, float(i + 1))
    reducer.reduce()
    sums, weights = 0.0, 0.0
    all_sum = [(p.sum(), p.numel()) for p in reducer.reduced_module.parameters()]
    for s, w in all_sum:
        sums += float(s)
        weights += float(w)
    pipe.send(sums / weights)
    dist.destroy_process_group()


def run_reduction_test(reducer, num_processes=1, num_models=4):
    """
    Used in multiprocess test only.
    Runs a simple scenario in multiple processes.
    Models are sequentially initilized with
    a constant value for params, i.e. 1, 2, ..., num_models.
    """
    _, tmpfile = mkstemp(dir="/tmp")
    pipe_out, pipe_in = mp.Pipe(False)
    # reducer.reduced_module.share_memory()
    models = [utils.SampleNet(utils.TwoFC()) for _ in range(num_models)]
    for i, m in enumerate(models):
        m.fl_get_module().fill_all(float(i + 1))
    processes = []
    results = []
    FLDistributedUtils.WORLD_SIZE = num_processes
    for pid in range(num_processes):
        p = mp.Process(
            target=init_process,
            args=(pid, num_processes, reducer, models, tmpfile, pipe_in),
        )
        p.start()
        processes.append(p)
        results.append(pipe_out)
    for p in processes:
        p.join()
    res = [r.recv() for r in results]
    return res


class TestRoundReducerBase:
    def _fake_client(self, global_value, client_value, weight):
        clnt = Client(dataset=None, **OmegaConf.structured(ClientConfig()))

        def fill(model, *args):
            model.fl_get_module().fill_all(global_value - client_value)
            return model, weight

        clnt.generate_local_update = MagicMock(side_effect=fill)
        return clnt

    def _create_fake_clients(
        self, global_param_value, num_clients, client_param_value, client_weight
    ) -> List[Client]:
        # initialize clients, each with model parameters equal to client_param_values
        return [
            self._fake_client(
                global_value=global_param_value,
                client_value=client_param_value,
                weight=client_weight,
            )
            for _ in range(num_clients)
        ]


class TestRoundReducer(TestRoundReducerBase):
    def get_round_reducer(
        self,
        model=None,
        reduction_type=ReductionType.WEIGHTED_AVERAGE,
        reset=True,
    ):
        model = model or utils.SampleNet(utils.TwoFC())
        round_reducer = RoundReducer(
            **OmegaConf.structured(RoundReducerConfig(reduction_type=reduction_type)),
            global_model=model,
        )
        return round_reducer

    def test_reset(self):
        rr = self.get_round_reducer()
        mismatched = utils.model_parameters_equal_to_value(rr.reduced_module, 0.0)
        assertEqual(mismatched, "", mismatched)
        # do it again
        rr.reduced_module.fill_all(1.0)
        rr.reset(utils.SampleNet(utils.TwoFC()))
        mismatched = utils.model_parameters_equal_to_value(rr.reduced_module, 0.0)
        assertEqual(mismatched, "", mismatched)

    def test_receive_through_channel(self):
        # expected channel effects,
        rr = self.get_round_reducer()
        model = utils.SampleNet(utils.TwoFC())
        # check channel is pass through
        # TODO modify when there is actuall channel
        model2 = rr.receive_through_channel(model)
        mismatched = utils.verify_models_equivalent_after_training(model2, model)
        assertEqual(mismatched, "", mismatched)

    def test_update_reduced_module(self):
        model = utils.SampleNet(utils.TwoFC())
        rr = self.get_round_reducer(model)
        model.fl_get_module().fill_all(0.2)
        rr.update_reduced_module(model.fl_get_module(), 3.0)
        model.fl_get_module().fill_all(0.3)
        rr.update_reduced_module(model.fl_get_module(), 2.0)
        mismatched = utils.model_parameters_equal_to_value(
            rr.reduced_module, 3 * 0.2 + 2 * 0.3
        )
        assertEqual(mismatched, "", mismatched)

    def test_collect_update(self):
        param_values = [0.1 * i for i in range(100)]
        weights = [i % 10 for i in range(100)]
        global_param = 1.0
        clients = [
            self._fake_client(global_param, p, w) for p, w in zip(param_values, weights)
        ]
        ref_model = utils.SampleNet(utils.TwoFC())
        rr = self.get_round_reducer(ref_model)
        for clnt in clients:
            model, weight = clnt.generate_local_update(utils.SampleNet(utils.TwoFC()))
            rr.collect_update(model, weight)
        expected_sum_weights = sum(weights)
        expected_param_values = sum(
            (global_param - param_value) * w
            for param_value, w in zip(param_values, weights)
        )
        experiment_model, experiment_weight = rr.current_results
        assertAlmostEqual(expected_sum_weights, experiment_weight, 5)
        mismatched = utils.model_parameters_equal_to_value(
            experiment_model, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)

    def test_reduction_types_sum(self):
        model = utils.SampleNet(utils.TwoFC())
        rr = self.get_round_reducer(model, reduction_type=ReductionType.SUM)
        results = run_reduction_test(rr, num_processes=1, num_models=2)
        value_expected = float(sum(i + 1 for i in range(2)))
        for r in results:
            assertAlmostEqual(r, value_expected, 5)

    def test_reduction_types_avg(self):
        model = utils.SampleNet(utils.TwoFC())
        rr = self.get_round_reducer(model, reduction_type=ReductionType.AVERAGE)
        results = run_reduction_test(rr, num_processes=2, num_models=4)
        value_expected = sum(i + 1 for i in range(4)) / 4
        for r in results:
            assertAlmostEqual(r, value_expected, 5)

    def test_reduction_types_weighted_sum(self):
        model = utils.SampleNet(utils.TwoFC())
        rr = self.get_round_reducer(model, reduction_type=ReductionType.WEIGHTED_SUM)
        results = run_reduction_test(rr, num_processes=3, num_models=6)
        value_expected = float(sum((i + 1) ** 2 for i in range(6)))
        for r in results:
            assertAlmostEqual(r, value_expected, 5)

    def test_reduction_types_weighted_avg(self):
        model = utils.SampleNet(utils.TwoFC())
        rr = self.get_round_reducer(
            model, reduction_type=ReductionType.WEIGHTED_AVERAGE
        )
        results = run_reduction_test(rr, num_processes=4, num_models=8)
        value_expected = float(sum((i + 1) ** 2 for i in range(8))) / sum(
            i + 1 for i in range(8)
        )
        for r in results:
            assertAlmostEqual(r, value_expected, 5)

    def test_multiprocess_reduce(self):
        # test multi processing.
        model = utils.SampleNet(utils.TwoFC())
        num_models = 4
        value_expected = float(sum((i + 1) ** 2 for i in range(num_models))) / sum(
            i + 1 for i in range(num_models)
        )
        # test 1 process
        r1 = self.get_round_reducer(model, reset=False)
        results = run_reduction_test(r1, num_processes=1, num_models=num_models)
        for r in results:
            assertAlmostEqual(r, value_expected, 5)
        # test 4 processes
        r2 = self.get_round_reducer(model)
        results = run_reduction_test(r2, num_processes=2, num_models=num_models)
        for r in results:
            assertAlmostEqual(r, value_expected, 5)

    def test_logging_level(self):
        rr = self.get_round_reducer()
        assertTrue(utils.check_inherit_logging_level(rr, 50))
        assertTrue(utils.check_inherit_logging_level(rr, 10))


class TestSecureRoundReducer(TestRoundReducerBase):
    def get_round_reducer(
        self,
        model,
        fixedpoint,
    ):
        model = model or utils.SampleNet(utils.TwoFC())
        round_reducer = SecureRoundReducer(
            **OmegaConf.structured(
                SecureRoundReducerConfig(
                    reduction_type=ReductionType.WEIGHTED_AVERAGE,
                    fixedpoint=fixedpoint,
                )
            ),
            global_model=model,
        )
        return round_reducer

    def test_secure_aggregator_reducer_init(self):
        """
        Tests whether secure aggeragtion object is initiated
        correctly in round reducer
        """
        model = utils.SampleNet(utils.TwoFC())
        # test secure aggregation with flat FP config
        fixedpoint = FixedPointConfig(num_bytes=2, scaling_factor=100)
        r1 = self.get_round_reducer(model, fixedpoint=fixedpoint)
        assertEqual(len(r1.secure_aggregator.converters), 4)
        assertEqual(  # verify an arbitrary layer of the model
            r1.secure_aggregator.converters["fc2.bias"].scaling_factor, 100
        )

    def test_secure_aggregator_reducer_update_reduced_module(self):
        """
        Tests whether secure aggeragtion operations work correctly
        when a model update is received and reduced module is updated
        """
        model = utils.SampleNet(utils.TwoFC())

        fixedpoint = FixedPointConfig(num_bytes=2, scaling_factor=100)
        # hence minValue = -32768, maxValue = 32767
        rr = self.get_round_reducer(model, fixedpoint=fixedpoint)

        model.fl_get_module().fill_all(7.2345)
        rr.update_reduced_module(model.fl_get_module(), 3.0)
        model.fl_get_module().fill_all(-3.45612)
        rr.update_reduced_module(model.fl_get_module(), 7.0)
        mismatched = utils.model_parameters_equal_to_value(
            rr.reduced_module, -249.0
        )  # round(7.2345*100*3 - 3.45612*100*7) = round(-248.934)
        assertEqual(mismatched, "", mismatched)

        rr.reduce()
        mismatched = utils.model_parameters_equal_to_value(
            rr.reduced_module, -0.2490  # -249.0 / 100 / (3.0 + 0.7)
        )
        assertEqual(mismatched, "", mismatched)

    def test_secure_aggregator_reducer_reduce_large_range(self):
        """
        Tests whether secure aggeragtion operations work correctly
        when the reduce() method is called, and when the num_bytes is
        big, so we do not have a possible fixedpoint overflow
        """
        ref_model = utils.SampleNet(utils.TwoFC())
        fixedpoint = FixedPointConfig(num_bytes=4, scaling_factor=10)
        rr = self.get_round_reducer(ref_model, fixedpoint=fixedpoint)

        global_param = 8.0
        client_param = 2.123
        num_clients = 50

        clients = self._create_fake_clients(
            global_param_value=global_param,
            num_clients=num_clients,
            client_param_value=client_param,
            client_weight=1.0,
        )
        for client in clients:
            delta, weight = client.generate_local_update(utils.SampleNet(utils.TwoFC()))
            rr.collect_update(delta, weight)
        rr.reduce()
        # update is 5.877. In current fixedpoint setup, it is 59.
        # when we reduce, we convert from fixedpoint to floating point, so 59 -> 5.9
        expected_param_values = (5.9 * num_clients) / num_clients

        model_after_noise, sum_weights = rr.current_results
        mismatched = utils.model_parameters_equal_to_value(
            model_after_noise, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)

    def test_secure_aggregator_reducer_reduce_small_range(self):
        """
        Tests whether secure aggeragtion operations work correctly
        when the reduce() method is called, and when the num_bytes is
        small so we have possible fixedpoint overflows
        """
        ref_model = utils.SampleNet(utils.TwoFC())
        fixedpoint = FixedPointConfig(num_bytes=1, scaling_factor=100)
        # the range of numbers would be (-128,127)
        rr = self.get_round_reducer(ref_model, fixedpoint=fixedpoint)

        global_param = 8.0
        client_param = 2.123
        num_clients = 50

        clients = self._create_fake_clients(
            global_param_value=global_param,
            num_clients=num_clients,
            client_param_value=client_param,
            client_weight=1.0,
        )
        for client in clients:
            delta, weight = client.generate_local_update(utils.SampleNet(utils.TwoFC()))
            rr.collect_update(delta, weight)
        rr.reduce()
        # update is 5.877. In current fixedpoint setup, it is 127, as 587.7 > 127
        # when we reduce, we convert from fixedpoint to floating point, so 127 -> 1.27
        expected_param_values = (1.27 * num_clients) / num_clients

        model_after_noise, sum_weights = rr.current_results

        mismatched = utils.model_parameters_equal_to_value(
            model_after_noise, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)


class TestDPRoundReducer(TestRoundReducerBase):
    def test_dp_off(self):
        ref_model = create_ref_model(ref_model_param_value=3.0)
        # when clipping_value is inf, sensetivity is inf -> dp not supported
        dp_rr = get_dp_round_reducer(
            ref_model, clipping_value=float("inf"), noise_multiplier=0
        )
        assertFalse(dp_rr.privacy_on)
        # noise < 0 means no dp
        dp_rr = get_dp_round_reducer(
            ref_model, clipping_value=10.0, noise_multiplier=-1
        )
        assertFalse(dp_rr.privacy_on)

    def test_collect_update_with_clipping(self):
        """
        Tests whether the model updates associated with the new models sent
        from clients are clipped correctly.
        """
        num_clients = 100
        global_value = 5.0
        clients = self._create_fake_clients(
            global_param_value=global_value,
            num_clients=num_clients,
            client_param_value=3.0,
            client_weight=1.0,
        )
        ref_model = create_ref_model(ref_model_param_value=global_value)

        dp_rr = get_dp_round_reducer(
            ref_model,
            clipping_value=6.0,
            num_users_per_round=num_clients,
            total_number_of_users=num_clients,
        )
        for client in clients:
            delta, weight = client.generate_local_update(utils.SampleNet(utils.TwoFC()))
            dp_rr.collect_update(delta, weight)
            """
            delta = global (all 5.0) - local (all 3.0) = all 2.0
            delta norm = sqrt(num_params*delta^2)=sqrt(21*2^2)=sqrt(84)= 9.16515138991168
            and this will be clipped to clipping_value of 6.0, which
            means that the parameters of the clipped update will be all equal
            to sqrt(36/21)= 1.309307341415954
            """

        expected_param_values = (1.309307341415954) * num_clients
        collected_model_updates, _ = dp_rr.current_results
        mismatched = utils.model_parameters_equal_to_value(
            collected_model_updates, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)

    def test_clipping_when_noise_zero(self):
        """
        Tests when noise multiplier is zero, calling add_noise() in reduce()
        does not change the model after clipping.
        """
        num_clients = 50
        global_value = 8.0
        clients = self._create_fake_clients(
            global_param_value=global_value,
            num_clients=num_clients,
            client_param_value=2.0,
            client_weight=1.0,
        )
        ref_model = create_ref_model(ref_model_param_value=global_value)

        dp_rr = get_dp_round_reducer(
            ref_model,
            clipping_value=15.0,
            noise_multiplier=0,
            num_users_per_round=num_clients,
            total_number_of_users=num_clients,
        )
        for client in clients:
            delta, weight = client.generate_local_update(utils.SampleNet(utils.TwoFC()))

            dp_rr.collect_update(delta, weight)
            """
            update = global (all 8.0) - local (all 2.0) = all 6.0
            update norm = sqrt(num_params*delta^2)=sqrt(21*6^2)=sqrt(756)= 27.49545416973504
            and this will be clipped to clipping_value of 15, which
            means that the parameters of the clipped update will be all equal
            to sqrt(15^2/21)= 3.273268353539886
            """

        dp_rr.reduce()
        # asserts callnig add_noise did not change anything
        expected_param_values = ((3.273268353539886 * num_clients) / num_clients) + 0

        model_after_noise, sum_weights = dp_rr.current_results
        mismatched = utils.model_parameters_equal_to_value(
            model_after_noise, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)

    def test_noise_when_clipping_large_value(self):
        """
        Tests 2 things: 1) whether clipping does not happen when
        clipping threshold is set to a large value, 2) whether we get
        a different model when we add noise and clipping is ineffective
        (clipping threshold is set to a large value).
        """
        num_clients = 20
        global_value = 5.0
        clients = self._create_fake_clients(
            global_param_value=global_value,
            num_clients=num_clients,
            client_param_value=3.0,
            client_weight=1.0,
        )
        ref_model = create_ref_model(ref_model_param_value=global_value)

        ref_model_before = copy.deepcopy(ref_model)
        dp_rr = get_dp_round_reducer(
            ref_model,
            clipping_value=10.0,
            num_users_per_round=num_clients,
            total_number_of_users=num_clients,
        )
        for client in clients:
            delta, weight = client.generate_local_update(utils.SampleNet(utils.TwoFC()))
            dp_rr.collect_update(delta, weight)
            """
            update = global (all 5.0) - local (all 3.0) = all 2.0
            update norm = sqrt(num_params*delta^2)=sqrt(21*2^2)=sqrt(84)= 9.16515138991168
            and this will not be clipped, because the clipping_value
            is set as a larger value (10 > 9.16515138991168). So the parameters
            of the model update will not change and all be equal to 2.
            """

        # asserts clipping does not happen
        expected_param_values = 2.0 * num_clients
        collected_model_updates, _ = dp_rr.current_results
        mismatched = utils.model_parameters_equal_to_value(
            collected_model_updates, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)

        dp_rr.reduce()
        ref_module_after_noise, _ = dp_rr.current_results
        # asserts by adding noise, we get a different model
        mismatched = utils.verify_models_equivalent_after_training(
            ref_model_before.fl_get_module(), ref_module_after_noise
        )
        assertNotEqual(mismatched, "")

    def test_noise_added_correctly(self):
        """
        Tests that the noise is added correctly to model.
        """
        num_clients = 100
        global_value = 5.0
        clients = self._create_fake_clients(
            global_param_value=global_value,
            num_clients=num_clients,
            client_param_value=3.0,
            client_weight=1.0,
        )
        ref_model = create_ref_model(ref_model_param_value=global_value)

        dp_rr = get_dp_round_reducer(
            ref_model,
            clipping_value=7.0,
            num_users_per_round=num_clients,
            total_number_of_users=num_clients,
        )
        for client in clients:
            delta, weight = client.generate_local_update(utils.SampleNet(utils.TwoFC()))
            client.compute_delta(ref_model, delta, delta)
            dp_rr.collect_update(delta, weight)
            """
            update = global (all 5.0) - local (all 3.0) = all 2.0
            update norm = sqrt(num_params*delta^2)=sqrt(21*2^2)=sqrt(84)= 9.16515138991168
            and this will be clipped to clipping_value of 7, which
            means that the parameters of the clipped update will be all equal
            to sqrt(49/21)= 1.527525231651947
            """

        dp_rr.privacy_engine._generate_noise = MagicMock(return_value=0.8)
        expected_param_values = ((1.527525231651947 * num_clients) / num_clients) + 0.8
        dp_rr.reduce()
        ref_module_after_noise, _ = dp_rr.current_results
        mismatched = utils.model_parameters_equal_to_value(
            ref_module_after_noise, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)

    def test_multiprocess_dp_all_processes_the_same(self):
        # test multi processing.
        model = utils.SampleNet(utils.TwoFC())
        num_models = 4
        # test 4 processes
        r = get_dp_round_reducer(
            model,
            clipping_value=1.0,
            reduction_type=ReductionType.AVERAGE,
            noise_multiplier=1,
            num_users_per_round=4,
            total_number_of_users=4,
            reset=False,
        )
        results = run_reduction_test(r, num_processes=4, num_models=num_models)
        same_value = results[0]
        for r in results:
            assertAlmostEqual(r, same_value, 5)


@dataclass
class WeightReducerTestSetting:
    num_clients: int = 10
    clients_per_round: int = 5
    clipping_value: float = 1.0
    max_staleness: int = 100
    noise: float = 1.0
    max_weight: float = 1.0
    min_weight: float = 0.0
    mean_weight: float = 0.5


class TestWeightedDPRoundReducer(TestRoundReducerBase):
    def _get_reducer(
        self,
        ref_model=None,
        clipping_value=1e10,
        reduction_type=ReductionType.WEIGHTED_SUM,
        estimator_type=EstimatorType.UNBIASED,
        noise_multiplier=0,
        num_users_per_round: int = 1,
        total_number_of_users: int = 1,
        max_weight: float = 10,
        min_weight: float = 1e-6,
        mean_weight: float = 1e-6,
    ):
        ref_model = ref_model or utils.SampleNet(utils.TwoFC())
        privacy_setting = PrivacySetting(
            noise_multiplier=noise_multiplier, clipping_value=clipping_value
        )
        reducer = WeightedDPRoundReducer(
            **OmegaConf.structured(
                WeightedDPRoundReducerConfig(
                    reduction_type=reduction_type,
                    privacy_setting=privacy_setting,
                    estimator_type=estimator_type,
                    max_weight=max_weight,
                    min_weight=min_weight,
                    mean_weight=mean_weight,
                )
            ),
            global_model=ref_model,
            num_users_per_round=num_users_per_round,
            total_number_of_users=total_number_of_users,
        )
        return reducer

    def _get_async_weight(self, exponent, avg_staleness=0):
        return AsyncWeight(
            **OmegaConf.structured(
                AsyncWeightConfig(
                    staleness_weight=PolynomialStalenessWeightConfig(
                        exponent=exponent, avg_staleness=avg_staleness
                    ),
                    example_weight=EqualExampleWeightConfig(),
                )
            )
        )

    def _get_num_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def _reduce_weighted_models(
        self,
        global_model,
        settings: WeightReducerTestSetting,
        reduction_type: ReductionType,
        estimator_type: EstimatorType = EstimatorType.UNBIASED,
        global_param: float = 1.0,
        client_param: float = 1.0,
        client_weight: float = 1.0,
    ):
        clients = self._create_fake_clients(
            global_param_value=global_param,
            num_clients=settings.num_clients,
            client_param_value=client_param,
            client_weight=client_weight,
        )

        reducer = self._get_reducer(
            global_model,
            reduction_type=reduction_type,
            clipping_value=settings.clipping_value,
            estimator_type=estimator_type,
            num_users_per_round=settings.clients_per_round,
            total_number_of_users=settings.num_clients,
            max_weight=settings.max_weight,
            min_weight=settings.min_weight,
            mean_weight=settings.mean_weight,
        )
        async_weight = self._get_async_weight(exponent=0.5)

        weights = []
        for client in clients:
            delta, model_weight = client.generate_local_update(
                utils.SampleNet(utils.TwoFC())
            )
            staleness = np.random.randint(1, settings.max_staleness)
            weight = async_weight.weight(num_examples=model_weight, staleness=staleness)
            assertTrue(0.0 <= weight <= 1.0)

            reducer.collect_update(delta, weight)
            weights.append(weight)

        reducer.privacy_engine._generate_noise = MagicMock(return_value=settings.noise)
        reducer.reduce()
        return reducer, weights

    def _test_weighted_avg_reduction(
        self, estimator_type, global_param, client_param, max_clip_norm
    ):
        delta = global_param - client_param
        global_model = create_ref_model(ref_model_param_value=global_param)
        num_params = self._get_num_params(global_model.fl_get_module())
        user_norm = math.sqrt(num_params * delta ** 2)

        settings = WeightReducerTestSetting(
            noise=np.random.sample(),
            clipping_value=max_clip_norm,
            max_weight=1,
            min_weight=1e-6,
            mean_weight=1e-6,
        )
        reducer, model_updates = self._reduce_weighted_models(
            global_model=global_model,
            settings=settings,
            reduction_type=ReductionType.WEIGHTED_AVERAGE,
            estimator_type=estimator_type,
            client_param=client_param,
            client_weight=10,
            global_param=global_param,
        )
        if max_clip_norm <= user_norm:
            expected_param_values = (
                delta * (settings.clipping_value / user_norm) + settings.noise
            )
        else:
            expected_param_values = delta + settings.noise

        ref_module_after_noise, _ = reducer.current_results
        return utils.model_parameters_equal_to_value(
            ref_module_after_noise, expected_param_values
        )

    def test_clipped_models_weighted_sum(self):
        """
        Test when models get clipped with weighted sum

        1. Compute the expected per user L2 Norm
        2. Set the clipping threshold to be between 1 and user norm
        3. Expected model param should be
        global = init_global - sum(clipped norms * weights) - noise
        """

        global_param = 5
        client_param = 1
        delta = global_param - client_param

        global_model = create_ref_model(ref_model_param_value=global_param)
        num_params = self._get_num_params(global_model.fl_get_module())
        user_norm = math.sqrt(num_params * delta ** 2)
        settings = WeightReducerTestSetting(
            num_clients=10,
            clients_per_round=10,
            noise=np.random.sample(),
            # clipping value is between 1 and user norm
            clipping_value=np.random.randint(1, user_norm),
        )
        reducer, weights = self._reduce_weighted_models(
            global_model=global_model,
            settings=settings,
            reduction_type=ReductionType.WEIGHTED_SUM,
            client_param=client_param,
            global_param=global_param,
        )

        clipped_deltas = math.sqrt(settings.clipping_value ** 2 / num_params)

        model_updates = sum((w * clipped_deltas for w in weights))
        expected_param_values = model_updates + settings.noise

        ref_module_after_noise, _ = reducer.current_results
        mismatched = utils.model_parameters_equal_to_value(
            ref_module_after_noise, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)

    def test_clipped_models_weighted_avg_with_biased_estimator(self):
        """
        Test when models get clipped with weighted avg with biased estimator
        where the sensitivity = (clipping_value * max_weight) / (mean_weight * num_clients_per_round)

        1. Compute the expected per user L2 Norm
        2. Set the clipping threshold to be between 1 and user norm
        3. Expected model param should be
            delta = init_global - client
            clipped_norm = min(max_clip_norm / user_norm, 1.0)
            global = delta * clipped_norms - noise
        """
        global_param = 5
        client_param = 1
        mismatched = self._test_weighted_avg_reduction(
            EstimatorType.BIASED, global_param, client_param, max_clip_norm=1
        )
        assertEqual(mismatched, "", mismatched)

    def test_clipped_models_weighted_avg_with_unbiased_estimator(self):
        """
        Test when models get clipped with weighted avg with unbiased estimator
        where the sensitivity = (clipping_value * max_weight) / (min_weight * users_per_round)

        1. Compute the expected per user L2 Norm
        2. Set the clipping threshold to be between 1 and user norm
        3. Expected model param should be
            delta = init_global - client
            clipped_norm = min(max_clip_norm / user_norm, 1.0)
            global = delta * clipped_norms - noise
        4.
        """
        global_param = 5
        client_param = 1
        mismatched = self._test_weighted_avg_reduction(
            EstimatorType.UNBIASED, global_param, client_param, max_clip_norm=1
        )
        assertEqual(mismatched, "", mismatched)

    def test_unclipped_models_weighted_avg_with_biased_estimator(self):
        """
        Test when max_clip_norm is greater than user norm with weighted avg

        When the models are unclipped then the expected global model is
            delta = init_global - client
            global = init_global - delta - noise
        """
        global_param = 5
        client_param = 1

        mismatched = self._test_weighted_avg_reduction(
            EstimatorType.BIASED, global_param, client_param, max_clip_norm=100
        )
        assertEqual(mismatched, "", mismatched)

    def test_unclipped_models_weighted_avg_with_unbiased_estimator(self):
        """
        Test when max_clip_norm is greater than user norm with weighted avg

        When the models are unclipped then the expected global model is
            delta = init_global - client
            global = init_global - delta - noise
        """
        global_param = 5
        client_param = 1

        mismatched = self._test_weighted_avg_reduction(
            EstimatorType.UNBIASED, global_param, client_param, max_clip_norm=100
        )
        assertEqual(mismatched, "", mismatched)

    def test_unclipped_models_weighted_sum(self):
        """
        Test when max_clip_norm is greater than user norm with weighted sum

        When the models are unclipped then the expected global model is
            delta = init_global - client
            global = init_global - sum(delta * weights) - noise
        """
        global_param = np.random.randint(2, 10)
        client_param = np.random.randint(1, global_param)
        delta = global_param - client_param

        global_model = create_ref_model(ref_model_param_value=global_param)
        num_params = self._get_num_params(global_model.fl_get_module())
        user_norm = math.sqrt(num_params * delta ** 2)
        settings = WeightReducerTestSetting(
            num_clients=10,
            clients_per_round=10,
            noise=np.random.sample(),
            # clipping value is greater than user norm
            clipping_value=user_norm + 1,
        )
        reducer, weights = self._reduce_weighted_models(
            global_model=global_model,
            settings=settings,
            reduction_type=ReductionType.WEIGHTED_SUM,
            client_param=client_param,
            global_param=global_param,
        )

        model_updates = sum((w * delta for w in weights))
        expected_param_values = model_updates + settings.noise
        ref_module_after_noise, _ = reducer.current_results
        mismatched = utils.model_parameters_equal_to_value(
            ref_module_after_noise, expected_param_values
        )
        assertEqual(mismatched, "", mismatched)

    def test_weighted_dp_multiprocess_same(self):
        """
        Multiprocess test for weighted DP reducer
        """
        model = utils.SampleNet(utils.TwoFC())

        # test 4 processes
        r4 = get_dp_round_reducer(
            model,
            clipping_value=1.0,
            reduction_type=ReductionType.WEIGHTED_AVERAGE,
            noise_multiplier=1,
            num_users_per_round=4,
            total_number_of_users=4,
            reset=False,
        )
        results_4 = run_reduction_test(r4, num_processes=4, num_models=4)
        same_value = results_4[0]
        for r in results_4:
            assertAlmostEqual(r, same_value, places=5)


class FLRoundReducerTest:
    @pytest.mark.parametrize(
        "config, expected_type",
        [
            (RoundReducerConfig(), RoundReducer),
        ],
    )
    def test_reducer_creation_from_config(self, config: Type, expected_type: Type):
        ref_model = utils.SampleNet(utils.TwoFC())
        reducer = instantiate(config, global_model=ref_model)
        assertIsInstance(reducer, expected_type)
