#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from flsim.channels.message import Message
from flsim.clients.base_client import Client, ClientConfig
from flsim.clients.dp_client import DPClient, DPClientConfig
from flsim.clients.sync_fedshuffle_client import (
    FedShuffleClient,
    FedShuffleClientConfig,
)
from flsim.clients.sync_mime_client import MimeClient, MimeClientConfig
from flsim.clients.sync_mimelite_client import MimeLiteClient, MimeLiteClientConfig
from flsim.common.pytest_helper import (
    assertAlmostEqual,
    assertEmpty,
    assertEqual,
    assertFalse,
    assertIsInstance,
    assertNotEqual,
    assertTrue,
)
from flsim.common.timeout_simulator import (
    GaussianTimeOutSimulator,
    GaussianTimeOutSimulatorConfig,
    NeverTimeOutSimulator,
    NeverTimeOutSimulatorConfig,
)
from flsim.data.data_provider import IFLUserData
from flsim.optimizers.local_optimizers import (
    LocalOptimizerFedProxConfig,
    LocalOptimizerSGD,
    LocalOptimizerSGDConfig,
)
from flsim.optimizers.optimizer_scheduler import (
    ConstantLRScheduler,
    ConstantLRSchedulerConfig,
)
from flsim.privacy.common import ClippingSetting, PrivacySetting
from flsim.utils import test_utils as utils
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
)
from omegaconf import OmegaConf
from opacus.accountants.analysis import rdp as privacy_analysis
from opacus.optimizers import DPOptimizer


def calc_eps(
    sample_rate: float, noise_multiplier: float, steps: int, alphas, delta: float
) -> float:
    rdp = privacy_analysis.compute_rdp(
        q=sample_rate, noise_multiplier=noise_multiplier, steps=steps, orders=alphas
    )
    eps, _ = privacy_analysis.get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)
    return eps


@pytest.fixture(scope="class")
def prepare_client_test(request) -> None:
    request.cls.num_batches = 2
    request.cls.batch_size = 10


@pytest.mark.usefixtures("prepare_client_test")
class ClientTestBase:
    def _fake_data(self, num_batches=None, batch_size=None):
        num_batches = self.num_batches if num_batches is None else num_batches
        batch_size = batch_size or self.batch_size
        torch.manual_seed(0)
        dataset = [torch.rand(batch_size, 2) for _ in range(num_batches)]
        dataset = utils.DatasetFromList(dataset)
        return utils.DummyUserData(dataset, utils.SampleNet(utils.TwoFC()))

    def _get_client(
        self,
        data=None,
        store_models_and_optimizers: bool = False,
        timeout_simulator=None,
    ):
        data = data or self._fake_data()
        config = ClientConfig(
            store_models_and_optimizers=store_models_and_optimizers,
            lr_scheduler=ConstantLRSchedulerConfig(),
        )
        return Client(
            **OmegaConf.structured(config),
            dataset=data,
            timeout_simulator=timeout_simulator,
        )

    def _get_dp_client(
        self,
        data=None,
        noise_multiplier: int = 1,
        clipping_value: int = 1,
        store_models_and_optimizers: bool = False,
    ):
        privacy_setting = PrivacySetting(
            noise_multiplier=noise_multiplier,
            clipping=ClippingSetting(clipping_value=clipping_value),
        )
        config = DPClientConfig(
            store_models_and_optimizers=store_models_and_optimizers,
            privacy_setting=privacy_setting,
        )
        return DPClient(
            **OmegaConf.structured(config), dataset=(data or self._fake_data())
        )

    def _get_mime_client(
        self,
        data=None,
        store_models_and_optimizers: bool = False,
        timeout_simulator=None,
    ):
        data = data or self._fake_data()
        config = MimeClientConfig(
            store_models_and_optimizers=store_models_and_optimizers,
            lr_scheduler=ConstantLRSchedulerConfig(),
        )
        return MimeClient(
            **OmegaConf.structured(config),
            dataset=data,
            timeout_simulator=timeout_simulator,
        )

    def _get_mimelite_client(
        self,
        data=None,
        store_models_and_optimizers: bool = False,
        timeout_simulator=None,
    ):
        data = data or self._fake_data()
        config = MimeLiteClientConfig(
            store_models_and_optimizers=store_models_and_optimizers,
            lr_scheduler=ConstantLRSchedulerConfig(),
        )
        return MimeLiteClient(
            **OmegaConf.structured(config),
            dataset=data,
            timeout_simulator=timeout_simulator,
        )

    def _train(self, data: IFLUserData, model, optim) -> None:
        # basically re-write training logic
        model.fl_get_module().train()
        for batch in data.train_data():
            optim.zero_grad()
            _batch = model.fl_create_training_batch(batch)
            loss = model.fl_forward(_batch).loss
            loss.backward()
            optim.step()

    def _run_client_eval_test(self, client) -> None:
        """
        Test client eval will turn on eval mode and turn back into train mode
        after evaluation loop is finished
        """

        class Net(utils.SampleNet):
            def get_eval_metrics(self, batch):
                assert (
                    self.sample_nn.training is False
                ), "Client should call eval after setting model.eval()"
                return self.sample_nn(batch)

        input_dim = 2
        model = Net(nn.Linear(input_dim, 1))

        model.fl_get_module().train()
        client.eval(model=model)
        assert model.fl_get_module().training

    def _test_reload_server_state(self, client):
        model = utils.SampleNet(utils.TwoFC())
        base_optim = torch.optim.Adam(model.fl_get_module().parameters())

        # create a state in the optimizer by performing backprop
        for batch in client.dataset.train_data():
            base_optim.zero_grad()
            model.fl_forward(batch).loss.backward()
            base_optim.step()

        # check if the optimizer state is loaded
        client.server_opt_state = base_optim.state_dict()["state"]
        optim = torch.optim.Adam(model.fl_get_module().parameters())
        client._reload_server_state(optim)
        error_msg = utils.verify_optimizer_state_dict_equal(
            optim.state_dict()["state"], base_optim.state_dict()["state"]
        )
        assertEmpty(error_msg)

        # check if passing empty optimizer state works
        # used for the first round, at the start of training
        # since server has no optimizer state
        client.server_opt_state = {}
        optim = torch.optim.Adam(model.fl_get_module().parameters())
        client._reload_server_state(optim)
        assertEqual(optim.state_dict()["state"], {})


class TestBaseClient(ClientTestBase):
    def test_storage(self) -> None:
        client = self._get_client(store_models_and_optimizers=True)
        model0 = utils.SampleNet(utils.TwoFC())
        model1 = utils.SampleNet(utils.TwoFC())
        delta1, weight1 = client.generate_local_update(Message(model0))
        delta1 = FLModelParamUtils.clone(delta1)

        delta2, weight2 = client.generate_local_update(Message(model1))
        assertEqual(client.times_selected, 2)
        # model1 should be the first model stored
        assertAlmostEqual(weight1, client.weights[0])
        mismatched = utils.verify_models_equivalent_after_training(
            delta1, client.model_deltas[0]
        )
        assertEqual(mismatched, "", mismatched)
        # model2 should be second model stored
        assertAlmostEqual(weight2, client.weights[1])
        mismatched = utils.verify_models_equivalent_after_training(
            delta2, client.model_deltas[1]
        )
        assertEqual(mismatched, "", mismatched)

    def test_receive_through_channel(self) -> None:
        # expected channel effects,
        clnt = self._get_client()
        model = utils.SampleNet(utils.TwoFC())
        # check channel is pass through
        model2 = clnt.receive_through_channel(model)
        mismatched = utils.verify_models_equivalent_after_training(model2, model)
        assertEqual(mismatched, "", mismatched)

    def test_prepare_for_training(self) -> None:
        clnt = self._get_client()
        model = utils.SampleNet(utils.TwoFC())
        try:
            # should work
            model2, optim, optim_sch = clnt.prepare_for_training(model)
        except BaseException as e:
            assertTrue(False, e)
        # pyre-fixme[61]: `model2` is undefined, or not always defined.
        mismatched = utils.verify_models_equivalent_after_training(model2, model)
        assertEqual(mismatched, "", mismatched)
        # expect correct type of optimizer
        # pyre-fixme[61]: `optim` is undefined, or not always defined.
        assertIsInstance(optim, LocalOptimizerSGD)
        # pyre-fixme[61]: `optim_sch` is undefined, or not always defined.
        assertIsInstance(optim_sch, ConstantLRScheduler)

    def test_train(self) -> None:
        data = self._fake_data(num_batches=5, batch_size=10)
        clnt = self._get_client(data)
        model = utils.SampleNet(utils.TwoFC())
        model, optim, optim_sch = clnt.prepare_for_training(model)
        model2, optim2, _ = clnt.prepare_for_training(FLModelParamUtils.clone(model))
        # value chekd in previous test
        try:
            # should work
            model, weight = clnt.train(model, optim, optim_sch, None)
        except BaseException as e:
            assertTrue(False, e)
        # pyre-fixme[61]: `weight` is undefined, or not always defined.
        assertAlmostEqual(weight, 5 * 10)
        self._train(data, model2, optim2)
        mismatched = utils.verify_models_equivalent_after_training(model2, model)
        assertEqual(mismatched, "", mismatched)

    def test_generate_local_update(self) -> None:
        clnt = self._get_client()
        model = utils.SampleNet(utils.TwoFC())
        model.fl_get_module().fill_all(0.1)
        clnt.train = MagicMock(return_value=(model, 12.34))
        clnt.compute_delta = MagicMock(return_value=model)

        try:
            # should work
            delta, weight = clnt.generate_local_update(Message(model))
        except BaseException as e:
            assertTrue(False, e)
        # pyre-fixme[61]: `weight` is undefined, or not always defined.
        assertAlmostEqual(12.34, weight)
        mismatched = utils.verify_models_equivalent_after_training(model, model)
        assertEqual(mismatched, "", mismatched)
        # pyre-fixme[61]: `delta` is undefined, or not always defined.
        mismatched = utils.verify_models_equivalent_after_training(delta, model)
        assertEqual(mismatched, "", mismatched)

    def test_fed_prox_sgd_equivalent(self) -> None:
        """
        Test FedProx under the following scenarios:

        FedProx == SGD iff
        1. FedProx with mu = 0 == SGD
        2. FedProx with mu = x == SGD(weight_decay=mu)

        FedProx != SGD if
        1. mu > 0 and SGD(weight_decay=0)
        """
        # scenario 1
        data = self._fake_data()
        prox_client = Client(
            dataset=data,
            **OmegaConf.structured(
                ClientConfig(
                    optimizer=LocalOptimizerFedProxConfig(mu=0),
                )
            ),
        )
        client = Client(
            dataset=data,
            **OmegaConf.structured(
                ClientConfig(
                    optimizer=LocalOptimizerSGDConfig(),
                )
            ),
        )

        init_model = utils.SampleNet(utils.TwoFC())
        delta, weight = client.generate_local_update(
            Message(FLModelParamUtils.clone(init_model))
        )
        prox_delta, weight = prox_client.generate_local_update(
            Message(FLModelParamUtils.clone(init_model))
        )
        mismatched = utils.verify_models_equivalent_after_training(
            prox_delta, delta, init_model
        )
        assertEqual(mismatched, "", mismatched)

        # scenario 2
        init_model = utils.SampleNet(utils.TwoFC())
        init_model.fl_get_module().fill_all(0.0)
        mu = 1.0
        prox_client = Client(
            dataset=data,
            **OmegaConf.structured(
                ClientConfig(
                    optimizer=LocalOptimizerFedProxConfig(mu=mu),
                )
            ),
        )
        client = Client(
            dataset=data,
            **OmegaConf.structured(
                ClientConfig(
                    optimizer=LocalOptimizerSGDConfig(weight_decay=mu),
                )
            ),
        )

        delta, _ = client.generate_local_update(
            Message(FLModelParamUtils.clone(init_model))
        )
        prox_delta, _ = prox_client.generate_local_update(
            Message(FLModelParamUtils.clone(init_model))
        )

        mismatched = utils.verify_models_equivalent_after_training(
            prox_delta, delta, init_model
        )
        assertEqual(mismatched, "", mismatched)

        # negative case
        # FedProx != SGD if mu > 0 and SGD has no weight decay
        init_model = utils.SampleNet(utils.TwoFC())
        init_model.fl_get_module().fill_all(0.0)
        mu = 1.0
        prox_client = Client(
            dataset=data,
            **OmegaConf.structured(
                ClientConfig(
                    optimizer=LocalOptimizerFedProxConfig(mu=mu),
                )
            ),
        )
        client = Client(
            dataset=data,
            **OmegaConf.structured(
                ClientConfig(
                    optimizer=LocalOptimizerSGDConfig(weight_decay=0),
                )
            ),
        )

        delta, _ = client.generate_local_update(
            Message(FLModelParamUtils.clone(init_model))
        )
        prox_delta, _ = prox_client.generate_local_update(
            Message(FLModelParamUtils.clone(init_model))
        )

        mismatched = utils.verify_models_equivalent_after_training(
            prox_delta, delta, init_model
        )
        assertNotEqual(mismatched, "", mismatched)

    def test_device_perf_generation(self) -> None:
        """
        test either client.device_perf is always generated,
        either using the TimeOutSimulator given in __init__
        or creates a NeverTimeOutSimulator( if none is provided.
        """
        data = self._fake_data(num_batches=5, batch_size=10)
        # pass gaussian timeout into client
        cfg = GaussianTimeOutSimulatorConfig(
            timeout_wall_per_round=1.0,
            fl_stopping_time=1.0,
            duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                training_duration_mean=1.0, training_duration_sd=0.0
            ),
        )
        gaussian_timeout_simulator = GaussianTimeOutSimulator(
            **OmegaConf.structured(cfg)
        )
        clnt_gaussian_timeout = self._get_client(
            data, timeout_simulator=gaussian_timeout_simulator
        )
        assertEqual(clnt_gaussian_timeout.per_example_training_time, 1.0)
        # pass never timeout to clients
        clnt_never_timeout = self._get_client(
            data,
            timeout_simulator=NeverTimeOutSimulator(
                **OmegaConf.structured(NeverTimeOutSimulatorConfig())
            ),
        )
        assertEqual(clnt_never_timeout.per_example_training_time, 0.0)
        # default created never timeout within clients
        clnt_default = self._get_client(data)
        assertEqual(clnt_default.per_example_training_time, 0.0)

    def test_total_training_time(self) -> None:
        """
        total training time for Gaussian with mean 1.0 and std 0.0
        equals to min(number of client examples, timeout_wall_per_round)
        """
        data = self._fake_data(num_batches=5, batch_size=10)
        # pass gaussian timeout into client with a big timeout wall
        cfg_big_wall = GaussianTimeOutSimulatorConfig(
            timeout_wall_per_round=99999.0,
            fl_stopping_time=1.0,
            duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                training_duration_mean=1.0, training_duration_sd=0.0
            ),
        )
        gaussian_timeout_simulator = GaussianTimeOutSimulator(
            **OmegaConf.structured(cfg_big_wall)
        )
        clnt_gaussian_timeout = self._get_client(
            data, timeout_simulator=gaussian_timeout_simulator
        )
        num_examples = 5 * 10
        assertEqual(clnt_gaussian_timeout.get_total_training_time(), num_examples)
        # pass gaussian timeout into client with a small timeout wall
        timeout_wall = 25.0
        cfg_small_wall = GaussianTimeOutSimulatorConfig(
            timeout_wall_per_round=timeout_wall,
            fl_stopping_time=1.0,
            duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                training_duration_mean=1.0, training_duration_sd=0.0
            ),
        )
        gaussian_timeout_simulator = GaussianTimeOutSimulator(
            **OmegaConf.structured(cfg_small_wall)
        )
        clnt_gaussian_timeout = self._get_client(
            data, timeout_simulator=gaussian_timeout_simulator
        )
        assertEqual(clnt_gaussian_timeout.get_total_training_time(), timeout_wall)
        # pass never timeout to clients
        clnt_never_timeout = self._get_client(
            data,
            timeout_simulator=NeverTimeOutSimulator(
                **OmegaConf.structured(NeverTimeOutSimulatorConfig())
            ),
        )
        assertEqual(clnt_never_timeout.get_total_training_time(), 0.0)

    def test_partial_training(self) -> None:
        """
        producing two training instance expected same training results:
        1. client with n (even number) batches and a time out wall just
        enough for training half of the batches
        2. client with n/2 batches and no time out wall

        check model1 == model2
        """
        n_batches = 2
        bs = 10
        data = self._fake_data(num_batches=n_batches, batch_size=bs)
        # only feasible to process 3 batches, client sends back partial results
        expected_processed_samples = n_batches * bs / 2
        cfg = GaussianTimeOutSimulatorConfig(
            timeout_wall_per_round=expected_processed_samples,
            fl_stopping_time=1.0,
            duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                training_duration_mean=1.0, training_duration_sd=0.0
            ),
        )
        gaussian_timeout_simulator = GaussianTimeOutSimulator(
            **OmegaConf.structured(cfg)
        )
        clnt_gaussian_timeout = self._get_client(
            data, timeout_simulator=gaussian_timeout_simulator
        )
        torch.manual_seed(0)
        model_init = utils.SampleNet(utils.TwoFC())
        model1, optim1, optim_sch1 = clnt_gaussian_timeout.prepare_for_training(
            FLModelParamUtils.clone(model_init)
        )
        torch.manual_seed(0)
        partial_model, partial_weight = clnt_gaussian_timeout.train(
            model1, optim1, optim_sch1, None
        )
        assertEqual(partial_weight, expected_processed_samples)

        # no timeout, but client only has half of the data
        n_batches = int(n_batches / 2)
        data = self._fake_data(num_batches=n_batches, batch_size=bs)
        clnt = self._get_client(data)
        model2, optim2, optim_sch2 = clnt.prepare_for_training(
            FLModelParamUtils.clone(model_init)
        )
        torch.manual_seed(0)
        full_model, full_weight = clnt.train(model2, optim2, optim_sch2, None)
        assertEqual(full_weight, expected_processed_samples)

        mismatched = utils.verify_models_equivalent_after_training(
            full_model, partial_model
        )
        assertEqual(mismatched, "", mismatched)

    def test_logging_level(self) -> None:
        clnt = self._get_client()
        assertTrue(utils.check_inherit_logging_level(clnt, 50))
        assertTrue(utils.check_inherit_logging_level(clnt, 10))

    def test_base_client_eval(self) -> None:
        client = self._get_client()
        self._run_client_eval_test(client)

    def test_full_dataset_gradient(self):
        """Test whether average gradient over the training dataset works correctly"""
        data = self._fake_data(num_batches=3, batch_size=5)
        clnt = self._get_client(data)
        model = utils.SampleNet(utils.create_model_with_value(3))

        grads, num_examples = clnt.full_dataset_gradient(model)
        expected_grads = utils.TwoFC()

        expected_grads.fc1.weight.grad = torch.tensor([1.19955, 1.59725]).repeat(5, 1)
        expected_grads.fc2.weight.grad = torch.tensor([5.79683]).repeat(1, 5)
        expected_grads.fc1.bias.grad = torch.tensor([3.0]).repeat(5)
        expected_grads.fc2.bias.grad = torch.tensor([1.0])

        error_msg = utils.verify_gradients_equal(grads, expected_grads)
        assertEmpty(error_msg)

        data = self._fake_data(num_batches=0, batch_size=10)
        assertEqual(data.num_train_examples(), 0)
        clnt = self._get_client(data)

        try:
            grads, num_examples = clnt.full_dataset_gradient(model)
        except AssertionError:
            pass
        else:
            assert "full_dataset_gradient must throw an assertion error if called with empty data"


class TestDPClient(ClientTestBase):
    def test_privacy_engine_properly_initialized(self) -> None:
        data = self._fake_data(num_batches=11, batch_size=3)
        # pyre-fixme[6]: Expected `int` for 2nd param but got `float`.
        # pyre-fixme[6]: Expected `int` for 3rd param but got `float`.
        clnt = self._get_dp_client(data, noise_multiplier=0.1, clipping_value=2.0)
        model = utils.SampleNet(utils.TwoFC())
        model, optim, optim_sch = clnt.prepare_for_training(model)
        assertIsInstance(optim, DPOptimizer)
        assertEqual(optim.noise_multiplier, 0.1)
        assertEqual(optim.max_grad_norm, 2.0)
        assertEqual(optim.expected_batch_size, 3)

    def test_privacy_turned_off(self) -> None:
        data = self._fake_data(num_batches=11, batch_size=3)
        # clipping value of inf means privacy is off no matter what the noise multiplier
        clnt = self._get_dp_client(
            data,
            # pyre-fixme[6]: Expected `int` for 2nd param but got `float`.
            noise_multiplier=0.0,
            # pyre-fixme[6]: Expected `int` for 3rd param but got `float`.
            clipping_value=float("inf"),
        )
        assertFalse(clnt.privacy_on)
        clnt = self._get_dp_client(
            data,
            # pyre-fixme[6]: Expected `int` for 2nd param but got `float`.
            noise_multiplier=1.0,
            # pyre-fixme[6]: Expected `int` for 3rd param but got `float`.
            clipping_value=float("inf"),
        )
        assertFalse(clnt.privacy_on)
        # negative noise multiplier should turn of privacy engine
        # pyre-fixme[6]: Expected `int` for 2nd param but got `float`.
        # pyre-fixme[6]: Expected `int` for 3rd param but got `float`.
        clnt = self._get_dp_client(data, noise_multiplier=-1.0, clipping_value=0.1)
        assertFalse(clnt.privacy_on)

    def test_prepare_for_training(self) -> None:
        clnt = self._get_dp_client()
        model = utils.SampleNet(utils.TwoFC())
        model2, optim, optim_sch = clnt.prepare_for_training(model)
        mismatched = utils.verify_models_equivalent_after_training(model2, model)
        assertEqual(mismatched, "")
        # expect correct type of optimizer
        assertIsInstance(optim, DPOptimizer)
        assertIsInstance(optim.original_optimizer, LocalOptimizerSGD)
        assertIsInstance(optim_sch, ConstantLRScheduler)

    def test_storage(self) -> None:
        client = self._get_dp_client(store_models_and_optimizers=True)
        model0 = utils.SampleNet(utils.TwoFC())
        delta, weight1 = client.generate_local_update(Message(model0))

        assertEqual(client.times_selected, 1)
        # test existence of privacy_engine
        # model1 should be the first model stored
        optim = client.optimizers[0]
        assertIsInstance(optim, DPOptimizer)

    def test_no_noise_no_clip(self) -> None:
        data = self._fake_data(3, 4)
        model = utils.SampleNet(utils.TwoFC())
        private_model = FLModelParamUtils.clone(model)

        clnt = self._get_client(data)
        delta, weight = clnt.generate_local_update(Message(model))
        # set noise to 0 and clipping to a large number
        private_clnt = self._get_dp_client(
            data, noise_multiplier=0, clipping_value=1000
        )

        private_delta, private_weight = private_clnt.generate_local_update(
            Message(private_model)
        )
        mismatched = utils.verify_models_equivalent_after_training(model, private_model)
        mismatched_delta = utils.verify_models_equivalent_after_training(
            delta, private_delta
        )
        assertAlmostEqual(weight, private_weight)
        assertEqual(mismatched, "", mismatched)
        assertEqual(mismatched_delta, "", mismatched_delta)

    def test_only_clip(self) -> None:
        data = self._fake_data(4, 4)
        model = utils.SampleNet(utils.TwoFC())
        private_model = FLModelParamUtils.clone(model)

        clnt = self._get_client(data)
        delta, weight = clnt.generate_local_update(Message(model))
        private_clnt = self._get_dp_client(
            data,
            noise_multiplier=0,
            # pyre-fixme[6]: Expected `int` for 3rd param but got `float`.
            clipping_value=0.01,
        )

        private_delta, private_weight = private_clnt.generate_local_update(
            Message(private_model)
        )
        mismatched = utils.verify_models_equivalent_after_training(delta, private_delta)
        assertAlmostEqual(weight, private_weight)
        assertNotEqual(mismatched, "")

    def test_noise_and_clip(self) -> None:
        data = self._fake_data(4, 4)
        model = utils.SampleNet(utils.TwoFC())
        private_model = FLModelParamUtils.clone(model)

        clnt = self._get_client(data)
        delta, weight = clnt.generate_local_update(Message(model))
        private_clnt = self._get_dp_client(
            data,
            noise_multiplier=1,
            # pyre-fixme[6]: Expected `int` for 3rd param but got `float`.
            clipping_value=0.01,
        )
        private_delta, private_weight = private_clnt.generate_local_update(
            Message(private_model)
        )
        mismatched_delta = utils.verify_models_equivalent_after_training(
            delta, private_delta
        )
        assertAlmostEqual(weight, private_weight)
        assertNotEqual(mismatched_delta, "")

    def test_epsilon(self) -> None:
        noise_multiplier = 1.5
        clipping_value = 0.01
        num_batches = 5
        batch_size = 6
        data = self._fake_data(num_batches, batch_size)
        model = utils.SampleNet(utils.TwoFC())
        # pyre-fixme[6]: Expected `int` for 2nd param but got `float`.
        # pyre-fixme[6]: Expected `int` for 3rd param but got `float`.
        clnt = self._get_dp_client(data, noise_multiplier, clipping_value, True)
        model, weight = clnt.generate_local_update(Message(model))

        alphas = clnt.accountant.DEFAULT_ALPHAS
        delta = 1e-5

        eps_from_script = calc_eps(
            1.0 / num_batches, noise_multiplier, num_batches, alphas, delta
        )
        eps_from_client, _ = clnt.accountant.get_privacy_spent(
            delta=delta, alphas=alphas
        )
        assertAlmostEqual(eps_from_script, eps_from_client)

    def test_dp_client_eval(self) -> None:
        dp_client = self._get_dp_client()
        self._run_client_eval_test(dp_client)


class TestMimeClient(ClientTestBase):
    def test_reload_server_state(self):
        """Test whether server optimizer state is loading correctly in MIMEClient"""
        data = self._fake_data(num_batches=1, batch_size=10)
        clnt = self._get_mime_client(data)
        self._test_reload_server_state(clnt)

    def test_mime_generate_local_update(self):
        """The only change from base_client is to load mime_control_variate and
        server_opt_state in client, thus verifying the same
        """
        data = self._fake_data(num_batches=5, batch_size=10)
        clnt = self._get_mime_client(data)

        model = utils.SampleNet(utils.TwoFC())
        mime_control_variate, _ = clnt.full_dataset_gradient(model)
        server_opt_state = {}

        # Return value previously tested with base_client
        _ = clnt.generate_local_update(
            message=Message(
                model=model,
                mime_control_variate=mime_control_variate,
                server_opt_state=server_opt_state,
            )
        )

        # Verify server model and optimizer state did not change in MIMELite Client
        error_msg = utils.verify_models_equivalent_after_training(clnt.ref_model, model)
        assertEmpty(error_msg)
        error_msg = utils.verify_gradients_equal(
            clnt.mime_control_variate, mime_control_variate
        )
        assertEmpty(error_msg)
        error_msg = utils.verify_optimizer_state_dict_equal(
            clnt.server_opt_state, server_opt_state
        )
        assertEmpty(error_msg)


class TestMimeLiteClient(TestBaseClient):
    def test_reload_server_state(self):
        """Test whether server optimizer state is loaded correctly in MIMELite client"""
        data = self._fake_data(num_batches=5, batch_size=10)
        clnt = self._get_mimelite_client(data)
        self._test_reload_server_state(clnt)

    def test_mimelite_generate_local_update(self):
        """The only change from base_client is to load server_opt_state
        in client, thus verifying the same
        """
        data = self._fake_data(num_batches=5, batch_size=10)
        clnt = self._get_mimelite_client(data)

        model = utils.SampleNet(utils.TwoFC())
        server_opt_state = {}

        # Return value previously tested with base_client
        _ = clnt.generate_local_update(
            message=Message(
                model=model,
                server_opt_state=server_opt_state,
            )
        )

        # Verify server model and optimizer state did not change in MIMELite Client
        error_msg = utils.verify_models_equivalent_after_training(clnt.ref_model, model)
        assertEmpty(error_msg)

        error_msg = utils.verify_optimizer_state_dict_equal(
            clnt.server_opt_state, server_opt_state
        )
        assertEmpty(error_msg)


class TestFedShuffleClient(TestBaseClient):
    def test_fedshuffle_generate_local_update(self):
        # Verify that the deltas match with expected value
        model = utils.SampleNet(utils.linear_model(4.0))
        data = self._fake_data(3, 2)
        config = FedShuffleClientConfig(
            optimizer=LocalOptimizerSGDConfig(lr=0.2),
            shuffle_batch_order=False,
            epochs=2,
        )
        client = FedShuffleClient(**OmegaConf.structured(config), dataset=data)
        server_message = Message(model=model)
        delta, weight = client.generate_local_update(server_message)

        # Value obtained on FedShuffle's official implementation for same client data
        expected_delta = utils.linear_model(0.0)
        expected_delta.fc1.weight = nn.Parameter(
            torch.tensor([[0.07289219, 0.11549282]])
        )
        expected_delta.fc1.bias = nn.Parameter(torch.tensor([0.19999981]))

        error_msg = utils.verify_models_equivalent_after_training(
            delta, expected_delta, model
        )
        assertEmpty(error_msg, error_msg)

        # Verify that client does not change any part of the server_message
        expected_model = utils.SampleNet(utils.linear_model(4.0))
        error_msg = utils.verify_models_equivalent_after_training(
            server_message.model, expected_model
        )
        assertEmpty(error_msg, error_msg)
