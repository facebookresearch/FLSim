#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.


import copy
import unittest
from dataclasses import dataclass
from functools import partial
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F
from flsim.experimental.fel import utils
from flsim.experimental.fel.ensemble import (
    BasicRigidNNEnsemble,
    BasicArchEnsemble,
    ProcessingType,
)
from flsim.experimental.fel.interfaces import EnsembleType
from flsim.experimental.fel.sub_model import BasicNNSubModel


class TestUtils(unittest.TestCase):
    def assertTensorsEqual(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(torch.allclose(a, b), f"Tensor values are not equal: {a} {b}")

    def assertTensorsDifferent(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertFalse(
            torch.allclose(a, b, atol=0.01), f"Tensor values are equal: {a} {b}"
        )

    def generate_random_input(self, batch_size=None):
        return torch.rand(batch_size or torch.randint(1, 10, (1,)).item(), 1, 28, 28)

    def sample_model(self):
        class SampleModule(nn.Module):
            @dataclass
            class Results:
                conv2: torch.Tensor = torch.rand(1)
                fc1_relu: torch.Tensor = torch.rand(1)
                out: torch.Tensor = torch.rand(1)

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
                self.conv2 = nn.Conv2d(16, 32, 4, 2)
                self.fc1 = nn.Linear(32 * 4 * 4, 32)
                self.fc1_relu = nn.ReLU()
                self.fc2 = nn.Linear(32, 10)

            def forward(self, x):  # x of shape [B, 1, 28, 28]
                result = SampleModule.Results()
                x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
                x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
                x = self.conv2(x)  # -> [B, 32, 5, 5]
                result.conv2 = x.detach().clone()
                x = F.relu(x)  # -> [B, 32, 5, 5]
                x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
                x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
                x = self.fc1(x)  # -> [B, 32]
                x = self.fc1_relu(x)  # -> [B, 32]
                result.fc1_relu = x.detach().clone()
                x = self.fc2(x)  # -> [B, 10]
                result.out = x
                return result

        return SampleModule()

    def sample_sub_model_list(self, batch_size, num_models, num_features, mock=True):
        model = self.sample_model()
        sub_models = [
            BasicNNSubModel(copy.deepcopy(model), feature_layer="fc2")
            for _ in range(num_models)
        ]

        def fill(x, value):
            return torch.ones(batch_size, num_features) * value

        if mock:
            for i, model in enumerate(sub_models):
                model.feature = MagicMock(side_effect=partial(fill, value=i))
        return sub_models

    def sample_arch(self, takes_hint, num_models, model_output_size):
        class Arch(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(
                    num_models * (model_output_size + int(takes_hint)),
                    model_output_size,
                )
                # L + M X F if takes no hint, L + M X (F + 1) if takes hints
                # outputs F

            def forward(self, x):
                features, inputs, hint = x  # inputs are ignored for simplicity
                features = utils.multiple_batches_to_one(features, merge=True)
                if takes_hint:
                    assert hint is not None
                    features = utils.append_batches(hint, features)

                return self.linear(features)

        return Arch()


class UtilsTest(TestUtils):
    def test_to_onehot(self):
        x = [1, 2, 3]
        num_classes = 4
        expected = torch.Tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # works with list
        self.assertTensorsEqual(expected, utils.to_onehot(x, num_classes))
        # works with tensor
        x = torch.as_tensor(x)
        self.assertTensorsEqual(expected, utils.to_onehot(x, num_classes))
        # works with extra dim of size 1
        x = x.view(-1, 1)
        self.assertTensorsEqual(expected, utils.to_onehot(x, num_classes))
        # does not work with many dimensions
        x = torch.rand(5, 4, 3)
        with self.assertRaises(AssertionError):
            utils.to_onehot(x, 10)

    def test_multiple_batches_to_one_stack(self):
        x = torch.Tensor(
            [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]]
        )
        expected_merged = torch.Tensor(
            [[1.0, 1.0, 4.0, 4.0], [2.0, 2.0, 5.0, 5.0], [3.0, 3.0, 6.0, 6.0]]
        )
        expected_unmerged = torch.Tensor(
            [
                [[1.0, 1.0], [4.0, 4.0]],
                [[2.0, 2.0], [5.0, 5.0]],
                [[3.0, 3.0], [6.0, 6.0]],
            ]
        )
        self.assertTensorsEqual(
            expected_merged, utils.multiple_batches_to_one(x, merge=True)
        )
        self.assertTensorsEqual(
            expected_unmerged, utils.multiple_batches_to_one(x, merge=False)
        )

        # more dimensions
        x = torch.rand(3, 4, 5, 6, 7)
        y = utils.multiple_batches_to_one(x, merge=False)
        self.assertEqual(list(y.shape), [4, 3, 5, 6, 7])
        y = utils.multiple_batches_to_one(x, merge=True)
        self.assertEqual(list(y.shape), [4, 15, 6, 7])

    def test_multiple_batches_to_one_list(self):
        x = [
            torch.Tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            torch.Tensor([[4.0], [5.0], [6.0]]),
        ]
        expected_merged = torch.Tensor(
            [[1.0, 1.0, 4.0], [2.0, 2.0, 5.0], [3.0, 3.0, 6.0]]
        )
        self.assertTensorsEqual(
            expected_merged, utils.multiple_batches_to_one(x, merge=True)
        )

        # should not allow the no-merge option for list
        with self.assertRaises(AssertionError):
            utils.multiple_batches_to_one(x, merge=False)

    def test_append_batches(self):
        x = torch.rand(3, 5)
        y = torch.rand(3, 10)
        z = utils.append_batches(x, y)

        self.assertTensorsEqual(x, z[:, 0:5])
        self.assertTensorsEqual(y, z[:, 5:])

        # non matching batch size
        with self.assertRaises(AssertionError):
            utils.append_batches(torch.ones(5, 4), torch.zeros(2, 1))

        # feaure > 1D
        with self.assertRaises(AssertionError):
            utils.append_batches(torch.ones(5, 4, 2), torch.zeros(5, 4))


class BasicSubmodelTest(TestUtils):
    def setUp(self):
        self.model = self.sample_model()

    def test_standalone_forward_works(self):
        sub_model = BasicNNSubModel(self.model)
        x = self.generate_random_input()
        out1 = self.model(x)
        out2 = sub_model(x)
        self.assertTensorsEqual(out1.out, out2.out)

    def test_standalone_backward_works(self):
        sub_model = BasicNNSubModel(self.model)
        x = self.generate_random_input()
        sub_model(x).out.sum().backward()  # should not throw

    def test_default_output_match(self):
        sub_model = BasicNNSubModel(self.model)
        x = self.generate_random_input()
        result = self.model(x)
        y = sub_model.feature(x)
        self.assertTensorsEqual(result.out, y.out)

    def test_deep_copy_works(self):
        tmp = BasicNNSubModel(self.model)
        x = self.generate_random_input()
        result = self.model(x)
        # let's also see potential cross interactions on the hook
        _ = tmp.feature(x)
        sub_model = copy.deepcopy(tmp)
        y = sub_model.feature(x)
        self.assertTensorsEqual(result.out, y.out)

    def test_default_output_match_no_input_feature(self):
        sub_model = BasicNNSubModel(self.model)
        x = self.generate_random_input()
        result = self.model(x)
        sub_model(x)
        y = sub_model.feature()
        self.assertTensorsEqual(result.out, y.out)

    def test_mid_layer_1(self):
        sub_model = BasicNNSubModel(self.model, feature_layer="conv2")
        x = self.generate_random_input()
        result = self.model(x)
        y = sub_model.feature(x)  # will be a tensor type
        self.assertTensorsEqual(result.conv2, y)

    def test_mid_layer_2(self):
        sub_model = BasicNNSubModel(self.model, feature_layer="fc1_relu")
        x = self.generate_random_input()
        result = self.model(x)
        y = sub_model.feature(x)
        self.assertTensorsEqual(result.fc1_relu, y)


class BasicRigidEnsembleTest(TestUtils):
    def setUp(self):
        self.batch_size = 8
        self.num_models = 10
        self.num_features = 2
        self.sub_models = self.sample_sub_model_list(
            self.batch_size, self.num_models, self.num_features
        )

    def _expected(self, value):
        return torch.ones(self.batch_size, self.num_features) * value

    def test_models_included(self):
        ensemble = BasicRigidNNEnsemble(self.sub_models)
        self.assertEqual(ensemble.type(), EnsembleType.RIGID)
        for i, (original, wrapped) in enumerate(
            zip(self.sub_models, ensemble.sub_models())
        ):
            self.assertTrue(original is wrapped[1])
            self.assertEqual(f"{i}", wrapped[0])

    def test_backward_fails(self):
        ensemble = BasicRigidNNEnsemble(self.sub_models)
        x = self.generate_random_input(self.batch_size)
        out = ensemble(x)
        with self.assertRaises(RuntimeError):
            out.sum().backward()

    def test_predictions_match(self):
        ensemble = BasicRigidNNEnsemble(self.sub_models)
        x = self.generate_random_input(self.batch_size)
        for i, prediction in enumerate(ensemble.sub_model_features(x)):
            self.assertTensorsEqual(prediction, self._expected(i))

    def test_mean_med_max_match(self):
        x = self.generate_random_input(self.batch_size)
        value = {}
        value[ProcessingType.MEDIAN] = self._expected((self.num_models - 1) // 2)
        value[ProcessingType.MAX] = self._expected(self.num_models - 1)
        value[ProcessingType.MEAN] = self._expected(
            sum(range(self.num_models)) / self.num_models
        )

        for processing_type in [
            ProcessingType.MEDIAN,
            ProcessingType.MAX,
            ProcessingType.MEAN,
        ]:
            ensemble = BasicRigidNNEnsemble(self.sub_models, processing_type)
            self.assertTensorsEqual(value[processing_type], ensemble(x))

    def test_multiplex(self):
        x = self.generate_random_input(self.batch_size)
        ensemble = BasicRigidNNEnsemble(self.sub_models, ProcessingType.MULTIPLEX)
        ensemble.set_model_selector(
            lambda x: utils.to_onehot([3] * len(x), self.num_models)
        )  # select only model 3
        self.assertTensorsEqual(self._expected(3), ensemble(x))

    def test_multiplex_2(self):

        x = self.generate_random_input(self.batch_size)
        for i in range(self.batch_size):
            x[i, 0, 0, 0] = i
        ensemble = BasicRigidNNEnsemble(self.sub_models, ProcessingType.MULTIPLEX)
        ensemble.set_model_selector(
            lambda x: utils.to_onehot(x[:, 0, 0, 0], self.num_models)
        )  # select model 3
        expected = [[float(i)] * self.num_features for i in range(self.batch_size)]
        self.assertTensorsEqual(torch.tensor(expected), ensemble(x))


class BasicArchEnsembleTest(TestUtils):
    def setUp(self):
        """
        we have M as number of models, B  as batch size
        and F = as number of features
        """
        self.B = 11
        self.M = 3
        self.F = 10
        self.sub_models = self.sample_sub_model_list(
            batch_size=self.B, num_models=self.M, num_features=self.F, mock=False
        )
        self.arch_with_hint = self.sample_arch(
            takes_hint=True, num_models=self.M, model_output_size=self.F
        )
        self.arch_no_hint = self.sample_arch(
            takes_hint=False, num_models=self.M, model_output_size=self.F
        )

    def test_works_trainable_and_modules_const_no_hint(self):
        x = self.generate_random_input(self.B)
        ensemble = BasicArchEnsemble(
            models=self.sub_models,
            arch_model=self.arch_no_hint,
            processing_type=ProcessingType.NN,
        )
        sample_param_arch_start = next(self.arch_no_hint.parameters()).detach().clone()
        sample_param_models_start = torch.stack(
            [next(m.parameters()).detach().clone() for m in self.sub_models]
        )
        optim = torch.optim.SGD(ensemble.parameters(), lr=1)
        output = ensemble(x)
        output.sum().backward()
        optim.step()
        sample_param_arch_end = next(self.arch_no_hint.parameters()).detach().clone()
        sample_param_models_end = torch.stack(
            [next(m.parameters()).detach().clone() for m in self.sub_models]
        )

        self.assertTensorsEqual(sample_param_models_start, sample_param_models_end)
        self.assertTensorsDifferent(sample_param_arch_start, sample_param_arch_end)

    def test_works_trainable_and_modules_const_hint(self):
        x = self.generate_random_input(self.B)
        ensemble = BasicArchEnsemble(
            models=self.sub_models,
            arch_model=self.arch_with_hint,
            processing_type=ProcessingType.NN,
        )
        ensemble.set_model_selector(
            lambda x: utils.to_onehot([2] * self.B, self.M)
        )  # always select model 2 for test
        sample_param_arch_start = (
            next(self.arch_with_hint.parameters()).detach().clone()
        )
        sample_param_models_start = torch.stack(
            [next(m.parameters()).detach().clone() for m in self.sub_models]
        )
        optim = torch.optim.SGD(ensemble.parameters(), lr=1)
        output = ensemble(x)
        output.sum().backward()
        optim.step()
        sample_param_arch_end = next(self.arch_with_hint.parameters()).detach().clone()
        sample_param_models_end = torch.stack(
            [next(m.parameters()).detach().clone() for m in self.sub_models]
        )

        self.assertTensorsEqual(sample_param_models_start, sample_param_models_end)
        self.assertTensorsDifferent(sample_param_arch_start, sample_param_arch_end)
