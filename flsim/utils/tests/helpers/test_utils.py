#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, List, Optional

import torch
import torch.nn as nn
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.utils.fl.common import FLModelParamUtils
from opacus import PrivacyEngine


class FLTestUtils:
    @classmethod
    def compare_model_linear_comb(cls, model1: nn.Module, model2: nn.Module):
        temp_modelA = copy.deepcopy(model1)
        temp_modelB = copy.deepcopy(model1)
        temp_modelC = copy.deepcopy(model1)
        # model1 + 0*model2 = model1
        FLModelParamUtils.linear_comb_models(model1, 1, model2, 0, temp_modelA)
        assert FLModelParamUtils.get_mismatched_param([model1, temp_modelA]) == ""
        # model1 + model2 != model1
        FLModelParamUtils.linear_comb_models(model1, 1, model2, 1, temp_modelA)
        assert FLModelParamUtils.get_mismatched_param([model1, temp_modelA]) != ""
        # (2*model1 + 3*model1 ) - 4*model1 = model1
        FLModelParamUtils.linear_comb_models(model1, 2, model1, 3, temp_modelA)
        FLModelParamUtils.linear_comb_models(model1, 4, model1, 0, temp_modelB)
        FLModelParamUtils.linear_comb_models(
            temp_modelA, 1, temp_modelB, -1, temp_modelC
        )
        assert FLModelParamUtils.get_mismatched_param([model1, temp_modelC], 1e-5) == ""
        # test that resuing one of the input models as model_to_save also works
        # model1 = model1 - model2, followed by model2 = model1 + model2
        # model2 should be the same as original model1
        temp_modelA = copy.deepcopy(model1)
        FLModelParamUtils.linear_comb_models(model1, 1, model2, -1, model1)
        FLModelParamUtils.linear_comb_models(model1, 1, model2, 1, model1)
        assert FLModelParamUtils.get_mismatched_param([model1, temp_modelA], 1e-5) == ""

    @classmethod
    def random_grad(cls, model: nn.Module):
        for param in model.parameters():
            param.grad = torch.rand_like(param)

    @classmethod
    def compare_gradient_reconstruction(
        cls, model0: nn.Module, copy_model0: nn.Module, reconstructed_grad: nn.Module
    ):
        """Test that gradient reconstruction post-optimization works
        Moment-based optimizers for FL require approximate gradient reconstruction from
        two models: original model, and new model after FL optmization step
        approx_gradient = original_model - new_model
        This test checks that gradient reconstruction works as expected
        """
        # copy model.0 into copy_model.0
        # create optimizerA for model.0, take 1 step of gradient descent on model.0,
        # moving to model.1
        # reconstruct original gradients by reconstructred_grad = model.1 - model.0
        # set grad(copy_model.0) = reconstructed_grad
        # create optimizerB for copy_model.0
        # take 1 step of gradient descent on copy_model.0, moving to copy_model.1
        # check model.1 = copy_model.1
        learning_rate = 1.0
        FLModelParamUtils.copy_models(model0, [copy_model0])
        optimizer = torch.optim.SGD(model0.parameters(), lr=learning_rate)
        # take a few steps of gradient descent
        for _i in range(0, 10):
            optimizer.zero_grad()
            cls.random_grad(model0)
            optimizer.step()

        copy_optimizer = torch.optim.SGD(copy_model0.parameters(), lr=learning_rate)
        copy_optimizer.zero_grad()
        FLModelParamUtils.reconstruct_gradient(
            old_model=copy_model0, new_model=model0, grads=reconstructed_grad
        )
        FLModelParamUtils.set_gradient(
            model=copy_model0, reference_gradient=reconstructed_grad
        )
        copy_optimizer.step()
        assert (
            FLModelParamUtils.get_mismatched_param(
                [model0, copy_model0], rel_epsilon=1e-4
            )
            == ""
        )

    @classmethod
    def _verify_averaged_and_orig_models(
        cls, orig_models: List[nn.Module], new_models: List[nn.Module]
    ) -> None:
        """Verify that:
        a) Every model in new_models is the same
        b) Every model in new_models is the 'average' of models in orig_models
        """
        assert len(orig_models) == len(new_models)
        if len(orig_models) == 0:
            return
        orig_dicts = [dict(aModel.named_parameters()) for aModel in orig_models]
        new_dicts = [dict(aModel.named_parameters()) for aModel in new_models]
        assert len(orig_dicts) == len(new_dicts)
        if len(orig_dicts) == 0:
            return
        # verify new models have all params same
        assert FLModelParamUtils.get_mismatched_param(new_models) == ""

        # verify that new_models have average of old models
        for name, param in new_dicts[0].items():
            orig_tensors = torch.stack([thedict[name] for thedict in orig_dicts])
            orig_shape = orig_tensors[0].shape
            averaged = torch.mean(orig_tensors, 0, keepdim=True)
            averaged_reshaped = averaged.view(orig_shape)
            assert torch.allclose(averaged_reshaped, param, atol=1e-6)

    @classmethod
    def average_and_verify_models(cls, orig_models: List[nn.Module]) -> None:
        """Compute the average of models in orig_models, and verify the average"""
        if len(orig_models) == 0:
            return
        models = copy.deepcopy(orig_models)
        temp_model = copy.deepcopy(models[0])
        FLModelParamUtils.average_models(models, temp_model)
        FLModelParamUtils.copy_models(temp_model, models)
        cls._verify_averaged_and_orig_models(orig_models, models)

    @classmethod
    def do_two_models_have_same_weights(cls, model1, model2) -> bool:
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                return False
        return True

    @classmethod
    def train_non_fl(
        cls,
        data_provider: IFLDataProvider,
        global_model: IFLModel,
        optimizer: torch.optim.Optimizer,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
        epochs: int = 1,
        cuda_enabled: bool = False,
    ):
        if cuda_enabled:
            global_model.fl_cuda()

        for _ in range(epochs):
            for one_user_data in data_provider.train_data():
                for batch in one_user_data:
                    optimizer.zero_grad()
                    batch_metrics = global_model.fl_forward(batch)

                    if metrics_reporter is not None:
                        metrics_reporter.add_batch_metrics(batch_metrics)

                    batch_metrics.loss.backward()
                    optimizer.step()
        return global_model, metrics_reporter

    @classmethod
    def run_nonfl_training(
        cls,
        model: IFLModel,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> IFLModel:
        torch.manual_seed(1)
        for _ in range(epochs):
            for training_batch in data_loader:
                FLTestUtils.run_nonfl_training_one_batch(
                    model=model, optimizer=optimizer, training_batch=training_batch
                )
        return model

    @classmethod
    def run_nonfl_training_one_batch(
        cls, model: IFLModel, optimizer: torch.optim.Optimizer, training_batch: Any
    ):
        optimizer.zero_grad()
        batch_metrics = model.fl_forward(training_batch)
        loss = batch_metrics.loss
        loss.backward()
        optimizer.step()
