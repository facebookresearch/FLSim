#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import logging
import math
from typing import List, Optional, Union

import torch
from flsim.common.logger import Logger
from flsim.interfaces.model import IFLModel
from flsim.utils.fl.personalized_model import FLModelWithPrivateModules
from torch import nn
from torch.optim.optimizer import Optimizer


class FLModelParamUtils:
    logger: logging.Logger = Logger.get_logger(__name__)
    logger.setLevel(logging.WARNING)

    @classmethod
    def get_state_dict(cls, model: nn.Module, only_federated_params: bool):
        if only_federated_params and isinstance(model, FLModelWithPrivateModules):
            state_dict = model.federated_state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    @classmethod
    def load_state_dict(cls, model: nn.Module, state_dict, only_federated_params: bool):
        if only_federated_params and isinstance(model, FLModelWithPrivateModules):
            model.load_federated_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    @classmethod
    def zero_weights(cls, model: nn.Module, only_federated_params=False) -> None:
        state_dict = cls.get_state_dict(model, only_federated_params)
        for _name, param in state_dict.items():
            param.data.fill_(0.0)

    @classmethod
    def pseudo_random_weights(cls, model: nn.Module, seed: int = 1) -> None:
        torch.manual_seed(seed)
        for _name, param in model.state_dict().items():
            param.data.uniform_()

    @classmethod
    def get_mismatched_param(
        cls,
        models: List[nn.Module],
        rel_epsilon: Optional[float] = None,
        abs_epsilon: Optional[float] = None,
    ) -> str:
        """Compare all the models in the given list of models.
        It returns an empty string if all the models have the same parameters.
        It returns the name of the first parameter that is different if any.
        """
        if rel_epsilon is None and abs_epsilon is not None:
            print("WARNING: rel_epsilon is not specified, abs_epsilon is ignored.")

        if len(models) <= 1:
            return ""
        dicts = [aModel.state_dict() for aModel in models]
        # Verify new models have all params same
        rtol_atol = {}
        if rel_epsilon is not None:
            rtol_atol["rtol"] = rel_epsilon
        if abs_epsilon is not None:
            rtol_atol["atol"] = abs_epsilon
        for name, param in dicts[0].items():
            for adict in dicts[1:]:
                # If a parameter name does not exist in a model, return early
                if name not in adict.keys():
                    return name
                param_here = adict[name]
                # If epsilon is specified, do approx comparison
                if not torch.allclose(param.float(), param_here.float(), **rtol_atol):
                    return name
        return ""

    @classmethod
    def linear_comb_models(
        cls,
        model1: nn.Module,
        wt1: float,
        model2: nn.Module,
        wt2: float,
        model_to_save: nn.Module,
        only_federated_params: bool = False,
    ) -> None:
        """sets model_to_save = model1*wt1 + model2*wt2"""
        global_params = cls.get_state_dict(model_to_save, only_federated_params)
        params_model1 = cls.get_state_dict(model1, only_federated_params)
        params_model2 = cls.get_state_dict(model2, only_federated_params)

        assert (
            global_params.keys() == params_model1.keys() == params_model2.keys()
        ), "Models should have the same set of parameters, including order."

        with torch.no_grad():
            for name, global_param in global_params.items():
                global_param.data = (
                    params_model1[name].data * wt1 + params_model2[name].data * wt2
                )

        cls.load_state_dict(model_to_save, global_params, only_federated_params)

    @classmethod
    def average_models(
        cls,
        models: List[nn.Module],
        model_to_save: nn.Module,
        weights: Optional[List[float]] = None,
    ) -> None:
        """Averages parameters of input models. Saves the average model in model_to_save

        Args:
            models: collection of models. These will be changed in-place
            model_to_save: update this model with the average
            weights: (optional) use weighted average
        """
        assert weights is None or len(weights) == len(models), (
            "Weights should have the same length as models. len(wts):"
            # pyre-fixme[6]: For 1st argument expected
            #  `pyre_extensions.ReadOnly[Sized]` but got `Optional[List[float]]`.
            + str(len(weights))
            + ", len(models):"
            + str(len(models))
        )
        wts_divisor = len(models)
        if weights is not None:
            for w in weights:
                assert w >= 0, "Weights must be non-negative. Found:" + str(w)
            wts_divisor = sum(weights)
            assert wts_divisor > 0, "Sum of weights must be positive:" + str(weights)
        cls.zero_weights(model_to_save, only_federated_params=True)
        for idx, aModel in enumerate(models):
            wts_numerator = 1 if weights is None else weights[idx]
            wt = wts_numerator / wts_divisor
            cls.linear_comb_models(
                aModel, wt, model_to_save, 1, model_to_save, only_federated_params=True
            )

    @classmethod
    def copy_models(
        cls,
        from_model: nn.Module,
        to_models: List[nn.Module],
        only_federated_params: bool = False,
    ) -> None:
        """Copy from_model into every model in to_models

        Args:
            from_model: a model
            to_models: collection of models. These will be changed in-place
            only_federated_params: copy only federated params.
        """
        from_state_dict = cls.get_state_dict(from_model, only_federated_params)
        for m in to_models:
            cls.load_state_dict(m, from_state_dict, only_federated_params)

    @classmethod
    def clone(
        cls, model: Union[nn.Module, IFLModel], dtype: Optional[torch.dtype] = None
    ):
        """Clones a pytorch module, and allows for a change of precision.
        TODO If needed we can also add device here.
        """
        new_model = copy.deepcopy(model)
        if isinstance(new_model, IFLModel):
            if dtype == torch.float32:
                new_model.fl_get_module().float()
            elif dtype == torch.float64:
                new_model.fl_get_module().double()
            return new_model
        else:
            return (
                new_model.float()
                if dtype == torch.float32
                else (new_model.double() if dtype == torch.float64 else new_model)
            )

    @classmethod
    def set_gradient(cls, model: nn.Module, reference_gradient: nn.Module) -> None:
        """Set gradient of model to the parameters of reference_gradient
        Args:
            model: nn.Module
            reference_gradient: nn.Module - gradient is the parameters of this model
        """
        # Use parameters() since state_dict() may include non-learnable params.
        for m, ref in zip(model.parameters(), reference_gradient.parameters()):
            m.grad = ref.detach().clone().type(m.type())

    @classmethod
    def linear_combine_gradient(
        cls,
        model1: nn.Module,
        wt1: float,
        model2: nn.Module,
        wt2: float,
        model_to_save: nn.Module,
    ):
        """Sets model_to_save.grad = model1.grad * wt1 + model2.grad * wt2"""
        for save_p, model1_p, model2_p in zip(
            model_to_save.parameters(), model1.parameters(), model2.parameters()
        ):
            if save_p.requires_grad:
                grad = None
                if model1_p.grad is not None:
                    grad = wt1 * model1_p.grad
                if model2_p.grad is not None:
                    if grad is not None:
                        grad += wt2 * model2_p.grad
                    else:
                        grad = wt2 * model2_p.grad
                if grad is None:
                    cls.logger.warning(
                        "Parameter with requires_grad=True has gradient set to None"
                    )
                save_p.grad = grad

    @classmethod
    def multiply_gradient_by_weight(
        cls, model: nn.Module, weight: float, model_to_save: nn.Module
    ):
        """Sets model_to_save.grad = model.grad * weight"""
        for save_p, model_p in zip(model_to_save.parameters(), model.parameters()):
            if save_p.requires_grad:
                grad = None
                if model_p.grad is not None:
                    grad = weight * model_p.grad
                if grad is None:
                    cls.logger.warning(
                        "Parameter with requires_grad=True has gradient set to None"
                    )
                del save_p.grad
                save_p.grad = grad

    @classmethod
    def add_gradients(
        cls, model1: nn.Module, model2: nn.Module, model_to_save: nn.Module
    ):
        """Sets model_to_save.grad = model1.grad + model2.grad"""
        for save_p, model1_p, model2_p in zip(
            model_to_save.parameters(), model1.parameters(), model2.parameters()
        ):
            if save_p.requires_grad:
                grad = None
                if model1_p.grad is not None:
                    grad = model1_p.grad.detach().clone().type(save_p.type())
                if model2_p.grad is not None:
                    if grad is not None:
                        grad += model2_p.grad
                    else:
                        grad = model2_p.grad.detach().clone().type(save_p.type())
                if grad is None:
                    cls.logger.warning(
                        "Parameter with requires_grad=True has gradient set to None"
                    )
                del save_p.grad
                save_p.grad = grad

    @classmethod
    def subtract_gradients(
        cls, minuend: nn.Module, subtrahend: nn.Module, difference: nn.Module
    ):
        """Sets difference.grad = minuend.grad - subtrahend.grad"""
        for difference_p, minuend_p, subtrahend_p in zip(
            difference.parameters(), minuend.parameters(), subtrahend.parameters()
        ):
            if difference_p.requires_grad:
                grad = None
                if minuend_p.grad is not None:
                    grad = minuend_p.grad
                if subtrahend_p.grad is not None:
                    if grad is not None:
                        grad -= subtrahend_p.grad
                    else:
                        grad = (
                            -subtrahend_p.grad.detach()
                            .clone()
                            .type(difference_p.type())
                        )
                if grad is None:
                    cls.logger.warning(
                        "Parameter with requires_grad=True has gradient set to None"
                    )
                del difference_p.grad
                difference_p.grad = grad

    @classmethod
    def copy_gradients(cls, model: nn.Module, model_to_copy: nn.Module):
        """Sets model_to_copy.grad = model.grad"""
        for copy_p, model_p in zip(model_to_copy.parameters(), model.parameters()):
            if copy_p.requires_grad:
                grad = None
                if model_p.grad is not None:
                    grad = model_p.grad.detach().clone().type(copy_p.type())
                if grad is None:
                    cls.logger.warning(
                        "Parameter with requires_grad=True has gradient set to None"
                    )
                del copy_p.grad
                copy_p.grad = grad

    @classmethod
    def reconstruct_gradient(
        cls, old_model: nn.Module, new_model: nn.Module, grads: nn.Module
    ) -> None:
        # compute approximate gradient:
        # grads = old_model - new_model
        cls.subtract_model(old_model, new_model, grads)

    @classmethod
    def get_trainable_params(cls, model: nn.Module):
        return filter(lambda p: p.requires_grad, model.parameters())

    @classmethod
    def get_trainable_named_parameters(cls, model: nn.Module):
        return filter(lambda np: np[1].requires_grad, model.named_parameters())

    @classmethod
    def get_gradient_l2_norm_raw(cls, model: nn.Module) -> float:
        total_norm = 0
        for p in cls.get_trainable_params(model):
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    @classmethod
    def get_num_trainable_params(cls, model: nn.Module) -> int:
        total_params = 0
        for p in cls.get_trainable_params(model):
            total_params += p.numel()
        return total_params

    @classmethod
    def get_gradient_l2_norm_normalized(cls, model: nn.Module) -> float:
        """Compute l2-norm-of-gradient/sqrt(num-params)
        If gradients are all independent, l2 norm grows as sqrt() of number
        of parameters. Eg: in Xavier Initialization
        """
        return cls.get_gradient_l2_norm_raw(model) / math.sqrt(
            cls.get_num_trainable_params(model)
        )

    @classmethod
    def debug_model_norm(cls, model: nn.Module):
        norm = 0
        for p in model.parameters():
            norm += torch.sum(torch.abs(p))
        return norm

    @classmethod
    def get_mismatched_param_max_difference(cls, models: List[nn.Module]):

        if len(models) <= 1:
            return 0.0
        dicts = [aModel.state_dict() for aModel in models]
        max_diff = 0
        # compute maximum element-wise difference of model parameters
        for name, param in dicts[0].items():
            for adict in dicts[1:]:
                param_here = adict[name]
                param_diff = torch.max(torch.abs(param - param_here)).item()
                # pyre-fixme[58]: `<` is not supported for operand types
                #  `Union[float, int]` and `int`.
                max_diff = param_diff if (param_diff > max_diff) else max_diff
                # if epsilon is specified, do approx comparison
        return max_diff

    @classmethod
    def clip_gradients(cls, max_normalized_l2_norm: float, model: nn.Module) -> None:
        """Clip gradients in model parameters by maximum value for normalized
        L2 norm (max_normalized_norm).
        """
        max_unnormalized_l2_norm = max_normalized_l2_norm * math.sqrt(
            cls.get_num_trainable_params(model)
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_unnormalized_l2_norm)

    @classmethod
    def step_with_modified_lr(
        cls, optimizer: Optimizer, base_lr: float, lr_normalizer: float
    ) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * lr_normalizer
        optimizer.step()

    @classmethod
    def multiply_model_by_weight(
        cls,
        model: nn.Module,
        weight: float,
        model_to_save: nn.Module,
        only_federated_params: bool = False,
    ):
        """Returns model_to_save = model * weight."""
        cls.linear_comb_models(
            model, weight, model, 0, model_to_save, only_federated_params
        )

    @classmethod
    def subtract_model(
        cls,
        minuend: nn.Module,
        subtrahend: nn.Module,
        difference: nn.Module,
        only_federated_params: bool = False,
    ):
        """Returns difference = minuend - subtrahend."""
        cls.linear_comb_models(
            minuend, 1, subtrahend, -1, difference, only_federated_params
        )

    @classmethod
    def add_model(
        cls,
        model1: nn.Module,
        model2: nn.Module,
        model_to_save: nn.Module,
        only_federated_params: bool = False,
    ):
        """
        Returns model_to_save = model1 + model2
        """
        cls.linear_comb_models(
            model1, 1, model2, 1, model_to_save, only_federated_params
        )

    @classmethod
    def scale_optimizer_lr(
        cls, optimizer: torch.optim.Optimizer, scaling_factor: float
    ):
        """
        Set optimizer.lr = optimizer.lr / scaling_factor for all param groups
        """
        assert (
            scaling_factor > 0
        ), "Scaling factor needs to be postive to keep learning rate >= 0"
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / scaling_factor
