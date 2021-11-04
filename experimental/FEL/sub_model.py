#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import copy
from typing import Any, Iterable

from torch import nn

from .interfaces import ISubModel, SubModelState


class BasicNNSubModel(nn.Module, ISubModel):
    r"""
    Simple class to define sub module that extends
    torch.nn.Module
    """

    def __init__(
        self,
        module: nn.Module,
        feature_layer: str = "",
    ):
        super().__init__()
        r"""
        sets the layer that should be user for feature generation.
        The default value of empty string takes the output
        """
        self.main = module

        self._feature = None
        self.hook = None
        layer_names = [n for n, _ in self.main.named_modules()]
        assert feature_layer in layer_names, f"layer: {feature_layer} dose not exist!"
        self.feature_layer = feature_layer
        self.register_hook()

    def register_hook(self):
        def collect(module, module_in, module_out):
            self._feature = module_out

        for name, layer in self.main.named_modules():
            if name == self.feature_layer:
                handle = layer.register_forward_hook(collect)
                self.hook = handle

    def __deepcopy__(self, memo):
        return BasicNNSubModel(*copy.deepcopy((self.main, self.feature_layer)))

    def detach(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        return self

    def forward(self, x: Any):
        return self.main(x)

    def feature(self, x: Any = None):
        if x is not None:
            self.forward(x)
        return self._feature

    def set_state(
        self, state: SubModelState = SubModelState.PREDICT, tunable: Iterable[str] = ()
    ) -> None:
        tunable = list(tunable)

        def is_partially_tunable(name: str):
            return state == SubModelState.TUNE_PARTIALLY and any(
                name.startswith(t) for t in tunable
            )

        for name, param in self.main.named_parameters():
            param.requires_grad = state == SubModelState.TRAIN or is_partially_tunable(
                name
            )
