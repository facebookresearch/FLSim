#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from enum import IntEnum
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from .interfaces import IServerEnsemble, ISubModel, EnsembleType, SubModelState
from .sub_model import BasicNNSubModel


class ProcessingType(IntEnum):
    r"""
    Defines how the ouptut from different sub-modules
    are combined together to give a final output.
    Simple ways are max, mode, and mean.
    Multiplex is simply a selector.
    NN is a neural network based learnable aggregation.
    """
    MAX = 1
    MEDIAN = 2
    MEAN = 3
    MULTIPLEX = 4
    NN = 10


class ArchInput(NamedTuple):
    """
    Immutable tuple to wrap required input type
    for an [over-]arch layer
    """

    sub_model_features: List[Tensor]
    x: Any
    hint: Optional[Tensor] = None

    @property
    def batch_size(self):
        return self.sub_model_features[0].shape[0]


class BasicRigidNNEnsemble(nn.Module, IServerEnsemble):
    r"""
    Simple class to create naive ensembles.

    The overal architecture of a rigid ensemble looks like:

               ::::::::::::::  input  ::::::::::::::
             /           |            \              :
            /            |              \              :
    [Sub Model 1]    [Sub Model 2]  [Sub Model 3]  [model_selector]
           \             |               /               :
            \            |              /               :
              [------------------------]               :
              [------ Aggregate -------]  <<----------:


    This type of ensemble cannot learn how to mix sub models,
    it simply takes outputs from different sub-models and reduces
    them using min, max, mean, median, and selection.
    """

    def __init__(
        self,
        models: List[Union[Tuple[str, BasicNNSubModel], BasicNNSubModel]],
        processing_type: ProcessingType = ProcessingType.MEAN,
    ):
        super().__init__()
        self.check_processing_type(processing_type)
        if not isinstance(models[0], BasicNNSubModel):
            self.model_names, self.models = zip(*models)  # pyre-ignore
        else:
            self.model_names = [str(i) for i, _ in enumerate(models)]
            self.models = models
        for m in self.models:
            assert isinstance(m, BasicNNSubModel)
            m.set_state(SubModelState.PREDICT)
        self.models = nn.ModuleList(self.models)
        self.processing_type = processing_type
        self._sub_model_features = []
        self.model_selector = None  # Callable to be set

    def check_processing_type(self, processing_type: ProcessingType):
        assert (
            processing_type != ProcessingType.NN
        ), "type = ProcessingType.NN is not supported in this class."

    def sub_models(self) -> List[Tuple[str, ISubModel]]:
        r"""
        returns a named set of sub_modules that were used
        in the server
        """
        return list(zip(self.model_names, self.models))

    def forward(self, x):
        sub_model_features = self.sub_model_features(x)
        return self.post_process(x, sub_model_features)

    def sub_model_features(self, x: Any = None) -> List[Tensor]:
        r"""
        returns the prediction of all sub_module.
        """
        if x is not None:
            self._sub_model_features = [m.feature(x) for m in self.models]
        return self._sub_model_features

    def type(self) -> EnsembleType:
        return EnsembleType.RIGID

    def set_model_selector(self, method: Callable[[Any], Tensor]) -> None:
        r"""
        Set model selector, this should operate on batches.

        Model selector selects, one of the model outputs as the final output.

        The selector should return probability distribution over the selected
        models.

        This function only needs to be set if the processing_type is MULTIPLEX
        for this class and can return a one-hot vector.
        """
        self.model_selector = method

    def post_process(self, x: Any, sub_model_features: List[Tensor]) -> Tensor:
        r"""
        provides the final prediction value based on the
        processing type.

        if the processing type is multiplex the implementer needs
        to implement the select_model function.
        """
        stacked_features = torch.stack(sub_model_features, dim=0)
        if self.processing_type == ProcessingType.MEAN:
            result = stacked_features.mean(dim=0)
        elif self.processing_type == ProcessingType.MEDIAN:
            result = torch.median(stacked_features, dim=0).values
        elif self.processing_type == ProcessingType.MAX:
            result = stacked_features.max(dim=0).values
        elif self.processing_type == ProcessingType.MULTIPLEX:
            try:
                onehot = self.model_selector(x).to(stacked_features.device)
                onehot = onehot.transpose(0, 1)
            except BaseException as e:
                raise Exception("Have you set a proper set_model_selector") from e
            result = torch.einsum("mn, mn...->n...", onehot, stacked_features)
        else:
            raise NotImplementedError(
                "This class does not allow for proceesing type:"
                f" {self.processing_type}"
            )
        return result


class BasicArchEnsemble(BasicRigidNNEnsemble):
    r"""
    A simple trainable ensemble.

    This class takes a few sub-models, puts a given trainable
    arch on top of them and provides utilities to train them.

    The overal architecture of an ArchEnsemble looks like:

               ::::::::::::::  input  :::::::::::::::::::
             /           |            \         \         :
            /            |              \         \         :
    [Sub Model 1]    [Sub Model 2]  [Sub Model 3]  |   [model_weights]
           \             |               /        /        :
            \            |              /       /        :
              [----------------------------------------]
              [---------- Over Arch Layer -------------]
              [----------------------------------------]

    Attributes:
        arch_mode:
            an NN module that accepts a tuple of length 3 as input:
                1st element are the predictions from sub-models
                2nd is the input x
                3rd is an optional hint
            a hint input and a feature input.

    Note:
        The sub_modules themselves will not be trained in this class.
    """

    def __init__(
        self,
        models: List[Union[Tuple[str, BasicNNSubModel], BasicNNSubModel]],
        arch_model: nn.Module,
        processing_type: ProcessingType = ProcessingType.NN,
    ):
        super().__init__(models, processing_type)
        self.arch = arch_model

    def check_processing_type(self, processing_type: ProcessingType):
        assert processing_type in (
            ProcessingType.NN,
        ), f"processing_type: {processing_type} is the not supported for this class!"

    def type(self) -> EnsembleType:
        return EnsembleType.TRAINABLE

    def post_process(self, x: Any, sub_model_features: List[Tensor]) -> Tensor:
        r"""
        provides the final trainable prediction value based on the
        processing type.
        """
        assert len(sub_model_features) == len(
            self.models
        )  # m models X n samples X feature sizeß
        hint = None
        if self.model_selector is not None:
            try:
                hint = self.model_selector(x).to(sub_model_features[0].device)
            except BaseException as e:
                raise Exception("Have you set a proper set_model_selector") from e
        return self.arch(
            ArchInput(sub_model_features=sub_model_features, x=x, hint=hint)
        )
