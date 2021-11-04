#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import abc
from enum import IntEnum
from typing import Any, List, Tuple


class SubModelState(IntEnum):
    r"""
    Defines the state/mode of operation of a sub module.=

    If in PREDICT only mode the submodule will not support backprop
    and is only used for generating a feature. If in TRAIN mode the
    sub-module can be fully trained or tuned. If in TUNE_PARTIALLY mode
    parts of the model can be trained and parts will have frozen parameters.
    """
    PREDICT = 1
    TRAIN = 2
    TUNE_PARTIALLY = 3


class ISubModel(abc.ABC):
    r"""
    This interface needs to be implemented by a an ensemble's
    sub module. Each sub module should clearly state the feature
    that is going to be used in the ensemble als, it will need to
    determine if and how this feature could be learned by the
    ensemble.
    """

    @abc.abstractmethod
    def feature(self, x: Any = None) -> Any:
        r"""
        Module output that should be used in ensemble as a feature.

        Arguments:
            x: the input to generate the feature, if default value of None
               is provided the module can chose to return some stale resutls
               from previous runs

        Depending on the mode the feature is fully, partially, or non-trainable.
        """
        pass

    @abc.abstractmethod
    def set_state(
        self, state: SubModelState = SubModelState.PREDICT, tunable: Any = None
    ) -> None:
        r"""
        Sets how this submodule will be used. Look at `SubModelState`
        """
        pass


class EnsembleType(IntEnum):
    r"""
    Defines the type of the Ensemble.

    An Ensemble could be learnable or rigid.
    The rigid ensemble does an adhoc redution on the sub modules.
    The reduction process is rigid and not trainable.
    The trainable ensemble is able to be trained with server side
    training data.
    """
    RIGID = 1
    TRAINABLE = 2


class IServerEnsemble(abc.ABC):
    r"""
    This interface abstracts the functionalities that an
    ensemble model that runs on the server must implement.
    This will include how to combine features from different
    ensembles, and how/if learn the best combination parameters.
    """

    @abc.abstractmethod
    def sub_models(self) -> List[Tuple[str, ISubModel]]:
        r"""
        returns a named set of sub_modules that were used
        in the server
        """
        pass

    @abc.abstractmethod
    def sub_model_features(self, x: Any = None) -> List[Any]:
        r"""
        returns the prediction/features of all sub_module.

        Arguments:
            x: the input, that can have a default value of None.
               in the case of x = None last stale predicions should
               be returned.
        """
        pass

    @abc.abstractmethod
    def type(self) -> EnsembleType:
        r"""
        returns the type of the ensemble.

        This should be used to understand if the the ensemble
        is trainable or not.
        """
        pass
