#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.common.logger import Logger
from flsim.interfaces.model import IFLModel
from flsim.privacy.common import PrivacyBudget
from flsim.reducers.base_round_reducer import (
    ReductionType,
    RoundReducerConfig,
)
from flsim.reducers.dp_round_reducer import DPRoundReducerConfig
from flsim.reducers.weighted_dp_round_reducer import WeightedDPRoundReducerConfig
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.fl.stats import ModelSequenceNumberTracker
from hydra.utils import instantiate
from omegaconf import MISSING
from torch.nn import Module as Model  # @manual


class AsyncAggregationType(Enum):
    r"""
    There are two ways in which we can aggregate local model updates
    into the global model in AsyncFL.
    Assume we're trying to aggregate updates from userA
    local_model_before_training = model sent to userA by server
    final_local_model = model after local training by userA
    global_model = model on server when userA finishes training

    new_global_model = global_model - delta*lr
    The difference is in how `delta` is computed
    a) fed_buff_aggregation: delta = local_model_before_training - final_local_model
    b) fed_async_aggregation: delta = global_model - final_local_model

    Literature uses fed_async_aggregation (https://arxiv.org/abs/1903.03934)
    We find empirically that fed_buff_aggregation performs better
    """

    fed_buff_aggregation = 1
    fed_async_aggregation = 2

    @staticmethod
    def from_str(name: str):
        name_lower = name.lower()
        names = [e.name for e in AsyncAggregationType]
        assert name_lower in names, "Unknown async aggregation type:" + name
        return AsyncAggregationType[name_lower]


class AsyncAggregator:
    r"""
    Implements Asynchronous Federated Learning
        Input: local_model_before_training, final_local_model, global_model, wt
        Output: global_model = global_model - lr*delta, where
            delta = (local_model_before_training - final_local_model)
            lr: original_lr*wt (higher wt = higher learning rate)
            original_lr: lr given in config.
                smaller original_lr: small changes in global_model
                higher original_lr: large changes in global_model
                Eg: lr=1 means keep global_model=final_local_model (when wt=1)
        TODO: add adaptive learning rate based on staleness of gradient
    Attributes:

    Internal attributes:
        optimizer (torch.optim.Optimizer): an optimizer kept
        internally to prevent optimizer creation on every epoch
    """

    logger: logging.Logger = Logger.get_logger("AsyncAggregator")

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=AsyncAggregatorConfig,
            **kwargs,
        )

        self.optimizer = create_optimizer_for_async_aggregator(
            # pyre-fixme[16]: `AsyncAggregator` has no attribute `cfg`.
            self.cfg,
            global_model.fl_get_module(),
        )

        if self.cfg.aggregation_type not in [
            AsyncAggregationType.fed_buff_aggregation,
            AsyncAggregationType.fed_async_aggregation,
        ]:
            raise AssertionError(
                f"Unknown AsyncAggregationType:{self.cfg.aggregation_type}"
            )

        self.orig_lr = self.optimizer.param_groups[0]["lr"]
        self._global_model: IFLModel = global_model
        self._reconstructed_grad: IFLModel = copy.deepcopy(self._global_model)
        # there is no concept of a round in async, hence round reducer is not tied to a round
        self.reducer = instantiate(
            self.cfg.reducer,
            global_model=self._global_model,
            channel=channel,
            num_users_per_round=self.cfg.num_users_per_round,
            total_number_of_users=self.cfg.total_number_of_users,
        )
        self.seqnum_tracker = ModelSequenceNumberTracker()

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @property
    def global_model(self) -> IFLModel:
        return self._global_model

    def set_num_total_users(self, num_total_users: int) -> None:
        self.reducer.set_num_total_users(num_total_users)

    def model_staleness(self, model_seqnum: int):
        seqnum_diff = self.seqnum_tracker.get_staleness_and_update_stats(model_seqnum)
        return seqnum_diff

    def zero_grad(self):
        r"""
        Reset the reducer and optimizer to prepare to update
        the global model
        """
        self.reducer.reset(ref_model=self._global_model)
        self.optimizer.zero_grad()

    @property
    def privacy_budget(self) -> Optional[PrivacyBudget]:
        return self.reducer.privacy_budget if self.is_private else None

    @property
    def global_seqnum(self):
        return self.seqnum_tracker.current_seqnum

    @property
    def is_private(self):
        return self.cfg.reducer._target_ in {
            DPRoundReducerConfig._target_,
            WeightedDPRoundReducerConfig._target_,
        }

    def on_client_training_end(
        self,
        client_delta: IFLModel,
        final_local_model: IFLModel,
        weight: float,
    ) -> bool:
        """
        Collects the client update and updates the global model

        Note:
            Async updates the global model on every client hence return value is always true

        Args:
            client_delta (IFLModel): the difference between the client's before and after training
            final_local_model (IFLModel): client's model after local training
            weight (float): client's update weight

        Returns:
            bool: whether or not the global model was updated
        """
        client_delta = self.reducer.receive_through_channel(client_delta)
        # pyre-fixme[16]: `AsyncAggregator` has no attribute `cfg`.
        if self.cfg.aggregation_type == AsyncAggregationType.fed_buff_aggregation:
            FLModelParamUtils.set_gradient(
                model=self._global_model.fl_get_module(),
                reference_gradient=client_delta.fl_get_module(),
            )
        else:
            FLModelParamUtils.subtract_model(
                minuend=self._global_model.fl_get_module(),
                subtrahend=final_local_model.fl_get_module(),
                difference=self._reconstructed_grad.fl_get_module(),
            )
            FLModelParamUtils.set_gradient(
                model=self._global_model.fl_get_module(),
                reference_gradient=self._reconstructed_grad.fl_get_module(),
            )

        self._step_with_modified_lr(lr_normalizer=weight)
        return True

    def on_training_epoch_end(self) -> bool:
        """
        Base method for end of training epoch
        Returns whether some client updates were aggregated into global model
          (In k-async, up to (k-1) client updates may not have been aggregated
          at the end of the epoch, and need to be handled explicitly)
        """
        pass

    def _step_with_modified_lr(self, lr_normalizer: float):
        r"""
        Updates the learning rate based on the weight and
        increments global seqnum

        original_lr: lr given in config
        lr: original_lr * weight (higher weight = higher learning rate)

        smaller original_lr: small changes in global_model
        higher original_lr: large changes in global_model
        Eg: lr=1 means keep global_model=final_local_model (when wt=1)
        """
        # TODO: for optimizers that don't use momentum or adaptive learning rate,
        # scaling LR directly is optimal
        # For optimizers like Adam, or SGD+Momentum, add optimal weight-based scaling
        # (open research problem)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.orig_lr * lr_normalizer
        self.optimizer.step()
        self.seqnum_tracker.increment()


class FedAdamAsyncAggregator(AsyncAggregator):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAdamAsyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass


class FedAvgWithLRAsyncAggregator(AsyncAggregator):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAvgWithLRAsyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)


class FedAvgWithLRWithMomentumAsyncAggregator(FedAvgWithLRAsyncAggregator):
    """Implements the correct version of momentum"""

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAvgWithLRWithMomentumAsyncAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)
        self.orig_momentum = self.optimizer.param_groups[0]["momentum"]
        assert (
            # pyre-fixme[16]: `FedAvgWithLRWithMomentumAsyncAggregator` has no attribute
            #  `cfg`.
            self.cfg.aggregation_type
            == AsyncAggregationType.fed_buff_aggregation
        ), "Only delta direction is supported by "
        f"{FedAvgWithLRWithMomentumAsyncAggregatorConfig.__class__.__name__}"

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def on_client_training_end(
        self,
        client_delta: IFLModel,
        final_local_model: IFLModel,
        weight: float,
    ) -> bool:
        # TODO: better if this assert fires at config creation time
        assert (
            weight >= 0 and weight <= 1
        ), f"{FedAvgWithLRWithMomentumAsyncAggregatorConfig.__class__.__name__}"
        "only supports weights between 0 and 1"

        # scale delta by weight
        FLModelParamUtils.multiply_model_by_weight(
            model=client_delta.fl_get_module(),
            weight=weight,
            model_to_save=self._reconstructed_grad.fl_get_module(),
        )
        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=self._reconstructed_grad.fl_get_module(),
        )

        # change momentum parameter
        momentum = self.orig_momentum + ((1 - self.orig_momentum) * (1 - weight))
        for param_group in self.optimizer.param_groups:
            param_group["momentum"] = momentum
        self._step_with_modified_lr(lr_normalizer=weight)
        return True


class HybridAggregator(AsyncAggregator):
    r"""
    Aggregator for Hybrid-FL where client update and global update are decoupled

    Keeps track of number clients reported and take a global step after reaching the
    threshold set by the config
    """
    logger: logging.Logger = Logger.get_logger("HybridAggregator")

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=HybridAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)
        # pyre-fixme[16]: `HybridAggregator` has no attribute `cfg`.
        if self.cfg.aggregation_type != AsyncAggregationType.fed_buff_aggregation:
            raise ValueError("Hybrid Aggregator only supports delta direction")
        self.num_clients_reported = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def zero_grad(self):
        r"""
        Zero'd out the grads if it's the first update or if
        reaching buffer_size otherwise no-op
        """
        if self.num_clients_reported == 0 or self.should_update_global_model():
            super().zero_grad()

    def on_client_training_end(
        self,
        client_delta: IFLModel,
        final_local_model: IFLModel,
        weight: float,
    ) -> bool:
        """
        Collects client update and update global model if reaching buffer_size

        Args:
            client_delta (IFLModel): the difference between the client's before and after training
            final_local_model (IFLModel): client's model after local training
            weight (float): client's update weight

        Returns:
            bool: whether or not the global model was updated
        """
        # total_delta += delta
        self._collect_client_update(update=client_delta, weight=weight)
        if self.should_update_global_model():
            self._update_global_model()
            return True
        return False

    def should_update_global_model(self) -> bool:
        # pyre-fixme[16]: `HybridAggregator` has no attribute `cfg`.
        return self.num_clients_reported >= self.cfg.buffer_size

    def on_training_epoch_end(self) -> bool:
        """
        Updates the global model in case when
        there are remaining clients who didn't get aggregated
        into the global model at the end of an epoch.
        Return value:
          True if there were any such clients with pending updates. In this case,
            the global model was updated.
          False if there were no such clients. Global model was not update
        """
        if self.num_clients_reported != 0:
            self._update_global_model()
            return True
        return False

    def _update_global_model(self):
        total_delta, _ = self.reducer.reduce()
        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(), reference_gradient=total_delta
        )
        self._step_with_modified_lr(lr_normalizer=1.0)
        self.num_clients_reported = 0

    def _collect_client_update(self, update: IFLModel, weight: float) -> None:
        """
        Collects update from one client and aggregtes it internally.
        reduced model = reduced model + update * weight
        """
        self.reducer.collect_update(delta=update, weight=weight)
        self.num_clients_reported += 1


class FedAvgWithLRHybridAggregator(HybridAggregator):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAvgWithLRHybridAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass


class FedAdamHybridAggregator(HybridAggregator):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IdentityChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FedAdamHybridAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass


def create_optimizer_for_async_aggregator(config: AsyncAggregatorConfig, model: Model):
    if config._target_ in {
        FedAvgWithLRAsyncAggregatorConfig._target_,
        FedAvgWithLRHybridAggregatorConfig._target_,
        FedAvgWithLRWithMomentumAsyncAggregatorConfig._target_,
    }:
        return torch.optim.SGD(
            model.parameters(),
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `lr`.
            lr=config.lr,
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `momentum`.
            momentum=config.momentum,
        )
    elif config._target_ in {
        FedAdamAsyncAggregatorConfig._target_,
        FedAdamHybridAggregatorConfig._target_,
    }:
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `weight_decay`.
            weight_decay=config.weight_decay,
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `beta1`.
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `beta2`.
            betas=(config.beta1, config.beta2),
            # pyre-fixme[16]: `AsyncAggregatorConfig` has no attribute `eps`.
            eps=config.eps,
        )


@dataclass
class AsyncAggregatorConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    aggregation_type: AsyncAggregationType = AsyncAggregationType.fed_buff_aggregation
    # reducer to collect client updates, there is no concept of a round
    # in async, hence round reducer is not tied to a round
    reducer: RoundReducerConfig = RoundReducerConfig(
        reduction_type=ReductionType.WEIGHTED_SUM
    )
    num_users_per_round: int = 1
    total_number_of_users: int = 10000000000


@dataclass
class FedAvgWithLRAsyncAggregatorConfig(AsyncAggregatorConfig):
    _target_: str = fullclassname(FedAvgWithLRAsyncAggregator)
    lr: float = 0.001
    momentum: float = 0.0


@dataclass
class FedAvgWithLRWithMomentumAsyncAggregatorConfig(FedAvgWithLRAsyncAggregatorConfig):
    _target_: str = fullclassname(FedAvgWithLRWithMomentumAsyncAggregator)


@dataclass
class FedAdamAsyncAggregatorConfig(AsyncAggregatorConfig):
    _target_: str = fullclassname(FedAdamAsyncAggregator)
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class HybridAggregatorConfig(AsyncAggregatorConfig):
    # number of clients to collect before taking a global step
    buffer_size: int = 1


@dataclass
class FedAvgWithLRHybridAggregatorConfig(HybridAggregatorConfig):
    _target_: str = fullclassname(FedAvgWithLRHybridAggregator)
    lr: float = 0.001
    momentum: float = 0.0


@dataclass
class FedAdamHybridAggregatorConfig(HybridAggregatorConfig):
    _target_: str = fullclassname(FedAdamHybridAggregator)
    lr: float = 0.001
    weight_decay: float = 0.00001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
