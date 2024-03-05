# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import logging

from copy import deepcopy

from dataclasses import dataclass
from typing import List, Optional

from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import IdentityChannel, IFLChannel
from flsim.channels.message import Message
from flsim.common.logger import Logger
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import Metric
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import (
    ServerFTRLOptimizerConfig,
    ServerOptimizerConfig,
)
from flsim.privacy.common import (
    ClippingType,
    PrivacySetting,
    PrivateTrainingMetricsUtils,
)
from flsim.privacy.privacy_engine import CummuNoiseEffTorch, CummuNoiseTorch
from flsim.privacy.user_update_clip import IUserClipper
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import SyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class SyncFTRLServer(SyncServer):

    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IFLChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SyncFTRLServerConfig,
            **kwargs,
        )
        assert (
            self.cfg.aggregation_type == AggregationType.AVERAGE  # pyre-ignore[16]
        ), "As in https://arxiv.org/pdf/1710.06963.pdf, DP training must be done with simple averaging and uniform weights."

        assert (
            FLDistributedUtils.is_master_worker()
        ), "Distributed training is not supported for FTRL"

        self._optimizer = instantiate(
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
            record_last_noise=True,
        )
        self._global_model: IFLModel = global_model
        self._aggregator: Aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._active_user_selector = instantiate(self.cfg.active_user_selector)
        self._channel: IFLChannel = channel or IdentityChannel()
        self._restart_rounds = self.cfg.restart_rounds

        self._noise_std = self.cfg.privacy_setting.noise_multiplier
        self._clipping_value = self.cfg.privacy_setting.clipping.clipping_value

        self._user_update_clipper = IUserClipper.create_clipper(
            self.cfg.privacy_setting
        )
        self._shapes = [p.shape for p in global_model.fl_get_module().parameters()]
        self._device = next(global_model.fl_get_module().parameters()).device
        self._privacy_engine = None
        self._per_user_norms: List[float] = []
        if self.cfg.privacy_setting.clipping.clipping_type == ClippingType.ADAPTIVE:
            self.logger.warn(
                "Privacy accounting for adaptive clipping with DP-FTRL is unclear. Please note that the epsilon will be slightly higher with adaptive clipping."
            )

    def _create_tree(self, users_per_round):
        std = (self._noise_std * self._clipping_value) / users_per_round
        if self.cfg.efficient:
            self._privacy_engine = CummuNoiseEffTorch(
                std=std,
                shapes=self._shapes,
                device=self._device,
                test_mode=False,
                seed=self.cfg.privacy_setting.noise_seed,
            )
        else:
            self._privacy_engine = CummuNoiseTorch(
                std=std,
                shapes=self._shapes,
                device=self._device,
                test_mode=False,
                seed=self.cfg.privacy_setting.noise_seed,
            )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.active_user_selector, "_target_"):
            cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()
        if OmegaConf.is_missing(cfg.server_optimizer, "_target_"):
            cfg.server_optimizer = ServerFTRLOptimizerConfig()

    @property
    def global_model(self):
        return self._global_model

    def select_clients_for_training(
        self,
        num_total_users,
        users_per_round,
        data_provider: Optional[IFLDataProvider] = None,
        global_round_num: Optional[int] = None,
    ):
        if self._privacy_engine is None:
            self._create_tree(users_per_round)

        return self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_round_num=global_round_num,
        )

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()
        self._user_update_clipper.reset_clipper_stats()

        if self.should_restart():
            last_noise = None
            if self.cfg.tree_completion:
                actual_steps = self._privacy_engine.step * self._restart_rounds
                next_pow_2 = 2 ** (actual_steps - 1).bit_length()
                if next_pow_2 > actual_steps:
                    last_noise = self._privacy_engine.proceed_until(next_pow_2)
            self._optimizer.restart(last_noise)

    def receive_update_from_client(self, message: Message):
        message = self._channel.client_to_server(message)

        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

        self._user_update_clipper.clip(message.model.fl_get_module())
        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

    def mock_step(self):
        """
        Populates the grad attributes of the global model without performing
        an optimization step and returns the aggregated *sum* and not avearge.
        Useful for the CANIFE Empirical Privacy Measurement method.

        We populate p.grad with:
        buff = grad_sum_after + noise_after - (grad_sum_before + noise_before)
             = p.grad + noise_after - noise_before
        since grad_sum_after = grad_sum_before + p.grad
        """

        # mock step that updates the gradients of the global model
        self.step(mock=True)

        # get sum of gradients + noise for current mock round
        noise = self._privacy_engine()
        model = self._global_model.fl_get_module()

        for p, nz in zip(model.parameters(), noise):
            # copy difference of noisy sums in the grad attribute
            nz_before = self._optimizer.state[p]["last_noise"]
            p.grad.add_(nz - nz_before)

        # scale by sum_weights
        FLModelParamUtils.multiply_gradient_by_weight(
            model,
            self._aggregator.sum_weights.item(),
            model,
        )

        return model

    def step(self, mock=False) -> Optional[List[Metric]]:
        aggregated_model = self._aggregator.aggregate()
        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )

        if not mock:
            noise = self._privacy_engine()
            grad = deepcopy(aggregated_model)
            self._optimizer.step(noise)
            self._user_update_clipper.update_clipper_stats()

            return [
                Metric("Noise", sum([p.sum() for p in noise])),
                Metric(
                    "Signal_to_noise",
                    PrivateTrainingMetricsUtils.signal_to_noise_ratio(grad, noise),
                ),
                Metric("Num_Unclipped", self._user_update_clipper.unclipped_num),
            ]

    def should_restart(self):
        return (
            (self._privacy_engine is not None)
            and self._privacy_engine.step != 0
            and ((self._privacy_engine.step + 1) % self._restart_rounds) == 0
        )


@dataclass
class SyncFTRLServerConfig(SyncServerConfig):
    _target_: str = fullclassname(SyncFTRLServer)
    aggregation_type: AggregationType = AggregationType.AVERAGE
    server_optimizer: ServerOptimizerConfig = ServerFTRLOptimizerConfig()
    restart_rounds: int = 10000
    privacy_setting: PrivacySetting = PrivacySetting()
    efficient: bool = False
    tree_completion: bool = False
