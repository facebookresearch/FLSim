#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
This file defines the concept of a differentially private
client where a sample level dp is enforced during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from flsim.channels.base_channel import IdentityChannel
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.timeout_simulator import TimeOutSimulator
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.model import IFLModel
from flsim.privacy.common import PrivacyBudget, PrivacySetting
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER
from opacus import PrivacyEngine


class DPClient(Client):
    def __init__(
        self,
        *,
        dataset: IFLUserData,
        channel: Optional[IdentityChannel] = None,
        timeout_simulator: Optional[TimeOutSimulator] = None,
        store_last_updated_model: Optional[bool] = False,
        name: Optional[str] = None,
        cuda_manager: ICudaStateManager = DEFAULT_CUDA_MANAGER,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=DPClientConfig,
            **kwargs,
        )

        super().__init__(
            dataset=dataset,
            channel=channel,
            timeout_simulator=timeout_simulator,
            store_last_updated_model=store_last_updated_model,
            name=name,
            cuda_manager=cuda_manager,
            **kwargs,
        )
        self.dataset_length = -1
        self.privacy_steps = 0
        self._privacy_budget = PrivacyBudget()
        self.privacy_on = (
            # pyre-fixme[16]: `DPClient` has no attribute `cfg`.
            self.cfg.privacy_setting.noise_multiplier >= 0
            and self.cfg.privacy_setting.clipping_value < float("inf")
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _get_dataset_stats(self, model: IFLModel):
        for batch in self.dataset:
            batch_size = model.get_num_examples(batch)
            break
        # pyre-fixme[61]: `batch_size` may not be initialized here.
        return batch_size, self.dataset.num_examples()

    @property
    def privacy_budget(self) -> PrivacyBudget:
        return self._privacy_budget

    def prepare_for_training(self, model: IFLModel):
        """
        1- call parent's prepare_for_training
        2- attach the privacy_engine
        """
        model, optimizer, optimizer_scheduler = super().prepare_for_training(model)
        if self.privacy_on:
            batch_size, self.dataset_length = self._get_dataset_stats(model)
            privacy_engine = PrivacyEngine(
                module=model.fl_get_module(),
                batch_size=batch_size,
                sample_size=self.dataset_length,
                # pyre-fixme[16]: `DPClient` has no attribute `cfg`.
                alphas=self.cfg.privacy_setting.alphas,
                noise_multiplier=self.cfg.privacy_setting.noise_multiplier,
                max_grad_norm=self.cfg.privacy_setting.clipping_value,
                delta=self.cfg.privacy_setting.target_delta,
            )
            if self.cfg.privacy_setting.noise_seed is not None:
                privacy_engine.random_number_generator = privacy_engine._set_seed(
                    self.cfg.privacy_setting.noise_seed
                )
            privacy_engine.steps = self.privacy_steps
            privacy_engine.attach(optimizer)
        return model, optimizer, optimizer_scheduler

    def _get_privacy_budget(self, optimizer) -> PrivacyBudget:
        if self.privacy_on and self.dataset_length > 0:
            eps, delta = optimizer.privacy_engine.get_privacy_spent()
            return PrivacyBudget(epsilon=eps, delta=delta)
        else:
            return PrivacyBudget()

    def post_batch_train(
        self, epoch: int, model: IFLModel, sample_count: int, optimizer: Any
    ):
        if self.privacy_on and sample_count > optimizer.privacy_engine.batch_size:
            raise ValueError(
                "Batchsize was not properly calculated!"
                " Calculated Epsilons are not Correct"
            )

    def post_train(self, model: IFLModel, total_samples: int, optimizer: Any):
        if not self.privacy_on:
            self.logger.debug(f"Privacy Engine is not enabled for client: {self.name}!")
            return
        if self.dataset_length != total_samples:
            DPClient.logger.warning(
                "Calculated privacy budgets were not Accurate." " Fixing the problem."
            )
            sample_rate = float(optimizer.privacy_engine.batch_size) / total_samples
            optimizer.privacy_engine.sample_rate = sample_rate

        self._privacy_budget = self._get_privacy_budget(optimizer)
        DPClient.logger.debug(f"Privacy Budget: {self._privacy_budget}")

        # book keeping
        privacy_engine = optimizer.privacy_engine
        self.privacy_steps = privacy_engine.steps
        # detach the engine to be safe, (not necessary if model is not reused.)
        privacy_engine.detach()
        # re-add the detached engine so that can be saved along with optimizer
        optimizer.privacy_engine = privacy_engine


@dataclass
class DPClientConfig(ClientConfig):
    """
    Contains configurations for a dp user (sample-level dp)
    """

    _target_: str = fullclassname(DPClient)
    privacy_setting: PrivacySetting = PrivacySetting()
