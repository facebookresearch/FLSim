# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""This class defines a synchronous MIMELite Client for the MIME framework.
Should be used in conjunction with the synchronous MIMELite server.
Needs the server_opt_state to function
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

from flsim.channels.message import Message
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.fl.common import FLModelParamUtils

from .base_client import Client, ClientConfig


class MimeLiteClient(Client):
    def generate_local_update(
        self,
        message: Message,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ) -> Tuple[IFLModel, float]:
        """
        Message: Must contain server_opt_state
        """
        assert (
            message.server_opt_state is not None
        ), "Server message must contain server_opt_state"
        # pyre-ignore [16]
        self.server_opt_state = message.server_opt_state
        model = message.model
        updated_model, weight, optimizer = self.copy_and_train_model(
            model, metrics_reporter=metrics_reporter
        )
        # 4. Store updated model if being tracked
        if self.store_last_updated_model:
            self.last_updated_model = FLModelParamUtils.clone(updated_model)
        # 5. compute delta
        delta = self.compute_delta(
            before=model, after=updated_model, model_to_save=updated_model
        )
        # 6. track state of the client
        self.track(delta=delta, weight=weight, optimizer=optimizer)
        return delta, weight

    def _reload_server_state(self, optimizer):
        state_dict = deepcopy(optimizer.state_dict())
        state_dict["state"] = deepcopy(self.server_opt_state)
        optimizer.load_state_dict(state_dict)
        del state_dict

    def _batch_train(
        self,
        model,
        optimizer,
        training_batch,
        epoch,
        metrics_reporter,
        optimizer_scheduler,
    ) -> int:
        """Trainer for FL Tasks using MIMELite framework.
        Uses server optimizer state at every step
        Run a single iteration of minibatch-gradient descent on a single user.
        Compatible with the new tasks in which the model is responsible for
        arranging its inputs, targets and context.
        Return number of examples in the batch.
        """
        optimizer.zero_grad()
        batch_metrics = model.fl_forward(training_batch)
        loss = batch_metrics.loss
        loss.backward()

        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        if self.cfg.max_clip_norm_normalized is not None:
            max_norm = self.cfg.max_clip_norm_normalized
            FLModelParamUtils.clip_gradients(
                max_normalized_l2_norm=max_norm, model=model.fl_get_module()
            )

        num_examples = batch_metrics.num_examples

        # reload server_opt_state
        # Skip MIMELite if train is directly called on the client, such as in personalization
        use_mimelite = (
            hasattr(self, "server_opt_state")
            # pyre-ignore [16]
            and self.server_opt_state is not None
        )
        if use_mimelite:
            self._reload_server_state(optimizer)
        else:
            self.logger.debug(
                "Skipping MIMELite. Personalization might be enabled or copy_and_train_model is directly called"
            )

        # adjust lr and take a step
        optimizer_scheduler.step(batch_metrics, model, training_batch, epoch)
        optimizer.step()

        if metrics_reporter is not None:
            metrics_reporter.add_batch_metrics(batch_metrics)

        return num_examples


@dataclass
class MimeLiteClientConfig(ClientConfig):
    _target_: str = fullclassname(MimeLiteClient)
