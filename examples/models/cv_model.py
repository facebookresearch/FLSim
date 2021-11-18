#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from flsim.examples.data.data_providers import LEAFUserData
from flsim.interfaces.model import IFLModel
from flsim.utils.simple_batch_metrics import FLBatchMetrics


class FLModel(IFLModel):
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device

    def fl_forward(self, batch) -> FLBatchMetrics:
        features = batch["features"]  # [B, C, 28, 28]
        batch_label = batch["labels"]
        stacked_label = batch_label.view(-1).long().clone().detach()
        if self.device is not None:
            features = features.to(self.device)

        output = self.model(features)

        if self.device is not None:
            output, batch_label, stacked_label = (
                output.to(self.device),
                batch_label.to(self.device),
                stacked_label.to(self.device),
            )

        loss = F.cross_entropy(output, stacked_label)
        num_examples = self.get_num_examples(batch)
        output = output.detach().cpu()
        stacked_label = stacked_label.detach().cpu()
        del features
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=output,
            targets=stacked_label,
            model_inputs=[],
        )

    def fl_create_training_batch(self, **kwargs):
        features = kwargs.get("features", None)
        labels = kwargs.get("labels", None)
        return LEAFUserData.fl_training_batch(features, labels)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return LEAFUserData.get_num_examples(batch["labels"])
