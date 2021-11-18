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
    """
    Encapsulates a seq2seq model for benchmarking FL experiments.
    This includes a sequence based loss and perplexity measurement
    as an additional metric for reporting.
    """

    def __init__(
        self, model: nn.Module, pad_token_idx: int, device: Optional[str] = None
    ):
        self.model = model
        self.device = device
        self.pad_token_idx = pad_token_idx

    def fl_forward(self, batch) -> FLBatchMetrics:
        input = batch["features"]
        target, mask = torch.split(batch["labels"].long(), input.shape[1], 1)

        if self.device is not None:
            input, target, mask = (
                input.to(self.device),
                target.to(self.device),
                mask.to(self.device),
            )

        output = self.model(input)

        # Reshape `output`,`target` and `mask` to match loss inputs
        output = output.view(-1, output.shape[-1])
        target = target.flatten()
        mask = mask.flatten()

        num_examples = int(torch.sum(mask).item())

        # Filter `output` and `target` based on mask (discard padding based samples)
        nonzero_indices = torch.nonzero(mask, as_tuple=True)[0]
        output = torch.index_select(output, 0, nonzero_indices)
        target = torch.index_select(target, 0, nonzero_indices)

        loss = F.cross_entropy(output, target)

        output = output.detach().cpu()
        target = target.detach().cpu()

        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=output,
            targets=target,
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
        _, mask = torch.split(batch["labels"].long(), batch["features"].shape[1], 1)
        return int(torch.sum(mask).item())
