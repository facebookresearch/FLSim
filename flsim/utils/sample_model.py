#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from flsim.interfaces.model import IFLModel
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from torch import Tensor


class TestDataSetting:
    NUM_SHARDS: int = 10
    SHARDING_COL_INDEX: int = 2
    TEXT_COL_NAME: str = "text"
    LABEL_COL_NAME: str = "label"
    USER_ID_COL_NAME: str = "user_id"


class SimpleLinearNet(nn.Module):
    def __init__(self, D_in: int, D_out: int) -> None:
        """
        We create a simple linear model, with input dimension D_in and output dimension D_out
        """
        super(SimpleLinearNet, self).__init__()
        self.linear = nn.Linear(in_features=D_in, out_features=D_out, bias=False)

    def forward(self, x) -> Tensor:
        return self.linear(x)


class LinearFLModel(IFLModel):
    def __init__(
        self, D_in: int = 40, D_out: int = 1, use_cuda_if_available: bool = False
    ) -> None:
        """
        create a sample dummy FL model for alphabet dataset
        """
        self.model = SimpleLinearNet(D_in, D_out)
        self.use_cuda_if_available = use_cuda_if_available

    def fl_forward(self, batch) -> FLBatchMetrics:
        text = batch[TestDataSetting.TEXT_COL_NAME]
        batch_label = batch[TestDataSetting.LABEL_COL_NAME]
        # stacked_label = torch.tensor(batch_label.view(-1), dtype=torch.long)
        stacked_label = batch_label

        if self.use_cuda_if_available:
            text = text.cuda()

        out = self.model(text)
        if self.use_cuda_if_available:
            out, batch_label, stacked_label = (
                out.cuda(),
                batch[TestDataSetting.LABEL_COL_NAME].cuda(),
                stacked_label.cuda(),
            )

        loss = F.mse_loss(out, stacked_label)
        # loss = F.mse_loss(out, batch_label)
        num_examples = self.get_num_examples(batch)
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=out,
            targets=batch_label,
            model_inputs=batch,
        )

    def fl_create_training_batch(self, **kwargs) -> None:
        return kwargs.get("batch", None)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.cuda()

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return len(batch[TestDataSetting.LABEL_COL_NAME])


class TwoLayerNet(nn.Module):
    def __init__(self, D_in: int, H: int, D_out: int) -> None:
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        D_in: input dimension
        H: dimension of hidden layer
        D_out: output dimension
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x) -> Tensor:
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return F.log_softmax(y_pred, 1)


class DummyAlphabetFLModel(IFLModel):
    def __init__(
        self,
        embedding_size: int = 10,
        hidden_dim: int = 8,
        use_cuda_if_available: bool = False,
    ) -> None:
        """
        create a sample dummy FL model for alphabet dataset
        """
        self.model = TwoLayerNet(embedding_size, hidden_dim, 2)
        self.dummy_embedding = torch.rand(26, embedding_size)
        self.use_cuda_if_available = use_cuda_if_available

    def fl_forward(self, batch) -> FLBatchMetrics:
        text = batch[TestDataSetting.TEXT_COL_NAME]
        batch_label = batch[TestDataSetting.LABEL_COL_NAME]
        stacked_label = torch.tensor(batch_label.view(-1), dtype=torch.long)
        text_embeddings = self.dummy_embedding[text, :]

        if self.use_cuda_if_available:
            text_embeddings = text_embeddings.cuda()

        out = self.model(text_embeddings)
        if self.use_cuda_if_available:
            out, batch_label, stacked_label = (
                out.cuda(),
                batch[TestDataSetting.LABEL_COL_NAME].cuda(),
                stacked_label.cuda(),
            )

        loss = F.nll_loss(out, stacked_label)
        # produce a large loss, so gradients are large
        # this prevents unit tests from failing because of numerical issues
        loss.mul_(100.0)
        num_examples = self.get_num_examples(batch)
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=out,
            targets=batch_label,
            model_inputs=text_embeddings,
        )

    def fl_create_training_batch(self, **kwargs) -> None:
        return kwargs.get("batch", None)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.cuda()

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return len(batch[TestDataSetting.LABEL_COL_NAME])


class MockFLModel(IFLModel):
    r"""
    Mock IFLModel for testing that will return
    whatever the user pass into the constructor
    """

    def __init__(
        self,
        num_examples_per_user: int = 1,
        batch_labels: Optional[torch.Tensor] = None,
        model_output: Optional[torch.Tensor] = None,
        model_input: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = TwoLayerNet(10, 8, 2)
        self.num_examples_per_user = num_examples_per_user

        self.batch_labels = self._get_or_return_dummy_tensor(batch_labels)
        self.model_output = self._get_or_return_dummy_tensor(model_output)
        self.model_input = self._get_or_return_dummy_tensor(model_input)
        self.loss = self._get_or_return_dummy_tensor(loss)

    def fl_forward(self, batch) -> FLBatchMetrics:
        num_examples = self.get_num_examples(batch)
        return FLBatchMetrics(
            loss=self.loss,
            num_examples=num_examples,
            predictions=self.model_output,
            targets=self.batch_labels,
            model_inputs=self.model_input,
        )

    def fl_create_training_batch(self, **kwargs) -> None:
        return kwargs.get("batch", None)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        pass

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        return FLBatchMetrics(
            loss=self.loss,
            num_examples=self.num_examples_per_user,
            predictions=self.model_output,
            targets=self.batch_labels,
            model_inputs=self.model_input,
        )

    def get_num_examples(self, batch) -> int:
        return self.num_examples_per_user

    def _get_or_return_dummy_tensor(self, data: Optional[torch.Tensor]):
        return data if data is not None else torch.Tensor([1])


class BiasOnly(nn.Module):
    """This module has only a bias term
    It is useful for unit testing because the gradient will be constant.
    """

    def __init__(self):
        super(BiasOnly, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.bias


class ConstantGradientFLModel(IFLModel):
    def __init__(
        self,
    ) -> None:
        """A dummy model where the gradient is constant
        This is useful in testing because the parameters of the model move by a
        a constant amount in each optimizer.step()
        Initial value of self.model.bias = 0
        If using SGD, Final value of self.model.bias = lr * num_times_optimizer_step()_is_Called
        """
        self.model = BiasOnly()

    def fl_forward(self, batch) -> FLBatchMetrics:
        # self.model() will return the value of bias
        # we want the gradient to be negative, so self.model()['bias'] increase
        # with each optimizer.step
        loss = -1 * self.model()
        return FLBatchMetrics(
            loss=torch.Tensor(loss),
            num_examples=1,
            predictions=torch.Tensor(0),
            targets=torch.Tensor(0),
            model_inputs=torch.Tensor(0),
        )

    def fl_create_training_batch(self, **kwargs) -> None:
        return kwargs.get("batch", None)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.cuda()

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return 1
