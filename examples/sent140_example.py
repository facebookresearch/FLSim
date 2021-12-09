#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train a binary sentiment classifier on LEAF's Sent140 dataset with FLSim.

Before running this file, you need to download the dataset, and partition the data by users. We
provided the script get_data.sh for such task.

    Typical usage example:

    FedAvg
    python3 sent140_tutorial.py --config-file configs/sent140_config.json

    FedBuff + SGDM
    python3 sent140_tutorial.py --config-file configs/sent140_fedbuff_config.json
"""
import itertools
import json
import re
import string
import unicodedata
from typing import List

import flsim.configs  # noqa
import hydra  # @manual
import torch
import torch.nn as nn
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataProvider,
    FLModel,
    MetricsReporter,
    LEAFDataLoader,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset


class CharLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        n_hidden,
        num_embeddings,
        embedding_dim,
        max_seq_len,
        dropout_rate,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.fc = nn.Linear(self.n_hidden, self.num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        seq_lens = torch.sum(x != (self.num_embeddings - 1), 1) - 1
        x = self.embedding(x)  # [B, S] -> [B, S, E]
        out, _ = self.lstm(x)  # [B, S, E] -> [B, S, H]
        out = out[torch.arange(out.size(0)), seq_lens]
        out = self.fc(self.dropout(out))  # [B, S, H] -> # [B, S, C]
        return out


class Sent140Dataset(Dataset):
    def __init__(self, data_root, max_seq_len):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.all_letters = {c: i for i, c in enumerate(string.printable)}
        self.num_letters = len(self.all_letters)
        self.UNK: int = self.num_letters

        with open(data_root, "r+") as f:
            self.dataset = json.load(f)

        self.data = {}
        self.targets = {}

        self.num_classes = 2

        # Populate self.data and self.targets
        for user_id, user_data in self.dataset["user_data"].items():
            self.data[user_id] = self.process_x(list(user_data["x"]))
            self.targets[user_id] = self.process_y(list(user_data["y"]))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str):
        if user_id not in self.data or user_id not in self.targets:
            raise IndexError(f"User {user_id} is not in dataset")

        return self.data[user_id], self.targets[user_id]

    def unicodeToAscii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in self.all_letters
        )

    def line_to_indices(self, line: str, max_seq_len: int):
        line_list = self.split_line(line)  # split phrase in words
        line_list = line_list
        chars = self.flatten_list([list(word) for word in line_list])
        # padding
        indices: List[int] = [
            self.all_letters.get(letter, self.UNK)
            for i, letter in enumerate(chars)
            if i < max_seq_len
        ]
        indices = indices + ([self.UNK] * (max_seq_len - len(indices)))
        return indices

    def process_x(self, raw_x_batch):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [self.line_to_indices(e, self.max_seq_len) for e in x_batch]
        x_batch = torch.LongTensor(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        return y_batch

    def split_line(self, line):
        """split given line/phrase into list of words
        Args:
            line: string representing phrase to be split

        Return:
            list of strings, with each string representing a word
        """
        return re.findall(r"[\w']+|[.,!?;]", line)

    def flatten_list(self, nested_list):
        return list(itertools.chain.from_iterable(nested_list))


def build_data_provider(data_config, drop_last=False):

    train_dataset = Sent140Dataset(
        data_root="leaf/data/sent140/data/train/all_data_0_01_keep_1_train_9.json",
        max_seq_len=data_config.max_seq_len,
    )
    test_dataset = Sent140Dataset(
        data_root="leaf/data/sent140/data/test/all_data_0_01_keep_1_test_9.json",
        max_seq_len=data_config.max_seq_len,
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=drop_last,
    )

    data_provider = DataProvider(dataloader)
    return data_provider, train_dataset.num_letters


def main_worker(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
):
    data_provider, num_letters = build_data_provider(data_config)

    model = CharLSTM(
        num_classes=model_config.num_classes,
        n_hidden=model_config.n_hidden,
        num_embeddings=num_letters + 1,
        embedding_dim=100,
        max_seq_len=data_config.max_seq_len,
        dropout_rate=model_config.dropout_rate,
    )
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metric_reporter=metrics_reporter,
        num_total_users=data_provider.num_users(),
        distributed_world_size=distributed_world_size,
    )

    trainer.test(
        data_iter=data_provider.test_data(),
        metric_reporter=MetricsReporter([Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="sent140_tutorial")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    main_worker(trainer_config, model_config, data_config)


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
