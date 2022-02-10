#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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
import json
import re

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


class LSTMModel(nn.Module):
    def __init__(
        self, seq_len, num_classes, embedding_dim, n_hidden, vocab_size, dropout_rate
    ):
        super(LSTMModel, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim)
        self.stacked_lstm = nn.LSTM(
            self.embedding_dim,
            self.n_hidden,
            2,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.fc1 = nn.Linear(self.n_hidden, self.num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.out = nn.Linear(128, self.num_classes)

    def forward(self, features):
        seq_lens = torch.sum(features != (self.vocab_size - 1), 1) - 1
        x = self.embedding(features)
        outputs, _ = self.stacked_lstm(x)
        outputs = outputs[torch.arange(outputs.size(0)), seq_lens]
        pred = self.fc1(self.dropout(outputs))
        return pred


class Sent140Dataset(Dataset):
    def __init__(self, data_root, max_seq_len):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        with open(data_root, "r+") as f:
            self.dataset = json.load(f)

        self.data = {}
        self.targets = {}

        self.num_classes = 2
        self.word2id = self.build_vocab()
        self.vocab_size = len(self.word2id)

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

    def build_vocab(self):
        word2id = {}
        for user_data in self.dataset["user_data"].values():
            lines = [e[4] for e in user_data["x"]]
            for line in lines:
                line_list = self.split_line(line)
                for word in line_list:
                    if word not in word2id:
                        word2id[word] = len(word2id)
        return word2id

    def process_x(self, raw_x_batch):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [self.line_to_indices(e, self.max_seq_len) for e in x_batch]
        x_batch = torch.LongTensor(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        return y_batch

    def line_to_indices(self, line, max_words=25):
        unk_id = len(self.word2id)
        line_list = self.split_line(line)  # split phrase in words
        indl = [
            self.word2id[w] if w in self.word2id else unk_id
            for w in line_list[:max_words]
        ]
        indl += [unk_id - 1] * (max_words - len(indl))
        return indl

    def val_to_vec(self, size, val):
        assert 0 <= val < size
        vec = [0 for _ in range(size)]
        vec[int(val)] = 1
        return vec

    def split_line(self, line):
        return re.findall(r"[\w']+|[.,!?;]", line)


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
    return data_provider, train_dataset.vocab_size


def main_worker(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
):
    data_provider, vocab_size = build_data_provider(data_config)

    model = LSTMModel(
        num_classes=model_config.num_classes,
        n_hidden=model_config.n_hidden,
        vocab_size=vocab_size,
        embedding_dim=50,
        seq_len=data_config.max_seq_len,
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
