#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
Example usages:
FL Training
buck run @mode/dev-nosan papaya/toolkit/simulation/baselines:run_sent140_fl -- --config-file \
fblearner/flow/projects/papaya/examples/hydra_configs/sent140_single_process.json

FL Training with YAML config (only with Hydra 1.1)
buck run @mode/dev-nosan papaya/toolkit/simulation:run_sent140_fl -- --config-dir \
fblearner/flow/projects/papaya/examples/hydra_configs --config-name sent140_single_process
"""
from typing import NamedTuple, Dict

import flsim.configs  # noqa
import hydra
import torch
from flsim.baselines.data_providers import LEAFDataLoader, LEAFDataProvider
from flsim.baselines.models.cv_model import FLModel
from flsim.baselines.utils import train_non_fl
from flsim.fb.metrics_reporters.cv_reporter import CVMetricsReporter
from flsim.fb.process_state import FBProcessState
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from primal.datasets.leaf.sent140 import (
    Sent140Dataset,
    Sent140StackedLSTMModel,
)


class Sent140Output(NamedTuple):
    log_dir: str
    eval_scores: Dict[str, float]
    test_scores: Dict[str, float]


def build_data_provider_vocab(
    local_batch_size, vocab_size, num_users, user_dist, max_seq_len, drop_last
):
    vocab_file = "embeddings_10k" if vocab_size == 10000 else "embeddings"
    train_dataset = Sent140Dataset(
        split="train",
        num_users=num_users,
        user_dist=user_dist,
        max_seq_len=max_seq_len,
        vocab_file=vocab_file,
    )
    eval_dataset = Sent140Dataset(
        split="eval",
        num_users=num_users,
        user_dist=user_dist,
        max_seq_len=max_seq_len,
        vocab=train_dataset.vocab,
    )
    test_dataset = Sent140Dataset(
        split="test",
        num_users=num_users,
        user_dist=user_dist,
        vocab=train_dataset.vocab,
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_size=local_batch_size,
        drop_last=drop_last,
    )

    data_provider = LEAFDataProvider(dataloader)
    return data_provider, train_dataset.vocab_size, train_dataset.embedding_size


def main_worker(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
    fb_info=None,
    rank=0,
):
    assert data_config.vocab_size in {
        10000,
        400000,
    }, "Sent140 only support two vocab sizes: 10k and 400k"

    data_provider, vocab_size, emb_size = build_data_provider_vocab(
        local_batch_size=data_config.local_batch_size,
        vocab_size=data_config.vocab_size,
        num_users=data_config.num_users,
        user_dist=data_config.user_dist,
        max_seq_len=data_config.max_seq_len,
        drop_last=False,
    )
    model = Sent140StackedLSTMModel(
        seq_len=data_config.max_seq_len,
        num_classes=model_config.num_classes,
        emb_size=emb_size,
        n_hidden=model_config.n_hidden,
        vocab_size=vocab_size,
        dropout_rate=model_config.dropout_rate,
    )

    FBProcessState.getInstance(rank=rank, fb_info=fb_info)

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{rank}" if cuda_enabled else "cpu")

    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")

    train_reporter = CVMetricsReporter(
        [Channel.TENSORBOARD, Channel.STDOUT],
        target_eval=model_config.target_eval,
        window_size=model_config.window_size,
        average_type=model_config.average_type,
    )

    _, train_metric = trainer.train(
        data_provider=data_provider,
        metric_reporter=train_reporter,
        num_total_users=data_provider.num_users(),
        distributed_world_size=distributed_world_size,
        rank=rank,
    )
    test_metric = trainer.test(
        data_iter=data_provider.test_data(),
        metric_reporter=CVMetricsReporter([Channel.STDOUT]),
    )
    return Sent140Output(
        log_dir=train_reporter.writer.log_dir,
        eval_scores=train_metric,
        test_scores=test_metric,
    )


@hydra.main(config_path=None, config_name="sent140_single_process")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    assert cfg.distributed_world_size == 1, "Distributed training is not yet supported."

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    if cfg.non_fl:
        cuda_enabled = torch.cuda.is_available() and cfg.use_cuda_if_available
        device = torch.device("cuda:0" if cuda_enabled else "cpu")

        data_provider, vocab_size, emb_size = build_data_provider_vocab(
            local_batch_size=data_config.local_batch_size,
            vocab_size=data_config.vocab_size,
            num_users=data_config.num_users,
            user_dist=data_config.user_dist,
            max_seq_len=data_config.max_seq_len,
            drop_last=False,
        )
        model = Sent140StackedLSTMModel(
            seq_len=data_config.max_seq_len,
            num_classes=model_config.num_classes,
            emb_size=emb_size,
            n_hidden=model_config.n_hidden,
            vocab_size=vocab_size,
            dropout_rate=model_config.dropout_rate,
        )

        train_non_fl(
            data_provider=data_provider,
            model=model,
            device=device,
            cuda_enabled=cuda_enabled,
            lr=trainer_config.client.optimizer.lr,
            momentum=trainer_config.client.optimizer.momentum,
            epochs=trainer_config.epochs,
        )
    else:
        main_worker(
            trainer_config=trainer_config,
            data_config=data_config,
            model_config=model_config,
            use_cuda_if_available=cfg.use_cuda_if_available,
            distributed_world_size=cfg.distributed_world_size,
            fb_info=None,
        )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
