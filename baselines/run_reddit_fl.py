#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
Example usages:
FL Training with JSON config
buck run @mode/dev-nosan papaya/toolkit/simulation:run_reddit_fl -- --config-file \
fblearner/flow/projects/papaya/examples/hydra_configs/reddit_single_process.json

FL Training with YAML config (only with Hydra 1.1)
buck run @mode/dev-nosan papaya/toolkit/simulation:run_reddit_fl -- --config-dir \
fblearner/flow/projects/papaya/examples/hydra_configs --config-name reddit_single_process
"""
from typing import NamedTuple

import flsim.configs  # noqa
import hydra
import torch
from flsim.baselines.data_providers import LEAFDataLoader, LEAFDataProvider
from flsim.baselines.models.seq2seq_model import FLModel
from flsim.baselines.utils import train_non_fl_seq2seq
from flsim.fb.metrics_reporters.cv_reporter import Seq2SeqMetricsReporter
from flsim.fb.process_state import FBProcessState
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from primal.datasets.leaf.reddit import RedditDataset, RedditStackedLSTMModel


class RedditOutput(NamedTuple):
    log_dir: str
    eval_scores: float
    eval_perplexity: float
    test_scores: float
    test_perplexity: float


def build_data_provider_and_vocab(local_batch_size, drop_last, user_dist="niid"):

    train_dataset = RedditDataset(split="train", user_dist=user_dist)
    test_dataset = RedditDataset(split="test", user_dist=user_dist)
    eval_dataset = RedditDataset(split="eval", user_dist=user_dist)

    dataloader = LEAFDataLoader(
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_size=local_batch_size,
        drop_last=drop_last,
    )

    data_provider = LEAFDataProvider(dataloader)
    return data_provider, train_dataset.vocab_info


def train(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
    fb_info=None,
):
    FBProcessState.getInstance(rank=0, fb_info=fb_info)
    data_provider, vocab_info = build_data_provider_and_vocab(
        local_batch_size=data_config.local_batch_size,
        drop_last=False,
        user_dist=data_config.user_dist,
    )
    model = RedditStackedLSTMModel(
        n_hidden=model_config.n_hidden,
        num_layers=model_config.num_layers,
        seq_len=model_config.seq_len,
        vocab_info=vocab_info,
    )
    metrics_reporter = Seq2SeqMetricsReporter(
        vocab_info,
        [Channel.TENSORBOARD, Channel.STDOUT],
    )

    print("Created metrics reporter")
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")

    _, eval_metric = trainer.train(
        data_provider=data_provider,
        metric_reporter=metrics_reporter,
        num_total_users=data_provider.num_users(),
        distributed_world_size=distributed_world_size,
        rank=0,
    )

    test_metric = trainer.test(
        data_iter=data_provider.test_data(),
        metric_reporter=metrics_reporter,
    )

    return RedditOutput(
        log_dir=metrics_reporter.writer.log_dir,
        eval_scores=eval_metric[metrics_reporter.ACCURACY],
        test_scores=test_metric[metrics_reporter.ACCURACY],
        eval_perplexity=eval_metric[metrics_reporter.PERPLEXITY],
        test_perplexity=test_metric[metrics_reporter.PERPLEXITY],
    )


@hydra.main(config_path=None, config_name="reddit_config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    assert cfg.distributed_world_size == 1, "Distributed training is not yet supported."

    if cfg.non_fl:
        cuda_enabled = torch.cuda.is_available() and cfg.use_cuda_if_available
        device = torch.device("cuda:0" if cuda_enabled else "cpu")

        data_provider, vocab_info = build_data_provider_and_vocab(
            local_batch_size=cfg.data.local_batch_size,
            drop_last=False,
            user_dist=cfg.data.user_dist,
        )
        model = RedditStackedLSTMModel(
            n_hidden=cfg.model.n_hidden,
            num_layers=cfg.model.num_layers,
            seq_len=cfg.model.seq_len,
            vocab_info=vocab_info,
        )
        train_non_fl_seq2seq(
            data_provider=data_provider,
            model=model,
            device=device,
            cuda_enabled=cuda_enabled,
            lr=cfg.trainer.client.optimizer.lr,
            momentum=cfg.trainer.client.optimizer.momentum,
            epochs=cfg.trainer.epochs,
            vocab_info=vocab_info,
        )
    else:
        train(
            trainer_config=cfg.trainer,
            data_config=cfg.data,
            model_config=cfg.model,
            use_cuda_if_available=cfg.use_cuda_if_available,
            distributed_world_size=1,
        )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
