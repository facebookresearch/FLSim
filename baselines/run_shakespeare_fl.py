#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
Example usages:
FL Training
buck run @mode/dev-nosan papaya/toolkit/simulation/baselines:run_shakespeare_fl -- --config-file \
fblearner/flow/projects/papaya/examples/hydra_configs/shakespeare_single_process.json

FL Training with YAML config (only with Hydra 1.1)
buck run @mode/dev-nosan papaya/toolkit/simulation:run_shakespeare_fl -- --config-dir \
fblearner/flow/projects/papaya/examples/hydra_configs --config-name shakespeare_single_process
"""
from typing import NamedTuple

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
from primal.datasets.leaf.shakespeare import ShakespeareDataset, ShakespeareModel


class ShakespeareOutput(NamedTuple):
    log_dir: str
    eval_scores: float
    test_scores: float


def train(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
    fb_info=None,
):
    FBProcessState.getInstance(rank=0, fb_info=fb_info)

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device("cuda:0" if cuda_enabled else "cpu")
    model = ShakespeareModel(
        seq_len=model_config.seq_len,
        n_hidden=model_config.n_hidden,
        num_classes=model_config.num_classes,
        dropout_rate=model_config.dropout_rate,
    )

    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")

    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        drop_last=False,
        user_dist=data_config.user_dist,
    )
    train_reporter = CVMetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])
    _, train_metric = trainer.train(
        data_provider=data_provider,
        metric_reporter=train_reporter,
        num_total_users=data_provider.num_users(),
        distributed_world_size=distributed_world_size,
        rank=0,
    )
    test_metric = trainer.test(
        data_iter=data_provider.test_data(),
        metric_reporter=CVMetricsReporter([Channel.STDOUT]),
    )

    return ShakespeareOutput(
        log_dir=train_reporter.writer.log_dir,
        eval_scores=train_metric,
        test_scores=test_metric,
    )


def build_data_provider(local_batch_size, user_dist, drop_last):

    train_dataset = ShakespeareDataset(split="train", user_dist=user_dist)
    eval_dataset = ShakespeareDataset(split="eval", user_dist=user_dist)
    test_dataset = ShakespeareDataset(split="test", user_dist=user_dist)

    dataloader = LEAFDataLoader(
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_size=local_batch_size,
        drop_last=drop_last,
    )

    data_provider = LEAFDataProvider(dataloader)
    return data_provider


@hydra.main(config_path=None, config_name="shakespeare_config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    assert cfg.distributed_world_size == 1, "Distributed training is not yet supported."

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    if cfg.non_fl:
        cuda_enabled = torch.cuda.is_available() and cfg.use_cuda_if_available
        device = torch.device("cuda:0" if cuda_enabled else "cpu")
        model = ShakespeareModel(
            seq_len=model_config.seq_len,
            n_hidden=model_config.n_hidden,
            num_classes=model_config.num_classes,
            dropout_rate=model_config.dropout_rate,
        )
        data_provider = build_data_provider(
            local_batch_size=data_config.local_batch_size,
            drop_last=False,
            user_dist=data_config.user_dist,
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
        train(
            trainer_config=trainer_config,
            data_config=data_config,
            model_config=model_config,
            use_cuda_if_available=cfg.use_cuda_if_available,
            distributed_world_size=cfg.distributed_world_size,
        )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
