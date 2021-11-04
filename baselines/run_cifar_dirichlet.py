#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""This file runs cifar10 partioned using dirichlet.

The dataset must be created beforehand using this notebook https://fburl.com/anp/d8nxmsc3.

  Typical usage example:

  buck run papaya/toolkit/simulation/baselines:run_cifar_dirichlet -- --config-file \\
  fbcode/fblearner/flow/projects/papaya/examples/hydra_configs/cifar10_dirichlet.json
"""
from typing import Dict, NamedTuple

import flsim.configs  # noqa
import hydra  # @manual
import torch
from flsim.baselines.data_providers import LEAFDataLoader, LEAFDataProvider
from flsim.baselines.models.cnn import Resnet18, SimpleConvNet
from flsim.baselines.models.cv_model import FLModel
from flsim.baselines.utils import train_non_fl
from flsim.fb.metrics_reporters.cv_reporter import CVMetricsReporter
from flsim.fb.process_state import FBProcessState
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from hydra.utils import instantiate  # @manual
from omegaconf import DictConfig, OmegaConf
from primal.datasets.cifar import CIFAR10
from primal.datasets.leaf.constants import MANIFOLD_BUCKET
from primal.datasets.utils import SimpleManifoldClient
from torchvision import transforms


class CIFAROutput(NamedTuple):
    log_dir: str
    eval_scores: Dict[str, float]
    test_scores: Dict[str, float]


def build_data_provider(data_config, drop_last=False):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    client = SimpleManifoldClient(MANIFOLD_BUCKET)
    with client.open(data_config.train_file) as f:
        train_dataset = torch.load(f)
    with client.open(data_config.eval_file) as f:
        eval_dataset = torch.load(f)
    test_dataset = CIFAR10(train=False, download=True, transform=transform)

    data_loader = LEAFDataLoader(
        train_dataset,
        eval_dataset,
        [list(zip(*test_dataset))],
        data_config.local_batch_size,
    )
    data_provider = LEAFDataProvider(data_loader)
    print(f"Clients in total: {data_provider.num_users()}")
    return data_provider


def train(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available=True,
    fb_info=None,
    world_size=1,
    rank=0,
):
    FBProcessState.getInstance(rank=rank, fb_info=fb_info)
    data_provider = build_data_provider(data_config)

    for _ in range(model_config.num_trials):
        metrics_reporter = CVMetricsReporter(
            [Channel.TENSORBOARD, Channel.STDOUT],
            target_eval=model_config.target_eval,
            window_size=model_config.window_size,
            average_type=model_config.average_type,
        )
        print("Created metrics reporter")

        cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
        device = torch.device(f"cuda:{rank}" if cuda_enabled else "cpu")
        print(f"Training launched on device: {device}")

        model = (
            Resnet18(num_classes=10)
            if model_config.use_resnet
            else SimpleConvNet(in_channels=3, num_classes=10)
        )
        global_model = FLModel(model, device)
        if cuda_enabled:
            global_model.fl_cuda()

        trainer = instantiate(
            trainer_config, model=global_model, cuda_enabled=cuda_enabled
        )
        print(f"Created {trainer_config._target_}")

        final_model, eval_score = trainer.train(
            data_provider=data_provider,
            metric_reporter=metrics_reporter,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
            rank=rank,
        )
        test_metric = trainer.test(
            data_iter=data_provider.test_data(),
            metric_reporter=CVMetricsReporter([Channel.STDOUT]),
        )
        if eval_score[CVMetricsReporter.ACCURACY] <= model_config.target_eval:
            break
    return CIFAROutput(
        log_dir=metrics_reporter.writer.log_dir,
        eval_scores=eval_score,
        test_scores=test_metric,
    )


@hydra.main(config_path=None, config_name="cifar10_single_process")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    if cfg.non_fl:
        cuda_enabled = torch.cuda.is_available() and cfg.use_cuda_if_available
        device = torch.device("cuda:0" if cuda_enabled else "cpu")
        model = (
            Resnet18(num_classes=10)
            if model_config.use_resnet
            else SimpleConvNet(in_channels=3, num_classes=10)
        )
        data_provider = build_data_provider(data_config=data_config)
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
            use_cuda_if_available=True,
            fb_info=None,
            world_size=1,
        )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
