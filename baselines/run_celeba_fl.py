#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
Example usages:
FL Training with JSON config
buck run @mode/dev-nosan papaya/toolkit/simulation:run_celeba_fl -- --config-file \
fblearner/flow/projects/papaya/examples/hydra_configs/celeba_single_process.json

FL Training with YAML config (only with Hydra 1.1)
buck run @mode/dev-nosan papaya/toolkit/simulation:run_celeba_fl -- --config-dir \
fblearner/flow/projects/papaya/examples/hydra_configs --config-name celeba_single_process
"""
from typing import NamedTuple, Dict

import flsim.configs  # noqa
import hydra
import torch
from flsim.baselines.data_providers import LEAFDataLoader, LEAFDataProvider
from flsim.baselines.models.cnn import Resnet18, SimpleConvNet
from flsim.baselines.models.cv_model import FLModel
from flsim.baselines.utils import train_non_fl
from flsim.fb.metrics_reporters.cv_reporter import CVMetricsReporter
from flsim.fb.process_state import FBProcessState
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from primal.datasets.leaf.celeba import CelebaDataset
from torchvision import transforms

IMAGE_SIZE: int = 32


class CelebaOutput(NamedTuple):
    log_dir: str
    eval_scores: Dict[str, float]
    test_scores: Dict[str, float]


def build_data_provider(local_batch_size, user_dist, drop_last=False):
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = CelebaDataset(
        split="train",
        user_dist=user_dist,
        transform=transform,
        download=True,
    )
    eval_dataset = CelebaDataset(
        split="eval",
        user_dist=user_dist,
        transform=transform,
        download=True,
        image_root=train_dataset.image_root,
    )

    test_dataset = CelebaDataset(
        split="test",
        user_dist=user_dist,
        transform=transform,
        download=True,
        image_root=train_dataset.image_root,
    )

    print(
        f"Created datasets with {len(train_dataset)} train users, {len(eval_dataset)} eval users and {len(test_dataset)} test users"
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_size=local_batch_size,
        drop_last=drop_last,
    )
    data_provider = LEAFDataProvider(dataloader)
    print(f"Training clients in total: {data_provider.num_users()}")
    return data_provider


def main_worker(
    trainer_config,
    data_config,
    model_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
    fb_info=None,
    rank=0,
    is_unit_test=False,
) -> CelebaOutput:
    FBProcessState.getInstance(rank=rank, fb_info=fb_info)
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{rank}" if cuda_enabled else "cpu")

    model = (
        Resnet18(num_classes=2)
        if model_config.use_resnet
        else SimpleConvNet(
            in_channels=3, num_classes=2, dropout_rate=model_config.dropout
        )
    )
    # pyre-ignore[6]: Incompatible parameter type (device is a str)
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

    if is_unit_test:
        return CelebaOutput(
            log_dir=train_reporter.writer.log_dir,
            eval_scores={},
            test_scores={},
        )

    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        drop_last=False,
        user_dist=data_config.user_dist,
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
    return CelebaOutput(
        log_dir=train_reporter.writer.log_dir,
        eval_scores=train_metric,
        test_scores=test_metric,
    )


@hydra.main(config_path=None, config_name="celeba_config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    assert cfg.distributed_world_size == 1, "Distributed training is not yet supported."

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    if cfg.non_fl:
        cuda_enabled = torch.cuda.is_available() and cfg.use_cuda_if_available
        device = torch.device("cuda:0" if cuda_enabled else "cpu")
        model = (
            Resnet18(num_classes=2)
            if model_config.use_resnet
            else SimpleConvNet(
                in_channels=3, num_classes=2, dropout_rate=model_config.dropout
            )
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
        main_worker(
            trainer_config=trainer_config,
            data_config=data_config,
            model_config=model_config,
            use_cuda_if_available=cfg.use_cuda_if_available,
            distributed_world_size=cfg.distributed_world_size,
        )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
