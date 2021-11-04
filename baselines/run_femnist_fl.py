#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
Example Usages:
FL Training with JSON config
buck run @mode/dev-nosan papaya/toolkit/simulation:run_femnist_fl -- --config-file \
fblearner/flow/projects/papaya/examples/hydra_configs/femnist_single_process.json

FL Training with YAML config (only with Hydra 1.1)
buck run @mode/dev-nosan papaya/toolkit/simulation:run_femnist_fl -- --config-dir \
fblearner/flow/projects/papaya/examples/hydra_configs --config-name femnist_single_process
"""
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
from primal.datasets.leaf.femnist import FEMNISTDataset
from torchvision import transforms


def build_data_provider(local_batch_size, user_dist, drop_last=False):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = FEMNISTDataset(
        split="train", user_dist=user_dist, transform=transform, download=True
    )
    test_dataset = FEMNISTDataset(
        split="test", user_dist=user_dist, transform=transform, download=True
    )
    print(
        f"Created datasets with {len(train_dataset)} train users and {len(test_dataset)} test users"
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=local_batch_size,
        drop_last=drop_last,
    )
    return LEAFDataProvider(dataloader)


def train(
    trainer_config,
    data_config,
    model_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
    fb_info=None,
    rank=0,
):
    FBProcessState.getInstance(rank=rank, fb_info=fb_info)
    metrics_reporter = CVMetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    print("Created metrics reporter")
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if model_config.use_resnet:
        model = Resnet18(num_classes=62, pretrained=False)
        model.backbone.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
    else:
        model = SimpleConvNet(1, 62)

    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")

    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        user_dist=data_config.user_dist,
        drop_last=False,
    )

    final_model, _ = trainer.train(
        data_provider=data_provider,
        metric_reporter=metrics_reporter,
        num_total_users=data_provider.num_users(),
        distributed_world_size=distributed_world_size,
        rank=0,
    )
    return final_model, metrics_reporter


@hydra.main(config_path=None, config_name="femnist_config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    assert cfg.distributed_world_size == 1, "Distributed training is not yet supported."

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    assert data_config.user_dist in {
        "iid",
        "niid",
    }, "User distribution must be iid or niid"

    if cfg.non_fl:
        cuda_enabled = torch.cuda.is_available() and cfg.use_cuda_if_available
        device = torch.device("cuda:0" if cuda_enabled else "cpu")

        model: torch.nn.Module = SimpleConvNet(in_channels=1, num_classes=62)
        if model_config.use_resnet:
            model = Resnet18(num_classes=62, pretrained=False)
            model.backbone.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        data_provider = build_data_provider(
            local_batch_size=data_config.local_batch_size,
            user_dist=data_config.user_dist,
            drop_last=False,
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
