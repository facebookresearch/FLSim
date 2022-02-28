from typing import Dict, NamedTuple

import flsim.configs  # noqa
import hydra  # @manual
import torch
from flsim.utils.example_utils import (
    CIFAR10UserData,
    CIFAR10DirichletDataProvider,
    CIFARDirichletDataLoader,
    SimpleConvNet,
    FLModel,
    MetricsReporter,
)
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from hydra.utils import instantiate  # @manual
from omegaconf import DictConfig, OmegaConf
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms
import random

class CIFAROutput(NamedTuple):
    log_dir: str
    eval_scores: Dict[str, float]
    test_scores: Dict[str, float]

def build_data_provider(data_config):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    with open(data_config.train_file, "rb+") as f:
        train_dataset = torch.load(f)

    test_dataset = CIFAR10(root="./data/", train=False, download=True, transform=transform)

    data_loader = CIFARDirichletDataLoader(
        train_dataset,
        [list(zip(*test_dataset))],
        [list(zip(*test_dataset))],
        data_config.local_batch_size,
    )
    data_provider = CIFAR10DirichletDataProvider(data_loader, data_config.eval_split)

    print(f"Clients in total: {data_provider.num_users()}")
    return data_provider


def train(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available: bool = True,
    world_size: int = 1,
    rank: int = 0,
) -> CIFAROutput:

    data_provider = build_data_provider(data_config)

    metrics_reporter = MetricsReporter(
        [Channel.TENSORBOARD, Channel.STDOUT],
    )
    print("Created metrics reporter")

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{rank}" if cuda_enabled else "cpu")
    print(f"Training launched on device: {device}")

    model = SimpleConvNet(in_channels=3, num_classes=10)
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
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
        metric_reporter=MetricsReporter([Channel.STDOUT]),
    )
    return CIFAROutput(
        log_dir=metrics_reporter.writer.log_dir,
        eval_scores=eval_score,
        test_scores=test_metric,
    )


@hydra.main(config_path=None, config_name="cifar10_single_process")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    train(
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config,
        use_cuda_if_available=True,
        world_size=1,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
