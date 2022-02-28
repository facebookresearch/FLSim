import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest
from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import IdentityChannel
from flsim.channels.half_precision_channel import HalfPrecisionChannel
from flsim.channels.message import Message
from flsim.common.pytest_helper import assertEqual, assertEmpty
from flsim.optimizers.server_optimizers import (
    FedAvgOptimizerConfig,
    FedAdamOptimizerConfig,
    FedAvgWithLROptimizerConfig,
    FedLARSOptimizerConfig,
    FedLAMBOptimizerConfig,
    OptimizerType,
)

from flsim.servers.aggregator import AggregationType
from flsim.servers.cd_server import CDServerConfig
from flsim.clients.bilevel_client import BiLevelClientConfig
from flsim.servers.sync_servers import SyncServerConfig
from flsim.tests.utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
    verify_models_equivalent_after_training,
    SampleNet,
)
from flsim.clients.cd_client import CDClientConfig
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from flsim.optimizers.local_optimizers import (
    LocalOptimizerProximalConfig,
    LocalOptimizerConfig,
)
from flsim.tests import utils
import torch

@dataclass
class MockClientUpdate:
    deltas: List[float]
    weights: List[float]

def create_client_updates(num_clients):
    deltas = [i + 1 for i in range(num_clients)]
    weights = [i + 1 for i in range(num_clients)]
    return MockClientUpdate(deltas, weights)

def fake_data(num_batches, batch_size):
    torch.manual_seed(0)
    dataset = [
        torch.ones(batch_size, 2) for _ in range(num_batches)
    ]
    dataset = utils.DatasetFromList(dataset)
    return utils.DummyUserData(dataset, utils.SampleNet(utils.TwoFC()))

def test_cd_server():
    num_rounds = 2
    w = 0

    cd_server = instantiate(
        CDServerConfig(
            server_optimizer=FedAvgWithLROptimizerConfig(lr=1.0)
        ),
        global_model=SampleNet(create_model_with_value(w)),
    )
    cd_server.num_users = 10
    cd_server.users_per_round = 10

    bilevel_server = instantiate(
        SyncServerConfig(
            aggregation_type=AggregationType.WEIGHTED_AVERAGE,
            server_optimizer=FedAvgWithLROptimizerConfig(lr=1.0),
        ),
        global_model=SampleNet(create_model_with_value(w)),
    )
    weights = [i + 1 for i in range(cd_server.users_per_round)]

    for round_num in range(num_rounds):
        cd_server.init_round()
        bilevel_server.init_round()

        if round_num == 0:
            cd_deltas = [i + 1 for i in range(cd_server.users_per_round)]
        else:
            cd_deltas = [0 for _ in range(cd_server.users_per_round)]

        if round_num == 0:
            bi_updates = [w - (i + 1) for i in range(cd_server.users_per_round)]
        else:
            bi_updates = [7 - (i + 1) for i in range(cd_server.users_per_round)]

        print("CD", round_num, cd_deltas)
        print("BL", round_num, bi_updates)
        for delta, weight in zip(cd_deltas, weights):
            cd_server.receive_update_from_client(
                Message(model=SampleNet(create_model_with_value(delta)), weight=weight)
            )
        for delta, weight in zip(bi_updates, weights):
            bilevel_server.receive_update_from_client(
                Message(model=SampleNet(create_model_with_value(delta)), weight=weight)
            )

        bilevel_server.step()
        cd_server.step()

    print(f"Bilevel {[p for p in bilevel_server.global_model.fl_get_module().parameters()]}")
    print(f"CD {[p for p in cd_server.global_model.fl_get_module().parameters()]}")


def test_cd_client():
    cd_client = instantiate(
        CDClientConfig(
            optimizer=LocalOptimizerProximalConfig(lr=1.0, lambda_=1.0)
        ),
        dataset=fake_data(num_batches=1, batch_size=1)
    )
    global_model = SampleNet(create_model_with_value(0))
    delta, weight = cd_client.generate_local_update(global_model)
    print("Delta First Time", [p for p in delta.fl_get_module().parameters()])
    delta, weight = cd_client.generate_local_update(global_model)
    print("Delta Second Time", [p for p in delta.fl_get_module().parameters()])

    delta, weight = cd_client.generate_local_update(global_model)
    print("Delta Third Time", [p for p in delta.fl_get_module().parameters()])

if __name__ == "__main__":
    # test_cd_client()
    test_cd_server()
