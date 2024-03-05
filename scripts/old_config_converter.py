#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import copy
import json
from typing import Any, Dict


def _handle_optimizer(trainer):
    if "optimizer" not in trainer:
        return

    trainer["client"] = trainer["client"] if "client" in trainer else {}
    client = trainer["client"]
    client["optimizer"] = trainer["optimizer"]
    del trainer["optimizer"]
    optimizer = client["optimizer"]

    if "type" not in optimizer:
        pass
    elif "sgd" == optimizer["type"].lower():
        optimizer["_base_"] = "base_optimizer_sgd"
    elif "fedprox" == optimizer["type"].lower():
        optimizer["_base_"] = "base_optimizer_fedprox"
    optimizer.pop("type", None)


def _handle_optimizer_in_client(client):
    if "optim_config" not in client:
        return

    client["optimizer"] = client["optim_config"]
    del client["optim_config"]
    optimizer = client["optimizer"]

    if "type" not in optimizer:
        pass
    elif "sgd" == optimizer["type"].lower():
        optimizer["_base_"] = "base_optimizer_sgd"
    elif "fedprox" == optimizer["type"].lower():
        optimizer["_base_"] = "base_optimizer_fedprox"
    optimizer.pop("type", None)


def _handle_lr_scheduler(trainer):
    if "local_lr_scheduler" not in trainer:
        return

    trainer["client"] = trainer["client"] if "client" in trainer else {}
    client = trainer["client"]
    client["lr_scheduler"] = trainer["local_lr_scheduler"]
    del trainer["local_lr_scheduler"]
    lr_scheduler = client["lr_scheduler"]

    if "type" not in lr_scheduler:
        pass
    elif "constant" == lr_scheduler["type"].lower():
        lr_scheduler["_base_"] = "base_constant_lr_scheduler"
    elif "default" == lr_scheduler["type"].lower():
        lr_scheduler["_base_"] = "base_constant_lr_scheduler"
    elif "batch_size_normalizer" == lr_scheduler["type"].lower():
        lr_scheduler["_base_"] = "base_lr_batch_size_normalizer_scheduler"
    elif "armijo_line_search" == lr_scheduler["type"].lower():
        lr_scheduler["_base_"] = "base_armijo_line_search_lr_scheduer"
    lr_scheduler.pop("type", None)


def _handle_lr_scheduler_in_client(client):
    if "local_lr_scheduler" not in client:
        return

    client["lr_scheduler"] = client["local_lr_scheduler"]
    del client["local_lr_scheduler"]
    lr_scheduler = client["lr_scheduler"]

    if "type" not in lr_scheduler:
        pass
    elif "constant" == lr_scheduler["type"].lower():
        lr_scheduler["_base_"] = "base_constant_lr_scheduler"
    elif "default" == lr_scheduler["type"].lower():
        lr_scheduler["_base_"] = "base_constant_lr_scheduler"
    elif "batch_size_normalizer" == lr_scheduler["type"].lower():
        lr_scheduler["_base_"] = "base_lr_batch_size_normalizer_scheduler"
    elif "armijo_line_search" == lr_scheduler["type"].lower():
        lr_scheduler["_base_"] = "base_armijo_line_search_lr_scheduer"
    lr_scheduler.pop("type", None)


def _handle_trainer_to_client_params(trainer):
    trainer["client"] = trainer["client"] if "client" in trainer else {}
    client = trainer["client"]

    if "user_epochs_per_round" in trainer:
        client["epochs"] = trainer["user_epochs_per_round"]
        del trainer["user_epochs_per_round"]

    if "max_clip_norm_normalized" in trainer:
        client["max_clip_norm_normalized"] = trainer["max_clip_norm_normalized"]
        del trainer["max_clip_norm_normalized"]
        if client["max_clip_norm_normalized"] is False:
            del client["max_clip_norm_normalized"]

    if "random_seed" in trainer:
        client["random_seed"] = trainer["random_seed"]
        del trainer["random_seed"]

    if "store_local_models_and_optimizers" in trainer:
        client["store_models_and_optimizers"] = trainer[
            "store_local_models_and_optimizers"
        ]
        del trainer["store_local_models_and_optimizers"]

    if "shuffle_user_batch_ordering" in trainer:
        client["shuffle_batch_order"] = trainer["shuffle_user_batch_ordering"]
        del trainer["shuffle_user_batch_ordering"]


def _handle_renaming_client_params(client):
    if (
        "max_clip_norm_normalized" in client
        and client["max_clip_norm_normalized"] is False
    ):
        del client["max_clip_norm_normalized"]

    if "store_local_models_and_optimizers" in client:
        client["store_models_and_optimizers"] = client[
            "store_local_models_and_optimizers"
        ]
        del client["store_local_models_and_optimizers"]

    if "shuffle_user_batch_ordering" in client:
        client["shuffle_batch_order"] = client["shuffle_user_batch_ordering"]
        del client["shuffle_user_batch_ordering"]


def _handle_timeout_simulator(trainer):
    if "timeout_simulator_config" not in trainer:
        return

    trainer["timeout_simulator"] = trainer["timeout_simulator_config"]
    del trainer["timeout_simulator_config"]
    timeout_simulator = trainer["timeout_simulator"]

    if "type" not in timeout_simulator:
        pass
    elif "never" == timeout_simulator["type"].lower():
        timeout_simulator["_base_"] = "base_never_timeout_simulator"
    elif "default" == timeout_simulator["type"].lower():
        timeout_simulator["_base_"] = "base_never_timeout_simulator"
    elif "gaussian" == timeout_simulator["type"].lower():
        timeout_simulator["_base_"] = "base_gaussian_timeout_simulator"
    timeout_simulator.pop("type", None)

    if "base_gaussian_timeout_simulator" == timeout_simulator.get("_base_", None):
        timeout_simulator["duration_distribution_generator"] = {}
        if "mean_per_example" in timeout_simulator:
            timeout_simulator["duration_distribution_generator"][
                "training_duration_mean"
            ] = timeout_simulator["mean_per_example"]
            del timeout_simulator["mean_per_example"]
        if "std_per_example" in timeout_simulator:
            timeout_simulator["duration_distribution_generator"][
                "training_duration_sd"
            ] = timeout_simulator["std_per_example"]
            del timeout_simulator["std_per_example"]
        if "min_duration_per_example" in timeout_simulator:
            timeout_simulator["duration_distribution_generator"][
                "training_duration_min"
            ] = timeout_simulator["min_duration_per_example"]
            del timeout_simulator["min_duration_per_example"]


def _handle_active_user_selector(trainer):
    if "active_user_selector" not in trainer:
        return

    active_user_selector = trainer["active_user_selector"]
    if "type" not in active_user_selector:
        pass
    elif "uniformly_random" == active_user_selector["type"].lower():
        active_user_selector["_base_"] = "base_uniformly_random_active_user_selector"
    elif "sequential" == active_user_selector["type"].lower():
        active_user_selector["_base_"] = "base_sequential_active_user_selector"
    elif "random_round_robin" == active_user_selector["type"].lower():
        active_user_selector["_base_"] = "base_random_round_robin_active_user_selector"
    elif "number_of_samples" == active_user_selector["type"].lower():
        active_user_selector["_base_"] = "base_number_of_samples_active_user_selector"
    elif "high_loss" == active_user_selector["type"].lower():
        active_user_selector["_base_"] = "base_high_loss_active_user_selector"
    elif "diversity_reporting" == active_user_selector["type"].lower():
        active_user_selector["_base_"] = "base_diversity_reporting_user_selector"
    elif "diversity_statistics_reporting" == active_user_selector["type"].lower():
        active_user_selector["_base_"] = (
            "base_diversity_statistics_reporting_user_selector"
        )
    elif "uniformlydiversity_maximizing_random" == active_user_selector["type"].lower():
        active_user_selector["_base_"] = "base_diversity_maximizing_user_selector"

    active_user_selector.pop("type", None)


def _handle_aggregator_reducer(aggregator):
    if "reducer_config" not in aggregator:
        return

    aggregator["reducer"] = aggregator["reducer_config"]
    del aggregator["reducer_config"]
    reducer = aggregator["reducer"]

    if "type" not in reducer:
        pass
    elif "roundreducer" == reducer["type"].lower():
        reducer["_base_"] = "base_reducer"
    elif "dproundreducer" == reducer["type"].lower():
        reducer["_base_"] = "base_dp_reducer"
    elif "secureroundreducer" == reducer["type"].lower():
        reducer["_base_"] = "base_secure_reducer"
    elif "weighteddproundreducer" == reducer["type"].lower():
        reducer["_base_"] = "base_weighted_dp_reducer"
    reducer.pop("type", None)

    if "fixedpoint_config" in reducer:
        reducer["fixedpoint"] = reducer["fixedpoint_config"]
        del reducer["fixedpoint_config"]

        if (
            len(reducer["fixedpoint"].keys()) != 1
            or "all-layers" not in reducer["fixedpoint"].keys()
        ):
            raise Exception(
                "per-layer config for fixedpoint in secure round reducer "
                "is no longer supported. Your (old) fixedpoint config should "
                "have all-layers as the key for this script to work. "
                "Please reach out to FL Simulator Users workplace group if "
                "you need per-layer fixedpoint config support."
            )

        reducer["fixedpoint"] = reducer["fixedpoint"]["all-layers"]
        reducer["fixedpoint"]["_base_"] = "base_fixedpoint"


def _handle_aggregator(trainer):  # noqa
    is_async_trainer = "async" in trainer["_base_"]
    if "aggregator" not in trainer:
        return

    aggregator = trainer["aggregator"]
    if "type" not in aggregator:
        pass
    elif "default" == aggregator["type"].lower() and not is_async_trainer:
        aggregator["_base_"] = "base_fed_avg_sync_aggregator"
    elif "fedavg" == aggregator["type"].lower() and not is_async_trainer:
        aggregator["_base_"] = "base_fed_avg_sync_aggregator"
    elif "fedavgwithlr" == aggregator["type"].lower() and not is_async_trainer:
        aggregator["_base_"] = "base_fed_avg_with_lr_sync_aggregator"
    elif "fedadam" == aggregator["type"].lower() and not is_async_trainer:
        aggregator["_base_"] = "base_fed_adam_sync_aggregator"
    elif "fedlars" == aggregator["type"].lower() and not is_async_trainer:
        aggregator["_base_"] = "base_fed_lars_sync_aggregator"
    elif "fedlamb" == aggregator["type"].lower() and not is_async_trainer:
        aggregator["_base_"] = "base_fed_lamb_sync_aggregator"
    elif "fedavgwithlr" == aggregator["type"].lower() and is_async_trainer:
        aggregator["_base_"] = "base_fed_avg_with_lr_async_aggregator"
    elif "fedadam" == aggregator["type"].lower() and is_async_trainer:
        aggregator["_base_"] = "base_fed_adam_async_aggregator"
    elif (
        "asyncfedavgwithlrmithmomentum" == aggregator["type"].lower()
        and is_async_trainer
    ):
        aggregator["_base_"] = "base_fed_avg_with_lr_with_momentum_async_aggregator"
    elif "hybridfedavgwithlr" == aggregator["type"].lower() and is_async_trainer:
        aggregator["_base_"] = "base_fed_avg_with_lr_hybrid_aggregator"
    elif "hybridfedadam" == aggregator["type"].lower() and is_async_trainer:
        aggregator["_base_"] = "base_fed_adam_hybrid_aggregator"
    aggregator.pop("type", None)

    _handle_aggregator_reducer(aggregator)


def _handle_training_start_time_distribution(teg):
    if "training_start_time_distr" not in teg:
        return

    teg["training_start_time_distribution"] = teg["training_start_time_distr"]
    del teg["training_start_time_distr"]
    tstd = teg["training_start_time_distribution"]

    if "type" not in tstd:
        pass
    elif "constant" == tstd["type"].lower():
        tstd["_base_"] = "base_constant_training_start_time_distribution"
    elif "poisson" == tstd["type"].lower():
        tstd["_base_"] = "base_poisson_training_start_time_distribution"
    tstd.pop("type", None)


def _handle_duration_distribution_generator(teg):
    if "training_duration_distr" not in teg:
        return

    teg["duration_distribution_generator"] = teg["training_duration_distr"]
    del teg["training_duration_distr"]
    ddg = teg["duration_distribution_generator"]

    if "type" not in ddg:
        pass
    elif "per_example_gaussian" == ddg["type"].lower():
        ddg["_base_"] = "base_per_example_gaussian_duration_distribution"
    elif "per_user_half_normal" == ddg["type"].lower():
        ddg["_base_"] = "base_per_user_half_normal_duration_distribution"
    elif "per_user_gaussian" == ddg["type"].lower():
        ddg["_base_"] = "base_per_user_gaussian_duration_distribution"
    elif "per_user_uniform" == ddg["type"].lower():
        ddg["_base_"] = "base_per_user_uniform_duration_distribution"
    elif "per_user_exponential" == ddg["type"].lower():
        ddg["_base_"] = "base_per_user_exponential_duration_distribution"
    elif "training_duration_from_list" == ddg["type"].lower():
        ddg["_base_"] = "base_duration_distribution_from_list"
    ddg.pop("type", None)


def _handle_training_event_generator(trainer):
    if "training_event_generator_config" not in trainer:
        return

    trainer["training_event_generator"] = trainer["training_event_generator_config"]
    del trainer["training_event_generator_config"]
    teg = trainer["training_event_generator"]

    if "type" not in teg:
        pass
    elif "async_training_event_generator" == teg["type"].lower():
        teg["_base_"] = "base_async_training_event_generator"
    elif "async_training_event_generator_from_list" == teg["type"].lower():
        teg["_base_"] = "base_async_training_event_generator_from_list"
    teg.pop("type", None)

    _handle_training_start_time_distribution(teg)
    _handle_duration_distribution_generator(teg)


def _handle_async_weight(trainer):  # noqa
    if (
        "staleness_weight_config" not in trainer
        and "example_weight_config" not in trainer
    ):
        return

    trainer["async_weight"] = {}
    async_weight = trainer["async_weight"]

    if "staleness_weight_config" in trainer:
        async_weight["staleness_weight"] = trainer["staleness_weight_config"]
        del trainer["staleness_weight_config"]
        staleness_weight = async_weight["staleness_weight"]

        if "type" not in staleness_weight:
            pass
        elif "default" == staleness_weight["type"].lower():
            staleness_weight["_base_"] = "base_constant_staleness_weight"
        elif "constant" == staleness_weight["type"].lower():
            staleness_weight["_base_"] = "base_constant_staleness_weight"
        elif "threshold" == staleness_weight["type"].lower():
            staleness_weight["_base_"] = "base_threshold_staleness_weight"
        elif "polynomial" == staleness_weight["type"].lower():
            staleness_weight["_base_"] = "base_polynomial_staleness_weight"
        staleness_weight.pop("type", None)

    if "example_weight_config" in trainer:
        async_weight["example_weight"] = trainer["example_weight_config"]
        del trainer["example_weight_config"]
        example_weight = async_weight["example_weight"]

        if "type" not in example_weight:
            pass
        elif "default" == example_weight["type"].lower():
            example_weight["_base_"] = "base_equal_example_weight"
        elif "equal" == example_weight["type"].lower():
            example_weight["_base_"] = "base_equal_example_weight"
        elif "linear" == example_weight["type"].lower():
            example_weight["_base_"] = "base_linear_example_weight"
        elif "sqrt" == example_weight["type"].lower():
            example_weight["_base_"] = "base_sqrt_example_weight"
        elif "log10" == example_weight["type"].lower():
            example_weight["_base_"] = "base_log10_example_weight"
        example_weight.pop("type", None)


def _handle_private_client_config(trainer):
    if "private_client_config" not in trainer:
        return

    # check if client is already present through trainer params
    trainer["client"] = {
        **trainer.get("client", {}),
        **trainer["private_client_config"],
    }
    del trainer["private_client_config"]
    client = trainer["client"]

    client["_base_"] = "base_dp_client"
    client.pop("type", None)

    _handle_renaming_client_params(client)
    _handle_optimizer_in_client(client)
    _handle_lr_scheduler_in_client(client)


def _handle_private_reducer_config(trainer):
    if "private_reducer_config" not in trainer:
        return

    trainer["reducer"] = trainer["private_reducer_config"]
    del trainer["private_reducer_config"]
    reducer = trainer["reducer"]

    if "type" not in reducer:
        pass
    elif "dproundreducer" == reducer["type"].lower():
        reducer["_base_"] = "base_dp_reducer"
    elif "weighteddproundreducer" == reducer["type"].lower():
        reducer["_base_"] = "base_weighted_dp_reducer"
    else:
        raise Exception("invalid reducer type for private sync trainer")
    reducer.pop("type", None)


def _handle_data_and_model(new_config):
    if "data_config" in new_config:
        new_config["data"] = new_config["data_config"]
        del new_config["data_config"]

    if "model_config" in new_config:
        new_config["model"] = new_config["model_config"]
        del new_config["model_config"]

    if "local_batch_size" in new_config:
        new_config["data"] = new_config.get("data", {})
        new_config["data"]["local_batch_size"] = new_config["local_batch_size"]
        del new_config["local_batch_size"]

    if "use_resnet" in new_config:
        new_config["model"] = new_config.get("model", {})
        new_config["model"]["use_resnet"] = new_config["use_resnet"]
        del new_config["use_resnet"]


def convert_old_fl_trainer_config_to_new(trainer):
    # handle trainer types
    if "synctrainer" == trainer["type"].lower():
        trainer["_base_"] = "base_sync_trainer"
    elif "asynctrainer" == trainer["type"].lower():
        trainer["_base_"] = "base_async_trainer"
    elif "privatesynctrainer" == trainer["type"].lower():
        trainer["_base_"] = "base_private_sync_trainer"
    del trainer["type"]

    if "channel_config" in trainer:
        trainer["channel"] = trainer["channel_config"]
        del trainer["channel_config"]

    # handle trainer --> client params
    _handle_optimizer(trainer)
    _handle_lr_scheduler(trainer)
    _handle_trainer_to_client_params(trainer)

    # handle trainer base params
    _handle_timeout_simulator(trainer)
    _handle_active_user_selector(trainer)

    # handle sync/async/private trainer params
    _handle_aggregator(trainer)
    _handle_training_event_generator(trainer)
    _handle_async_weight(trainer)

    # handle private trainer params
    _handle_private_client_config(trainer)
    _handle_private_reducer_config(trainer)


def get_new_pytext_fl_trainer_config(fl_trainer):
    new_config = {}
    new_config["trainer"] = fl_trainer
    trainer = new_config["trainer"]
    convert_old_fl_trainer_config_to_new(trainer)
    return new_config


def get_new_fl_config(
    old_config: Dict[str, Any], flsim_example: bool = False
) -> Dict[str, Any]:  # noqa
    new_config = copy.deepcopy(old_config)

    new_config["trainer"] = new_config["trainer_config"]
    del new_config["trainer_config"]
    trainer = new_config["trainer"]

    convert_old_fl_trainer_config_to_new(trainer)

    # specifically for fl examples and baseline
    if flsim_example:
        _handle_data_and_model(new_config)
        new_config = {"config": new_config}

    return new_config


def create_new_fl_config_from_old_json(
    old_config_json_path,
    new_config_json_path=None,
    flsim_example=False,
):
    new_config = {}
    with open(old_config_json_path) as f:
        old_config = json.load(f)
        new_config = get_new_fl_config(old_config, flsim_example)

    if new_config_json_path is None:
        print(new_config)
    else:
        with open(new_config_json_path, "w") as fo:
            json.dump(new_config, fo, indent=4)

    return new_config


def main() -> None:
    warning_msg = """ NOTE:\n
    -----\n
    THIS CONFIG CONVERTER IS A HACK AND IS NOT FULLY TESTED. \n
    DO NOT RELY ON THIS CONVERTER SCRIPT BEING AVAILABLE FOR A LONG TIME. \n
    IF YOU HAVE A LOT OF CONFIGS TO CONVERT, PLEASE DO SO ASAP. \n
    \n
    WARNING:\n
    --------\n
    THIS SCRIPT BLINDLY CONVERTS THE INPUT CONFIG TO THE NEW FORMAT AND DOES NOT VALIDATE \n
    THE INPUT. IF YOU SUPPLY THE WRONG CONFIG, THE ERROR MESSAGE WILL BE THROWN BY FLSIM \n
    AND NOT THIS SCRIPT.\n
    \n
    ======================================================================================\n
    \n
    """
    parser = argparse.ArgumentParser(
        description="Convert old FLSim JSON config to new format."
    )
    parser.add_argument("-o", "--old", type=str, help="path to old json config")
    parser.add_argument(
        "-n", "--new", type=str, default=None, help="path to new json config"
    )
    parser.add_argument(
        "--flsim_example",
        default=False,
        action="store_true",
        help="also modify data and model configs for flsim repo examples",
    )

    args = parser.parse_args()
    if args.old is None:
        parser.print_help()
        exit(1)

    print(warning_msg)
    create_new_fl_config_from_old_json(args.old, args.new, args.flsim_example)
    print("Conversion successful")


if __name__ == "__main__":
    main()  # pragma: no cover
