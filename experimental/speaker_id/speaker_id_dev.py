#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import os
import pickle
import tempfile

import flsim.baselines.utils as butils
import flsim.common.active_user_selectors.simple_user_selector as aus
import flsim.data.data_sharder as dsh
import numpy as np
import papaya.toolkit.simulation.experimental.speaker_id.speaker_id_fl_dataset_utils as sdu
import papaya.toolkit.simulation.experimental.speaker_id.speaker_id_fl_model_utils as smu
import papaya.toolkit.simulation.experimental.speaker_id.speaker_id_training_utils as stu
import torch
from flsim.interfaces.metrics_reporter import Channel
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from portalai.speakerid.spkid import models, system
from portalai.speakerid.spkid.config import cfg
from portalai.speakerid.spkid.data_catalog import DATASET_CATALOG
from portalai.speakerid.spkid.data_sources import Disk


torch.manual_seed(0)
np.random.seed(0)
# to ensure random initialization of parameters is eliminated as a confounding factor


"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation:speaker_id_dev_bin -- --cfg /data/users/akashb/fbsource/fbcode/papaya/toolkit/simulation/experimental/speaker_id/poc_config.yaml --tdp /home/akashb/spkid_data_train --edp /home/akashb/spkid_data_eval --cuda --num_workers 0 --clients_per_round 1 --report_train_metrics --eval_epoch_frequency 1 --do_eval --shard_size 256 --num_shards 123 --local_batch_size 1000 --sharding_strategy random --global_optimizer_type Adam --optimizer_lr 0.2748

The --tdp and --edp arguments are optional, but not specifying them will require the download of large datasets from manifold and is not recommended. It is better to download them to local storage and specify the path instead.

--overfit flag must only be specified to ensure the model is trained and evaluated on the same dataset. This is typically done to screen for trivial bugs that prevent convergence.

When --sharding_strategy "column" is specified, each client sampled during each round of each epoch of FL will only train on data corresponding to one user in the speaker id dataset, and conversely, one user's data will only ever appear in its totality on one client.
This is useful to simulate the actual setting of the speaker ID model in production. However, when only a single client is sampled during each round (--clients_per_round 1), and sharding strategy is random, this should be equivalent to central server training.
"""


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a classification model")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Path to speaker ID config file.", type=str
    )
    parser.add_argument(
        "--tdp",
        dest="train_data_path",
        help="Path to the root directory containing the training data.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--edp",
        dest="eval_data_path",
        help="Path to the root directory containing the eval data.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--overfit",
        dest="overfit",
        action="store_true",
        help="Whether to overfit to same samples at train and test time. Note that "
        "this ensures the train and eval/test data is exactly the same.",
    )
    parser.set_defaults(overfit=False)
    parser.add_argument(
        "--cuda",
        dest="use_cuda_if_available",
        action="store_true",
        help="Whether to use cuda if it is available.",
    )
    parser.set_defaults(use_cuda_if_available=False)
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        help="Number of workers to use for loading data. This is NOT the number of clients used for FL. "
        "It is recommended to set this to 0 when debugging. 0 is also the default.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--optimizer_lr",
        dest="optimizer_lr",
        help="Learning rate for local optimizers on clients.",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--global_optimizer_lr",
        dest="global_optimizer_lr",
        help="Learning rate for optimizer on server.",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--local_optimizer_momentum",
        dest="local_optimizer_momentum",
        help="Optimizer momentum on client devices.",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--global_optimizer_momentum",
        dest="global_optimizer_momentum",
        help="Optimizer momentum on central server. Only relevant for certain global optimizers.",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--global_optimizer_beta1",
        dest="global_optimizer_beta1",
        help="Optimizer beta1 on central servers. Only relevant for certain global optimizers.",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--global_optimizer_beta2",
        dest="global_optimizer_beta2",
        help="Optimizer beta2 on central server. Only relevant for certain global optimizers.",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--global_optimizer_type",
        dest="global_optimizer_type",
        help="Federated aggregation strategy. Defaults to simple federated averaging.",
        default=str(butils.GlobalOptimizerType.SGD),
        choices=[
            str(butils.GlobalOptimizerType.FEDAVG),
            str(butils.GlobalOptimizerType.SGD),
            str(butils.GlobalOptimizerType.LAMB),
            str(butils.GlobalOptimizerType.LARS),
            str(butils.GlobalOptimizerType.ADAM),
        ],
        type=str,
    )
    parser.add_argument(
        "--clients_per_round",
        dest="users_per_round",
        help="Number of client devices to sample during each round within an epoch of "
        " FL by the central server. In practice, increasing learning rate along with the "
        "number of clients yields better results. Also, increasing clients per round when "
        " using column based sharding - which in this case ensures each client has data of "
        "only one speaker - is good for diversity of supervision per round of FL. However, "
        "if multiple epochs are run locally on each client, this may backfire. Also note "
        "that if client_page_size is also set, the number of samples processed per round "
        "will be <= users_per_round * client_page_size, with equality holding if "
        "client_page_size <= minimum number of samples on any client. This is useful for "
        "simulating a constant batch size during FL to maintain parity with central server "
        "settings. Large batches will require more epochs, but fewer rounds per epoch. Given "
        "that rounds are cheaper than epochs in terms of wall-clock time due to parallelization "
        "across multiple clients, smaller batches are expected to be more data efficient.",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        help="Number of epochs of FL on the central server.",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--user_epochs_per_round",
        dest="user_epochs_per_round",
        help="Number of epochs of training on local data by clients. It is recommended to "
        "restrict this to 1 in order to compare against  SGD on central server. Only increase "
        "this to test its effect before deploying to production. Higher values are of interest "
        "in production since they minimize communication overhead at the cost of compute overhead "
        "on specific devices and wall clock overhead for each round. Also, running multiple epochs "
        "on local clients means that federated averaging ceases to be equivalent to SGD (in addition "
        "to issues that specific sharding strategies might introduce), and no convergence guarantees "
        "hold when asynchronously updated models are averaged.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--always_keep_trained_model",
        dest="always_keep_trained_model",
        action="store_true",
    )
    parser.set_defaults(always_keep_trained_model=False)
    parser.add_argument(
        "--report_train_metrics", dest="report_train_metrics", action="store_true"
    )
    parser.set_defaults(report_train_metrics=True)
    parser.add_argument("--debug_output", dest="debug_output", action="store_true")
    parser.set_defaults(debug_output=False)
    parser.add_argument(
        "--max_clip_norm_normalized",
        dest="max_clip_norm_normalized",
        action="store_true",
    )
    parser.set_defaults(max_clip_norm_normalized=False)
    parser.add_argument(
        "--train_metrics_reported_per_epoch",
        dest="train_metrics_reported_per_epoch",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--eval_epoch_frequency", dest="eval_epoch_frequency", default=1, type=int
    )
    parser.add_argument("--do_eval", dest="do_eval", action="store_true")
    parser.set_defaults(do_eval=True)
    parser.add_argument(
        "--user_selector",
        dest="user_selector",
        help="Sampling strategy for clients. Defaults to random round robin to ensure "
        "coverage of all clients while having a different ordering of clients in each "
        "round. Note that the ideal choice here may depend on the sharding strategy chosen. "
        "See the help section associated with sharding_strategy parameter for details.",
        default=str(aus.ActiveUserSelectorType.random_round_robin),
        choices=[
            str(aus.ActiveUserSelectorType.uniformly_random),
            str(aus.ActiveUserSelectorType.sequential),
            str(aus.ActiveUserSelectorType.random_round_robin),
            str(aus.ActiveUserSelectorType.number_of_samples),
        ],
        type=str,
    )
    parser.add_argument(
        "--shard_size",
        dest="shard_size",
        help="Size of a shard for FL. Only relevant if sequential sharding strategy is "
        "chosen. It is useful to set shard size so that each client processes a single "
        "batch per epoch. Also, for speaker ID in particular, the model requires batches "
        "at least 2 samples due to the use of a batch norm layer. Thus, setting this to 1 "
        "is not recommended. If you do though, there is a hack to deal with it that simply "
        "replicates the sample to effectively create a shard size of 2.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num_shards",
        dest="num_shards",
        help="The number of shards that the data must be split into. This is only relevant "
        "for random, broadcast and round robin sharding. It is useful to set the number of "
        "shards so that no client receives a number of samples greater than the batch size "
        "locally on each client. This would ensure that each client applies a single update "
        "locally, as long as number of client epochs has been set to 1. For speaker ID in "
        "particular, shard size must be > 1 ideally so set this parameter accordingly. There "
        "is a hack to deal with singleton shards by replicating their sample.",
        type=int,
        default=314,
    )
    parser.add_argument(
        "--local_batch_size",
        dest="local_batch_size",
        help="The batch size for data processed locally on each client. It is worth setting "
        "this in consideration with num_shards or shard_size so that each client only processes "
        "one batch per epoch. If user_epochs_per_round is also set to 1, this sets up an "
        "equivalence with SGD, modulo any effects of the sharding strategy used (eg. column "
        "sharding will limit diversity of supervision by ensuring each client has only one "
        "speaker's data), need for maintaining batch sizes used by baselines etc. For speaker "
        "id, a batch must contain > 1 sample so set this parameter to be at least 2. There is "
        "a hack to overcome this by replicating the sample in a singleton batch.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--eval_batch_size",
        dest="eval_batch_size",
        help="The batch size for data processed locally on each client. It is worth setting "
        "this in consideration with num_shards or shard_size so that each client only processes "
        "one batch per epoch. If user_epochs_per_round is also set to 1, this sets up an "
        "equivalence with SGD, modulo any effects of the sharding strategy used (eg. column "
        "sharding will limit diversity of supervision by ensuring each client has only one "
        "speaker's data), need for maintaining batch sizes used by baselines etc.",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "opts",
        help="See lib/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--sharding_strategy",
        dest="sharding_strategy",
        help="How the data must be sharded for the purposes of FL training. "
        "Note that random sharding may produce empty shards since for each "
        "record in the dataset, the shard ID is sampled randomly WITH REPLACEMENT! "
        "This will almost certainly manifest when the number of shards is "
        "comparable to the number clients. This can lead to seemingly unrelated/cryptic "
        "errors later on. A viable alternative is to use round robin sharding along "
        "with uniformly random/random round robin user selector.",
        default=str(dsh.ShardingStrategyType.ROUND_ROBIN),
        choices=[
            str(dsh.ShardingStrategyType.COLUMN),
            str(dsh.ShardingStrategyType.RANDOM),
            str(dsh.ShardingStrategyType.BROADCAST),
            str(dsh.ShardingStrategyType.ROUND_ROBIN),
            str(dsh.ShardingStrategyType.SEQUENTIAL),
            sdu.COLUMN_CARDINALITY_RANDOM,
        ],
        type=str,
    )
    parser.add_argument(
        "--shard_cardinalitites_file",
        dest="shard_cardinalitites_file",
        help="Path to the pickle file containing the frequences of ",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    if args.eval_data_path is not None:
        DATASET_CATALOG["combined_eval"] = [
            Disk(
                clips=os.path.join(args.eval_data_path, "clips"),
                labels=os.path.join(args.eval_data_path, "all.csv"),
                veri_pairs=os.path.join(args.eval_data_path, "veri_pairs.txt"),
                probe_clips=os.path.join(args.eval_data_path, "td/clips")
                if not args.overfit
                else None,
                probe_labels=os.path.join(args.eval_data_path, "td/all.csv")
                if not args.overfit
                else None,
            )
        ]
    if args.train_data_path is not None:
        DATASET_CATALOG["hp12dftrim_ti_train_all"] = [
            Disk(
                clips=os.path.join(args.train_data_path, "clips")
                if not args.overfit
                else os.path.join(args.eval_data_path, "clips"),
                labels=os.path.join(args.train_data_path, "all.csv")
                if not args.overfit
                else os.path.join(args.eval_data_path, "all.csv"),
            )
        ]
    return args


def main(args):
    print("Called with args: {}".format(args))
    use_cuda_if_available = args.use_cuda_if_available and torch.cuda.is_available()
    if use_cuda_if_available:
        # To ensure determinism across multiple executions, assuming the same pytorch version, platform etc.
        # Note that even after setting random seeds for torch/numpy and the following flags, there are additional
        # sources of non-determinism, such as the order in which atomicAdd operations are process by CUDA runtime.
        # Such operations are prevalent in various torch functions.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    num_workers = args.num_workers if not args.use_cuda_if_available else 0
    # cuda can't be used with multiprocessing in the data loader
    report_channel = Channel.STDOUT

    if args.cfg_file is not None:
        print("Loading config file from {}".format(args.cfg_file))
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    print(cfg)
    system.init(
        rng_seed=cfg.SYSTEM.RNG_SEED, cudnn_benchmark=cfg.SYSTEM.CUDNN_BENCHMARK
    )
    with tempfile.TemporaryDirectory() as base_dir:
        print("Train base dir:", base_dir)
        shard_cardinalities = None
        if args.shard_cardinalitites_file is not None:
            shard_cardinalities = tuple(
                pickle.load(open(args.shard_cardinalitites_file, "rb"))
            )
        data_loader, num_classes, verif_pairs = sdu.get_loader(
            base_dir=base_dir,
            shard_size=args.shard_size,
            local_batch_size=args.local_batch_size,
            pin_memory=not use_cuda_if_available,
            num_workers=num_workers,
            use_cuda_if_available=use_cuda_if_available,
            overfit=args.overfit,
            sharding_strategy=args.sharding_strategy,
            num_shards=args.num_shards,
            shard_cardinalities=shard_cardinalities,
            eval_batch_size=args.eval_batch_size,
        )
        print("Done loading data!")
        print("Num classes = %d" % (num_classes))
        spkid_model = models.build(num_classes)
        model = smu.SpeakerIdFLModel(spkid_model)
        print("Created model")
        if use_cuda_if_available:
            model.fl_cuda()
        # need to create a metric reporter as well

        aggregator_dict = butils.create_global_aggregator(
            args.global_optimizer_type,
            args.global_optimizer_lr,
            args.global_optimizer_momentum,
            args.global_optimizer_beta1,
            args.global_optimizer_beta2,
        )

        metrics_reporter = stu.SpeakerIdMetricsReporter([report_channel])

        sync_trainer = SyncTrainer(
            model=model,
            cuda_enabled=use_cuda_if_available,
            config=SyncTrainerConfig(
                optimizer={
                    "lr": args.optimizer_lr,
                    "momentum": args.local_optimizer_momentum,
                },
                aggregator=aggregator_dict,
                users_per_round=args.users_per_round,
                epochs=args.epochs,
                user_epochs_per_round=args.user_epochs_per_round,
                always_keep_trained_model=args.always_keep_trained_model,
                train_metrics_reported_per_epoch=args.train_metrics_reported_per_epoch,
                report_train_metrics=args.report_train_metrics,
                max_clip_norm_normalized=args.max_clip_norm_normalized,
                eval_epoch_frequency=args.eval_epoch_frequency,
                do_eval=args.do_eval,
                active_user_selector={"type": args.user_selector},
                report_train_metrics_after_aggregation=True,
            ),
        )
        train_set = data_loader.fl_train_set()
        num_train_samples = len(train_set)
        eval_set = data_loader.fl_eval_set()
        test_set = data_loader.fl_test_set()
        print("Number of training samples is %d" % (num_train_samples))
        print("Number of eval samples is %d" % (len(eval_set)))
        print("Number of test samples is %d" % (len(test_set)))
        print("Beginning training")
        sync_trainer.train(
            data_provider=sdu.SpeakerIdFLDataProviderFromList(
                train_set, eval_set, test_set, model
            ),
            metric_reporter=metrics_reporter,
            num_total_users=num_train_samples,
            distributed_world_size=1,
        )
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
