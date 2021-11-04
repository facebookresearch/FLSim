#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import os
import random
from collections import defaultdict, namedtuple
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import flsim.data.data_sharder as dsh
import flsim.data.dataset_data_loader as ddl
import flsim.interfaces.data_loader as dl
import matplotlib.pyplot as plt
import numpy as np
import portalai.speakerid.spkid.data_frontend as df
import portalai.speakerid.spkid.data_transforms as sdt
import torch
from flsim.data.data_provider import FLDataProviderFromList
from flsim.data.data_sharder import (
    FLDataSharder,
    ShardingStrategyFactory,
    ShardingStrategyType,
)
from flsim.interfaces.dataset import FLDataset
from portalai.speakerid.spkid.config import cfg
from portalai.speakerid.spkid.data_transforms import (
    SpeakerDataset,
    build_torch_test_dataset,
)
from scipy import stats


COLUMN_CARDINALITY_RANDOM: str = "column_cardinality_random"


class SpeakerIdFLDataProviderFromList(FLDataProviderFromList):

    """
    We don't override train_data() herebecause that is dependent on data
    processing done by SpeakerIdFLDatasetDataLoaderWithBatch.fl_train_set().
    Thus, the inherited method from FLDataProviderFromList is retained.

    """

    def eval_data(
        self,
        # pyre-fixme[11]: Annotation `tensor` is not defined as a type.
    ) -> List[
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]
    ]:
        """
        Only works for SpeakerIdFLModel and speaker ID data
        """
        return [
            # pyre-fixme[16]: `IFLModel` has no attribute `fl_create_eval_batch`.
            self.model.fl_create_eval_batch(batch=batch)
            for batch in self.eval_user_list
        ]

    def test_data(
        self,
    ) -> List[
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]
    ]:
        return [
            # pyre-fixme[16]: `IFLModel` has no attribute `fl_create_test_batch`.
            self.model.fl_create_test_batch(batch=batch)
            for batch in self.test_user_list
        ]


class SpeakerIdFLDataset(FLDataset):
    shard_col_id = "user_id"
    data_id = "data"

    def __init__(self, speakerid_dataset: SpeakerDataset):
        self.speakerid_dataset = speakerid_dataset
        self.cache = {}

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        The original dataset returns either
        Tuple[int, torch.Tensor]
        or
        Tuple[int, torch.Tensor, str, str]

        The first integer is the user id
        The Tensor contains some processed form of the input audio file
        The last two optional str fields indicate the path of the input and the
        clip type. They are omitted by default.

        This method instead returns a dictionary containing what is called as
        "processed data" by the FL sharder. This structure is expected by
        the FL sharder.
        """
        if index not in self.cache:
            item = self.speakerid_dataset[index]
            self.cache[index] = item
        else:
            item = self.cache[index]
        return {
            SpeakerIdFLDataset.shard_col_id: torch.Tensor([item[0]])
            .type(torch.LongTensor)
            .to(device=item[1].device),
            SpeakerIdFLDataset.data_id: item[1],
        }

    def __len__(self) -> int:
        return len(self.speakerid_dataset)

    def get_num_classes(self) -> int:
        return self.speakerid_dataset.num_classes


class SpeakerIdValFLDataset(FLDataset):
    shard_col_id = SpeakerIdFLDataset.shard_col_id
    data_id = SpeakerIdFLDataset.data_id

    def __init__(
        self,
        speakerid_dataset: torch.utils.data.DataLoader,
        verif_pairs: List[Tuple[str, str, bool]],
    ):
        self.speakerid_dataset = speakerid_dataset
        self.verif_pairs = verif_pairs
        self.embs = {}
        for _, clip, (file_id,), _, user_id in self.speakerid_dataset:
            clip = clip[:, : cfg.TEST.PROBE_CLIP_LENGTH, :]
            self.embs[file_id] = (user_id, clip)
        self.index_to_verif_tuple = {
            index: val for index, val in enumerate(self.verif_pairs)
        }

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Returns a tuple containing
        (tensor for sample A, tensor for sample B,
         tensor of gold id for sample A, tensor of gold id for sample B,
         bool flag indicating if they are the same
        )
        TODO : @akashb @jesikmin
        Confirm that user ids in the eval set are 1-base indexed, as opposed ot the train set,
        where they are 0-base indexed
        """
        (afn, bfn, is_same) = self.index_to_verif_tuple[index]
        auid, afn_sample = self.embs[afn]
        buid, bfn_sample = self.embs[bfn]
        return (afn_sample, bfn_sample, auid, buid, is_same)

    def __len__(self) -> int:
        return len(self.index_to_verif_tuple)

    def get_num_classes(self) -> int:
        # pyre-fixme[16]: `DataLoader` has no attribute `num_classes`.
        return self.speakerid_dataset.num_classes


class SpeakerIdFLDatasetDataLoaderWithBatch(ddl.FLDatasetDataLoaderWithBatch):
    # TODO: (jesikmin) T58865105 [papaya][data] Add shuffling support to FL
    # dataset data loader
    def __init__(
        self,
        train_dataset: SpeakerIdFLDataset,
        test_dataset: SpeakerIdValFLDataset,
        eval_dataset: SpeakerIdValFLDataset,
        sharder: FLDataSharder,
        train_batch_size: int,
        eval_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ):
        assert train_batch_size > 0, "Batch size should be a positive integer."
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset
        self.sharder = sharder
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self._num_total_users: int = -1
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    @property
    def num_total_users(self):
        assert (
            self._num_total_users != -1
        ), "num_total_users is valid only after fl_train_set() has been called"
        return self._num_total_users

    def fl_train_set(self, **kwargs) -> Iterable[Iterable[Any]]:
        """
        Actually returns type Iterable[Iterable[Dict[str, torch.tensor]]]

        For each FL shard, for each batch within that shard, the batched processed data
        as a dictionary. In the Sync trainer, each shard corresponds to a client device's
        data.

        Processed data dictionary contains:
        {
            SpeakerIdFLDataset.shard_col_id: LongTensor of dim (batch_size, 1),
            SpeakerIdFLDataset.data_id: FloatTensor of dim (batch_size, ...)
        }

        Note that the speaker ID model doesn't tolerate singleton batches due to the use of
        batch norm. Thus, we replicate the sample in such batches in this wrapper function.
        It technically allows us to simulate clients with a single data sample. Apart from this,
        it compensates for the underlying data loader not dropping the last batch (which may be
        a single sample). This however is a hack.
        """
        final_train_batches = super().fl_train_set(**kwargs)
        ret_batches = []
        for orig_client_batches in final_train_batches:
            non_singleton_client_batches = []
            for client_batch in orig_client_batches:
                if client_batch[SpeakerIdFLDataset.data_id].shape[0] > 1:
                    non_singleton_client_batches.append(client_batch)
                else:
                    # TODO: @akashb
                    # This is a hack but since the speaker ID model doesn't handle batches with single element,
                    # the element is simply repeated.
                    new_client_batch = {
                        SpeakerIdFLDataset.shard_col_id: torch.cat(
                            2 * [client_batch[SpeakerIdFLDataset.shard_col_id]]
                        ),
                        SpeakerIdFLDataset.data_id: torch.cat(
                            2 * [client_batch[SpeakerIdFLDataset.data_id]]
                        ),
                    }
                    non_singleton_client_batches.append(new_client_batch)
            if len(non_singleton_client_batches) > 0:
                ret_batches.append(non_singleton_client_batches)
        return ret_batches

    def fl_eval_set(self, **kwargs) -> Iterable[Any]:
        collate_fn = kwargs.get("collate_fn", None)  # identity function
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def fl_test_set(self, **kwargs) -> Iterable[Any]:
        collate_fn = kwargs.get("collate_fn", None)  # identity function
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


def extract_data(
    base_dir: str, dataset_names: List[str]
) -> Dict[int, List[Tuple[str, str]]]:
    all_files = {}
    for data_set_name in dataset_names:
        source = df._find_source(data_set_name)
        speaker_files = source.get_local_files(base_dir)
        print(f"Total Ids from {data_set_name}: {len(speaker_files)}")
        for sid, items in speaker_files.items():
            all_files.setdefault(sid, []).extend(items)
    return all_files


def analyze_dataset(
    train_dataset: SpeakerIdFLDataset,
    # pyre-fixme[9]: desired_percentiles has type `Tuple[int]`; used as `Tuple[int,
    #  int, int, int]`.
    desired_percentiles: Tuple[int] = (50, 75, 90, 99),
):
    """
    Tracks the average and median number of samples per user.
    Plots the histogram of # samples vs. no. of users with those many samples
    Also tracks the variance in number of samples
    Outliers:
    - Users with most number of samples
    - Users with the least number of samples
    """
    uid_to_count = {}
    for sample in train_dataset:
        uid = sample[0][SpeakerIdFLDataset.shard_col_id][0].item()
        uid_to_count[uid] = uid_to_count.get(uid, 0.0) + 1.0
    count_to_num_users = {}
    for uid in uid_to_count.keys():
        count = uid_to_count[uid]
        count_to_num_users[count] = count_to_num_users.get(count, 0.0) + 1.0
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    counts = []
    for count in count_to_num_users.keys():
        for _i in range(int(count_to_num_users[count])):
            counts.append(count)
    counts = sorted(counts)
    num_users_with_count = [count_to_num_users[count] for count in counts]
    ax.bar(counts, num_users_with_count)
    ax.set_ylabel("Num users with given number of samples")
    ax.set_xlabel("Number of samples")
    ax.set_yticks(np.arange(0, counts[-1] + 10, 10))
    ax.set_title("Histogram of local dataset sizes when sharding by user ID")
    # plt.show()
    plt.savefig("/home/akashb/dump/count_hist.png", format="png")
    percentile_counts = []
    counts = np.array(counts)
    for pval in desired_percentiles:
        percentile_counts.append((pval, np.percentile(counts, pval)))
    top_outlier_percentile_thresh = np.percentile(counts, 90)
    bottom_outlier_percentile_thresh = np.percentile(counts, 10)
    top_outliers = []
    bottom_outliers = []
    for uid in uid_to_count.keys():
        count = uid_to_count[uid]
        if count > top_outlier_percentile_thresh:
            top_outliers.append(uid)
        elif count < bottom_outlier_percentile_thresh:
            bottom_outliers.append(uid)
    median_count = np.percentile(counts, 50)
    mean = np.mean(counts)
    var = np.var(counts)
    mode = stats.mode(counts)
    return (
        mean,
        median_count,
        mode,
        var,
        percentile_counts,
        top_outliers,
        bottom_outliers,
    )


@lru_cache(maxsize=None)
def get_test_loader(base_dir, num_workers=8, pin_memory=cfg.DATALOADER.PIN_MEMORY):
    base_dir = os.path.join(base_dir, "test")
    os.makedirs(base_dir, exist_ok=True)

    speaker_files = {}
    for eval_dataset in cfg.DATASETS.TEST:
        source = df._find_source(eval_dataset)
        curr_speaker_files = source.get_local_files(base_dir)
        for sid, items in curr_speaker_files.items():
            speaker_files.setdefault(sid, []).extend(items)

    speaker_files = source.get_local_files(base_dir)
    dataset = build_torch_test_dataset(speaker_files)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return data_loader


@lru_cache(maxsize=None)
def get_loader(
    base_dir: str,
    shard_size: int,
    local_batch_size: int,
    eval_batch_size: int,
    pin_memory: bool = False,
    num_workers: int = 0,
    use_cuda_if_available: bool = True,
    overfit: bool = False,
    sharding_strategy: str = dsh.ShardingStrategyType.COLUMN,
    num_shards: Optional[int] = None,
    shard_cardinalities: Optional[Tuple[int]] = None,
) -> Tuple[dl.IFLDataLoader, int, List[Tuple[str, str, bool]]]:
    train_base_dir = os.path.join(base_dir, "train")
    base_dir = os.path.join(base_dir, "test")
    os.makedirs(train_base_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    speaker_files = {}
    verif_pairs = []
    for eval_dataset in cfg.DATASETS.TEST:
        source = df._find_source(eval_dataset)
        curr_speaker_files = source.get_local_files(base_dir)
        for sid, items in curr_speaker_files.items():
            speaker_files.setdefault(sid, []).extend(items)
        curr_verif_pairs = df.get_verification_pairs(base_dir, eval_dataset)
        verif_pairs.extend(curr_verif_pairs)
    test_ids = speaker_files.keys()

    all_train_files = extract_data(train_base_dir, cfg.DATASETS.TRAIN)

    if not overfit:
        skip_ids = all_train_files.keys() & test_ids
        for sid in skip_ids:
            all_train_files.pop(sid)

    print(f"Total Train Ids:{len(all_train_files)}")
    ir_paths = (
        df.get_aug_dataset(cfg.DATASETS.IR_AUG, base_dir)
        if cfg.DATASETS.IR_AUG
        else None
    )
    noise_paths = (
        df.get_aug_dataset(cfg.DATASETS.BG_NOISE_AUG, base_dir)
        if cfg.DATASETS.BG_NOISE_AUG
        else None
    )

    train_dataset = SpeakerIdFLDataset(
        sdt.build_torch_train_dataset(all_train_files, ir_paths, noise_paths)
    )
    test_data_loader = get_test_loader(base_dir, num_workers, pin_memory=pin_memory)
    test_dataset = SpeakerIdValFLDataset(test_data_loader, verif_pairs)
    dev_dataset = SpeakerIdValFLDataset(test_data_loader, verif_pairs)
    # TODO : @akashb dev set should have no overlap with the test set.

    fl_data_sharder = SpeakerIdFlDataSharder(
        sharding_strategy,
        num_shards,
        None,
        SpeakerIdFLDataset.shard_col_id,
        shard_size,
        shard_cardinalities,
    )

    data_loader = SpeakerIdFLDatasetDataLoaderWithBatch(
        train_dataset,
        test_dataset,
        dev_dataset,
        fl_data_sharder,
        # TODO: @akashb
        # will revisit the proper batch size later when we revisit this flow
        # for flow testing, baselines, and etc
        # In the meantime, we will keep the previous behavior as-is.
        local_batch_size,
        eval_batch_size,
        eval_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return data_loader, train_dataset.get_num_classes(), verif_pairs


class SpeakerIdFlDataSharder(FLDataSharder):

    Config = namedtuple(
        "Config",
        [
            "sharding_strategy",
            "num_shards",
            "sharding_colindex",
            "sharding_col_name",
            "shard_size_for_sequential",
            "shard_cardinalities",
        ],
    )

    def __init__(
        self,
        sharding_strategy: str,
        num_shards: Optional[int] = None,
        sharding_colindex: Optional[int] = None,
        sharding_col_name: Optional[str] = None,
        shard_size_for_sequential: Optional[int] = None,
        shard_cardinalities: Optional[Tuple[int]] = None,
    ):
        """
        This initializer is a temporary workaround until we decide on Config
        design. Note that in order to avoid any PyText dependency, this
        initializer requires each property unpacked from FLDataSharderConfig,
        which currently has dependency on PyText's ConfigBase.
        """
        self.config = SpeakerIdFlDataSharder.Config(
            sharding_strategy,
            num_shards,
            sharding_colindex,
            sharding_col_name,
            shard_size_for_sequential,
            shard_cardinalities,
        )

    def shard_rows(
        self, data_rows: Iterable[Dict[str, Any]]
    ) -> Iterable[Tuple[str, List[Dict[str, Any]]]]:
        """Partition a set of rows into mulitple sets using a sharding strategy.

        Args:
            data_rows: Iterable[Dict[str, Any]]]: iterable over dictionary mapping column
            name to value.
        """
        shard_id_to_rows = defaultdict(list)
        sharding_strategy = SpeakerIdShardingStrategyFactory.create(self.config)
        for one_row in data_rows:
            for shard_id in sharding_strategy.shard_for_row(one_row):
                shard_id_to_rows[str(shard_id)].append(one_row)
        return shard_id_to_rows.items()

    class ColumnCardinalityRandomSharder(FLDataSharder.ShardingStrategy):
        """Specify a column name used to shard.
        It should be the last column in the file,
        and sharding_column must be specified.
        """

        def __init__(self, shard_cardinalities: Tuple[int]):
            self.shard_cardinalities = {}
            for id, cardinality in enumerate(shard_cardinalities):
                self.shard_cardinalities[id] = cardinality
            self.viable_shards = set(range(len(shard_cardinalities)))

        # pyre-fixme[14]: `shard_for_row` overrides method defined in
        #  `ShardingStrategy` inconsistently.
        def shard_for_row(self, csv_row: List[Any]) -> List[Any]:
            shard_idx = random.sample(self.viable_shards, 1)[0]
            self.shard_cardinalities[shard_idx] = (
                self.shard_cardinalities[shard_idx] - 1
            )
            if self.shard_cardinalities[shard_idx] <= 0:
                self.viable_shards.remove(shard_idx)
            return [shard_idx]


class SpeakerIdShardingStrategyFactory(ShardingStrategyFactory):
    @staticmethod
    # pyre-fixme[14]: `create` overrides method defined in `ShardingStrategyFactory`
    #  inconsistently.
    def create(config: SpeakerIdFlDataSharder.Config):
        if config.sharding_strategy == COLUMN_CARDINALITY_RANDOM:
            return SpeakerIdFlDataSharder.ColumnCardinalityRandomSharder(
                config.shard_cardinalities
            )
        if config.sharding_strategy == ShardingStrategyType.RANDOM:
            return FLDataSharder.RandomSharding(config.num_shards)
        elif config.sharding_strategy == ShardingStrategyType.BROADCAST:
            return FLDataSharder.BroadcastSharding(config.num_shards)
        elif config.sharding_strategy == ShardingStrategyType.COLUMN:
            # pyre-fixme[19]: Expected 1 positional argument.
            return FLDataSharder.ColumnSharding(
                config.sharding_colindex, config.sharding_col_name
            )
        elif config.sharding_strategy == ShardingStrategyType.ROUND_ROBIN:
            return FLDataSharder.RoundRobinSharding(config.num_shards)
        elif config.sharding_strategy == ShardingStrategyType.SEQUENTIAL:
            return FLDataSharder.SequentialSharding(config.shard_size_for_sequential)
        else:
            assert f"Invalid sharding strategy: {config.sharding_strategy}."
