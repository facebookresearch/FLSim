#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import random
from collections import Counter
from typing import Dict, List, Optional, Callable

from torch.utils.data import Subset
from tqdm import tqdm


class CustomSSLSharding:
    """Custom SSL sharder which can be adjusted based on different data split constraints
    The input dataset is the training set to be split into segments which are distributed across clients.
    The function shard_for_row() can be used to map each sample to its appropriate shard
    """

    def __init__(
        self,
        dataset,  # dataset must be a Pytorch map-style dataset
        num_clients: int,
        num_samples_skewness_client: Callable,  # mapping client to #examples/client (unnormalized)
        frac_server: Optional[float] = 0.0,  # Fraction of labeled data at server (0->1)
        frac_client_labeled: Optional[
            float
        ] = 0.0,  # Fraction of labeled data at client (0->1)
        iidness_client_labeled: Optional[
            float
        ] = 0.0,  # IIDness of labeled data at client (0->1)
        iidness_client_unlabeled: Optional[
            float
        ] = 0.0,  # IIDness of unlabeled data at client (0->1)
        iidness_server: Optional[
            float
        ] = 0.0,  # IIDness of (labeled) data at server (0->1)
        open_set: Optional[
            bool
        ] = False,  # Whether open-set phenomenon observed at client
        random_seed: Optional[int] = 1,
    ):
        self.dataset = dataset
        self.frac_server = frac_server
        self.frac_client_labeled = frac_client_labeled
        self.iidness_client_labeled = iidness_client_labeled
        self.iidness_client_unlabeled = iidness_client_unlabeled
        self.iidness_server = iidness_server
        self.num_samples_skewness_client = num_samples_skewness_client
        self.num_clients = num_clients
        self.open_set = open_set
        random.seed(random_seed)

        self.cls_pnt = {}  # Current pointer for each label's examples
        self.dataset_size = len(self.dataset)

        self._populate_dataset_map()
        self.labels_remaining = set(
            self.dataset_map.keys()
        )  # For sampling without replacement

    def _populate_dataset_map(self):
        self.dataset_map = {label: [] for label in self.dataset.targets}
        for idx in tqdm(range(self.dataset_size)):
            _, label = self.dataset[idx]
            self.dataset_map[label].append(idx)

    def draw_sample_with_label(self, label):
        # Draw sample with input label and advance the pointer
        if label not in self.cls_pnt:
            self.cls_pnt[label] = 0
        if self.cls_pnt[label] < len(self.dataset_map[label]):
            idx = self.dataset_map[label][self.cls_pnt[label]]
            self.cls_pnt[label] += 1
            return (idx, label)
        self.labels_remaining.discard(label)
        return None

    def generate_dataset_counts(self) -> Dict[int, int]:
        # Given counts and skewness determine which shards get how many examples
        shard_sizes = {}
        client_sizes = {}
        shard_sizes[0] = int(self.frac_server * self.dataset_size)
        shard_sizes[1] = 0
        norm_sum = 0.0
        for i in range(self.num_clients):  # 0->2,3 1->4,5 , 2->6,7
            client_sizes[i] = self.num_samples_skewness_client(i)
            norm_sum += client_sizes[i]
        # Normalize shard sizes for clients
        for i in range(self.num_clients):
            client_sizes[i] *= (self.dataset_size - shard_sizes[0]) / norm_sum
        for i in range(self.num_clients):
            shard_sizes[2 * i + 2] = int(client_sizes[i] * self.frac_client_labeled)
            shard_sizes[2 * i + 3] = int(client_sizes[i] - shard_sizes[2 * i + 2])
        return shard_sizes

    def generate_iidness_degree(self, iid: float, num_samples: int) -> List[int]:
        # Pick num_samples from classes determined by iid-ness
        samples = []
        labels_selected = random.sample(
            self.labels_remaining,
            k=min(max(1, int(iid * len(self.dataset_map))), len(self.labels_remaining)),
        )
        samples_allocated = label_idx = 0
        while samples_allocated < num_samples:
            label = labels_selected[label_idx]
            sample_selected = self.draw_sample_with_label(label)
            if sample_selected:
                samples.append(sample_selected)
                samples_allocated += 1
            label_idx = (label_idx + 1) % len(labels_selected)
            if self.labels_remaining.isdisjoint(set(labels_selected)):
                break
        return samples

    def create_sharding(
        self,
    ):  # Return a mapping of shard number -> list of (sample idx, sample labels)
        shard_sizes = self.generate_dataset_counts()
        self.shards = {}
        for shard_idx, num_samples in shard_sizes.items():
            if shard_idx == 0:
                self.shards[shard_idx] = self.generate_iidness_degree(
                    self.iidness_server, num_samples
                )
            elif shard_idx % 2 == 0:  # Labeled data
                self.shards[shard_idx] = self.generate_iidness_degree(
                    self.iidness_client_labeled, num_samples
                )
            else:
                self.shards[shard_idx] = self.generate_iidness_degree(
                    self.iidness_client_unlabeled, num_samples
                )

    def get_shard_dataset(self, shard_idx):
        return Subset(self.dataset, self.shards[shard_idx])

    def get_num_shards(self):
        return self.num_clients * 2 + 2

    def get_shard_summary(self, shard_idx):
        return {
            "shard_idx": shard_idx,
            "labeled": True if shard_idx % 2 == 0 else False,
            "num_samples": len(self.shards[shard_idx]),
            "class_freq": Counter([label for _, label in self.shards[shard_idx]]),
        }

    def get_allocation_utilization(self):
        # Returns fraction of samples allocated to a shard (tracks utilization)
        tot_allocated = sum(
            len(self.get_shard_dataset(shard_idx))
            for shard_idx in range(self.get_num_shards())
        )
        return tot_allocated / self.dataset_size
