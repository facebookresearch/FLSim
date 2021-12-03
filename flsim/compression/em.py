#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import random
from collections import Counter
from typing import Tuple

import torch


class EM:
    """
    EM algorithm used to quantize the columns of W to minimize

                         ||W - W_quantized||^2

    Args:
        - W: weight matrix of size (n_features x n_samples)
        - n_iter: number of k-means iterations
        - n_centroids: number of centroids (size of codebook)
        - n_iter: number of E/M steps to perform
        - eps: for cluster reassignment when an empty cluster is found
        - max_tentatives for cluster reassignment when an empty cluster is found
        - verbose: print quantization error after each iteration

    Remark:
        - If one cluster is empty, the most populated cluster is split into
          two clusters.
    """

    def __init__(
        self,
        W: torch.Tensor,
        n_centroids: int = 256,
        n_iter: int = 20,
        eps: float = 1e-6,
        max_tentatives: int = 30,
        verbose: bool = False,
    ):
        self.W = W
        self.n_centroids = n_centroids
        self.n_iter = n_iter
        self.eps = eps
        self.max_tentatives = max_tentatives
        self.verbose = verbose
        self.objectives = []
        self.assignments = torch.Tensor()
        self.centroids = torch.Tensor()
        self._initialize_centroids()

    def learn(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs n_iter E/M steps.
        """
        self._initialize_centroids()
        for i in range(self.n_iter):
            self._step(i)
        return self.centroids, self.assignments

    def _initialize_centroids(self):
        """
        Initializes the centroids by sampling random columns from W.
        """

        n_features, n_samples = self.W.size()

        # centroids
        indices = torch.randint(low=0, high=n_samples, size=(self.n_centroids,)).long()
        self.centroids = self.W[:, indices].t()  # (n_centroids, n_features)

    def _step(self, i: int):
        """
        There are two standard steps for each iteration: expectation (E) and
        minimization (M). The E-step (assignment) is performed with an exhaustive
        search and the M-step (centroid computation) is performed with
        the exact solution.

        Args:
            - i: step number

        Notes:
            - The E-step heavily uses PyTorch broadcasting to speed up computations
              and reduce the memory overhead
        """

        # assignments (E-step)
        distances = self._compute_distances()  # (n_centroids, n_samples)
        self.assignments = torch.argmin(distances, dim=0)  # (n_samples)
        n_empty_resolved_clusters = self._resolve_empty_clusters()

        # centroids (M-step)
        for k in range(self.n_centroids):
            W_q = self.W[:, self.assignments == k]  # (n_features, size_of_cluster_k)
            self.centroids[k] = W_q.mean(dim=1)  # (n_features)

        # book-keeping
        obj = (self.centroids[self.assignments].t() - self.W).norm(p=2).item()
        self.objectives.append(obj)
        if self.verbose:
            print(
                f"Iteration: {i},\t"
                f"objective: {obj:.6f},\t"
                f"resolved empty clusters: {n_empty_resolved_clusters}"
            )

    def _compute_distances(self) -> torch.Tensor:
        """
        For every centroid c, computes

                          ||W - c[None, :]||_2

        Notes:
            - We rely on PyTorch's broadcasting to speed up computations
              and reduce the memory overhead
            - We use the following trick: ||a - b|| = ||a||^2 + ||b||^2 - 2 * <a, b>
        """

        W_sqr = (self.W ** 2).sum(0)  # (n_samples,)
        centroids_sqr = (self.centroids ** 2).sum(1)  # (n_centroids,)
        corr = self.centroids.mm(self.W)  # (n_centroids, n_samples)

        # return squared distances of size (n_centroids, n_samples)
        return W_sqr[None, :] + centroids_sqr[:, None] - 2 * corr

    def _resolve_empty_clusters(self) -> int:
        """
        If one cluster is empty, the most populated cluster is split into
        two clusters by shifting the respective centroids. This is done
        iteratively for a fixed number of tentatives.
        """

        # empty clusters
        counts = Counter(map(lambda x: x.item(), self.assignments))
        empty_clusters = set(range(self.n_centroids)) - set(counts.keys())
        n_empty_clusters = len(empty_clusters)

        tentatives = 0
        while len(empty_clusters) > 0:
            # given an empty cluster, find most populated cluster and split it into two
            empty_cluster = random.choice(list(empty_clusters))
            biggest_cluster = counts.most_common(1)[0][0]
            shift = torch.randn_like(self.centroids[biggest_cluster]) * self.eps
            self.centroids[empty_cluster] = self.centroids[biggest_cluster].clone()
            self.centroids[empty_cluster] += shift
            self.centroids[biggest_cluster] -= shift

            # recompute assignments
            distances = self._compute_distances()  # (n_centroids, n_samples)
            self.assignments = torch.argmin(distances, dim=0)  # (n_samples,)

            # check for empty clusters
            counts = Counter(map(lambda x: x.item(), self.assignments))
            empty_clusters = set(range(self.n_centroids)) - set(counts.keys())

            # increment tentatives
            if tentatives == self.max_tentatives:
                print(f"Could not resolve empty clusters, {len(empty_clusters)} left")
                raise EmptyClusterResolveError
            tentatives += 1

        return n_empty_clusters


class EmptyClusterResolveError(Exception):
    pass
