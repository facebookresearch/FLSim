#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import math

import torch
from sklearn.cluster import KMeans


class PQ:
    """
    Quantizes the layer weights W with the standard Product Quantization
    technique. This learns n_dict codebooks of codewords of size
    block_size from W.

    For further reference on using PQ to quantize neural networks, see
    "And the Bit Goes Down: Revisiting the Quantization of Neural Networks",
    ICLR 2020.

    PQ is performed in two steps:
    (1) The matrix W (weights or fully-connected or convolutional layer)
        is reshaped to (block_size, -1).
            - If W is fully-connected (2D), its rows are split into
              blocks of size block_size.
            - If W is convolutional (4D), its filters are split along the
              spatial dimension.
    (2) We apply the standard EM/k-means algorithm to the resulting reshaped matrix.

    Once W is reshaped to (block_size, n_samples), we learn num_codebooks codebooks
    each of size n_samples // num_codebooks (except the last which may have a variable
    size). More specifically, the first num_codebooks samples belong to the first dict,
    and so on. We use a trick to recover the quantized matrix from the knowledge of
    its centroids and assignments: we shift the assignments by a factor n_centroids
    every dict. See the decode() function.

    Args:
        - sizes: sizes of the weight matrix to quantize
        - max_block_size: max allowed block size (for the subvectors)
        - num_codebooks: number of dicts
        - max_num_centroids: max allowed number of centroids
        - num_k_means_iter: number of k-means iterations
        - verbose: print information after each iteration

    Notes:
        - PQ works for tensors that are on the CPU or on the GPU.
        - We need the original size of the weight matrix to decode, that's why
          we include it in the class state.
        - We compute internally the actual block_size in _determine_block_size.
          The actual block size is defined as the largest block size that is
          compatible with the shape of W while being less or equal than max_block_size.
        - We compute internally the actual number of centroids in _determine_num_centroids
          to avoid quantizing small layers with too much centroids.
    """

    def __init__(
        self,
        sizes: torch.Size,
        max_block_size: int = 9,
        num_codebooks: int = 1,
        max_num_centroids: int = 256,
        num_k_means_iter: int = 20,
        verbose: bool = False,
    ):
        self.sizes = sizes
        self.ndim = len(sizes)
        self.num_codebooks = num_codebooks
        self.num_k_means_iter = num_k_means_iter
        self.verbose = verbose
        self.block_size = self._determine_block_size(max_block_size)
        self.n_centroids = self._determine_num_centroids(max_num_centroids)

    def _determine_block_size(self, max_block_size):
        """
        Return the largest block size that is compatible with
        the shape of W while being less than or equal to max_block_size.
        """

        if self.ndim == 2:
            _out_features, in_features = self.sizes
            allowed_block_sizes = filter(
                lambda block_size: in_features % block_size == 0,
                range(1, max_block_size + 1),
            )
            block_size = list(allowed_block_sizes)[-1]

        elif self.ndim == 4:
            _out_channels, in_channels, kh, kw = self.sizes
            allowed_block_sizes = filter(
                lambda block_size: (in_channels * kh * kw) % block_size == 0,
                range(1, max_block_size + 1),
            )
            block_size = list(allowed_block_sizes)[-1]

        else:
            raise NotImplementedError(self.sizes)

        if self.verbose:
            print(f"Selected block size {block_size} for W of shape {self.sizes}")

        return block_size

    def _determine_num_centroids(self, max_num_centroids, max_centroid_factor_bound=4):
        """
        W is split into n_subvectors per dict. Returns n_centroids such that:
            - n_centroids is a power of two (greater or equal than 2)
            - n_centroids <= max_num_centroids
            - n_centroids * max_centroid_factor_bound < n_subvectors

        Notes:
            - This is to avoid quantizing small layers with too much centroids.
            - Must be called after determining self.block_size.
        """

        n_tot_subvectors = math.prod(self.sizes) // self.block_size
        n_subvectors = n_tot_subvectors // self.num_codebooks
        assert n_subvectors >= 8, "Not enough subvectors, consider not quantizing."
        n_centroids = 2 ** int(math.log2(n_subvectors // max_centroid_factor_bound))
        n_centroids = min(max_num_centroids, n_centroids)

        if self.verbose:
            print(f"Selected n_centroids {n_centroids} for W of shape {self.sizes}")

        return n_centroids

    def _reshape_and_split(self, W) -> torch.Tensor:
        """
        Reshapes the matrix W as expained in step (1).
        """

        # fully connected: by convention the weight has size out_features x in_features
        if self.ndim == 2:
            out_features, in_features = self.sizes
            assert (
                in_features % self.block_size == 0
            ), "Linear: in_features must be a multiple of block_size"
            W_unsplit = (
                W.reshape(out_features, -1, self.block_size)
                .permute(2, 1, 0)
                .flatten(1, 2)
            )

        # convolutional: we reshape along the spatial dimension
        elif self.ndim == 4:
            out_channels, in_channels, kh, kw = self.sizes
            assert (
                in_channels * kh * kw
            ) % self.block_size == 0, (
                "Conv: kernel_size kh * kw must be a multiple of block_size"
            )
            W_unsplit = (
                W.reshape(out_channels, -1, self.block_size)
                .permute(2, 0, 1)
                .flatten(1, 2)
            )

        # not implemented
        else:
            raise NotImplementedError(self.sizes)

        # split into self.num_codebooks blocks (last block may be larger)
        split = W_unsplit.size(1) // self.num_codebooks
        last_split = W_unsplit.size(1) - split * (self.num_codebooks - 1)
        splits = [split] * (self.num_codebooks - 1) + [last_split]
        return torch.split(W_unsplit, splits, dim=1)

    def _offset_assignments(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        See ``decode`` for an explanation and illustration.
        """

        n_assignments = len(assignments)
        subvectors_per_dict = int(math.ceil(n_assignments / self.num_codebooks))
        offset = torch.arange(
            0, self.num_codebooks * self.n_centroids, self.n_centroids
        )
        offset = offset.type_as(assignments)
        offset = offset.repeat_interleave(subvectors_per_dict)[:n_assignments]
        return assignments + offset

    def encode(self, W):
        """
        Performs num_k_means_iter EM steps as explained in step (2).
        """

        # reshape and split W as expained in step (1).
        W_reshaped = self._reshape_and_split(W)
        # compute centroids for all dicts
        all_centroids = []
        all_assignments = []
        for d in range(self.num_codebooks):
            if self.verbose:
                print(
                    f"Building dict {d+1}/{self.num_codebooks} with {self.n_centroids} "
                    f"centroids for {W_reshaped[d].size(1)} vectors"
                )

            # current weight
            W_curr = W_reshaped[d]

            # run k-means
            kmeans = KMeans(
                n_clusters=self.n_centroids,
                init="k-means++",
                n_init=1,
                max_iter=self.num_k_means_iter,
                tol=0.0001,
                verbose=self.verbose,
            )
            assignments = kmeans.fit_predict(W_curr.t())
            centroids = kmeans.cluster_centers_
            assignments = torch.LongTensor(assignments)
            centroids = torch.Tensor(centroids)

            # remember centroids and assignments
            all_centroids.append(centroids)
            all_assignments.append(assignments)

        # cat centroids and assignments
        assignments = torch.cat(all_assignments)
        assignments = self._offset_assignments(assignments)
        centroids = torch.cat(all_centroids)
        return centroids, assignments

    def decode(
        self,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the encoded full weight matrix. Must be called after
        the encode function.

        We offset assignments, let us illustrate this on an example.
        Say num_codebooks = 2 with 3 centroids per dict, and assume that
        assignments = [1, 2, 3, 3, 1, 1, 3, 2]. Then, after the offset
        the assignments would be [1, 2, 3, 3, 4, 4, 6, 5].

        Thus, we can call centroids[assignments] to properly recover W.

        Args:
            - centroids has size (num_codebooks x n_centroids, block_size)
            - assignments has size (n_samples)
        """

        # decode in the fully connected case
        if self.ndim == 2:
            out_features, _ = self.sizes
            return (
                centroids[assignments]
                .reshape(-1, out_features, self.block_size)
                .permute(1, 0, 2)
                .flatten(1, 2)
            )

        # decode in the convolutional case
        elif self.ndim == 4:
            out_channels, in_channels, kh, kw = self.sizes
            return (
                centroids[assignments]
                .reshape(-1, in_channels, self.block_size)
                .permute(0, 1, 2)
                .reshape(out_channels, in_channels, kh, kw)
            )

        # not implemented
        else:
            raise NotImplementedError(
                f"Only supports 2D convolutions and linear layers, but got size {self.sizes}"
            )
