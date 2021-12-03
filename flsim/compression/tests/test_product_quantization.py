#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import torch
from flsim.common.pytest_helper import assertEqual, assertRaises
from flsim.compression.em import EM, EmptyClusterResolveError
from flsim.compression.pq import PQ

torch.manual_seed(0)


def test_em_constant() -> None:
    """
    We test our k-means (EM) algorithm on a dummy dataset
    with 3 groups of 100 identical vectors and check that
    the learnt centroids represent the 3 groups.
    """

    # define W as 100 * 3 vectors of dimension 2, the first 100 are
    # filled with 1, the next 100 with 2 and the final 100 with 3
    in_features = 2
    out_features = 100
    seed = torch.ones(out_features, in_features)
    W = torch.cat([seed, 2 * seed, 3 * seed]).t()

    # we quantize with 2 centroids
    em = EM(W, n_centroids=3)
    em.learn()

    # we know the centroids, up to a permutation
    seed = torch.ones(in_features)
    true_centroids = torch.stack([seed, 2 * seed, 3 * seed])
    norm = (em.centroids.sort(dim=0).values - true_centroids).norm()
    assertEqual(norm, 0)


def test_em_error() -> None:
    """
    We try to cluster identical vectors with 3 centroids and check that
    our program raises an error.
    """

    W = torch.ones(2, 500)
    em = EM(W, n_centroids=2)
    with assertRaises(EmptyClusterResolveError):
        em.learn()


def test_pq_linear_constant() -> None:
    """
    We check that PQ on dummy vectors of dimension 2 with a block size 2
    in the linera case yields the expected centroids and that successively
    quantizing and dequantizing amounts to the identity operation.
    """

    # define W as 100 * 3 vectors of dimension 2, the first 100 are
    # filled with 1, the next 100 with 2 and the final 100 with 3
    in_features = 2
    out_features = 100
    seed = torch.ones(out_features, in_features)
    W = torch.cat([seed, 2 * seed, 3 * seed])

    # we quantize with block size 2 (each colun of W is split into 2)
    pq = PQ(W.size(), max_block_size=2, num_codebooks=1, max_num_centroids=3)
    centroids, assignments = pq.encode(W)

    # we now the centroids, up to a permutation
    seed = torch.ones(in_features)
    true_centroids = torch.stack([seed, 2 * seed, 3 * seed])
    norm = (centroids.sort(dim=0).values - true_centroids).norm()

    assertEqual(norm, 0)

    # we should recover exactly W
    diff = (pq.decode(centroids, assignments) - W).norm()
    assertEqual(diff, 0)


def test_pq_conv_constant() -> None:
    """
    We check that PQ on dummy vectors of dimension 9 with a block size 9
    in the convolutional case yields the expected centroids and that
    successively quantizing and dequantizing amounts to the identity operation.
    """

    # define W as 10 * 5 * 3 vectors of dimension 9, the first 100 are
    # filled with 1, the next 100 with 2 and the final 100 with 3
    in_features = 10
    out_features = 5
    kernel_size = 3
    seed = torch.ones(out_features, in_features, kernel_size, kernel_size)
    W = torch.cat([seed, 2 * seed, 3 * torch.ones(4, 10, 3, 3)])

    # we quantize with block size 2 (each colun of W is split into 2)
    pq = PQ(W.size(), max_block_size=9, num_codebooks=1, max_num_centroids=3)
    centroids, assignments = pq.encode(W)

    # we now the centroids, up to a permutation
    seed = torch.ones(kernel_size * kernel_size)
    true_centroids = torch.stack([seed, 2 * seed, 3 * seed])
    norm = (centroids.sort(dim=0).values - true_centroids).norm()
    assertEqual(norm, 0)

    # we should recover exactly W
    diff = (pq.decode(centroids, assignments) - W).norm()
    assertEqual(diff, 0)


def test_pq_many_dicts() -> None:
    """
    We check that PQ on dummy vectors of dimension 4 with a block size 2
    and two learnt codebooks in the linear case yields the expected centroids
    (6 in total, 3 per codebook) and that successively quantizing and
    dequantizing amounts to the identity operation.
    """

    # define W as 300 vectors of dimension 4, the first 100 are filled with
    # 1, the next 100 with 2 and the final 100 with 3
    in_features = 4
    out_features = 100
    seed = torch.ones(out_features, in_features)
    W = torch.cat([seed, 2 * seed, 3 * seed])

    # we quantize with block size 2 (each colun of W is split into 2)
    num_codebooks = 2
    pq = PQ(
        W.size(), max_block_size=2, num_codebooks=num_codebooks, max_num_centroids=3
    )
    centroids, assignments = pq.encode(W)

    # we now the centroids, up to a permutation
    seed = torch.ones(in_features // num_codebooks)
    true_centroids = torch.stack([seed, seed, 2 * seed, 2 * seed, 3 * seed, 3 * seed])
    norm = (centroids.sort(dim=0).values - true_centroids).norm()

    assertEqual(norm, 0)

    # we should recover exactly W
    diff = (pq.decode(centroids, assignments) - W).norm()
    assertEqual(diff, 0)


def test_pq_conv_block_size() -> None:
    """
    We test that the effective block size is 9 when providing a maximum
    block size of 10 in the convolutional case with 1 input feature.
    """

    # define W as 10 * 5 * 3 vectors of dimension 9, the first 100 are
    # filled with 1, the next 100 with 2 and the final 100 with 3
    in_features = 1
    out_features = 5
    kernel_size = 3
    seed = torch.ones(out_features, in_features, kernel_size, kernel_size)
    W = torch.cat([seed, 2 * seed, 3 * seed])

    # we quantize with block size 2 (each colun of W is split into 2)
    pq = PQ(W.size(), max_block_size=10, num_codebooks=1, max_num_centroids=3)
    centroids, assignemnts = pq.encode(W)

    # centroids should have dimension 9, which is the largest acceptable block size
    assertEqual(centroids.size(1), kernel_size * kernel_size)


def test_pq_linear_n_centroids() -> None:
    """
    We check that PQ on 100 random vectors yields a number of centroids equal
    to 16 when providing a larger max_num_centroids equal to 32. Indeed, we
    require n_centroids to be less than n_subvectors // 4 and a power of two.
    """

    # define W as 100 * 3 vectors of dimension 2, the first 100 are
    # filled with 1, the next 100 with 2 and the final 100 with 3
    in_features = 2
    out_features = 100
    W = torch.rand(out_features, in_features)

    # we quantize with block size 2 (each colun of W is split into 2)
    pq = PQ(W.size(), max_block_size=2, num_codebooks=1, max_num_centroids=32)
    centroids, assignments = pq.encode(W)

    # we now the centroids, up to a permutation
    n_centroids = centroids.size(0)
    assertEqual(n_centroids, 16)
