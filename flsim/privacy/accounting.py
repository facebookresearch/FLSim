# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Script for computing privacy of a model trained with DP-FTRL and DP-SGD.
The script applies the RDP accountant to estimate privacy budget of an iterated
Sampled Gaussian Mechanism.

The code is mainly based on Google's TF Privacy and https://colab.research.google.com/github/google-research/federated/blob/master/dp_ftrl/blogpost_supplemental_privacy_accounting.ipynb
"""
import collections
import math
from typing import Collection, Dict, Tuple, Union

import numpy as np
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


# RDP orders to consider
ORDERS = np.arange(1.1, 200, 0.1)


def _check_nonnegative(value: Union[int, float], name: str):
    if value < 0:
        raise ValueError(f"Provided {name} must be non-negative, got {value}")


def _check_possible_tree_participation(
    num_participation: int, min_separation: int, start: int, end: int, steps: int
) -> bool:
    """Check if participation is possible with `min_separation` in `steps`.

    This function checks if it is possible for a sample to appear
    `num_participation` in `steps`, assuming there are at least `min_separation`
    nodes between the appearance of the same sample in the streaming data (leaf
    nodes in tree aggregation). The first appearance of the sample is after
    `start` steps, and the sample won't appear in the `end` steps after the given
    `steps`.

    Args:
      num_participation: The number of times a sample will appear.
      min_separation: The minimum number of nodes between two appearance of a
        sample. If a sample appears in consecutive x, y steps in a streaming
        setting, then `min_separation=y-x-1`.
      start:  The first appearance of the sample is after `start` steps.
      end: The sample won't appear in the `end` steps after the given `steps`.
      steps: Total number of steps (leaf nodes in tree aggregation).

    Returns:
      True if a sample can appear `num_participation` with given conditions.
    """
    return start + (min_separation + 1) * num_participation <= steps + end


def _tree_sensitivity_square_sum(
    num_participation: int,
    min_separation: int,
    start: int,
    end: int,
    steps: int,
    hist_buffer: Dict[Tuple[int, int, int, int], float],
) -> float:
    """Compute the worst-case sum of sensitivtiy square for `num_participation`.

    This is the key algorithm for DP accounting for DP-FTRL tree aggregation
    without restart, which recurrently counts the worst-case occurence of a sample
    in all the nodes in a tree. This implements a dynamic programming algorithm
    that exhausts the possible `num_participation` appearance of a sample in
    `steps` leaf nodes. See Appendix D of
    "Practical and Private (Deep) Learning without Sampling or Shuffling"
    https://arxiv.org/abs/2103.00039.

    Args:
      num_participation: The number of times a sample will appear.
      min_separation: The minimum number of nodes between two appearance of a
        sample. If a sample appears in consecutive x, y steps in a streaming
        setting, then `min_separation=y-x-1`.
      start:  The first appearance of the sample is after `start` steps.
      end: The sample won't appear in the `end` steps after the given `steps`.
      steps: Total number of steps (leaf nodes in tree aggregation).
      hist_buffer: A dictionary stores the worst-case sum of sesentivity square
        keyed by (num_participation, start, end, steps).

    Returns:
      The worst-case sum of sesentivity square for the given input.
    """
    key_tuple = (num_participation, start, end, steps)
    if key_tuple in hist_buffer:
        return hist_buffer[key_tuple]
    if not _check_possible_tree_participation(
        num_participation, min_separation, start, end, steps
    ):
        sum_value = -np.inf
    elif num_participation == 0:
        sum_value = 0.0
    elif num_participation == 1 and steps == 1:
        sum_value = 1.0
    else:
        steps_log2 = math.log2(steps)
        max_2power = math.floor(steps_log2)
        if max_2power == steps_log2:
            sum_value = num_participation**2
            max_2power -= 1
        else:
            sum_value = 0.0
        candidate_sum = []
        for right_part in range(num_participation + 1):
            for right_start in range(min_separation + 1):
                left_sum = _tree_sensitivity_square_sum(
                    num_participation=num_participation - right_part,
                    min_separation=min_separation,
                    start=start,
                    end=right_start,
                    steps=2**max_2power,
                    hist_buffer=hist_buffer,
                )
                if np.isinf(left_sum):
                    candidate_sum.append(-np.inf)
                    continue  # Early pruning for dynamic programming
                right_sum = _tree_sensitivity_square_sum(
                    num_participation=right_part,
                    min_separation=min_separation,
                    start=right_start,
                    end=end,
                    steps=steps - 2**max_2power,
                    hist_buffer=hist_buffer,
                )
                candidate_sum.append(left_sum + right_sum)
        sum_value += max(candidate_sum)
    hist_buffer[key_tuple] = sum_value
    return sum_value


def _max_tree_sensitivity_square_sum(
    max_participation: int, min_separation: int, steps: int
) -> float:
    """Compute the worst-case sum of sensitivtiy square in tree aggregation.

    See Appendix D of
    "Practical and Private (Deep) Learning without Sampling or Shuffling"
    https://arxiv.org/abs/2103.00039.

    Args:
      max_participation: The maximum number of times a sample will appear.
      min_separation: The minimum number of nodes between two appearance of a
        sample. If a sample appears in consecutive x, y steps in a streaming
        setting, then `min_separation=y-x-1`.
      steps: Total number of steps (leaf nodes in tree aggregation).

    Returns:
      The worst-case sum of sesentivity square for the given input.
    """
    num_participation = max_participation
    while not _check_possible_tree_participation(
        num_participation, min_separation, 0, min_separation, steps
    ):
        num_participation -= 1
    candidate_sum, hist_buffer = [], collections.OrderedDict()
    for num_part in range(1, num_participation + 1):
        candidate_sum.append(
            _tree_sensitivity_square_sum(
                num_part, min_separation, 0, min_separation, steps, hist_buffer
            )
        )
    return max(candidate_sum)


def _compute_gaussian_rdp(
    sigma: float, sum_sensitivity_square: float, alpha: Collection[float]
) -> float:
    """Computes RDP of Gaussian mechanism."""
    if np.isinf(alpha):
        return np.inf
    # pyre-ignore[58]
    return alpha * sum_sensitivity_square / (2 * sigma**2)


def compute_rdp_single_tree(
    noise_multiplier: float,
    num_rounds: int,
    max_participation: int,
    min_separation: int,
    orders: Collection[float] = ORDERS,
) -> Tuple[Union[float, Collection[float]], float]:
    """Computes RDP of the Tree Aggregation Protocol for a single tree.

    The accounting assume a single tree is constructed for `num_rounds` leaf
    nodes, where the same sample will appear at most `max_participation` times,
    and there are at least `min_separation` nodes between two appearance. The key
    idea is to (recurrently) count the worst-case occurence of a sample
    in all the nodes in a tree, which implements a dynamic programming algorithm
    that exhausts the possible `num_participation` appearance of a sample in
    `rounds` leaf nodes.

    See Appendix D of
    "Practical and Private (Deep) Learning without Sampling or Shuffling"
    https://arxiv.org/abs/2103.00039.

    Args:
      noise_multiplier: A non-negative float representing the ratio of the
        standard deviation of the Gaussian noise to the l2-sensitivity of a single
        contribution (a leaf node).
      num_rounds: Total number of rounds (leaf nodes in tree aggregation).
      max_participation: The maximum number of times any device can participate in `num_rounds`.
      min_separation: The smallest number of training rounds completed in any `x` hour period, where x is the delay
      in how long a client should wait before participate in training.
      orders: An array (or a scalar) of RDP orders.

    Returns: The RDPs at all orders. Can be `np.inf`.
    """
    _check_nonnegative(noise_multiplier, "noise_multiplier")
    if noise_multiplier == 0:
        return np.inf
    _check_nonnegative(num_rounds, "num_rounds")
    _check_nonnegative(max_participation, "max_participation")
    _check_nonnegative(min_separation, "min_separation")
    sum_sensitivity_square = _max_tree_sensitivity_square_sum(
        max_participation, min_separation, num_rounds
    )
    if np.isscalar(orders):
        rdp = _compute_gaussian_rdp(noise_multiplier, sum_sensitivity_square, orders)
    else:
        rdp = np.array(
            [
                _compute_gaussian_rdp(
                    noise_multiplier,
                    sum_sensitivity_square,
                    # pyre-fixme[6]: For 3rd param expected `Collection[float]` but
                    #  got `float`.
                    alpha,
                )
                for alpha in orders
            ]
        )
    return rdp, sum_sensitivity_square


def compute_zcdp(noise_multiplier, sensitivity_sq):
    """Computes zCDP of the Tree Aggregation Protocol for a single tree, using
       Lemma 2.4 from https://arxiv.org/pdf/1605.02065.pdf.
    Args:
      noise_multiplier: A non-negative float representing the ratio of the
        standard deviation of the Gaussian noise to the l2-sensitivity of a single
        contribution (a leaf node).
      sensitivity_sq: The sum of squared sensitivity of the nodes in the binary tree, assuming
        each minibatch has \ell_2 sensitivity of one.
    Returns: zCDP parameter.
    """
    return sensitivity_sq / (2 * pow(noise_multiplier, 2))


def get_dp_sgd_eps(sigma, rounds, q=1.0, delta=1e-6):
    """
    Computes the privacy Epsilon at a given delta via RDP accounting and
    converting to an (epsilon, delta) guarantee for a target Delta.

    Args:
        q : The sampling rate in SGD
        sigma : The ratio of the standard deviation of the Gaussian
            noise to the L2-sensitivity of the function to which the noise is added
        rounds : The number of rounds:
        alphas : A list of RDP orders
        delta : Target delta
    """
    rdp = compute_rdp(q=q, noise_multiplier=sigma, steps=rounds, orders=ORDERS)
    eps, alpha = get_privacy_spent(orders=ORDERS, rdp=rdp, delta=delta)
    print(f"ε = {eps}")


def get_dp_ftrl_eps(sigma, rounds, max_participation, min_separation, delta):
    """
    Computes privacy Epsilon at a given delta and zCDP
    of the Tree Aggregation Protocol for a single tree.

    Args:
      sigma: A non-negative float representing the ratio of the
        standard deviation of the Gaussian noise to the l2-sensitivity of a single
        contribution (a leaf node).
      num_rounds: Total number of rounds (leaf nodes in tree aggregation).
      max_participation: The maximum number of times any device can participate in `num_rounds`.
      min_separation: The smallest number of training rounds completed in any `x` hour period, where x is the delay
      in how long a client should wait before participate in training.
      orders: An array (or a scalar) of RDP orders.
    """

    rdp, sensitivity_sq = compute_rdp_single_tree(
        sigma, rounds, max_participation, min_separation
    )
    eps, alpha = get_privacy_spent(orders=ORDERS, rdp=rdp, delta=delta)
    zcdp = compute_zcdp(sigma, sensitivity_sq)
    print(f"ε = {eps}")
    print(f"zCDP = {zcdp:.2f}")
