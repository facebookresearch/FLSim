#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.


"""An implementation of Trie Heavy Hitters (TrieHH).
This is intended to implement and simulate the Trie Heavy Hitters (TrieHH)
protocol presented in Federated Heavy Hitters Discovery with Differential Privacy.
https://arxiv.org/abs/1902.08534
"""

import json
import math
import os
import random
from collections import defaultdict

import numpy as np


class ServerState(object):
    def __init__(self):
        self.quit_sign = False
        self.trie = {}


class SimulateTrieHH(object):
    """Simulation for TrieHH."""

    def __init__(
        self,
        ARTIFACTS_PATH,
        max_word_len=10,
        epsilon=1.0,
        delta=2.3e-12,
        num_runs=5,
        alphabet_size=1,
        max_beam_size=-1,
    ):
        self.ARTIFACTS_PATH = ARTIFACTS_PATH
        self.MAX_L = math.ceil(
            max_word_len * 1.0 / alphabet_size
        )  # maximum number of rounds
        self.delta = delta
        self.epsilon = epsilon
        self.num_runs = num_runs
        self.alphabet_size = alphabet_size
        self.clients = []
        self.client_num = 0
        self.max_beam_size = max_beam_size
        self.server_state = ServerState()
        self._init_clients()
        self._set_theta()
        self._set_batch_size()
        self.verify_epsilon()

    def verify_epsilon(self, tolerance=0.1):
        """
        Following corollary 1 in the paper while setting gamma and theta
        should guarantee agreement with the theoretical epsilon governing
        privacy. In practice though, a divergence is observed, especially
        as theta approaches/exceeds the sqrt(self.client_num).

        It is also worth noting that the privacy utility tradeoff is a little
        odd here. Specifically, increasing epsilon arbitrarily does not lead
        to 100% precision and recall. This is because there is a tradeoff between
        three things: privacy, utility and successful convergence. In fact, it is
        possible for the chosen hyper-parameters to guarantee non-convergence. This
        is especially noticeable at high epsilons or if the number of rounds of
        the protocol is too small (due to large alphabet size). Another reason
        for sub-optimal precision and recall is that evaluation is done on wrt
        a gold list from a different problem: ranking of items by frequency. Heavy
        hitters invovles identifying an unordered set based on a non-rigorous
        definition of what it means to be a heavy hitter. A top ranked item by
        frequency may not cross the threshold desired.
        """
        gamma = self.batch_size / math.sqrt(self.client_num)
        srn = math.sqrt(self.client_num)
        if self.theta < 4 or self.theta > srn or gamma > srn / (self.theta + 1):
            raise Exception("Conditions not met!")
        theoretical_epsilon = self.MAX_L * np.log(
            1 + (1.0 / (-1 + (srn / (gamma * self.theta))))
        )
        if abs(self.epsilon - theoretical_epsilon) / self.epsilon > tolerance:
            raise Exception(
                "Deviation from theoretical epsilon is "
                + str(abs(self.epsilon - theoretical_epsilon) * 100.0 / self.epsilon)
                + " percent!"
            )

    def _init_clients(self):
        """Initialization of the dictionary."""
        with open(self.ARTIFACTS_PATH + "clients_triehh.txt", "r") as fp:
            self.clients = json.loads(fp.read())
        self.client_num = len(self.clients)
        # print(f"Total number of clients: {self.client_num}")

    def lambert_W0(self, x, prec=1e-12, maxiters=200):
        """
        Lambert's omega function seeks to solve for w that satisfies:
        w * exp(w) = x

        This isn't a function in the range (-1/e, 0) since it takes
        multiple values. However, we are primarily interested in the
        domain (0, +INF) in which it does behave like a function. That
        said, it is still not possible to solve for analytically, so
        we resort to numerical optimization techniques to solve for it.

        This function evaluates the principal branch of lambert's omega
        function up to the desired precision. It is possible that we
        fail to converge, in which case an exception is thrown.
        """
        w = 0
        for _ in range(maxiters):
            we = w * math.exp(w)
            w1e = (w + 1) * math.exp(w)
            if prec > abs((x - we) / w1e):
                return w
            w = w - (we - x) / (w1e - (w + 2) * (we - x) / (2 * w + 2))
        # Sanity check here is to evaluate the difference
        # (abs(w * math.exp(w) - x)) and ensure it is 0
        raise Exception("W doesn't converge fast enough for %f" % (x))

    def _set_theta(self):
        """
        Note that for small epsilon, theta doesn't depend on it
        For large epsilon, theta scales exponentially with it.
        """

        # Implementation pursuant to corollary 1 in paper
        C = math.exp(-1.0) * np.log(8.0 / (self.delta * 7 * math.sqrt(2.0 * np.pi)))
        w = self.lambert_W0(C)
        self.theta = math.ceil(
            max(
                [
                    10.0,
                    math.exp(w + 1) - 0.5,
                    math.exp(self.epsilon * 1.0 / self.MAX_L) - 1,
                ]
            )
        )
        # print(f"Vote threshold used by TrieHH: {self.theta}")

    def _set_batch_size(self):
        """
        As epsilon becomes very large, gamma becomes
        sqrt(population_size)/theta.
        """
        # check Corollary 1 in paper.
        # gamma = int(
        #     math.sqrt(self.client_num)
        #     * (math.exp(self.epsilon * 1.0 / self.MAX_L) - 1)
        #     / (self.theta * np.e ** (self.epsilon / self.MAX_L))
        # self.batch_size = math.sqrt(self.client) * gamma

        # the above two steps are combined to yield
        exp_val = np.e ** (-self.epsilon * 1.0 / self.MAX_L)
        self.batch_size = int(self.client_num * (1 - exp_val) / (self.theta))
        if self.batch_size < self.theta:
            raise Exception("Batch size < vote threshold")
        # print(f"Batch size used by TrieHH: {self.batch_size}")

    def client_vote(self, word, r):
        # r*alphabet size is the number of characters we expect to be covered in this round
        if len(word) <= (r - 1) * self.alphabet_size:
            return 0  # current client's data is too short to participate in this round

        pre = word[
            0 : (r - 1) * self.alphabet_size
        ]  # checking if the prefix of the client's word so far is in the
        # server's trie
        if pre and (pre not in self.server_state.trie):
            return 0

        return 1

    def client_updates(self, r):
        votes = defaultdict(int)
        voters = []
        for word in random.sample(self.clients, self.batch_size):
            voters.append(word)

        for word in voters:
            vote_result = self.client_vote(word, r)
            if vote_result > 0:
                votes[word[0 : r * self.alphabet_size]] += vote_result
        if self.max_beam_size > 0:
            # only retaining self.max_beam_size number of non-terminal prefixes,
            # while prioritizing based on number of votes
            num_prefixes = 0
            all_prefix_vote_counts = list(votes.items())
            word_frequencies = sorted(
                all_prefix_vote_counts, key=lambda item: item[1], reverse=True
            )
            pruned_votes = {}
            for word, freq in word_frequencies:
                if word[-1:] == "$":
                    # legitimate word so retain without penalizing beam
                    pruned_votes[word] = freq
                elif num_prefixes < self.max_beam_size:
                    # only include top words by frequency up to this beam
                    # size
                    num_prefixes += 1
                    pruned_votes[word] = freq
            return pruned_votes
        return votes

    def server_update(self, votes):
        self.server_state.quit_sign = True
        for prefix in votes:
            if votes[prefix] >= self.theta:
                self.server_state.trie[prefix] = votes[prefix]
                self.server_state.quit_sign = False

    def start(self, batch_size):
        """Implementation of TrieHH."""
        self.server_state.trie.clear()
        r = 1
        while True:
            votes = self.client_updates(r)
            self.server_update(votes)
            r += 1
            if self.server_state.quit_sign or r > self.MAX_L:
                break

    def get_heavy_hitters(self):
        heavy_hitters = []
        for _ in range(self.num_runs):
            self.start(self.batch_size)
            raw_result = self.server_state.trie.items()
            results = []
            for word, vote_count in raw_result:
                if word[-1:] == "$":
                    results.append((word.rstrip("$"), vote_count))
            results = [
                item[0]
                for item in sorted(results, key=lambda tup: tup[1], reverse=True)
            ]
            # results are sorted in descending order based on the number of votes received
            # print(f"Discovered {len(results)} heavy hitters in run #{run+1}")
            heavy_hitters.append(results)
        return heavy_hitters


class SimulateTrieHHFBC(SimulateTrieHH):
    """
    Same as SimulateTrieHH, but instantiates clients from a
    differently formatted file that contains data with high
    fidelity to FBC's production data
    """

    def __init__(
        self,
        country_code,
        tile_id,
        ARTIFACTS_PATH,
        FBC_ARTIFACTS_PATH,
        max_word_len=10,
        epsilon=1.0,
        delta=2.3e-12,
        num_runs=5,
        alphabet_size=1,
        max_beam_size=-1,
    ):
        self.country_code = country_code
        self.tile_id = tile_id
        self.FBC_ARTIFACTS_PATH = FBC_ARTIFACTS_PATH
        super().__init__(
            ARTIFACTS_PATH=ARTIFACTS_PATH,
            max_word_len=max_word_len,
            epsilon=epsilon,
            delta=delta,
            num_runs=num_runs,
            alphabet_size=alphabet_size,
            max_beam_size=max_beam_size,
        )

    @staticmethod
    def binarize_client_word(word):
        word_nums = ["{0:08b}".format(int(w)) for w in word.split("/")[0].split(".")]
        bin_word = "".join(word_nums) + "$"
        return bin_word

    def _init_clients(self):
        """Initialization of the dictionary."""
        with open(
            os.path.join(
                self.FBC_ARTIFACTS_PATH,
                self.country_code + "_" + self.tile_id + "_clients_per_carrier.pkl",
            ),
            "r",
        ) as fp:
            word_to_carrier_to_stats = json.loads(fp.read())
            for word in word_to_carrier_to_stats:
                for carrier in word_to_carrier_to_stats[word]:
                    (
                        num_samples,
                        mean_download_speed,
                        std_dev_download_speed,
                    ) = word_to_carrier_to_stats[word][carrier]
                    bin_word = SimulateTrieHHFBC.binarize_client_word(word)
                    for _ in range(num_samples):
                        self.clients.append(bin_word)
        np.random.shuffle(self.clients)
        self.client_num = len(self.clients)

        # print(f"Total number of clients: {self.client_num}")


class SimulateTrieHHLDP(SimulateTrieHH):
    def __init__(
        self,
        ARTIFACTS_PATH,
        max_word_len,
        epsilon,
        delta,
        num_runs,
        alphabet_size,
        alphabet_set,
        local_epsilon,
        theta=None,
        batch_size=None,
        max_beam_size=-1,
    ):
        super().__init__(
            ARTIFACTS_PATH=ARTIFACTS_PATH,
            max_word_len=max_word_len,
            epsilon=epsilon,
            delta=delta,
            num_runs=num_runs,
            alphabet_size=alphabet_size,
            max_beam_size=max_beam_size,
        )
        self.ldp_epsilon = np.exp(local_epsilon)
        self.alphabet_set = alphabet_set
        if theta is not None:
            self.theta = theta
        if batch_size is not None:
            self.batch_size = batch_size
        self.alphabet_list = self.get_alphabet_list()

    def get_alphabet_list(self, curr_pos=0):
        if curr_pos == self.alphabet_size - 1:
            return list(self.alphabet_set)
        tlist = self.get_alphabet_list(curr_pos + 1)
        ret_list = []
        for alphabet in tlist:
            for char in self.alphabet_set:
                ret_list.append(alphabet + char)
        return ret_list

    def client_vote(self, word, r, randomization_candidates):
        # randomization candidates is a randomly accessible list of strings of length
        # self.alphabet_size*r
        # r*alphabet size is the number of characters we expect to be covered in this round
        has_word = True
        if len(word) <= (r - 1) * self.alphabet_size:
            has_word = False  # ought to vote 0 here

        pre = word[
            0 : (r - 1) * self.alphabet_size
        ]  # checking if the prefix of the client's word so far is in the
        # server's trie
        if pre and (pre not in self.server_state.trie):
            has_word = False

        toss_one = random.random() < (
            self.ldp_epsilon
            * 1.0
            / (
                len(randomization_candidates) * len(self.alphabet_list)
                + self.ldp_epsilon
                + 1
            )
        )
        if toss_one:  # answering honestly
            if has_word:
                return word[0 : r * self.alphabet_size]
            else:
                return None

        # answering randomly

        num_possibilities = len(randomization_candidates) * len(self.alphabet_list) + 1
        # cartesian product of number of trie prefixes and potential continuations, and
        # the option to abstain
        randomized_response = random.randint(0, num_possibilities - 1)
        # samples from the inclusive range and is 0-base indexed
        if randomized_response == len(randomization_candidates) * len(
            self.alphabet_list
        ):
            return None  # abstaining

        prefix_id = int(randomized_response / len(self.alphabet_list))
        alphabet_id = randomized_response % len(self.alphabet_list)

        return randomization_candidates[prefix_id] + self.alphabet_list[alphabet_id]

    def client_updates(self, r):

        votes = defaultdict(int)
        voters = []
        randomization_candidates = []
        for candidate in self.server_state.trie:
            if len(candidate) == (r - 1) * self.alphabet_size:
                randomization_candidates.append(candidate)

        if len(randomization_candidates) == 0 and r == 1:
            randomization_candidates.append("")
        elif len(randomization_candidates) == 0:
            raise Exception("No randomization candidates in intermediate round!")

        for word in random.sample(self.clients, self.batch_size):
            # TODO: @akashb Should probably sample with replacement
            voters.append(word)

        for word in voters:
            vote_word = self.client_vote(word, r, randomization_candidates)
            if vote_word is not None:
                assert (
                    len(vote_word) == r * self.alphabet_size
                ), "Reported word not of appropriate length!"
                votes[vote_word] += 1

        if self.max_beam_size > 0:
            # only retaining self.max_beam_size number of non-terminal prefixes,
            # while prioritizing based on number of votes
            num_prefixes = 0
            word_frequencies = sorted(
                votes.items(), key=lambda item: item[1], reverse=True
            )
            pruned_votes = {}
            for word, freq in word_frequencies:
                if word[-1:] == "$":
                    # legitimate word so retain without penalizing beam
                    pruned_votes[word] = freq
                elif num_prefixes < self.max_beam_size:
                    # only include top words by frequency up to this beam
                    # size
                    num_prefixes += 1
                    pruned_votes[word] = freq
            return pruned_votes

        return votes


class SimulateTrieHHLDPFBC(SimulateTrieHHFBC):
    def __init__(
        self,
        country_code,
        tile_id,
        alphabet_set,
        ARTIFACTS_PATH,
        FBC_ARTIFACTS_PATH,
        max_word_len=10,
        epsilon=1.0,
        delta=2.3e-12,
        num_runs=5,
        alphabet_size=1,
        max_beam_size=-1,
        local_epsilon=1.0,
    ):
        self.country_code = country_code
        self.tile_id = tile_id
        super().__init__(
            country_code=country_code,
            tile_id=tile_id,
            max_word_len=max_word_len,
            ARTIFACTS_PATH=ARTIFACTS_PATH,
            FBC_ARTIFACTS_PATH=FBC_ARTIFACTS_PATH,
            epsilon=epsilon,
            delta=delta,
            num_runs=num_runs,
            alphabet_size=alphabet_size,
            max_beam_size=max_beam_size,
        )
        self.ldp_epsilon = np.exp(local_epsilon)
        self.alphabet_set = alphabet_set
        self.alphabet_list = self.get_alphabet_list()

    def get_alphabet_list(self, curr_pos=0):
        if curr_pos == self.alphabet_size - 1:
            return list(self.alphabet_set)
        tlist = self.get_alphabet_list(curr_pos + 1)
        ret_list = []
        for alphabet in tlist:
            for char in self.alphabet_set:
                ret_list.append(alphabet + char)
        return ret_list

    def client_vote(self, word, r, randomization_candidates):
        # randomization candidates is a randomly accessible list of strings of length
        # self.alphabet_size*r
        # r*alphabet size is the number of characters we expect to be covered in this round
        has_word = True
        if len(word) <= (r - 1) * self.alphabet_size:
            has_word = False  # ought to vote 0 here

        pre = word[
            0 : (r - 1) * self.alphabet_size
        ]  # checking if the prefix of the client's word so far is in the
        # server's trie
        if pre and (pre not in self.server_state.trie):
            has_word = False

        toss_one = random.random() < (
            self.ldp_epsilon
            * 1.0
            / (
                len(randomization_candidates) * len(self.alphabet_list)
                + self.ldp_epsilon
                + 1
            )
        )
        if toss_one:  # answering honestly
            if has_word:
                return word[0 : r * self.alphabet_size]
            else:
                return None

        # answering randomly

        num_possibilities = len(randomization_candidates) * len(self.alphabet_list) + 1
        # cartesian product of number of trie prefixes and potential continuations, and
        # the option to abstain
        randomized_response = random.randint(0, num_possibilities - 1)
        # samples from the inclusive range and is 0-base indexed
        if randomized_response == len(randomization_candidates) * len(
            self.alphabet_list
        ):
            return None  # abstaining

        prefix_id = int(randomized_response / len(self.alphabet_list))
        alphabet_id = randomized_response % len(self.alphabet_list)

        return randomization_candidates[prefix_id] + self.alphabet_list[alphabet_id]

    def client_updates(self, r):

        votes = defaultdict(int)
        voters = []
        randomization_candidates = []
        for candidate in self.server_state.trie:
            if len(candidate) == (r - 1) * self.alphabet_size:
                randomization_candidates.append(candidate)

        if len(randomization_candidates) == 0 and r == 1:
            randomization_candidates.append("")
        elif len(randomization_candidates) == 0:
            raise Exception("No randomization candidates in intermediate round!")

        for word in random.sample(self.clients, self.batch_size):
            # TODO: @akashb Should probably sample with replacement
            voters.append(word)

        for word in voters:
            vote_word = self.client_vote(word, r, randomization_candidates)
            if vote_word is not None:
                assert (
                    len(vote_word) == r * self.alphabet_size or vote_word[-1] == "$"
                ), "Reported word not of appropriate length!"
                votes[vote_word] += 1

        if self.max_beam_size > 0:
            # only retaining self.max_beam_size number of non-terminal prefixes,
            # while prioritizing based on number of votes
            num_prefixes = 0
            word_frequencies = sorted(
                votes.items(), key=lambda item: item[1], reverse=True
            )
            pruned_votes = {}
            for word, freq in word_frequencies:
                if word[-1:] == "$":
                    # legitimate word so retain without penalizing beam
                    pruned_votes[word] = freq
                elif num_prefixes < self.max_beam_size:
                    # only include top words by frequency up to this beam
                    # size
                    num_prefixes += 1
                    pruned_votes[word] = freq
            return pruned_votes

        return votes
