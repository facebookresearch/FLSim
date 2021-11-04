#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.


import argparse
import math
import os
import pickle as pickle
import random

import numpy as np
import scipy.stats as sstats

"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation/experimental/analytics/median:bin_quant_bin --  --eval_csv --csv "/data/users/akashb/fa_data/pdp_data.csv" --op_dir "/data/users/akashb/fbsource/fbcode/papaya/toolkit/simulation/experimental/analytics/median/data/results/canonical_dist" --max_freq 5000 --desired_percentiles 0.05 0.5 0.95 --epsilon 1.0 --lamda 0.001 --tau 0.00000001 --upr 1000 --min 0 --max 1200
"""


class Dataset:
    def __init__(self):
        self.exp_cache = {}
        self.samples = []

    def evaluate_cumulative_probab(self, t):
        n = 0.0
        for sample in self.samples:
            if sample < t:
                n += 1.0
        return n / len(self.samples)

    def get_randomized(self, id, threshold, epsilon):
        if epsilon not in self.exp_cache:
            self.exp_cache[epsilon] = math.exp(epsilon) / (1 + math.exp(epsilon))
        p = self.exp_cache[epsilon]
        data = 1.0 if self.samples[id] < threshold else 0.0
        if random.random() < p:
            return data
        else:
            return 1 - data

    def get_empirical_quantile(self, p):
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p)
        return sorted_samples[idx]

    def sample_randomized_responses(self, clients_per_round, t, epsilon, fid, **kwargs):
        round_sids = random.sample(range(len(self.samples)), clients_per_round)
        round_data = np.array(
            [self.get_randomized(sid, t, epsilon) for sid in round_sids]
        )
        return round_data


class CsvDataset:
    def __init__(self, csv_path, max_freq=None, max_lines=None):
        """
        Reads in data from the specified csv file. The first
        column MUST be the device id. All other columns
        correspond to features. The first row must name the
        columns.

        If a device has more than max_freq number of samples,
        we randomly select a subset of them to respect this upper
        bound.
        """

        self.users = {}
        self.tot_samples = 0
        self.tot_users = 0
        self.exp_cache = {}

        with open(csv_path, "r") as ipf:
            lid = 0
            for line in ipf:
                if lid == 0:
                    feat_ids = line.strip(" \t\r\n").split(",")[1:]
                    self.feat_id_to_samples = {feat_id: [] for feat_id in feat_ids}
                    self.feat_pos_to_id = {
                        i: feat_id for (i, feat_id) in enumerate(feat_ids)
                    }
                    self.feat_id_to_pos = {
                        feat_id: i for (i, feat_id) in enumerate(feat_ids)
                    }
                    lid += 1
                    continue
                lid += 1
                fields = line.split(",")

                if fields[0] not in self.users:
                    self.users[fields[0]] = [[] for fid in self.feat_id_to_pos] + [0]
                curr_user = self.users[fields[0]]
                curr_user[-1] = curr_user[-1] + 1
                self.tot_samples += 1
                for i in range(1, len(fields)):
                    try:
                        curr_user[i - 1].append(float(fields[i]))
                    except Exception:
                        continue
                if lid % 10000000 == 0:
                    print("Processed %d lines" % (lid))
                if max_lines is not None and lid >= max_lines:
                    break

            if max_freq is not None:
                orig_users = list(self.users.keys())
                for user in orig_users:
                    if self.users[user][-1] > max_freq:
                        for feat_pos in range(len(self.users[user]) - 1):
                            feat_samples = self.users[user][feat_pos]
                            if len(feat_samples) > max_freq:
                                self.users[user][feat_pos] = [
                                    feat_samples[i]
                                    for i in random.sample(
                                        range(len(feat_samples)), max_freq
                                    )
                                ]
                        self.tot_samples -= self.users[user][-1]
                        self.tot_samples += max_freq
                        self.users[user][-1] = max_freq

            for feat_pos in self.feat_pos_to_id:
                feat_id = self.feat_pos_to_id[feat_pos]
                self.feat_id_to_samples[feat_id] = []
                for user in self.users:
                    self.feat_id_to_samples[feat_id].extend(self.users[user][feat_pos])

            self.tot_users = len(self.users)
            self.uids = list(self.users.keys())
            self.fids = [
                self.feat_pos_to_id[i] for i in range(len(self.feat_id_to_samples))
            ]
            print(
                "Total number of samples = %d, Total number of users = %d"
                % (self.tot_samples, self.tot_users)
            )

    def get_empirical_quantile_for_feat(self, p, fid):
        all_feats = self.feat_id_to_samples[fid]
        sorted_samples = sorted(all_feats)
        idx = int(len(sorted_samples) * p)
        return (sorted_samples[idx] + sorted_samples[idx + 1]) * 0.5

    def get_empirical_quantile(self, p):
        quantiles = []
        for fid in self.fids:
            quantiles.append(self.get_empirical_quantile_for_feat(p, fid))
        return quantiles

    def get_multiple_empirical_quantile(self, all_p):
        quantiles = {}
        for fid in self.fids:
            quantiles[fid] = self.get_multiple_empirical_quantile_for_feat(all_p, fid)
        return quantiles

    def get_multiple_empirical_quantile_for_feat(self, all_p, fid):
        all_feats = self.feat_id_to_samples[fid]
        sorted_samples = sorted(all_feats)
        ret = []
        for p in all_p:
            idx = int(len(sorted_samples) * p)
            ret.append((sorted_samples[idx] + sorted_samples[idx + 1]) * 0.5)
        return ret

    def evaluate_cumulative_probab_for_feat(self, t, fid):
        n = 0.0
        for sample in self.feat_id_to_samples[fid]:
            if sample < t:
                n += 1.0
        return n / len(self.feat_id_to_samples[fid])

    def evaluate_cumulative_probab(self, t):
        cp = []
        for fid in self.fids:
            cp.append(self.evaluate_cumulative_probab_for_feat(t, fid))
        return cp

    def get_raw(self, id, findex):
        user_data = self.users[id][findex]
        if len(user_data) == 0:
            orig_data = 0.0
        else:
            random_index = random.randint(0, len(user_data) - 1)
            orig_data = user_data[random_index]
        return orig_data

    def get_randomized(self, id, fid, threshold, epsilon):
        orig_data = self.get_raw(id, self.feat_id_to_pos[fid])
        if epsilon not in self.exp_cache:
            self.exp_cache[epsilon] = math.exp(epsilon) / (1 + math.exp(epsilon))
        p = self.exp_cache[epsilon]
        data = 1.0 if orig_data < threshold else 0.0
        if random.random() < p:
            return data
        else:
            return 1 - data

    def sample_randomized_responses(self, clients_per_round, t, epsilon, fid, **kwargs):
        round_uids = [
            self.uids[i]
            for i in random.sample(range(len(self.users)), clients_per_round)
        ]
        round_data = np.array(
            [self.get_randomized(uid, fid, t, epsilon) for uid in round_uids]
        )
        return round_data


class GaussianDataset(Dataset):
    def __init__(self, num_samples, mu=0.0, sigma=1.0):
        super().__init__()
        self.samples = np.random.normal(mu, sigma, num_samples)
        self.mu = mu
        self.sigma = sigma

    def get_analytic_quantile(self, p):
        return sstats.norm.ppf(p, self.mu, self.sigma)


class ExponentialDataset(Dataset):
    def __init__(self, num_samples, mean, left=0.0):
        super().__init__()
        self.samples = np.random.exponential(mean, num_samples)
        self.beta = mean
        self.left = left

    def get_analytic_quantile(self, p):
        return sstats.expon.ppf(p, self.left, self.beta)


class BinQuant:
    def __init__(
        self,
        dataset,
        epsilon,
        lamda,
        clients_per_round,
        q_min=-1000.0,
        q_max=1000.0,
        beta=0.0001,
        tau=0.00001,
        feat_id=None,
    ):
        """
        This algorithm uncovers the threshold corresponding to a quantile of interest
        through multiple iterations of a binary search procedure. At a high level, a
        continuous search range is discretized up to a specified lowest level of
        granularity. This discretized range is now ordered both in terms of the real
        values contained in each unit of this range, as well as because the cdf
        function is monotonically increasing. This is why binary search makes sense.

        At each round the algorithm guess the desired to be the mid point of the current
        search range. A specified number of users are sampled and each discloses whether
        its data is above/below the threshold through a binary bit that is also randomized
        for privacy. If the proportion of users with value below the threshold is below
        the desired quantile, we recurse on (mid, high), else, we recurse on (low, mid).
        Note that while evaluating this proportion, we applying post-processing to
        recover utility despite randomization of responses.

        The algorithm stops under two circumstances:

        - The PERCENTAGE of clients with value < threshold at the end of the current round
          is within lamda of the desired quantile

        - Recursing again would divide the range into a unit smaller in width than beta.
          The guess for the quantile in the last round is returned as the result

        The algorithm's output can be evaluated based on three metrics:

        1. Empirical lambda: Deviation between the EMPIRICALLY observed PERCENTAGE of clients
           with value < threshold and the desired quantile. This is an online metric

        2. True lambda: Deviation between the quantile that the returned threshold corresponds
           to and the desired quantile. This is an offline metric that needs access to the ground
           truth dataset

        3. True absolute error: The absolute error between the returned threshold and the true
           threshold associated with the desired quantile. This is also an offline metric that
           needs access to the ground truth dataset

        Which metric to choose is tricky and depends on the nature of the data distribution and
        the use case.

        In moderately multi-modal distributions where most values are concentrated
        around/at a values. Even a small change in the threshold can result in a huge change in
        lambda metrics. Here, the true absolute error is more useful.

        For a feature normalization use case, we care more about the true absolute error rather
        than lambda errors. However, for business analytics, we may care more about lambda metrics.

        Some clarifications about parameters:

        lambda: This is the tolerance for absolute quantile error. If Lamda is 0.05
        and the quantile of interest is the 50th percentile, then any value in the
        45th to 55th percentile is returned.

        q_min: The lower bound for the starting range within which the algorithm will
        search for the quantile of interest.

        q_max: THe upper bound for the starting range within which the algorithm will
        search for the quantile of interest.

        tau: Specifies the smallest discrete unit up to which we can divide any search
        range [starting with (qmin -> qmax)].

        beta: Probability with which we'd like the result to be within lambda quantile
        error of the desired quantile. It's best not to fiddle with this. Moreover, if
        you aren't setting epsilon using the formula comment out below, it is unimportant.

        The following is technically supposed to be a lower bound on epsilon
        to ensure decent utility for a dataset of size N. However, better utility is
        empirically observed for most canonical distributions of real valued random
        variables.

        self.epsilon = max(
            epsilon,
            math.log(
                1
                + 2
                * math.sqrt(8 * self.T * math.log((4 * tau) / beta))
                / (lamda * math.sqrt(N))
            ),
        )
        """
        self.T = math.ceil(math.log2((q_max - q_min) / tau))
        self.epsilon = epsilon
        self.label = "BQ"
        self.clients_per_round = int(clients_per_round)
        self.exp = math.exp(self.epsilon)
        self.lamda = lamda
        self.q_min = q_min
        self.q_max = q_max
        self.beta = beta
        self.tau = tau
        self.dataset = dataset
        self.feat_id = feat_id

    def post_process(self, vals):
        emp_tot = sum(vals)
        exp_tot = ((self.exp + 1.0) / (self.exp - 1.0)) * (emp_tot) - len(vals) / (
            self.exp - 1
        )
        return exp_tot

    def get_quantile_estimate(self, p):
        m = self.q_min
        M = self.q_max
        tot_num_clients = 0.0
        for _ in range(self.T):
            t = (m + M) / 2.0
            round_data = self.dataset.sample_randomized_responses(
                self.clients_per_round, t, self.epsilon, self.feat_id
            )
            tot_num_clients += len(round_data)
            p_emp = self.post_process(round_data) / len(round_data)
            if p_emp > (p + self.lamda / 2.0):
                M = t
            elif p_emp < (p - self.lamda / 2.0):
                m = t
            elif M <= m:
                break
            else:
                break
        return t, tot_num_clients


class FrequencyOracle:
    def __init__(self, epsilon, domain_size):
        self.epsilon = epsilon
        self.domain_size = domain_size


class HadamardOracle(FrequencyOracle):
    @staticmethod
    def get_oracle_label():
        return "HM"

    @staticmethod
    def fwht(a) -> None:
        """In-place Fast Walsh–Hadamard Transform of array a."""
        h = 1
        while h < len(a):
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2

    def __init__(self, epsilon, domain_size):
        super().__init__(epsilon, domain_size)
        self.truth_probability = math.exp(self.epsilon) / (1 + math.exp(self.epsilon))
        self.noisy_hadamard_coeffs = np.zeros(domain_size, dtype=int)
        self.coeff_counts = np.zeros(domain_size, dtype=int)

    def update(self, quantized_value):
        hadamard_coeff = random.randint(0, self.domain_size - 1)
        hadamard_value = bin(hadamard_coeff & quantized_value).count("1") & 1
        noisy_hadamard_value = (
            hadamard_value
            if (random.random() < self.truth_probability)
            else (1 - hadamard_value)
        )
        self.noisy_hadamard_coeffs[hadamard_coeff] += noisy_hadamard_value
        self.coeff_counts[hadamard_coeff] += 1

    def unbias(self):
        ## next unbias the noisy arrays, and do fast inverse hadamard transform to get estimated level counts
        self.frequency_vector = np.zeros(self.domain_size)
        for hadamard_coeff in range(self.domain_size):
            if self.coeff_counts[hadamard_coeff] > 0:
                self.frequency_vector[hadamard_coeff] = -(
                    2.0 * self.noisy_hadamard_coeffs[hadamard_coeff]
                    - self.coeff_counts[hadamard_coeff]
                ) / (
                    (2.0 * self.truth_probability - 1.0)
                    * self.coeff_counts[hadamard_coeff]
                    * self.domain_size
                )
                # unbiasing the noisy counts, see http://dimacs.rutgers.edu/~graham/pubs/papers/ldprange.pdf
        self.fwht(self.frequency_vector)
        return self.frequency_vector


class UnaryOracle(FrequencyOracle):
    @staticmethod
    def get_oracle_label():
        return "OUE"

    def __init__(self, epsilon, domain_size):
        super().__init__(epsilon, domain_size)
        self.zero_to_one_probability = 1.0 / (1.0 + math.exp(self.epsilon))
        self.one_to_zero_probability = 0.5
        self.count = 0
        self.frequency_vector = np.zeros(self.domain_size, dtype=int)

    def update(self, quantized_value):
        self.count += 1
        if random.random() >= self.one_to_zero_probability:
            self.frequency_vector[quantized_value] += 1
        bits_to_flip = np.random.binomial(
            self.domain_size - 1, self.zero_to_one_probability
        )
        flip_ids = random.sample(range(self.domain_size - 1), bits_to_flip)
        for i in flip_ids:  ## add one to the sampled bits
            if i != quantized_value:
                self.frequency_vector[i] += 1
            else:
                self.frequency_vector[self.domain_size - 1] += 1

    def unbias(self):
        for i in range(self.domain_size):  ## next unbias the noisy array
            self.frequency_vector[i] = (
                self.frequency_vector[i] - self.count * self.zero_to_one_probability
            ) / (self.one_to_zero_probability - self.zero_to_one_probability)
        level_sum = sum(self.frequency_vector)
        for i in range(self.domain_size):  ## normalize
            self.frequency_vector[i] = self.frequency_vector[i] / level_sum
        return self.frequency_vector


class FOQuant(BinQuant):
    def __init__(
        self,
        oracle_type,
        dataset,
        epsilon,
        lamda,
        users_per_round,
        q_min,
        q_max,
        beta=0.0001,
        tau=0.01,
        feat_id=None,
    ):
        super().__init__(
            dataset, epsilon, lamda, users_per_round, q_min, q_max, beta, tau, feat_id
        )
        self.depth = self.T
        self.depth = min(self.T, math.ceil(math.log2(users_per_round)) + 1)
        # heuristic to avoid accuracy errors when domain gets too large
        self.levels = [[0] * 2 ** i for i in range(self.depth + 1)]
        self.label = oracle_type.get_oracle_label()
        self.oracle_type = oracle_type
        self.process_dataset()  ## process the input dataset in the init function
        self.post_process_counts()

    def process_dataset(self):
        m = self.q_min
        M = self.q_max
        findex = self.dataset.feat_id_to_pos[self.feat_id]
        for j in range(1, self.depth):
            level_range = 2 ** j
            clients_per_round = self.clients_per_round
            sample_set = random.sample(
                range(len(self.dataset.users)), clients_per_round
            )
            oracle = self.oracle_type(self.epsilon, level_range)
            for uid_index in sample_set:
                orig_data = self.dataset.get_raw(self.dataset.uids[uid_index], findex)
                truncated_data_item = min(M, (max(m, orig_data)))
                quantized_value = int(
                    math.floor(
                        (level_range) * (truncated_data_item - m) / (M + self.tau - m)
                    )
                )
                oracle.update(quantized_value)
            self.levels[j] = oracle.unbias()

    def post_process_counts(self):
        # perform optional post-processing on the tree of counts to improve accuracy (usually)
        # see https://arxiv.org/pdf/0904.0942.pdf
        fbar = [[0] * 2 ** i for i in range(self.depth + 1)]
        fbar[self.depth - 1] = self.levels[self.depth - 1]
        for j in range(self.depth - 2, 0, -1):
            scale_factor = (2 ** j - 2 ** (j - 1)) / (2 ** j - 1)
            for i in range(2 ** j):
                fbar[j][i] = scale_factor * self.levels[j][i] + (1.0 - scale_factor) * (
                    fbar[j + 1][2 * i] + fbar[j + 1][2 * i + 1]
                )
        self.levels[0][0] = 1.0
        for j in range(1, self.depth - 1):
            for i in range(2 ** j):
                self.levels[j][i] = fbar[j][i] + 0.5 * (
                    self.levels[j - 1][i // 2] - (fbar[j][i] + fbar[j][i ^ 1])
                )

    def get_quantile_estimate(self, p):
        # binary search through tree of counts to find quantile p
        m = self.q_min
        M = self.q_max
        q = 0.0
        index = 0
        tot_num_clients = 0
        for j in range(1, self.depth):
            t = (m + M) / 2.0
            tot_num_clients += self.clients_per_round
            normalized_weight = max(self.levels[j][index], 0.0)
            if q + normalized_weight < p:
                q += normalized_weight
                index = (2 * index) + 2
                m = t
            else:
                index = 2 * index
                M = t
            if abs(p - q) < self.lamda / 2.0:
                break
        return t, tot_num_clients


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a quantile with differential privacy"
    )
    parser.add_argument(
        "--csv",
        dest="csv_file",
        help="Path to csv file containing two columns, the first with a "
        "user id and the second with the feature value. The first row must specify the names of "
        "the columns",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--op_dir",
        dest="op_dir",
        help="Path to directory into which pickle files containing results of runs and confidence intervals"
        " should be retained",
        type=str,
    )
    parser.add_argument(
        "--max_freq",
        dest="max_freq",
        help="If the same user holds multiple data samples, this field caps the number of samples per user",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_lines",
        dest="max_lines",
        help="The maximum number of lines to read from the csv. Defaults to reading in all lines",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eval_csv",
        dest="eval_csv",
        action="store_true",
        help="Whether to simulate results on distributions from CSV file",
    )
    parser.set_defaults(eval_csv=False)
    parser.add_argument(
        "--eval_mock",
        dest="eval_mock",
        action="store_true",
        help="Whether to simulate results on canonical distributions",
    )
    parser.set_defaults(eval_mock=False)
    parser.add_argument(
        "--num_runs",
        dest="num_runs",
        help="The number of times to repeat the simulation so as to evaluate confidence intervals",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--desired_percentiles",
        dest="desired_percentiles",
        help="The IDs for features that must be collected",
        nargs="*",
        type=float,
        default=[0.05, 0.5, 0.95],
    )
    parser.add_argument(
        "--epsilon",
        dest="epsilon",
        help="Epsilon for simulation with CSV data",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lamda",
        dest="lamda",
        help="Quantile error tolerance for simulation with CSV data",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--tau",
        dest="tau",
        help="Absolute error tolerance for simulation with CSV data",
        type=float,
        default=0.00000001,
    )
    parser.add_argument(
        "--upr",
        dest="users_per_round",
        help="Number of users to sample per round for simulation with CSV data",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--min",
        dest="data_min",
        help="Min of range we think contains the quantile for simulation with CSV data",
        type=int,
        default=-100,
    )
    parser.add_argument(
        "--max",
        dest="data_max",
        help="Max of range we think contains the quantile for simulation with CSV data",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    return args


def test_from_csv(args) -> None:

    D = CsvDataset(args.csv_file, max_freq=args.max_freq, max_lines=args.max_lines)

    feat_ids = list(D.feat_id_to_pos.keys())

    feat_id_to_perc_to_emp_lam = {
        feat_id: {p: [] for p in args.desired_percentiles} for feat_id in feat_ids
    }
    feat_id_to_perc_to_emp_tau = {
        feat_id: {p: [] for p in args.desired_percentiles} for feat_id in feat_ids
    }

    for feat_id in feat_ids:
        BinQ = BinQuant(
            D,
            args.epsilon,
            args.lamda,
            args.users_per_round,
            args.data_min,
            args.data_max,
            tau=args.tau,
            feat_id=feat_id,
        )
        HadQ = FOQuant(
            HadamardOracle,
            D,
            args.epsilon,
            args.lamda,
            args.users_per_round,
            args.data_min,
            args.data_max,
            tau=args.tau,
            feat_id=feat_id,
        )
        for p in args.desired_percentiles:
            true_quantile = D.get_empirical_quantile_for_feat(p, feat_id)
            print(
                "For feature %s the true %f th quantile is %f"
                % (
                    feat_id,
                    p,
                    true_quantile,
                )
            )
            # for nr in range(args.num_runs):
            for BQ in [BinQ, HadQ]:
                thresh, num_clients = BQ.get_quantile_estimate(p)
                empirical_lam = abs(
                    D.evaluate_cumulative_probab_for_feat(thresh, feat_id) - p
                )
                emp_tau = abs(thresh - true_quantile)
                print(
                    f"({BQ.label}) For feat {feat_id}, desired_p={p}, uncovered_threshold={thresh}, num_clients_sampled={num_clients}, emp_lam = {empirical_lam}, emp_tau = {emp_tau}"
                )
                feat_id_to_perc_to_emp_lam[feat_id][p].append(empirical_lam)
                feat_id_to_perc_to_emp_tau[feat_id][p].append(emp_tau)

    with open(os.path.join(args.op_dir, "emp_lambdas.pkl"), "wb") as opf:
        pickle.dump(feat_id_to_perc_to_emp_lam, opf)

    with open(os.path.join(args.op_dir, "emp_taus.pkl"), "wb") as opf:
        pickle.dump(feat_id_to_perc_to_emp_tau, opf)


def test_canonicalDistributions(args) -> None:
    epsilons = [0.1, 1.0, 5.0]
    desired_percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    lamdas = [0.01, 0.1, 0.2, 0.4, 0.5]
    population_sizes = [1000, 10000, 100000, 1000000, 10000000]
    distributions = ["gaussian", "exponential"]
    client_percs = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9]
    num_runs = 5
    expo_lambda_inv = 0.5
    gauss_mu = 5.0
    gauss_std = 10.0
    data_min = -10000
    data_max = 10000

    op_dir = args.op_dir

    # privacy vs. utility tradeoff without restrictions on sample complexity

    pop_dist_perc_eps_to_emplams = {
        pop: {
            dist: {perc: {eps: [] for eps in epsilons} for perc in desired_percentiles}
            for dist in distributions
        }
        for pop in population_sizes
    }
    # assumes very low tolerance. We want to understand how good utility is, assuming no restrictions on the
    # number of clients sampled

    pop_dist_perc_eps_to_taus = {
        pop: {
            dist: {perc: {eps: [] for eps in epsilons} for perc in desired_percentiles}
            for dist in distributions
        }
        for pop in population_sizes
    }
    # assumes very low tolerance. We want to understand how good utility is, assuming no restrictions on the
    # number of clients sampled

    pop_dist_perc_eps_to_emptaus = {
        pop: {
            dist: {perc: {eps: [] for eps in epsilons} for perc in desired_percentiles}
            for dist in distributions
        }
        for pop in population_sizes
    }
    # assumes very low tolerance. We want to understand how good utility is, assuming no restrictions on the
    # number of clients sampled

    pop_dist_perc_eps_to_min_num_clients = {
        pop: {
            dist: {perc: {eps: [] for eps in epsilons} for perc in desired_percentiles}
            for dist in distributions
        }
        for pop in population_sizes
    }
    # assumes very low tolerance. We want to understand how good utility is, assuming no restrictions on the
    # number of clients sampled

    # sample complexity for desired level of utility (with a cap enforced by limit of bin quant )

    pop_dist_perc_lam_to_min_num_clients = {
        pop: {
            dist: {perc: {lam: [] for lam in lamdas} for perc in desired_percentiles}
            for dist in distributions
        }
        for pop in population_sizes
    }

    for N in population_sizes:
        for dist in distributions:
            for p in desired_percentiles:
                for _ in range(num_runs):
                    if dist == "gaussian":
                        D = GaussianDataset(N, gauss_mu, gauss_std)
                    else:
                        D = ExponentialDataset(N, expo_lambda_inv)
                    for eps in epsilons:
                        print(
                            "Attempting N=%f, dist=%s, p=%f, eps=%f, lambda=%f"
                            % (N, dist, p, eps, 0.01)
                        )
                        min_qe = None
                        min_tau = None
                        min_emptau = None
                        min_cpr = -1.0
                        for client_perc in client_percs:
                            clients_per_round = client_perc * N
                            BQ = BinQuant(
                                D,
                                eps,
                                0.01,
                                clients_per_round,
                                data_min,
                                data_max,
                                tau=0.00000001,
                            )
                            # fixing lambda to be very low to evaluate for utility
                            thresh, num_clients = BQ.get_quantile_estimate(p)
                            empirical_lam = abs(
                                D.evaluate_cumulative_probab(thresh) - p
                            )
                            tau = abs(thresh - D.get_analytic_quantile(p))
                            emp_tau = abs(thresh - D.get_empirical_quantile(p))

                            if min_qe is None or empirical_lam < min_qe:
                                min_qe = empirical_lam
                                min_tau = tau
                                min_emptau = emp_tau
                                min_cpr = num_clients

                            if empirical_lam < 0.005:
                                break

                        print(
                            "Empirical lambda = %f, tau = %f, num_clients = %f"
                            % (min_qe, min_tau, min_cpr)
                        )
                        pop_dist_perc_eps_to_emplams[N][dist][p][eps].append(min_qe)
                        pop_dist_perc_eps_to_taus[N][dist][p][eps].append(min_tau)
                        pop_dist_perc_eps_to_emptaus[N][dist][p][eps].append(min_emptau)
                        pop_dist_perc_eps_to_min_num_clients[N][dist][p][eps].append(
                            min_cpr
                        )

                    for lam in lamdas:
                        print(
                            "Attempting N=%f, dist=%s, p=%f, eps=%f, lambda=%f"
                            % (N, dist, p, 1.0, lam)
                        )
                        achieved_lam = False
                        for client_perc in client_percs:
                            clients_per_round = client_perc * N
                            BQ = BinQuant(
                                D,
                                1.0,
                                lam,
                                clients_per_round,
                                data_min,
                                data_max,
                            )
                            # fixing epsilon to be reasonable
                            thresh, num_clients = BQ.get_quantile_estimate(p)
                            empirical_lam = abs(
                                D.evaluate_cumulative_probab(thresh) - p
                            )
                            if empirical_lam <= lam / 2.0:
                                achieved_lam = True
                                pop_dist_perc_lam_to_min_num_clients[N][dist][p][
                                    lam
                                ].append(num_clients)
                                break
                        if not achieved_lam:
                            pop_dist_perc_lam_to_min_num_clients[N][dist][p][
                                lam
                            ].append(
                                100 * N
                            )  # so that confidence interval gets widened to include N

    with open(os.path.join(op_dir, "pop_dist_perc_eps_to_emplams.pkl"), "wb") as opf:
        pickle.dump(pop_dist_perc_eps_to_emplams, opf)

    with open(os.path.join(op_dir, "pop_dist_perc_eps_to_taus.pkl"), "wb") as opf:
        pickle.dump(pop_dist_perc_eps_to_taus, opf)

    with open(os.path.join(op_dir, "pop_dist_perc_eps_to_emptaus.pkl"), "wb") as opf:
        pickle.dump(pop_dist_perc_eps_to_emptaus, opf)

    with open(
        os.path.join(op_dir, "pop_dist_perc_eps_to_min_num_clients.pkl"), "wb"
    ) as opf:
        pickle.dump(pop_dist_perc_eps_to_min_num_clients, opf)

    with open(
        os.path.join(op_dir, "pop_dist_perc_lam_to_min_num_clients.pkl"), "wb"
    ) as opf:
        pickle.dump(pop_dist_perc_lam_to_min_num_clients, opf)

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    if args.eval_csv:
        test_from_csv(args)
    if args.eval_mock:
        test_canonicalDistributions(args)
