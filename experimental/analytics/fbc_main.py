#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import json
import os
import pickle as pkl
from ast import literal_eval
from collections import OrderedDict

import numpy as np
from papaya.toolkit.simulation.experimental.analytics.triehh import (
    SimulateTrieHHFBC,
    SimulateTrieHHLDPFBC,
)


class Plot(object):
    def __init__(self, max_k, country, tile, FBC_ARTIFACTS_PATH):
        self.confidence = 0.95
        self.max_k = max_k
        self.country = country
        self.tile = tile
        self.FBC_ARTIFACTS_PATH = FBC_ARTIFACTS_PATH
        self._load_true_frequencies()

    def _load_true_frequencies(self):
        self.true_frequencies = {}
        with open(
            os.path.join(
                self.FBC_ARTIFACTS_PATH,
                self.country + "_" + self.tile + "_clients_per_carrier.pkl",
            ),
            "rb",
        ) as fp:
            word_to_carrier_to_stats = json.loads(fp.read())
            for word in word_to_carrier_to_stats:
                for carrier in word_to_carrier_to_stats[word]:
                    (
                        num_samples,
                        mean_download_speed,
                        std_dev_download_speed,
                    ) = word_to_carrier_to_stats[word][carrier]
                    bin_word = SimulateTrieHHFBC.binarize_client_word(word).rstrip(
                        "$"
                    )  # excluding the terminal symbol
                    self.true_frequencies[bin_word] = (
                        self.true_frequencies.get(bin_word, 0.0) + num_samples
                    )

    def precision(self, result, k, beam_size):
        sorted_all = OrderedDict(
            sorted(self.true_frequencies.items(), key=lambda x: x[1], reverse=True)
        )
        top_words = list(sorted_all.keys())[:k]
        precision = 0.0
        # proportion of identified beam of heavy hitters (of maximum size = beam_size)
        # that are actually in the top K. Note that it is assumed the words in result
        # are in sorted order of their estimated frequency.
        for word in result[:beam_size]:
            if word in top_words:
                precision += 1.0
        precision /= min(len(result), beam_size)
        return precision

    def get_plot_info_for_results_by_hyperparam_setting(
        self,
        trieehh_all_results_per_setting,
        recall_values_triehh,
        # for each K, holds a list of recall wrt the top K words in top_words
        # of each trie HH result identified
        precision_values_triehh,
        # for each K, holds the precision wrt the top K words in top_words
        # of each trie HH result identified
        f1_values_triehh,
        # for each K, holds the f1 wrt the top K words in top_words
        # of each trie HH result identified
    ):
        sorted_all = OrderedDict(
            sorted(self.true_frequencies.items(), key=lambda x: x[1], reverse=True)
        )
        # sorted list of words, each of which is held by a single client. Each client
        # holds a single word.
        top_words = list(sorted_all.keys())[: self.max_k - 1]

        for K in range(1, self.max_k):
            if K not in recall_values_triehh:
                recall_values_triehh[K] = []
            if K not in precision_values_triehh:
                precision_values_triehh[K] = []
            if K not in f1_values_triehh:
                f1_values_triehh[K] = []
        index = 0
        for triehh_result in trieehh_all_results_per_setting:
            if len(triehh_result) > 0:
                triehh_result_set = set(triehh_result)
                index = index + 1
                for K in range(1, self.max_k):
                    recall = 0
                    # for each (trieehh_result, K) combo, evaluating separate recall
                    for i in range(min(K, len(top_words))):
                        if top_words[i] in triehh_result_set:
                            # since counting the proportion of true top K items that
                            # were identified as heavy hitters.
                            recall += 1
                    recall = recall * 1.0 / min(K, len(top_words))
                    # since the number of true heavy hitters is K by design. If K > max no. of possible heavy hitters
                    # recall saturates since it would be unfair to penalize simply for arbitrarily scaling up K
                    recall_values_triehh[K].append(recall)
                    precision_values_triehh[K].append(
                        self.precision(
                            triehh_result,
                            min(K, len(top_words)),
                            beam_size=K,
                        )
                    )
            else:
                for K in range(1, self.max_k):
                    recall_values_triehh[K].append(0.0)
                    precision_values_triehh[K].append(0.0)

        for K in range(1, self.max_k):
            f1_values_triehh[K] = []
            for i in range(len(recall_values_triehh[K])):
                f1_values_triehh[K].append(
                    self.calculate_fscore(
                        precision_values_triehh[K][i], recall_values_triehh[K][i]
                    )
                )

        return (
            recall_values_triehh,
            precision_values_triehh,
            f1_values_triehh,
            find_jaccard_index(trieehh_all_results_per_setting),
        )

    def calculate_fscore(self, prec, rec):
        return 2.0 * prec * rec / (prec + rec + 0.0000000001)


def find_jaccard_index(heavy_hitters):
    intersection_set = set(heavy_hitters[0])
    union_set = set(heavy_hitters[0])
    for i in range(1, len(heavy_hitters)):
        intersection_set = intersection_set.intersection(set(heavy_hitters[i]))
        union_set = union_set.union(set(heavy_hitters[i]))
    if len(union_set) > 0:
        return len(intersection_set) * 1.0 / len(union_set)
    else:
        return 0.0


"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation/experimental/analytics:fbc_main -- --FBC_ARTIFACTS_PATH "/data/users/akashb/fa_data/processed/" --output_dir "/data/users/akashb/fbsource/fbcode/papaya/toolkit/simulation/experimental/analytics"

See arg parser help descriptions for other customizable options
"""


def main():
    parser = argparse.ArgumentParser(description="Spawn an FA query for Trie HH")

    parser.add_argument(
        "--max_k",
        dest="max_k",
        help="The maximum number of heavy hitters we may be interested in",
        default=51,
        type=int,
    )
    parser.add_argument(
        "--max_word_len",
        dest="max_word_len",
        help="The length of the longest word + 1",
        default=33,
        type=int,
    )
    parser.add_argument(
        "--delta",
        dest="delta",
        help="Desired delta DP parameter",
        default=1e-8,
        type=float,
    )
    parser.add_argument(
        "--min_epsilon",
        dest="min_epsilon",
        help="Minimum epsilon DP parameter for the output trie",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--max_epsilon",
        dest="max_epsilon",
        help="Maximum epsilon DP parameter for the output trie",
        default=10.0,
        type=float,
    )
    parser.add_argument(
        "--local_epsilon",
        dest="local_epsilon",
        help="Local epsilon DP parameter for the partial disclosures made by clients in each round trie",
        default=10.0,
        type=float,
    )
    parser.add_argument(
        "--min_alpha_len",
        dest="min_alpha_len",
        help="Minimum for range of alphabet lengths to consider",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--max_alpha_len",
        dest="max_alpha_len",
        help="Maximum for range of alphabet lengths to consider",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--num_runs",
        dest="num_runs",
        help="The number of times to run Trie HH for each tile in each country",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--num_ensemble_runs",
        dest="num_ensemble_runs",
        help="If using enmsebling, this is the number of runs to ensemble together",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--ensemble_method",
        dest="ensemble_method",
        help="How to ensemble results of multiple runs. Turned off by default by setting to None",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_extrema",
        dest="num_extrema",
        help="How many of the most populous and least populous tiles in each regime to include",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--num_tiles",
        dest="num_tiles",
        help="How many non-extrema tiles to include per regime",
        default=6,
        type=int,
    )
    parser.add_argument(
        "--max_beam_size",
        dest="max_beam_size",
        help="The maximum number of incomplete prefixes to pass through to next round during the "
        "Trie HH protocol. Note that complete words will always be passed through without diminishing "
        "the beam size",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--use_LDP",
        dest="use_LDP",
        action="store_true",
    )
    parser.set_defaults(use_LDP=False)
    parser.add_argument(
        "--FBC_ARTIFACTS_PATH",
        dest="FBC_ARTIFACTS_PATH",
        help="Path to directory containing json file that maps regime to info about tiles in that regime",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Path to directory where output files are written out",
        type=str,
    )

    args = parser.parse_args()

    FBC_ARTIFACTS_PATH = args.FBC_ARTIFACTS_PATH

    # maximum K value (non-inclusive) up to which F1 score is calculated
    max_k = args.max_k

    # length of longest word
    max_word_len = args.max_word_len
    # 4 bytes for ipv4 address in binary form represented as a string + terminal character

    delta = args.delta
    # same delta is maintained across all other hyper-parameter settings. In practice, this shouldn't cause much
    # of an issue since it scales inversely with the factorial of the voting threshold which convergences to zero
    # fairly quickly

    epsilons = np.linspace(
        args.min_epsilon,
        args.max_epsilon,
        int((args.max_epsilon - args.min_epsilon) + 1.0),
    )
    # different epsilon settings to ablate over. Increasing arbitrarily causes breakdown in convergence behavior

    # alphabet_lens = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    alphabet_lens = list(range(args.min_alpha_len, args.max_alpha_len + 2, 2))
    # different alphabet lengths to ablate over. Increasing arbitrarily causes breakdown in convergence behavior

    # repeat simulation for num_runs times. Used for evaluating confidence intervals
    num_runs = args.num_runs

    num_ensemble_runs = args.num_ensemble_runs

    ensemble_method = args.ensemble_method

    num_extrema = args.num_extrema
    # Within each regime, the experiments below make sure to include tiles near the two ends. This dictates how many
    # such fringe tiles should be considered.

    num_tiles = args.num_tiles
    # At most these many other tiles in a given regime are sampled

    max_beam_size = args.max_beam_size
    # when > 0, it forces the Trie HH algorithm to only consider at most the specified number of prefixes

    ensemble = args.ensemble_method is not None

    local_epsilon = args.local_epsilon

    use_LDP = args.use_LDP

    alphabet_set = {"1", "0"}  # since data is binarized

    # If not binarizing, uncomment the following two lines to read in the char vocab for the raw data
    # with open(os.path.join(FBC_ARTIFACTS_PATH, "char_vocab.pkl"), "r") as cf:
    #     char_vocab = json.loads(cf.read())

    results_tensor = {}

    """
    (regime, epsilon, alphabet_len, K) ->
                    (
                        [recall@K for all runs for all tiles],
                        [precision@K for all runs for all tiles],
                        [f1@K for all runs for all tiles],
                        [jaccard_index for all runs for all tiles],
                    )

    Once we have this table, we can evaluate any marginal we like and analyse trends accordingly.

    Note that in the experiments below, it is possible that no results are returned. In such a case, precision, recall and f1 are all set
    to 0, so as to penalize the regime appropriately. Also note that since the algorithm is probabilistic, it is possible that results are
    inconsistent. Hence, we plot confidence intervals. If the confidence interval is of non-zero length and contains 0, then one can assume
    that often the algorithm fails to find any heavy hitters at all.

    Also note that in some cases, the desired theoretical epsilon is not met based on the heuristic method for setting hyper-parameters,
    in part due to the need to numerical optimization techniques for evaluating some of them. In the experiments below, we tolerate a 10%
    deviation from the desired epsilon. The maximum deviation observed in the experiments below is ~16%. Luckily, deviations such as this
    only occur in the smallest of tiles (between 1000 and 10000 samples). when both epsilon and alphabet length settings are high - i.e.
    typically when epsilon > 6.0 and alpha_len > 8.

    In addition to deviations from the desired epsilon, sometimes the algorithm will trivially fail to converge because the vote threshold
    is greater than the number of clients sampled per round. Once again, this mostly happens for tiles with sample count in the range
    (1000, 10000) when using epsilon > 6.0 and alphabet length > 12. For the smallest of tiles with <= 1000 samples, this can also occur
    with epsilon = 1.0 and alphabet length = 4.

    Luckily, the above issues vanish when dealing with tiles that have samples >= 50000.
    """

    with open(
        os.path.join(FBC_ARTIFACTS_PATH, "sample_regime_to_tiles_info.pkl"), "r"
    ) as cf:
        sample_regime_to_tiles_info_string_keys = json.loads(cf.read())

    sample_regime_to_tiles_info = {}
    for string_key in sample_regime_to_tiles_info_string_keys:
        sample_regime_to_tiles_info[
            literal_eval(string_key)
        ] = sample_regime_to_tiles_info_string_keys[string_key]

    regime_epsilon_alpha_recall_values_triehh = {}
    # maps (regime, epsilon, alpha len) -> {K: [recall across all tiles]}

    regime_epsilon_alpha_precision_values_triehh = {}
    # maps (regime, epsilon, alpha len) -> {K: [precision across all tiles]}

    regime_epsilon_alpha_f1_values_triehh = {}
    # maps (regime, epsilon, alpha len) -> {K: [f1 across all tiles]}

    for regime in sample_regime_to_tiles_info:
        print("In regime ", regime)
        # regime specifies a range for the number of logs from a tile
        all_tiles = sample_regime_to_tiles_info[regime]
        # list of tiles with number of samples in this regime. Tiles are in sorted order of this cardinality.
        lower = all_tiles[:num_extrema]
        upper = all_tiles[-num_extrema:]
        range_size = min(num_tiles, len(all_tiles) - 2 * num_extrema)
        indices = np.random.randint(
            max(1, len(all_tiles) - 2 * num_extrema),
            size=max(0, range_size),
        )
        candidate_tiles = lower + [all_tiles[ind] for ind in indices] + upper
        # Ensuring coverage at both ends of the regime being considered. Useful since we evaluate confidence intervals
        for candidate in candidate_tiles:
            country, tile, sample_count = candidate
            print("In tile %s for country %s" % (tile, country))
            plotter = Plot(max_k, country, tile, FBC_ARTIFACTS_PATH=FBC_ARTIFACTS_PATH)
            for epsilon in epsilons:
                for alpha_len in alphabet_lens:
                    if (
                        regime,
                        epsilon,
                        alpha_len,
                    ) not in regime_epsilon_alpha_recall_values_triehh:
                        regime_epsilon_alpha_recall_values_triehh[
                            (regime, epsilon, alpha_len)
                        ] = {}

                    if (
                        regime,
                        epsilon,
                        alpha_len,
                    ) not in regime_epsilon_alpha_precision_values_triehh:
                        regime_epsilon_alpha_precision_values_triehh[
                            (regime, epsilon, alpha_len)
                        ] = {}

                    if (
                        regime,
                        epsilon,
                        alpha_len,
                    ) not in regime_epsilon_alpha_f1_values_triehh:
                        regime_epsilon_alpha_f1_values_triehh[
                            (regime, epsilon, alpha_len)
                        ] = {}
                    try:
                        if ensemble:
                            hh_sim = (
                                SimulateTrieHHFBC(
                                    country_code=country,
                                    tile_id=tile,
                                    ARTIFACTS_PATH=None,
                                    FBC_ARTIFACTS_PATH=FBC_ARTIFACTS_PATH,
                                    max_word_len=max_word_len,
                                    epsilon=epsilon,
                                    delta=delta,
                                    num_runs=num_ensemble_runs,
                                    alphabet_size=alpha_len,
                                    max_beam_size=max_beam_size,
                                )
                                if not use_LDP
                                else SimulateTrieHHLDPFBC(
                                    country_code=country,
                                    tile_id=tile,
                                    alphabet_set=alphabet_set,
                                    ARTIFACTS_PATH=None,
                                    FBC_ARTIFACTS_PATH=FBC_ARTIFACTS_PATH,
                                    max_word_len=max_word_len,
                                    epsilon=epsilon,
                                    delta=delta,
                                    num_runs=num_ensemble_runs,
                                    alphabet_size=alpha_len,
                                    max_beam_size=max_beam_size,
                                    local_epsilon=local_epsilon,
                                )
                            )
                            triehh_heavy_hitters = []
                            for _ in range(num_runs):
                                ensemble_heavy_hitters = hh_sim.get_heavy_hitters()
                                if ensemble_method == "union":
                                    tot_hh = set()
                                    for hh in ensemble_heavy_hitters:
                                        tot_hh = tot_hh.union(set(hh))
                                    triehh_heavy_hitters.append(list(tot_hh))
                                elif ensemble_method == "intersection":
                                    tot_hh = set(ensemble_heavy_hitters[0])
                                    for hid in range(1, len(ensemble_heavy_hitters)):
                                        tot_hh = tot_hh.intersection(
                                            ensemble_heavy_hitters[hid]
                                        )
                                    triehh_heavy_hitters.append(list(tot_hh))
                        else:
                            hh_sim = (
                                SimulateTrieHHFBC(
                                    country_code=country,
                                    tile_id=tile,
                                    ARTIFACTS_PATH=None,
                                    FBC_ARTIFACTS_PATH=FBC_ARTIFACTS_PATH,
                                    max_word_len=max_word_len,
                                    epsilon=epsilon,
                                    delta=delta,
                                    num_runs=num_runs,
                                    alphabet_size=alpha_len,
                                    max_beam_size=max_beam_size,
                                )
                                if not use_LDP
                                else SimulateTrieHHLDPFBC(
                                    country_code=country,
                                    tile_id=tile,
                                    alphabet_set=alphabet_set,
                                    ARTIFACTS_PATH=None,
                                    FBC_ARTIFACTS_PATH=FBC_ARTIFACTS_PATH,
                                    max_word_len=max_word_len,
                                    epsilon=epsilon,
                                    delta=delta,
                                    num_runs=num_runs,
                                    alphabet_size=alpha_len,
                                    max_beam_size=max_beam_size,
                                    local_epsilon=local_epsilon,
                                )
                            )
                            triehh_heavy_hitters = hh_sim.get_heavy_hitters()
                        del hh_sim
                        # better to get the raw precision/rec/f1 scores for all runs and evaluate a final confidence interval
                        # rather than evaluating min/max of confidence intervals
                        (
                            _,
                            _,
                            _,
                            jaccard_index,
                        ) = plotter.get_plot_info_for_results_by_hyperparam_setting(
                            triehh_heavy_hitters,
                            regime_epsilon_alpha_recall_values_triehh[
                                (
                                    regime,
                                    epsilon,
                                    alpha_len,
                                )
                            ],
                            regime_epsilon_alpha_precision_values_triehh[
                                (
                                    regime,
                                    epsilon,
                                    alpha_len,
                                )
                            ],
                            regime_epsilon_alpha_f1_values_triehh[
                                (
                                    regime,
                                    epsilon,
                                    alpha_len,
                                )
                            ],
                        )
                        for j1 in range(1, max_k):
                            if (
                                regime,
                                epsilon,
                                alpha_len,
                                j1,
                            ) not in results_tensor:
                                results_tensor[(regime, epsilon, alpha_len, j1,)] = (
                                    regime_epsilon_alpha_recall_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    regime_epsilon_alpha_precision_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    regime_epsilon_alpha_f1_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    [jaccard_index],
                                )
                            else:
                                prev_result = results_tensor[
                                    (
                                        regime,
                                        epsilon,
                                        alpha_len,
                                        j1,
                                    )
                                ]
                                prev_result[3].append(jaccard_index)
                                results_tensor[(regime, epsilon, alpha_len, j1,)] = (
                                    regime_epsilon_alpha_recall_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    regime_epsilon_alpha_precision_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    regime_epsilon_alpha_f1_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    prev_result[3],
                                )
                    except Exception as e:
                        print(
                            "Encountered exception when using epsilon=%f, alpha len=%d in tile with sample count %d: "
                            % (epsilon, alpha_len, sample_count),
                            str(e),
                        )
                        for j1 in range(1, max_k):
                            if (
                                j1
                                not in regime_epsilon_alpha_recall_values_triehh[
                                    (
                                        regime,
                                        epsilon,
                                        alpha_len,
                                    )
                                ]
                            ):
                                regime_epsilon_alpha_recall_values_triehh[
                                    (
                                        regime,
                                        epsilon,
                                        alpha_len,
                                    )
                                ][j1] = []
                            regime_epsilon_alpha_recall_values_triehh[
                                (
                                    regime,
                                    epsilon,
                                    alpha_len,
                                )
                            ][j1].extend([0.0 for r in range(num_runs)])

                            if (
                                j1
                                not in regime_epsilon_alpha_precision_values_triehh[
                                    (
                                        regime,
                                        epsilon,
                                        alpha_len,
                                    )
                                ]
                            ):
                                regime_epsilon_alpha_precision_values_triehh[
                                    (
                                        regime,
                                        epsilon,
                                        alpha_len,
                                    )
                                ][j1] = []
                            regime_epsilon_alpha_precision_values_triehh[
                                (
                                    regime,
                                    epsilon,
                                    alpha_len,
                                )
                            ][j1].extend([0.0 for r in range(num_runs)])

                            if (
                                j1
                                not in regime_epsilon_alpha_f1_values_triehh[
                                    (
                                        regime,
                                        epsilon,
                                        alpha_len,
                                    )
                                ]
                            ):
                                regime_epsilon_alpha_f1_values_triehh[
                                    (
                                        regime,
                                        epsilon,
                                        alpha_len,
                                    )
                                ][j1] = []
                            regime_epsilon_alpha_f1_values_triehh[
                                (
                                    regime,
                                    epsilon,
                                    alpha_len,
                                )
                            ][j1].extend([0.0 for r in range(num_runs)])
                            jaccard_index = 0.0
                            if (
                                regime,
                                epsilon,
                                alpha_len,
                                j1,
                            ) not in results_tensor:
                                results_tensor[(regime, epsilon, alpha_len, j1,)] = (
                                    regime_epsilon_alpha_recall_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    regime_epsilon_alpha_precision_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    regime_epsilon_alpha_f1_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    [jaccard_index],
                                )
                            else:
                                prev_result = results_tensor[
                                    (
                                        regime,
                                        epsilon,
                                        alpha_len,
                                        j1,
                                    )
                                ]
                                prev_result[3].append(jaccard_index)
                                results_tensor[(regime, epsilon, alpha_len, j1,)] = (
                                    regime_epsilon_alpha_recall_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    regime_epsilon_alpha_precision_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    regime_epsilon_alpha_f1_values_triehh[
                                        (
                                            regime,
                                            epsilon,
                                            alpha_len,
                                        )
                                    ][j1],
                                    prev_result[3],
                                )

        with open(
            os.path.join(
                args.output_dir,
                "triehh_fbc_results_full_ablation_upto_k_100_upto_regime_%s_%s_%s_ensemble_beam_%s_%s_ldp.pkl"
                % (
                    str(regime[0]),
                    str(regime[1]),
                    ensemble_method if ensemble else "no",
                    str(max_beam_size),
                    "no" if not use_LDP else str(local_epsilon),
                ),
            ),
            "wb",
        ) as opf:
            pkl.dump(results_tensor, opf)
    with open(
        os.path.join(
            args.output_dir,
            "triehh_fbc_results_full_ablation_upto_k_100_%s_ensemble_beam_%s_%s_ldp.pkl"
            % (
                ensemble_method if ensemble else "no",
                str(max_beam_size),
                "no" if not use_LDP else str(local_epsilon),
            ),
        ),
        "wb",
    ) as opf:
        pkl.dump(results_tensor, opf)
    print("Done!")


if __name__ == "__main__":
    main()
