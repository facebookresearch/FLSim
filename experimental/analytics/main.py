#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from papaya.toolkit.simulation.experimental.analytics.triehh import SimulateTrieHH


ARTIFACTS_PATH: str = "/tmp/papaya/"


class Plot(object):
    def __init__(self, max_k):
        self.confidence = 0.95
        self.max_k = max_k
        self._load_true_frequencies()

    def _load_true_frequencies(self):
        """Initialization of the dictionary."""
        with open(ARTIFACTS_PATH + "word_frequencies.txt", "r") as fp:
            self.true_frequencies = json.loads(fp.read())
        self.plotFrequencies()

    def plotFrequencies(self):
        eps = np.arange(0, 300, 1)
        all_true_freqs = list(self.true_frequencies.values())
        freq = sorted(all_true_freqs, reverse=True)
        df = pd.DataFrame({"idx": eps, "freq": freq[:300]})
        plt.title("Words: Frequency vs. Words (Single Word)", fontsize=14)
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.plot("idx", "freq", data=df)
        plt.savefig("freq.png", bbox_inches="tight")
        plt.close()

    def plotTrie(self, trie):
        triehh_freq = {}
        eps = np.arange(0, len(trie), 1)
        for item in trie:
            triehh_freq[item] = self.true_frequencies[item]
        all_freqs = list(triehh_freq.values())
        freq = sorted(all_freqs, reverse=True)
        df = pd.DataFrame({"idx": eps, "freq": freq})
        plt.title("Trie Learnt: Frequency vs. Words (Single Word)", fontsize=14)
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.plot("idx", "freq", data=df)
        plt.text(df.idx.max() + 0.4, df.freq.min(), df.freq.min())
        plt.savefig("trie.png", bbox_inches="tight")
        plt.close()

    def get_mean_u_l(self, recall_values):
        data_mean = []
        ub = []
        lb = []
        for K in range(10, self.max_k):
            curr_mean = np.mean(recall_values[K])
            data_mean.append(curr_mean)
            n = len(recall_values[K])
            std_err = scipy.stats.sem(recall_values[K])
            h = std_err * scipy.stats.t.ppf((1 + self.confidence) / 2, n - 1)
            lb.append(curr_mean - h)
            ub.append(curr_mean + h)
        mean_u_l = [data_mean, ub, lb]
        return mean_u_l

    def precision(self, result, k):
        sorted_all = OrderedDict(
            sorted(self.true_frequencies.items(), key=lambda x: x[1], reverse=True)
        )
        top_words = list(sorted_all.keys())[:k]
        precision = 0
        for word in result:
            if word in top_words:
                precision += 1
        precision /= len(result)
        return precision

    def plot_precision_recall(self, triehh_all_results, epsilon):
        sorted_all = OrderedDict(
            sorted(self.true_frequencies.items(), key=lambda x: x[1], reverse=True)
        )
        top_words = list(sorted_all.keys())[: self.max_k]

        all_recall_triehh = []
        k_values = []

        for K in range(10, self.max_k):
            k_values.append(K)

        recall_values_triehh = {}
        precision_values_triehh = {}

        for K in range(10, self.max_k):
            recall_values_triehh[K] = []
            precision_values_triehh[K] = []
        triehh_freq = {}
        index = 0
        for triehh_result in triehh_all_results:
            index = index + 1
            for item in triehh_result:
                triehh_freq[item] = self.true_frequencies[item]
            for K in range(10, self.max_k):
                recall = 0
                for i in range(K):
                    if top_words[i] in triehh_result:
                        recall += 1
                recall = recall * 1.0 / K
                recall_values_triehh[K].append(recall)
                precision_values_triehh[K].append(self.precision(triehh_result, K))
            print("iteration: ", index, " out of ", len(triehh_all_results))
        all_recall_triehh = self.get_mean_u_l(recall_values_triehh)
        all_precision_triehh = self.get_mean_u_l(precision_values_triehh)
        self.plotTrie(triehh_all_results[0])

        _, ax1 = plt.subplots(figsize=(10, 7))
        ax1.set_xlabel("K", fontsize=16)
        ax1.set_ylabel("Recall Score", fontsize=16)

        ax1.plot(
            k_values,
            all_recall_triehh[0],
            color="purple",
            alpha=1,
            label=r"TrieHH, $\varepsilon$ = " + str(epsilon),
        )
        ax1.fill_between(
            k_values,
            all_recall_triehh[2],
            all_recall_triehh[1],
            color="violet",
            alpha=0.3,
        )

        plt.legend(loc=4, fontsize=14)

        plt.title("Top K Recall Score vs. K (Single Word)", fontsize=14)
        plt.savefig("recall_single.png", bbox_inches="tight")
        plt.close()

        _, ax1 = plt.subplots(figsize=(10, 7))
        ax1.set_xlabel("K", fontsize=16)
        ax1.set_ylabel("Precision Score", fontsize=16)

        ax1.plot(
            k_values,
            all_precision_triehh[0],
            color="purple",
            alpha=1,
            label=r"TrieHH, $\varepsilon$ = " + str(epsilon),
        )
        ax1.fill_between(
            k_values,
            all_precision_triehh[2],
            all_precision_triehh[1],
            color="violet",
            alpha=0.3,
        )

        plt.legend(loc=4, fontsize=14)

        plt.title("Top K Precision Score vs. K (Single Word)", fontsize=14)
        plt.savefig("precision_single.png", bbox_inches="tight")
        plt.close()


def main():
    # maximum value at which F1 score is calculated
    max_k = 300

    # length of longest word
    max_word_len = 10
    # epsilon for differential privacy
    epsilon = 4
    # delta for differential privacy
    delta = 2.3e-12

    # repeat simulation for num_runs times
    num_runs = 5

    simulate_triehh = SimulateTrieHH(
        max_word_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs
    )
    triehh_heavy_hitters = simulate_triehh.get_heavy_hitters()

    plot = Plot(max_k)
    plot.plot_precision_recall(triehh_heavy_hitters, epsilon)


if __name__ == "__main__":
    main()
