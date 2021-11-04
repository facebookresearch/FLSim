#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import os
import pickle as pkl

import numpy as np
import plotly.graph_objects as go
import scipy
import scipy.stats


def make_plotly_surface(orig_x, orig_y, orig_z, name, opacity=1.0):
    """
    orig_z has values for corresponding entries in orig_x and orig_y.
    """
    uniq_x = list(set(orig_x))
    x = sorted(uniq_x)
    uniq_y = list(set(orig_y))
    y = sorted(uniq_y)
    z = np.zeros((len(x), len(y)))
    xind_dict = {j: i for (i, j) in enumerate(x)}
    yind_dict = {j: i for (i, j) in enumerate(y)}
    for i in range(len(orig_z)):
        x_ind = xind_dict[orig_x[i]]
        y_ind = yind_dict[orig_y[i]]
        z[x_ind][y_ind] = orig_z[i]
    return go.Surface(z=z, x=x, y=y, name=name, opacity=opacity)


def get_mean_u_l(recall_values, confidence=0.95):
    mean = np.mean(recall_values)
    n = len(recall_values)
    std_err = scipy.stats.sem(recall_values)
    h = std_err * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean + h, mean - h


"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation/experimental/analytics:fbc_plot_results -- --results_file "/data/users/akashb/fbsource/fbcode/papaya/toolkit/simulation/experimental/analytics/triehh_fbc_results_full_ablation_upto_k_100_upto_regime_500000_1000000_no_ensemble_beam_50_10.0_ldp.pkl" --plot_dir "/data/users/akashb/fbsource/fbcode/papaya/toolkit/simulation/experimental/analytics/plots/"

The sequence of steps to produce plots is as follows:
1. Run hive_reader to fetch relevant data from Hive tables into a csv file
2. Run fbc_agg_preprocessor to produce processed data for simulation
3. Run fbc_main to run simulations and store results for evaluation
4. Run fbc_plot_results to produce 3D surface plots examining the effect of varying hyper-parameters on performance metrics for tiles in different population regimes
"""


def main():
    parser = argparse.ArgumentParser(description="Plot results of FA query for Trie HH")

    parser.add_argument(
        "--results_file",
        dest="results_file",
        help="Path to output file produced by fbc_main",
        type=str,
    )
    parser.add_argument(
        "--plot_dir",
        dest="plot_dir",
        help="Path to directory where plots must be stored. Note that this directory must already exist and must contain a directory for each regime of interest with naming "
        " convention regime_<lower_bound>_<upper_bound> (eg. regime_1000_10000), each of which has sub-directories named 'f1', 'prec', 'rec'.",
        type=str,
    )

    args = parser.parse_args()

    with open(args.results_file, "rb") as ipf:
        results = pkl.load(ipf)

    all_keys = list(results.keys())
    max_k = max(item[-1] for item in all_keys)
    epsilons = sorted({item[1] for item in all_keys})
    alphabet_lens = sorted({item[2] for item in all_keys})
    regimes = list({item[0] for item in all_keys})

    """
    (regime, epsilon, alphabet_len, K) ->
                    (
                        [recall@K for all runs for all tiles],
                        [precision@K for all runs for all tiles],
                        [f1@K for all runs for all tiles],
                        [jaccard_index for all runs for all tiles],
                    )

    Using these results, we'd like guidance on how to pick epsilon and alphabet len

    Plots we are interested in:
    1. Within each regime:
        3-D plots of (epsilon, alphabet len) vs. average prec/rec/f1 across multiple tiles in a given regime at fixed
        K (can plot for K at intervals of 5).

        Note that all the above plots will include the 95% confidence intervals

    2. Across regimes:
        Trends in optimal epsilon/alphabet len values as a function of population size. It's probably simpler to analyse
        the above per-regime plots and draw inferences.


    A note on interpreting the plots produced:
    If the confidence interval at any given hyper-parameter setting is non-trivial (width>0) and contains 0, then it
    means that hyper-parameters can be successfully set but results are often empty. For the same reason, the mean F1
    is biased towards 0.

    If the confidence interval has 0 width, then it means that either the algorithm trivially fails to converge at the
    specified (epsilon, delta, alphabet len) or that theoretical epsilon guarantees are not met within a 10% tolerance.
    This can happen since some numerical optimization is involved in deriving vote threshold and batch size at specified
    epsilon and there can be inaccuracies.

    More generally, narrow confidence intervals mean predictable behavior.

    Another issue is that when operating in low sample regimes like the (1000, 10000) range, it is possible that the
    number of unique prefixes < max_k. In such a case, the evaluation of recall does NOT penalize simply for arbitrarily
    scaling up K, but rather limits it to whatever the recall was at the total number of unique prefixes.

    Precision is unaffected by this. Keep in mind that when evaluating precision at K, the systems predictions (assumed to
    be sorted by vote count) is also truncated to only inlude the top K.

    We could also experiment with ensembling multiple runs to see if performance improves that way (approaches mean, or
    corresponds to improved performance at a lower K). For example, if we choose hyper-parameters in a region with wide
    confidence intervals that include zero, union of multiple runs could help achieve performance comparable or even
    greater than the mean (which is usually biased towards 0). Assuming secure aggregation is used, repeated disclosures
    by clients is not an issue.

    Lastly, when choosing hyper-parameters, it is probably worth choosing a setting in a region of the manifold that
    has low gradients along all dimensions, while having an acceptable mean f1 and narrow confidence intervals that
    preferably don't contain 0. Moreover, given that we don't necessarily know the cardinality of samples from a given
    tile accurately in real time, it might be worth choosing a region that gives somewhat reliable performance across
    multiple regimes at a desired K (assuming desired K is the same across regimes).
    """
    if max_k > 5:
        K_milestones = [1, 2, 3, 4] + list(range(5, max_k + 1, 5))
    else:
        K_milestones = list(range(1, max_k + 1))

    for regime in regimes:
        for K in K_milestones:
            plot_x = []
            plot_y = []

            plot_z_mean_prec = []
            plot_z_prec_uc = []
            plot_z_prec_lc = []

            plot_z_mean_rec = []
            plot_z_rec_uc = []
            plot_z_rec_lc = []

            plot_z_mean_f1 = []
            plot_z_f1_uc = []
            plot_z_f1_lc = []

            for epsilon in epsilons:
                for alpha_len in alphabet_lens:
                    plot_x.append(epsilon)
                    plot_y.append(alpha_len)
                    all_tiles_results = results[(regime, epsilon, alpha_len, K)]
                    mean_rec, rec_hc, rec_lc = get_mean_u_l(all_tiles_results[0])
                    mean_prec, prec_hc, prec_lc = get_mean_u_l(all_tiles_results[1])
                    mean_f1, f1_hc, f1_lc = get_mean_u_l(all_tiles_results[2])

                    plot_z_mean_rec.append(mean_rec)
                    plot_z_rec_uc.append(rec_hc)
                    plot_z_rec_lc.append(rec_lc)

                    plot_z_mean_prec.append(mean_prec)
                    plot_z_prec_uc.append(prec_hc)
                    plot_z_prec_lc.append(prec_lc)

                    plot_z_mean_f1.append(mean_f1)
                    plot_z_f1_uc.append(f1_hc)
                    plot_z_f1_lc.append(f1_lc)

            fig = go.Figure()
            fig.add_trace(
                make_plotly_surface(plot_x, plot_y, plot_z_mean_rec, name="Mean recall")
            )
            fig.add_trace(
                make_plotly_surface(
                    plot_x,
                    plot_y,
                    plot_z_rec_uc,
                    name="Confidence interval upper bound",
                    opacity=0.7,
                )
            )
            fig.add_trace(
                make_plotly_surface(
                    plot_x,
                    plot_y,
                    plot_z_rec_lc,
                    name="Confidence interval lower bound",
                    opacity=0.7,
                )
            )

            fig.update_layout(
                title="Recall vs. epsilon (x) and alphabet length (y) at K = %d for tile with samples in range (%d, %d)"
                % (K, regime[0], regime[1]),
                xaxis_title="Epsilon",
                yaxis_title="Alphabet length",
                autosize=False,
                width=1000,
                height=1000,
                margin={"l": 65, "r": 50, "b": 65, "t": 90},
            )

            fig.write_html(
                os.path.join(
                    os.path.join(
                        args.plot_dir, "regime_%s_%s" % (str(regime[0]), str(regime[1]))
                    ),
                    "rec/rec_at_k_%s_regime_%s_%s_plot.html"
                    % (str(K), str(regime[0]), str(regime[1])),
                )
            )

            fig = go.Figure()
            fig.add_trace(
                make_plotly_surface(
                    plot_x, plot_y, plot_z_mean_prec, name="Mean precision"
                )
            )
            fig.add_trace(
                make_plotly_surface(
                    plot_x,
                    plot_y,
                    plot_z_prec_uc,
                    name="Confidence interval upper bound",
                    opacity=0.7,
                )
            )
            fig.add_trace(
                make_plotly_surface(
                    plot_x,
                    plot_y,
                    plot_z_prec_lc,
                    name="Confidence interval lower bound",
                    opacity=0.7,
                )
            )

            fig.update_layout(
                title="Precision vs. epsilon (x) and alphabet length (y) at K = %d for tile with samples in range (%d, %d)"
                % (K, regime[0], regime[1]),
                xaxis_title="Epsilon",
                yaxis_title="Alphabet length",
                autosize=False,
                width=1000,
                height=1000,
                margin={"l": 65, "r": 50, "b": 65, "t": 90},
            )

            fig.write_html(
                os.path.join(
                    os.path.join(
                        args.plot_dir, "regime_%s_%s" % (str(regime[0]), str(regime[1]))
                    ),
                    "prec/prec_at_k_%s_regime_%s_%s_plot.html"
                    % (str(K), str(regime[0]), str(regime[1])),
                )
            )

            fig = go.Figure()
            fig.add_trace(
                make_plotly_surface(plot_x, plot_y, plot_z_mean_f1, name="Mean f1")
            )
            fig.add_trace(
                make_plotly_surface(
                    plot_x,
                    plot_y,
                    plot_z_f1_uc,
                    name="Confidence interval upper bound",
                    opacity=0.7,
                )
            )
            fig.add_trace(
                make_plotly_surface(
                    plot_x,
                    plot_y,
                    plot_z_f1_lc,
                    name="Confidence interval lower bound",
                    opacity=0.7,
                )
            )

            fig.update_layout(
                title="F1 vs. epsilon (x) and alphabet length (y) at K = %d for tile with samples in range (%d, %d)"
                % (K, regime[0], regime[1]),
                xaxis_title="Epsilon",
                yaxis_title="Alphabet length",
                autosize=False,
                width=1000,
                height=1000,
                margin={"l": 65, "r": 50, "b": 65, "t": 90},
            )

            fig.write_html(
                os.path.join(
                    os.path.join(
                        args.plot_dir, "regime_%s_%s" % (str(regime[0]), str(regime[1]))
                    ),
                    "f1/f1_at_k_%s_regime_%s_%s_plot.html"
                    % (str(K), str(regime[0]), str(regime[1])),
                )
            )


if __name__ == "__main__":
    main()
