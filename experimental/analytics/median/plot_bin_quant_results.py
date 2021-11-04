#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import os
import pickle

import numpy as np
import plotly.graph_objects as go
import scipy
import scipy.stats

"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation/experimental/analytics/median:plot_bin_quant_results --  --op_dir "/data/users/akashb/fbsource/fbcode/papaya/toolkit/simulation/experimental/analytics/median/data/results/canonical_dist" --plot_dir "/data/users/akashb/fbsource/fbcode/papaya/toolkit/simulation/experimental/analytics/median/data/results/canonical_dist_plots"
"""


def get_mean_u_l(recall_values, confidence=0.95):
    mean = np.mean(recall_values)
    n = len(recall_values)
    std_err = scipy.stats.sem(recall_values)
    h = std_err * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean + h, mean - h


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


def make_fig(
    x, y, mz, ubz, lbz, op_param_name, fig_title, x_title, y_title, html_name, plot_dir
):
    fig = go.Figure()
    fig.add_trace(make_plotly_surface(x, y, mz, name="Mean " + op_param_name))
    fig.add_trace(
        make_plotly_surface(
            x,
            y,
            ubz,
            name="CI upper bound for " + op_param_name,
            opacity=0.7,
        )
    )
    fig.add_trace(
        make_plotly_surface(
            x,
            y,
            lbz,
            name="CI lower bound for " + op_param_name,
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=fig_title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        autosize=False,
        width=1000,
        height=1000,
        margin={"l": 65, "r": 50, "b": 65, "t": 9},
    )

    fig.write_html(os.path.join(plot_dir, html_name))


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a quantile with differentia privacy"
    )
    parser.add_argument(
        "--op_dir",
        dest="op_dir",
        help="Path to directory into which pickle files containing results of runs and confidence intervals"
        " should be retained",
        type=str,
    )
    parser.add_argument(
        "--plot_dir",
        dest="plot_dir",
        help="Path to directory into which pickle files containing results of runs and confidence intervals"
        " should be retained",
        type=str,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    with open(
        os.path.join(args.op_dir, "pop_dist_perc_eps_to_emplams.pkl"), "rb"
    ) as opf:
        pop_dist_perc_eps_to_emplams = pickle.load(opf)

    with open(os.path.join(args.op_dir, "pop_dist_perc_eps_to_taus.pkl"), "rb") as opf:
        pop_dist_perc_eps_to_taus = pickle.load(opf)

    with open(
        os.path.join(args.op_dir, "pop_dist_perc_eps_to_emptaus.pkl"), "rb"
    ) as opf:
        _ = pickle.load(opf)

    with open(
        os.path.join(args.op_dir, "pop_dist_perc_eps_to_min_num_clients.pkl"), "rb"
    ) as opf:
        pop_dist_perc_eps_to_num_clients = pickle.load(opf)

    with open(
        os.path.join(args.op_dir, "pop_dist_perc_lam_to_min_num_clients.pkl"), "rb"
    ) as opf:
        pop_dist_perc_lam_to_num_clients = pickle.load(opf)

    pop_sizes = list(pop_dist_perc_eps_to_emplams.keys())
    dists = list(pop_dist_perc_eps_to_emplams[pop_sizes[0]].keys())

    for N in pop_sizes:
        for dist in dists:
            # plotting manifold for desired_quantile vs. epsilon vs. empirical quantile error
            quantiles = sorted(pop_dist_perc_eps_to_emplams[N][dist].keys())
            epsilons = sorted(
                pop_dist_perc_eps_to_emplams[N][dist][quantiles[0]].keys()
            )
            x = []
            y = []
            mean_emplams = []
            ub_emplams = []
            lb_emplams = []
            for q in quantiles:
                for eps in epsilons:
                    x.append(q)
                    y.append(eps)
                    m, ub, lb = get_mean_u_l(
                        pop_dist_perc_eps_to_emplams[N][dist][q][eps]
                    )
                    mean_emplams.append(m)
                    ub_emplams.append(ub)
                    lb_emplams.append(lb)

            make_fig(
                x,
                y,
                mean_emplams,
                ub_emplams,
                lb_emplams,
                "Quantile Error",
                "Desired Quantile vs. Epsilon vs. Empirical Quantile Error",
                "Desired Quantile",
                "Epsilon",
                "Quantile_Epsilon_Emp_lambda_for_%s_at_N=%d.html" % (dist, N),
                args.plot_dir,
            )

            # plotting manifold for desired_quantile vs. epsilon vs. tau
            quantiles = sorted(pop_dist_perc_eps_to_taus[N][dist].keys())
            epsilons = sorted(pop_dist_perc_eps_to_taus[N][dist][quantiles[0]].keys())
            x = []
            y = []
            mean_taus = []
            ub_taus = []
            lb_taus = []
            for q in quantiles:
                for eps in epsilons:
                    x.append(q)
                    y.append(eps)
                    m, ub, lb = get_mean_u_l(pop_dist_perc_eps_to_taus[N][dist][q][eps])
                    mean_taus.append(m)
                    ub_taus.append(ub)
                    lb_taus.append(lb)

            make_fig(
                x,
                y,
                mean_taus,
                ub_taus,
                lb_taus,
                "Absolute Error",
                "Desired Quantile vs. Epsilon vs. Empirical Absolute Error",
                "Desired Quantile",
                "Epsilon",
                "Quantile_Epsilon_Tau_for_%s_at_N=%d.html" % (dist, N),
                args.plot_dir,
            )

            # plotting manifold for desired_quantile vs. epsilon vs. num samples
            quantiles = sorted(pop_dist_perc_eps_to_num_clients[N][dist].keys())
            epsilons = sorted(
                pop_dist_perc_eps_to_num_clients[N][dist][quantiles[0]].keys()
            )
            x = []
            y = []
            mean_nc = []
            ub_nc = []
            lb_nc = []
            for q in quantiles:
                for eps in epsilons:
                    x.append(q)
                    y.append(eps)
                    m, ub, lb = get_mean_u_l(
                        pop_dist_perc_eps_to_num_clients[N][dist][q][eps]
                    )
                    mean_nc.append(m)
                    ub_nc.append(ub)
                    lb_nc.append(lb)

            make_fig(
                x,
                y,
                mean_nc,
                ub_nc,
                lb_nc,
                "# samples",
                "Desired Quantile vs. Epsilon vs. # samples",
                "Desired Quantile",
                "Epsilon",
                "Quantile_Epsilon_num_samples_for_%s_at_N=%d.html" % (dist, N),
                args.plot_dir,
            )

            # plotting manifold for desired_quantile vs. tolerance vs. num samples
            quantiles = sorted(pop_dist_perc_lam_to_num_clients[N][dist].keys())
            tolerances = sorted(
                pop_dist_perc_lam_to_num_clients[N][dist][quantiles[0]].keys()
            )
            x = []
            y = []
            mean_nc = []
            ub_nc = []
            lb_nc = []
            for q in quantiles:
                for eps in tolerances:
                    x.append(q)
                    y.append(eps)
                    m, ub, lb = get_mean_u_l(
                        pop_dist_perc_lam_to_num_clients[N][dist][q][eps]
                    )
                    mean_nc.append(m)
                    ub_nc.append(ub)
                    lb_nc.append(lb)

            make_fig(
                x,
                y,
                mean_nc,
                ub_nc,
                lb_nc,
                "# samples",
                "Desired Quantile vs. QE Tolerance vs. # samples",
                "Desired Quantile",
                "Quantile Error Tolerance",
                "Quantile_Tolerance_Num_samples_for_%s_at_N=%d.html" % (dist, N),
                args.plot_dir,
            )
    print("Done!")


if __name__ == "__main__":
    main()
