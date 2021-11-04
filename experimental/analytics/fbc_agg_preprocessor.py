#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.


"""Preprocessing data."""
import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


"""
Dataset to be used is derived from the cell_tower_info_tile_ip_clean_ni_agg:mobile table on Hive. Note that
the partition to be used is ds='2020-11-15'.

Columns include:
1. country
2. carrier_id
3. client_prefix
4. quadkey_10
5. num_samples
6. rank_prefix
7. median_download_speed_mbps
8. avg_download_speed_mbps
9. variance_download_speed_mbps
10. ds

This data was collected from both Android and iOS devices, after taking sufficient precautions
to ensure privacy.

This particular table also benefits from various preprocessing steps, such as:
1. Rows with non-existent country, quadkey, carrier_id and client_prefix have been removed
2. Quadkey_10 actually occurs within the country of interest
3. Smaller countries in which FBC doesn't have partners have been removed

Some additional pre-processing to do:
1. Eliminate quadkeys with too few samples (<100 say)
2. Handle ipv4 and ipv6 addresses appropriately. Both are mixed in this table. Embedding ipv4s
   in ipv6 addresses will probably lead to ipv6 addresses being drowned out. For now ipv6 addresses
   are discarded.

Some outstanding issues include:
1. Samples include data from users on WiFi on iOS
2. Download speed can be NULL if download speed is faster that shortest clock tick, leading to
   division by 0

From command line, while in fbcode directory, run:
buck run papaya/toolkit/simulation/experimental/analytics:fbc_agg_preprocessor

This script produces

1. A dictionary of the following form:
{
    country: {
                tile: {
                    word: {
                            carrier: (count, stat_avg, stat_stddev)
                        }
                }
            }
}

2a. A dictionary of the following form
{
    country: avg_samples_per_tile_in_country
}

2b. A dictionary of the following form
{
    country: {
        tile: num_samples_for_tile
    }
}

sorted on the number of samples. This will be used to analyse the effect of population size
in a tile on algorithm performance/convergence characteristics at a given K.

Note that the number of unique IPs is not tracked since it is not known in production.

3. Char vocab
Vocabulary of all characters used in the dataset.

"""

FILENAME = ""
ARTIFACTS_PATH = ""

char_vocab = set()


def add_end_symbol(word):
    return word + "$"


def get_clients(filename, prefix_len=-1):
    """
    Returns a dictionary of the following form:
    {
        country: {
                    tile: {
                        word: {
                            carrier: (count, stat_avg, stat_stddev)
                            }
                    }
                }
    }
    Note that for this experiment, we pretend that each client has a single
    IP. In practice, a given client has multiple IPs. However, TrieHH allows
    us to simplify this to the single IP case by simply treating each IP as
    belonging to a different client.

    If prefix_len > 0, then words are truncated to that length.
    """
    ret = {}
    tot_num_clients = 0
    tot_ipv6_skipped = 0
    tot_none_skipped = 0
    country_to_tile_counts = {}
    country_tile_count_averages = {}

    def process_row(row):
        """
        Internal helper function to parse one row of the CSV and update
        the main data structures:
            ret, country_to_tile_counts, country_tile_count_averages
        """
        nonlocal tot_num_clients, tot_ipv6_skipped, tot_none_skipped
        if len(row) != 11:
            del row[3]
            # handle data quality error where Mocabique Telecom, SA is parsed as two fields instead of one
            # print(row)
        (
            country,
            id_number,
            carrier,
            client_prefix,
            tile,
            num_samples,
            rank_prefix,
            median_download_speed_mbps,
            average_download_speed_mpbs,
            variance_download_speed_mbps,
            ds,
        ) = row

        if country not in country_to_tile_counts:
            country_to_tile_counts[country] = {}
        tile_counts = country_to_tile_counts[country]
        num_samples = int(num_samples)
        tot_num_clients += num_samples
        # for the FBC case, IP address is both the client id and the data we are
        # trying to recover
        try:
            float(average_download_speed_mpbs), float(variance_download_speed_mbps)
        except BaseException:
            tot_none_skipped += num_samples
            return  # download speed is None (skipping)

        if ":" in client_prefix:
            # indicates ipv6
            tot_ipv6_skipped += num_samples
            return
        word = client_prefix if prefix_len <= 0 else client_prefix[: prefix_len + 1]
        [char_vocab.add(c) for c in word]

        tile_counts[tile] = tile_counts.get(tile, 0.0) + num_samples
        if country not in ret:
            ret[country] = {}
        if tile not in ret[country]:
            ret[country][tile] = {}
        clients = ret[country][tile]
        if word not in clients:
            clients[word] = {}

        if carrier in clients[word]:
            print(
                "Repeated carrier for IP %s for same country %s and tile %s!"
                % (word, country, tile)
            )
            return
        clients[word][carrier] = (
            num_samples,
            float(average_download_speed_mpbs),
            math.sqrt(float(variance_download_speed_mbps)),
        )
        return

    with open(filename, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for rid, row in enumerate(csv_reader):
            if rid == 0:
                continue
            if rid % 100000 == 0:
                print("Processed %d rows" % (rid))
            process_row(row)
    if tot_num_clients > 0:
        print(f"Total number of clients processed: {tot_num_clients}")
        print(
            "Skipped percent of clients for having ipv6 addresses:",
            (tot_ipv6_skipped * 100.0 / tot_num_clients),
        )
        print(
            "Skipped percent of clients due to missing network quality statistic:",
            (tot_none_skipped * 100.0 / tot_num_clients),
        )
    for country in country_to_tile_counts:
        country_tile_count_averages[country] = float(
            np.mean(list(country_to_tile_counts[country].values()))
        )
    return ret, country_to_tile_counts, country_tile_count_averages


def get_file_arguments():
    """
    Build a parser to capture the input and output file destinations
    """
    global FILENAME, ARTIFACTS_PATH
    parser = argparse.ArgumentParser(description="Preprocess FBC data for Trie HH")

    parser.add_argument(
        "--fbc_data_file",
        dest="fbc_data_file",
        help="Path to CSV file containing raw data from Hive",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Path to directory where processed data will be stored",
        type=str,
    )

    args = parser.parse_args()

    FILENAME = args.fbc_data_file
    ARTIFACTS_PATH = args.output_dir

    if not ARTIFACTS_PATH:
        raise Exception("Output directory must be specified with --output_dir")
    if not FILENAME:
        raise Exception("Input CSV file must be specified with --fbc_data_file")
    return (ARTIFACTS_PATH, FILENAME)


def write_country_to_tile(
    country_to_tile_to_clients,
    country_to_tile_counts,
    country_tile_count_averages,
):
    global ARTIFACTS_PATH
    for country in country_to_tile_to_clients:
        for tile in country_to_tile_to_clients[country]:
            with open(
                os.path.join(
                    ARTIFACTS_PATH,
                    country + "_" + tile + "_clients_per_carrier.pkl",
                ),
                "w",
            ) as opf:
                opf.write(json.dumps(country_to_tile_to_clients[country][tile]))
    with open(
        os.path.join(ARTIFACTS_PATH, "country_to_tile_to_clients.pkl"), "w"
    ) as opf:
        opf.write(json.dumps(country_to_tile_to_clients))
    with open(os.path.join(ARTIFACTS_PATH, "country_to_tile_counts.pkl"), "w") as opf:
        opf.write(json.dumps(country_to_tile_counts))
    with open(
        os.path.join(ARTIFACTS_PATH, "country_tile_count_averages.pkl"), "w"
    ) as opf:
        opf.write(json.dumps(country_tile_count_averages))
    with open(os.path.join(ARTIFACTS_PATH, "char_vocab.pkl"), "w") as opf:
        opf.write(json.dumps(list(char_vocab)))
    return


def write_all_tile_counts(country_to_tile_counts, clipping_number=5000):
    global ARTIFACTS_PATH
    all_tile_counts = []
    # Dictionary the groups tiles by the regime in which the number of samples exists
    sample_regime_to_tiles_info = {
        (0, 100): [],
        (1000, 50000): [],
        (50001, 100000): [],
        (100001, 1000000): [],
        (1000001, 10000000000): [],
    }
    for country in country_to_tile_counts:
        all_tile_counts.extend(list(country_to_tile_counts[country].values()))
        for tile in country_to_tile_counts[country]:
            for key in sample_regime_to_tiles_info.keys():
                if (
                    country_to_tile_counts[country][tile] >= key[0]
                    and country_to_tile_counts[country][tile] <= key[1]
                ):
                    sample_regime_to_tiles_info[key].append(
                        (country, tile, country_to_tile_counts[country][tile])
                    )
    for key in sample_regime_to_tiles_info:
        sample_regime_to_tiles_info[key].sort(key=lambda item: item[2])
    sample_regime_to_tiles_info_string_keys = {
        str(x): sample_regime_to_tiles_info[x] for x in sample_regime_to_tiles_info
    }
    with open(
        os.path.join(ARTIFACTS_PATH, "sample_regime_to_tiles_info.pkl"), "w"
    ) as opf:
        opf.write(json.dumps(sample_regime_to_tiles_info_string_keys))

    if len(all_tile_counts) < 2 * clipping_number:
        clipping_number = 0

    clipped_tile_counts = sorted(all_tile_counts)[clipping_number:-clipping_number]
    return clipped_tile_counts


def generate_tile_statistics(all_tile_counts):
    global ARTIFACTS_PATH
    print(
        "Min=%f, max=%f, mean=%f, median=%f, std dev=%f, num_tiles=%f,"
        % (
            min(all_tile_counts),
            max(all_tile_counts),
            np.mean(all_tile_counts),
            np.median(all_tile_counts),
            np.std(all_tile_counts),
            len(all_tile_counts),
        )
    )

    n, bins, patches = plt.hist(
        x=all_tile_counts, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("# samples for tile")
    plt.ylabel("# tiles")
    plt.title("Distribution of number of samples per tile")
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    print(bins)
    plt.savefig(os.path.join(ARTIFACTS_PATH, "tile_frequency_hist.png"))
    return


"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation/experimental/analytics:fbc_agg_preprocessor -- --fbc_data_file "/data/users/akashb/fa_data/cell_tower_info_tile_ip_clean_ni_agg_updated.csv" --output_dir "/data/users/akashb/fa_data/processed/"
"""


def main():
    (ARTIFACTS_PATH, FILENAME) = get_file_arguments()
    if not os.path.isdir(ARTIFACTS_PATH):
        os.mkdir(ARTIFACTS_PATH)
    (
        country_to_tile_to_clients,
        country_to_tile_counts,
        country_tile_count_averages,
    ) = get_clients(FILENAME)
    write_country_to_tile(
        country_to_tile_to_clients,
        country_to_tile_counts,
        country_tile_count_averages,
    )

    clipped_tile_counts = write_all_tile_counts(country_to_tile_counts)
    if clipped_tile_counts:
        generate_tile_statistics(clipped_tile_counts)
    print("Done!")


if __name__ == "__main__":
    main()
