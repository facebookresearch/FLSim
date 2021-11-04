#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import os

import koski.dataframes as kd

"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation/experimental/analytics/median:hive_reader_pdp -- --output_file "/data/users/akashb/fa_data/pdp_data.csv" --table_name "pdp_fb_cp_data_for_feat_193_country_123" --table_namespace "growth" --oncall "pdp_ml_infra" --feat_ids 193 192 9
"""


def main():
    parser = argparse.ArgumentParser(description="Download FAM data from HIVE")

    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Path to directory where plots must be stored",
        type=str,
        default="/data/users/akashb/fbsource/fbcode/papaya/toolkit/simulation/experimental/analytics/median/data/pdp_data_try2/",
    )

    parser.add_argument(
        "--table_name",
        dest="table_name",
        help="Name of Hive table with desired data",
        type=str,
        default="pdp_fb_cp_data_for_feat_193_country_123",
    )

    parser.add_argument(
        "--table_namespace",
        dest="table_namespace",
        help="Namespace for Hive table",
        type=str,
        default="growth",
    )

    parser.add_argument(
        "--oncall",
        dest="oncall",
        help="Oncall for hive table",
        type=str,
        default="pdp_ml_infra",
    )

    parser.add_argument(
        "--feat_ids",
        dest="required_features",
        help="The IDs for features that must be collected",
        nargs="*",
        type=int,
        default=[193, 192, 9],
    )

    args = parser.parse_args()

    ctx = kd.create_ctx(
        use_case=kd.UseCase.PROD,
        description="PDP hive reader",
        oncall=args.oncall,
    )
    kdf = kd.data_warehouse(
        namespace=args.table_namespace, table=args.table_name, session_ctx=ctx
    )

    column_names = [col.name for col in kdf.columns()]
    print(column_names)

    num_skipped = 0.0
    tot_samples = 0.0
    country_specific_files = {}
    for row in kdf.rows():
        # may need to update this based on schema of table, which
        # can change from time to time
        tot_samples += 1
        missing_at_least_one_feat = False
        for feat in args.required_features:
            if feat not in row[0] or 5 not in row[0]:
                missing_at_least_one_feat = True
                continue
            if str(row[0][5]) + "_" + str(feat) not in country_specific_files:
                country_specific_files[str(row[0][5]) + "_" + str(feat)] = open(
                    os.path.join(
                        args.output_dir,
                        "pdp_feats_for_country_"
                        + str(row[0][5])
                        + "_and_feat_"
                        + str(feat)
                        + ".csv",
                    ),
                    "w",
                )
                country_specific_files[str(row[0][5]) + "_" + str(feat)].write(
                    ",".join([column_names[-1], "feat_" + str(feat)]) + "\n"
                )
            country_specific_files[str(row[0][5]) + "_" + str(feat)].write(
                ",".join([str(row[1]), str(row[0][feat])]) + "\n"
            )
        if missing_at_least_one_feat:
            num_skipped += 1
        if tot_samples % 10000000 == 0:
            print(
                "Finished %d records and skipped at most %d records"
                % (tot_samples, num_skipped)
            )
    for country in country_specific_files:
        country_specific_files[country].close()


if __name__ == "__main__":
    main()
