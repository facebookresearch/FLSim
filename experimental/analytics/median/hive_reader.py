#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse

import koski.dataframes as kd

"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation/experimental/analytics/median:hive_reader -- --output_file "/data/users/akashb/fa_data/fam_data.csv" --table_name "FAM_cvr_app_ads_after_sl_train_set_rid_with_weight_April_24" --table_namespace "ad_delivery" --oncall "Looper" --feat_ids 21179536 3029 4160 4461 11225961 11532367 11607667 11655148 11980884 12213457 12511372 13134225 13697760 15048077 15145804 16062953 17136232 17324224 17568074 19251365
"""


def main():
    parser = argparse.ArgumentParser(description="Download FAM data from HIVE")

    parser.add_argument(
        "--output_file",
        dest="output_file",
        help="Path to directory where plots must be stored",
        type=str,
        default="/data/sandcastle/boxes/fbsource/fbcode/papaya/toolkit/simulation/experimental/analytics/median/data/fam_data.csv",
    )

    parser.add_argument(
        "--table_name",
        dest="table_name",
        help="Name of Hive table with desired data",
        type=str,
        default="FAM_cvr_app_ads_after_sl_train_set_rid_with_weight_April_24",
    )

    parser.add_argument(
        "--table_namespace",
        dest="table_namespace",
        help="Namespace for Hive table",
        type=str,
        default="ad_delivery",
    )

    parser.add_argument(
        "--oncall",
        dest="oncall",
        help="Oncall for hive table",
        type=str,
        default="Looper",
    )

    parser.add_argument(
        "--feat_ids",
        dest="required_features",
        help="The IDs for features that must be collected",
        nargs="*",
        type=int,
        default=[
            21179536,
            3029,
            4160,
            4461,
            11225961,
            11532367,
            11607667,
            11655148,
            11980884,
            12213457,
            12511372,
            13134225,
            13697760,
            15048077,
            15145804,
            16062953,
            17136232,
            17324224,
            17568074,
            19251365,
        ],
    )

    args = parser.parse_args()

    ctx = kd.create_ctx(
        use_case=kd.UseCase.PROD,
        description="FAM hive reader",
        oncall=args.oncall,
    )
    kdf = kd.data_warehouse(
        namespace=args.table_namespace, table=args.table_name, session_ctx=ctx
    )

    column_names = [col.name for col in kdf.columns()]
    print(column_names)

    required_non_feature_columns = [1]

    rid = 0
    with open(args.output_file, "w") as opf:
        opf.write(
            ",".join(
                [str(column_names[i]) for i in required_non_feature_columns]
                + [str(id) for id in args.required_features]
            )
            + "\n"
        )
        for row in kdf.rows():
            # may need to update this based on schema of table, which
            # can change from time to time
            str1 = ",".join([str(row[i]) for i in required_non_feature_columns])
            str2 = ",".join(
                [
                    str(row[2][feat]) if feat in row[2] else "None"
                    for feat in args.required_features
                ]
            )
            opf.write(",".join([str1, str2]) + "\n")
            rid += 1
            if rid % 50000 == 0:
                print("Finished %d records" % (rid))


if __name__ == "__main__":
    main()
