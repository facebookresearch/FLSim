#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse

import koski.dataframes as kd

"""
Sample command to run this script from fbcode directory:
buck run @mode/dev-nosan papaya/toolkit/simulation/experimental/analytics:hive_reader -- --output_file "/data/users/akashb/fa_data/cell_tower_info_tile_ip_clean_ni_agg_updated.csv"
"""


def main():
    parser = argparse.ArgumentParser(description="Download FBC data from HIVE")

    parser.add_argument(
        "--output_file",
        dest="output_file",
        help="Path to directory where plots must be stored",
        type=str,
    )

    parser.add_argument(
        "--table_name",
        dest="table_name",
        help="Name of Hive table with desired data",
        type=str,
        default="cell_tower_info_tile_ip_clean_ni_agg",
    )

    parser.add_argument(
        "--table_namespace",
        dest="table_namespace",
        help="Namespace for Hive table",
        type=str,
        default="mobile",
    )

    parser.add_argument(
        "--partition_date",
        dest="partition_date",
        help="Date associated with partition of Hive table to read",
        type=str,
        default="'2021-04-12'",
    )

    parser.add_argument(
        "--oncall",
        dest="oncall",
        help="Oncall for hive table",
        type=str,
        default="edge_insights",
    )

    args = parser.parse_args()

    ctx = kd.create_ctx(
        use_case=kd.UseCase.PROD,
        description="FBC hive reader",
        oncall=args.oncall,
    )
    kdf = kd.data_warehouse(
        namespace=args.table_namespace, table=args.table_name, session_ctx=ctx
    )
    kdf_partition = kdf.filter(
        "ds=%s" % (args.partition_date)
    )  # replace with appropriate partition date
    column_names = [col.name for col in kdf_partition.columns()]
    print(column_names)

    with open(args.output_file, "a") as opf:
        for row in kdf_partition.rows():
            # may need to update this based on schema of table, which
            # can change from time to time
            row = list(row)
            # print(row)
            row[1] = str(row[1])
            row[5] = str(row[5])
            row[6] = str(row[6])
            row[7] = str(row[7])
            row[8] = str(row[8])
            row[9] = str(row[9])
            opf.write(",".join(row) + "\n")


if __name__ == "__main__":
    main()
