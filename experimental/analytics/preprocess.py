#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.


"""Preprocessing data."""
import collections
import csv
import json
import operator
import os
import re

import numpy as np


# sentiment 140 dataset from here
# http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
FILENAME: str = "/mnt/vol/gfsfblearner-carolina/users/harishs/training.1600000.processed.noemoticon.csv"
# tmp path
ARTIFACTS_PATH: str = "/tmp/papaya/"


def is_valid(word):
    if (
        len(word) < 3
        or (word[-1] in ["?", "!", ".", ";", ","])
        or word.startswith("http")
        or word.startswith("www")
    ):
        return False
    if re.match(r"^[a-z_\@\#\-\;\(\)\*\:\.\'\/]+$", word):
        return True
    return False


def get_clients(filename):
    """Returns a dictionary of dictionaries containing per client word frequencies."""

    clients = {}
    with open(filename, encoding="ISO-8859-1") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            client = row[4]
            comment = row[5]

            raw_words = comment.lower().split()
            raw_words = [word.strip(",.;?!") for word in raw_words]
            raw_words = [x for x in raw_words if is_valid(x)]

            # don't create client if he/she has no valid words
            word_len = len(raw_words)
            if word_len > 0 and client not in clients:
                clients[client] = {}
            for word in raw_words:
                if word not in clients[client]:
                    clients[client][word] = 1
                else:
                    clients[client][word] += 1
    # change word counts to percentages
    for client in clients:
        num_words = sum(clients[client].values())
        for word in clients[client]:
            clients[client][word] = clients[client][word] * 1.0 / num_words
    return clients


def add_end_symbol(word):
    return word + "$"


def generate_triehh_clients(clients):
    clients_num = len(clients)
    triehh_clients = [add_end_symbol(clients[i]) for i in range(clients_num)]
    word_freq = collections.defaultdict(lambda: 0)
    for word in triehh_clients:
        word_freq[word] += 1
    word_freq = dict(word_freq)
    with open(ARTIFACTS_PATH + "clients_triehh.txt", "w") as fp:
        fp.write(json.dumps(triehh_clients))


def main():
    if not os.path.isdir(ARTIFACTS_PATH):
        os.mkdir(ARTIFACTS_PATH)
    clients = get_clients(FILENAME)

    clients_top_word = []
    top_word_counts = {}
    # get the top word for every client
    for client in clients:
        top_word = max(clients[client].items(), key=operator.itemgetter(1))[0]
        clients_top_word.append(top_word)
        if top_word not in top_word_counts:
            top_word_counts[top_word] = 1
        else:
            top_word_counts[top_word] += 1

    # compute frequencies of top words
    top_word_frequencies = {}
    sum_num = sum(top_word_counts.values())
    for word in top_word_counts:
        top_word_frequencies[word] = top_word_counts[word] * 1.0 / sum_num

    clients_top_word = np.array(clients_top_word)
    with open(ARTIFACTS_PATH + "word_frequencies.txt", "w") as fp:
        fp.write(json.dumps(top_word_frequencies))

    generate_triehh_clients(clients_top_word)

    print("client count:", len(clients_top_word))
    print("top word count:", len(top_word_counts))


if __name__ == "__main__":
    main()
