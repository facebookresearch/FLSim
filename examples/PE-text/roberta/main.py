# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import sys
import time

import numpy as np
from accelerate import Accelerator
from datasets import concatenate_datasets, load_from_disk
from nn_histogram import NN_Histogram
from sentence_transformers import SentenceTransformer
from transformers import RobertaForMaskedLM, RobertaTokenizer
from variation import Variation


# pyre-ignore[C901]
def main():
    accelerator = Accelerator()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large", use_fast=True)
    model = RobertaForMaskedLM.from_pretrained("roberta-large", load_in_8bit=True)

    mpnet_model = SentenceTransformer("all-mpnet-base-v2")
    mpnet_model_pool = mpnet_model.start_multi_process_pool()

    datasetname = "twitter"
    datadir = ""
    outputdir = ""
    seq_len = 64
    with open(f"{datadir}{datasetname}_train.json") as file:
        private_samples = json.load(file)
    accelerator.print(
        "Num private train samples", len(private_samples), file=sys.stderr
    )
    accelerator.print("Private samples", private_samples[:5], file=sys.stderr)
    if datasetname == "reddit":
        accelerator.print("loading non-reddit samples", file=sys.stderr)
        # non-reddit samples in c4
        all_data_file = load_from_disk(f"{datadir}/noreddit-split-c4")["train"]
        load_list = []
        for i, text in enumerate(all_data_file):
            if i > 400000:
                break
            load_list.append(text["chunks"])
        load_list = [x for x in load_list if len(x.split(" ")) > 5]
    else:
        # reddit samples in c4
        all_data_raw = load_from_disk(f"{datadir}reddit-split-c4")

        def chunk_examples(examples):
            chunks = []
            for example in examples["chunks"]:
                words = example.split(" ")
                chunks.extend(
                    [" ".join(words[i : i + 100]) for i in range(0, len(words), 100)]
                )
            return {"chunks": chunks}

        all_data_chunked = all_data_raw.map(
            chunk_examples,
            batched=True,
            remove_columns=all_data_raw["train"].column_names,
            num_proc=90,
        )
        all_data = concatenate_datasets(
            [all_data_chunked["train"], all_data_chunked["validation"]]
        )
        load_list = []
        for line in all_data:
            load_list.append(line["chunks"])
        load_list = [x for x in load_list if len(x.split(" ")) > 5]

    if accelerator.is_main_process:
        accelerator.print("Amount of reddit data", len(load_list), file=sys.stderr)

    init_pop = load_list
    model = model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)
    begin_mask = 0.6
    end_mask = 0.3
    config = {
        "model": model,
        "tokenizer": tokenizer,
        "accelerator": accelerator,
        "batch_size": 256,
        "max_length": seq_len,
        "num_workers": 8,
        "embed_batch_size": 32,
        "embed": mpnet_model,
        "embedpool": mpnet_model_pool,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "nearest_neighbors_print": 3,
        "sigma": 1.541 * np.sqrt(2),
        "H": 4.0,
        "embed_dim": mpnet_model.get_sentence_embedding_dimension(),
        "lookahead": 16,
        "T": 11,
        "multiplier": 1,
    }
    config["nsyn"] = config["batch_size"] * 8 * config["multiplier"]
    output_dir = (
        f"{outputdir}/roberta-generated-{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}-{8}/".format(
            config["top_p"],
            config["temperature"],
            round(config["sigma"], 3),
            config["H"],
            begin_mask,
            end_mask,
            config["lookahead"],
            config["nsyn"],
            datasetname,
        )
    )

    schedule = np.linspace(begin_mask, end_mask, num=config["T"])
    accelerator.print(output_dir, file=sys.stderr)
    if accelerator.is_main_process:
        accelerator.print(init_pop[0], file=sys.stderr)
        accelerator.print("Schedule", schedule, file=sys.stderr)

    parent_texts = random.choices(init_pop, k=config["nsyn"])
    parent_texts = sorted(parent_texts, key=lambda x: len(x))
    accelerator.print("Num parent texts", len(parent_texts), file=sys.stderr)
    parent_set = tokenizer(
        parent_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )
    num_steps_current = 0
    for t in range(config["T"]):
        t0 = time.time()

        histogram, meandist, nearest_idx = NN_Histogram.dp_nn_histogram(
            private_samples,
            parent_set["input_ids"],
            parent_set["attention_mask"],
            schedule[t],
            config,
        )
        accelerator.print(
            "Nearest generated samples",
            [
                tokenizer.batch_decode(
                    [parent_set["input_ids"][idx, :]], skip_special_tokens=True
                )
                for idx in nearest_idx
            ],
        )
        accelerator.print("Mean dist from nearest neighbor", meandist, file=sys.stderr)

        accelerator.print("Histogram sum", np.sum(histogram), file=sys.stderr)
        t1 = time.time()
        if accelerator.is_main_process:
            accelerator.print("Histogram time:", t1 - t0, file=sys.stderr)
            accelerator.print("Producing surviving parents...", file=sys.stderr)
        t0 = time.time()

        indices = np.random.choice(
            config["nsyn"], config["nsyn"], p=histogram / np.sum(histogram)
        )
        indices = np.sort(indices)
        surviving_parents_ids = parent_set["input_ids"][indices, :]
        surviving_parents_mask = parent_set["attention_mask"][indices, :]
        t1 = time.time()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Choosing survivors time:", t1 - t0, file=sys.stderr)
            accelerator.print("Producing variations...", file=sys.stderr)
        t0 = time.time()
        new_variations = Variation.produce_variation(
            {
                "input_ids": surviving_parents_ids,
                "attention_mask": surviving_parents_mask,
            },
            schedule[t],
            config,
        )

        t1 = time.time()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Producing variations time", t1 - t0, file=sys.stderr)
            accelerator.print("Checking similarity...", file=sys.stderr)

        generated_samples = tokenizer.batch_decode(
            new_variations, skip_special_tokens=True
        )
        surviving_samples = tokenizer.batch_decode(
            surviving_parents_ids, skip_special_tokens=True
        )

        parent_set["input_ids"] = new_variations
        parent_set["attention_mask"] = surviving_parents_mask
        # if len(l2_list) > 1:
        #     if l2_dist <= l2_list[-2]:
        #         accelerator.print('Improved this iteration!', file=sys.stderr)
        #         parent_set = new_variations
        #     else:
        #         accelerator.print('Got worse this iteration, keeping previous.', file=sys.stderr)
        #         l2_list[-1] = l2_list[-2]
        # else:
        #     parent_set = new_variations

        num_steps_current += 1
        if accelerator.is_main_process:
            text_list = generated_samples
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(
                os.path.join(output_dir, f"generated_text_it{t}.json"),
                "w+",
                encoding="utf8",
            ) as json_file:
                json.dump(text_list, json_file, ensure_ascii=False)
            with open(
                os.path.join(output_dir, f"surviving_text_it{t}.json"),
                "w+",
                encoding="utf8",
            ) as json_file:
                json.dump(list(set(surviving_samples)), json_file, ensure_ascii=False)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
