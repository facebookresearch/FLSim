import sys
import time

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import numpy as np
from accelerate import Accelerator
from sentence_transformers import models, SentenceTransformer
from similarity import Similarity
from transformers import RobertaForMaskedLM, RobertaTokenizer


class NN_Histogram:
    @staticmethod
    def dp_nn_histogram(
        private_samples, parent_set, attention_mask, mlm_probability, config
    ):
        """
        private_samples: list of private texts
        synthetic_samples: list of synthetic texts
        config: config dict
        """
        sigma = config["sigma"]
        H = config["H"]
        embed_dim = config["embed_dim"]
        accelerator = config["accelerator"]
        nearest_neighbors_print = config["nearest_neighbors_print"]
        index_flat = faiss.IndexFlatL2(embed_dim)
        accelerator.print("Making private embeddings", file=sys.stderr)
        t0 = time.time()
        private_embeddings = Similarity.sentence_embedding(
            private_samples, config["embed"], config["embedpool"], config
        )
        t1 = time.time()
        accelerator.print("Time for private embeddings", t1 - t0, file=sys.stderr)
        t0 = time.time()
        lookahead_embeddings = Similarity.lookahead_embedding(
            parent_set, attention_mask, mlm_probability, config
        )
        t1 = time.time()
        accelerator.print("Time for synthetic embeddings:", t1 - t0, file=sys.stderr)
        index_flat.add(lookahead_embeddings)
        # n_priv x 1
        D, I = index_flat.search(private_embeddings, 1)
        resulting_histogram, _ = np.histogram(
            I[:, 0],
            bins=[-0.5] + [x + 0.5 for x in range(lookahead_embeddings.shape[0])],
        )
        resulting_mean_distance = np.mean(D, axis=0).squeeze()
        accelerator.print(
            f"First {nearest_neighbors_print} private samples",
            private_samples[:nearest_neighbors_print],
            file=sys.stderr,
        )
        first_nearest_neighbors_idx = I[:nearest_neighbors_print, 0]
        resulting_histogram_noised = (
            resulting_histogram.astype(np.float32)
            + np.random.standard_normal(size=resulting_histogram.shape) * sigma
        )
        noised_histogram_thresh = np.maximum(resulting_histogram_noised - H, 0.0)

        return (
            noised_histogram_thresh,
            resulting_mean_distance,
            first_nearest_neighbors_idx,
        )


"""
This unit test tests whether the returned histogram in NN_histogram.dp_nn_histogram() satisfies sanity checks.
"""
if __name__ == "__main__":
    population = ["The capital of France is Paris.", "The meaning of life is 42."]
    private_samples = [
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The meaning of life is 42.",
    ]
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large", use_fast=True)
    model = RobertaForMaskedLM.from_pretrained(
        "roberta-large",
        # device_map=device_map,
        load_in_8bit=True,
    )
    accelerator = Accelerator()
    model = model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)
    bart_embedding_model = models.Transformer("facebook/bart-base")
    pooling_model = models.Pooling(bart_embedding_model.get_word_embedding_dimension())
    bart_model = SentenceTransformer(modules=[bart_embedding_model, pooling_model])
    bart_model_pool = bart_model.start_multi_process_pool()
    seq_len = 32

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "accelerator": accelerator,
        "batch_size": 1,
        "max_length": seq_len,
        "num_workers": 8,
        "embed_batch_size": 32,
        "embed": bart_model,
        "embedpool": bart_model_pool,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "embed_dim": bart_model.get_sentence_embedding_dimension(),
        "sigma": 0.0,
        "H": 0.0,
        "nearest_neighbors_print": 3,
        "lookahead": 2,
    }
    model = accelerator.prepare(model)

    population_inputs = tokenizer(
        population, return_tensors="pt", truncation=True, padding=True, max_length=20
    )
    curr_hist, mean_dist, nearest_idx = NN_Histogram.dp_nn_histogram(
        private_samples,
        population_inputs["input_ids"],
        population_inputs["attention_mask"],
        0.3,
        config,
    )
    accelerator.print(curr_hist)
    accelerator.print(mean_dist)
    assert np.linalg.norm(curr_hist - np.array([3.0, 1.0])) < 1e-10
    accelerator.print("Test passed!")
