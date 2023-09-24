# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from accelerate import Accelerator
from sentence_transformers import models, SentenceTransformer
from transformers import RobertaForMaskedLM, RobertaTokenizer
from variation import Variation


class Similarity:
    @staticmethod
    def sentence_embedding(texts, embedding_model, pool, config):
        """
        texts: list of texts
        config: holds all the various things you need
        returns: sentence embeddings
        """
        sentence_embeddings = embedding_model.encode_multi_process(
            texts, pool, batch_size=config["embed_batch_size"]
        )
        return sentence_embeddings

    @staticmethod
    def lookahead_embedding(parent_set, attention_mask, mlm_probability, config):
        """
        texts: list of texts:
        config: holds the configurations
        returns: sentence_embeddings
        """
        tokenizer = config["tokenizer"]
        embeddings_list = []
        for _ in range(config["lookahead"]):
            curr_variation = Variation.produce_variation(
                {"input_ids": parent_set, "attention_mask": attention_mask},
                mlm_probability,
                config,
            )
            curr_variation_texts = tokenizer.batch_decode(
                curr_variation, ignore_special_tokens=True
            )
            curr_variation_embedding = Similarity.sentence_embedding(
                curr_variation_texts, config["embed"], config["embedpool"], config
            )[None, :, :]
            embeddings_list.append(curr_variation_embedding)
        embeddings_cat = np.concatenate(embeddings_list, axis=0)
        embeddings_mean = np.mean(embeddings_cat, axis=0)
        return embeddings_mean


if __name__ == "__main__":
    population = [
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The meaning of life is 42.",
        "The meaning of life is 42.",
        "The meaning of life is 42.",
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
        "batch_size": 256,
        "max_length": seq_len,
        "num_workers": 8,
        "embed_batch_size": 32,
        "embed": bart_model,
        "embedpool": bart_model_pool,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "embed_dim": bart_model.get_sentence_embedding_dimension(),
        "lookahead": 2,
    }
    population_tok = tokenizer(
        population,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )

    """
    This tests Similarity.sentence_embedding
    """
    sentence_embeds = Similarity.sentence_embedding(
        population, config["embed"], config["embedpool"], config
    )
    if accelerator.is_main_process:
        print(
            "Norm between same items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[1, :]),
        )
        print(
            "Norm between diff items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[4, :]),
        )
        assert np.linalg.norm(
            sentence_embeds[0, :] - sentence_embeds[1, :]
        ) < np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[4, :])
        print("Test 1 passed!")

    """
    This tests Similarity.lookahead_embedding
    """

    sentence_embeds = Similarity.lookahead_embedding(
        population_tok["input_ids"], population_tok["attention_mask"], 0.3, config
    )
    if accelerator.is_main_process:
        print(
            "Norm between same items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[1, :]),
        )
        print(
            "Norm between diff items",
            np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[4, :]),
        )
        assert np.linalg.norm(
            sentence_embeds[0, :] - sentence_embeds[1, :]
        ) < np.linalg.norm(sentence_embeds[0, :] - sentence_embeds[4, :])
        print("Test 2 passed!")

    bart_model.stop_multi_process_pool(bart_model_pool)
