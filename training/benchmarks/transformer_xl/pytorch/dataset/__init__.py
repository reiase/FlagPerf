# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from itertools import chain

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data import default_data_collator


class WorkerInitializer(object):
    _instance = None

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, idx):
        np.random.seed(seed=self.seed + idx)
        random.seed(self.seed + idx)

    @classmethod
    def default(cls, seed=0):
        if cls._instance is None:
            cls._instance = cls(seed)
        return cls._instance


def create_dataset(config, data_dir=None, config_name=None, model_dir=None):
    data_dir = data_dir or config.data_dir + "/data"
    config_name = config_name or config.dataset_config_name
    model_dir = model_dir or config.data_dir + "/model"
    raw_dataset = load_dataset(data_dir, config_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def tokenize(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="running tokenizer on dataset",
    )

    block_size = 1024

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    train_dataset = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        batch_size=config.train_batch_size,
        collate_fn=default_data_collator,
    )
    test_dataset = DataLoader(
        lm_datasets["test"],
        shuffle=True,
        batch_size=config.train_batch_size,
        collate_fn=default_data_collator,
    )
    return train_dataset, test_dataset


if __name__ == "__main__":
    create_dataset(
        None,
        "/Users/houjie02/.cache/huggingface/datasets/wikitext/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/",
        "wikitext-103-v1",
        "/Users/houjie02/.cache/huggingface/hub/models--transfo-xl-wt103/snapshots/40a186da79458c9f9de846edfaea79c412137f97/",
    )
