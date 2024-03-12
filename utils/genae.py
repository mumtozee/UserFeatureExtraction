import re
import json
from collections import defaultdict
from tqdm import tqdm
import typing as tp

import pandas as pd
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from datasets import Dataset, load_dataset
import numpy as np
import torch

from ..dataset.gtky import process_data, load_gtky, to_pandas

RAW_DATA_PATH = "../data/GTKY_final"
DATA_PATH = "../data/GTKY_final/processed"


def get_data(toker: T5Tokenizer, preproc: bool = True) -> Dataset:
    if preproc:
        process_data(RAW_DATA_PATH, "train")
        process_data(RAW_DATA_PATH, "valid")
        process_data(RAW_DATA_PATH, "test")
        train = load_gtky(DATA_PATH, "train")
        test = load_gtky(DATA_PATH, "test")
        val = load_gtky(DATA_PATH, "valid")
        to_pandas(train).to_csv(f"{DATA_PATH}/train.tsv", sep="\t", index=False)
        to_pandas(test).to_csv(f"{DATA_PATH}/test.tsv", sep="\t", index=False)
        to_pandas(val).to_csv(f"{DATA_PATH}/valid.tsv", sep="\t", index=False)
    data_files = {
        "train": f"{DATA_PATH}/train.tsv",
        "val": f"{DATA_PATH}/valid.tsv",
        "test": f"{DATA_PATH}/test.tsv",
    }
    gtky = load_dataset("csv", data_files=data_files, sep="\t")

    def tokenize(batch):
        input_enc = toker.batch_encode_plus(
            batch["utterance"], padding="max_length", truncation=True, max_length=128
        )
        target_enc = toker.batch_encode_plus(
            batch["target"], padding="max_length", truncation=True, max_length=64
        )
        encodings = {
            "input_ids": input_enc["input_ids"],
            "attention_mask": input_enc["attention_mask"],
            "labels": target_enc["input_ids"],
            "decoder_attention_mask": target_enc["attention_mask"],
        }
        return encodings

    gtky_enc = gtky.map(
        tokenize, batched=True, num_proc=4, remove_columns=gtky["train"].column_names
    )

    columns = ["input_ids", "labels", "attention_mask", "decoder_attention_mask"]
    gtky_enc.set_format(type="torch", columns=columns)
    return gtky_enc


def gen_triplet(
    model: MT5ForConditionalGeneration, utt_batch: tp.List[str], toker: T5Tokenizer
) -> tp.List[str]:
    enc = toker(
        utt_batch,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    with torch.inference_mode():
        out = model.generate(
            enc["input_ids"].to(model.device),
            max_length=128,
        )
    decoded = toker.batch_decode(out)
    result = []
    for sentence in decoded:
        found_res = re.findall(r"<pad>\s*(.*)</s>", sentence)
        if len(found_res) == 0:
            found_res = re.findall(r"<pad>\s*(.*)", sentence)
        result.append(found_res[0].strip())
    return result


def get_batch(data: np.array, batch_size: int = 32) -> np.array:
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield batch_idx, data[batch_idx]


def make_preds(
    model: tp.Callable,
    data_df: pd.DataFrame,
    batch_size: int = 32,
) -> pd.DataFrame:
    model.eval()
    utterances = data_df["utterance"].to_numpy()
    result = data_df.copy()
    model_preds = np.empty(len(utterances), dtype=object)
    for idx, batch in tqdm(get_batch(utterances, batch_size=batch_size)):
        model_preds[idx] = gen_triplet(model, batch.tolist())
    result["model_preds"] = model_preds
    return result
