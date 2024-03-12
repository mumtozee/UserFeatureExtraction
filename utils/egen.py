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

from ..evaluation.metrics import compute_metrics

RAW_DATA_PATH = "../data/GTKY_final/processed"
DATA_PATH = "../data/GTKY_final/for_egen_2"
TOKER_PATH = "google/mt5-small"
MODEL_PATH = "google/mt5-small"

EXTRA_ID_0 = "<extra_id_0>"
EXTRA_ID_1 = "<extra_id_1>"
SEP_TOKEN = "<sep>"

with open("../data/GTKY_final/mappers.json", "r") as f:
    rel2desc, desc2rel = json.load(f)


def ccat_predicate(data_df: pd.DataFrame) -> pd.DataFrame:
    new_data = defaultdict(list)
    for _, row in data_df.iterrows():
        if row["target"] == "<none>":
            continue
        trg_splitted = row["target"].split(";")
        entity_list = defaultdict(set)
        for triplet in trg_splitted:
            rel = re.findall(r"<rel> (.*) <obj>", triplet)[0]
            rel = rel2desc[rel]
            subj = re.findall(r"<subj> (.*) <rel>", triplet)[0]
            obj = re.findall(r"<obj> (.*)", triplet)[0]
            entity_list[rel].add((subj, obj))
        for rel in entity_list.keys():
            subj_list = [pair[0] for pair in entity_list[rel]]
            obj_list = [pair[1] for pair in entity_list[rel]]
            subjects = "; ".join(subj_list)
            objects = "; ".join(obj_list)
            new_data["dialogue_id"].append(row["dialogue_id"])
            new_data["utterance"].append(
                f"{row['utterance']} {SEP_TOKEN} <subj> {EXTRA_ID_0} <rel> {rel} <obj> {EXTRA_ID_1}"
            )
            new_data["target"].append(f"{EXTRA_ID_0} {subjects} {EXTRA_ID_1} {objects}")
    return pd.DataFrame(new_data)


def get_data(toker: T5Tokenizer, preproc: bool = True) -> Dataset:
    if preproc:
        train_df = pd.read_csv(f"{RAW_DATA_PATH}/train.tsv", sep="\t")
        test_df = pd.read_csv(f"{RAW_DATA_PATH}/test.tsv", sep="\t")
        val_df = pd.read_csv(f"{RAW_DATA_PATH}/valid.tsv", sep="\t")
        ccat_predicate(val_df).to_csv(f"{DATA_PATH}/val.tsv", sep="\t", index=False)
        ccat_predicate(test_df).to_csv(f"{DATA_PATH}/test.tsv", sep="\t", index=False)
        ccat_predicate(train_df).to_csv(f"{DATA_PATH}/train.tsv", sep="\t", index=False)
    data_files = {
        "train": f"{DATA_PATH}/train.tsv",
        "val": f"{DATA_PATH}/val.tsv",
        "test": f"{DATA_PATH}/test.tsv",
    }
    gtky = load_dataset("csv", data_files=data_files, sep="\t")

    def tokenize(batch):
        input_enc = toker(
            batch["utterance"], padding="max_length", truncation=True, max_length=128
        )
        target_enc = toker(
            text_target=batch["target"],
            padding="max_length",
            truncation=True,
            max_length=64,
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


def get_batch(data: np.array, batch_size: int = 32) -> np.array:
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield batch_idx, data[batch_idx]


def gen_entities(
    model: MT5ForConditionalGeneration, batch: np.array, toker: T5Tokenizer
) -> np.array:
    enc = toker(
        batch.tolist(),
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = (
            model.generate(enc["input_ids"].to(model.device), max_new_tokens=64)
            .cpu()
            .numpy()
        )
    decoded = toker.batch_decode(out)
    result = []
    for sentence in decoded:
        found_res = re.findall(r"<pad>(.*)</s>", sentence)
        if len(found_res) == 0:
            found_res = re.findall(r"<pad>(.*)", sentence)
        if len(found_res) == 0:
            print(f"CORRUPT SENTENCE!: {sentence}")
        result.append(found_res[0].strip())
    return result


def make_preds(
    model: MT5ForConditionalGeneration, data_df: pd.DataFrame, batch_size: int = 32
) -> pd.DataFrame:
    utterances = data_df["utterance"].to_numpy()
    result = data_df.copy()
    model_preds = np.empty(len(utterances), dtype=object)
    for idx, batch in tqdm(get_batch(utterances, batch_size=batch_size)):
        model_preds[idx] = gen_entities(model, batch)
    result["model_preds"] = model_preds
    return result

def evaluate(
    model: MT5ForConditionalGeneration,
    test_data_path: str,
    toker: T5Tokenizer,
    device: torch.cuda.device = "cuda:0",
) -> tp.Tuple[float]:
    model.eval()
    model.to(device)
    test_df = pd.read_csv(test_data_path, sep="\t")
    preds = make_preds(model, test_df, batch_size=128)
    return compute_metrics(preds, toker)
