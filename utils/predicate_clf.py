import ast
import gc
import random
import re
import json
import pickle
from pathlib import Path
from collections import defaultdict
from time import time
from dataclasses import dataclass
from tqdm.notebook import tqdm
import typing as tp

import numpy as np
import numpy.typing as ntp
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, Dataset

CHECKPOINTS_PATH = "../checkpoints/ranker_mdeberta_base"

GLOBAL_RANDOM_SEED = 42
RAW_DATA_PATH = "../data/GTKY_final/for_pred_cls"
DATA_PATH = "../data/GTKY_final/for_pred_ranker"
SEP_TOKEN = "[SEP]"
K = 4

with open("../data/GTKY_final/mappers.json", "r") as f:
    rel2desc, desc2rel = json.load(f)

with open("../data/GTKY_final/predicates.json", "r") as f:
    labels = json.load(f)


def negative_sample(
    row: pd.Series, k: int = K, seed: int = GLOBAL_RANDOM_SEED, skip_p: float = 0.0
) -> tp.Tuple:
    new_row = defaultdict(list)
    rd = np.random.RandomState(seed=seed)
    all_labels = row.drop(labels=["dialogue_id", "utterance"])
    pos_labels = all_labels.index[all_labels > 0].to_numpy()
    neg_labels = all_labels.index[all_labels == 0].to_numpy()
    if len(pos_labels) == 0:
        to_skip = rd.binomial(n=1, p=skip_p)
        if to_skip == 1:
            return {}, seed
        neg_labels = rd.choice(a=neg_labels, size=(k // 2), replace=False)
    else:
        neg_labels = rd.choice(a=neg_labels, size=k, replace=False)
    for lbl in pos_labels:
        new_row["dialogue_id"].append(row["dialogue_id"])
        new_row["utterance"].append(row["utterance"])
        new_row["relation"].append(f"{rel2desc[lbl]}.")
        new_row["label"].append(1)
    for lbl in neg_labels:
        new_row["dialogue_id"].append(row["dialogue_id"])
        new_row["utterance"].append(row["utterance"])
        new_row["relation"].append(f"{rel2desc[lbl]}.")
        new_row["label"].append(0)
    return new_row, seed


def create_pairs(
    data_df: pd.DataFrame, k: int = K, skip_p: float = 0.0
) -> pd.DataFrame:
    new_data = defaultdict(list)
    seed = 0
    for _, row in data_df.iterrows():
        new_row, seed = negative_sample(row, k, seed=seed + 1, skip_p=skip_p)
        for key in new_row:
            new_data[key].extend(new_row[key])
    return pd.DataFrame(new_data)


def get_data(toker: AutoTokenizer, preproc=True) -> Dataset:
    if preproc:
        train_df = pd.read_csv(f"{RAW_DATA_PATH}/train.tsv", sep="\t")
        val_df = pd.read_csv(f"{RAW_DATA_PATH}/val.tsv", sep="\t")
        test_df = pd.read_csv(f"{RAW_DATA_PATH}/test.tsv", sep="\t")
        labels = list(set(train_df.columns) - set(["dialogue_id", "utterance"]))

        cols_to_explode = ["dialogue_id", "pair", "label"]

        train_df.apply(negative_sample, args=([5]), axis="columns").explode(
            column=cols_to_explode
        ).to_csv(f"{DATA_PATH}/train.tsv", sep="\t", index=False)

        test_df.apply(negative_sample, args=([5]), axis="columns").explode(
            column=cols_to_explode
        ).to_csv(f"{DATA_PATH}/test.tsv", sep="\t", index=False)

        val_df.apply(negative_sample, args=([5]), axis="columns").explode(
            column=cols_to_explode
        ).to_csv(f"{DATA_PATH}/val.tsv", sep="\t", index=False)

        create_pairs(train_df).to_csv(f"{DATA_PATH}/train.tsv", sep="\t", index=False)
        create_pairs(test_df).to_csv(f"{DATA_PATH}/test.tsv", sep="\t", index=False)
        create_pairs(val_df).to_csv(f"{DATA_PATH}/val.tsv", sep="\t", index=False)
    data_files = {
        "train": f"{DATA_PATH}/train.tsv",
        "val": f"{DATA_PATH}/val.tsv",
        "test": f"{DATA_PATH}/test.tsv",
    }
    gtky = load_dataset("csv", data_files=data_files, sep="\t")

    def tokenize(batch):
        utt_rel = zip(batch["utterance"], batch["relation"])
        encoded = toker.batch_encode_plus(
            [f"{utt} {SEP_TOKEN} {lbl}" for utt, lbl in utt_rel],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        encoded["labels"] = [[float(lbl)] for lbl in batch["label"]]
        return encoded

    gtky_enc = gtky.map(
        tokenize, batched=True, num_proc=4, remove_columns=gtky["train"].column_names
    )
    gtky_enc.set_format("torch")
    return gtky_enc


def predict_probas(
    model: tp.Callable,
    utt_batch: tp.List[str],
    rel_batch: tp.List[str],
    toker: AutoTokenizer,
) -> np.array:
    encoding = toker.batch_encode_plus(
        list(zip(utt_batch, rel_batch)),
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits.squeeze().cpu()
    probs = logits.sigmoid().numpy()
    return probs


def make_preds(
    model: tp.Callable, data_df: pd.DataFrame, batch_size: int = 32
) -> pd.DataFrame:
    desc_rels = list(map(lambda x: f"{rel2desc[x]}.", labels))
    result_df = data_df.copy()
    for i in tqdm(range(0, data_df.shape[0], batch_size)):
        end = min(i + batch_size, data_df.shape[0])
        batch_df = data_df.iloc[i:end]
        utt_batch = np.repeat(batch_df["utterance"].to_numpy(), len(labels)).tolist()
        rel_batch = np.tile(desc_rels, batch_size).tolist()
        scores = predict_probas(model, utt_batch, rel_batch)

        result_df.loc[i : end - 1, labels] = scores.reshape((end - i, len(labels)))
    return result_df


def eval_dataset(
    gold_df: pd.DataFrame,
    pred_scores: pd.DataFrame,
    average: str = "macro",
    threshold: tp.Union[float, ntp.ArrayLike] = 0.5,
) -> tp.Dict[str, float]:
    probs = pred_scores[labels].to_numpy()
    y_true = gold_df[labels].to_numpy()
    y_pred = (probs >= np.expand_dims(np.array(threshold), axis=0)).astype(np.float32)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics = {"f1": f1_micro_average, "accuracy": accuracy}
    return metrics

# a more correct way to tune threshold: tune one threshold for all predicates
def tune_threshold_v2(
    preds_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    metrics: str = "accuracy",
    grid_size: int = 200,
) -> tp.Tuple[float]:
    grid = np.linspace(0., 1., grid_size)
    max_metric = 0.0
    best_thres = 0.0
    for val in grid:
        curr_metrics = eval_dataset(gold_df, preds_df, threshold=val, average="micro")
        curr_metric = curr_metrics[metrics]
        if curr_metric > max_metric:
            best_thres = val
            max_metric = curr_metric
    return best_thres, max_metric
