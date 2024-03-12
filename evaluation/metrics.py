import typing as tp

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

def compute_metrics(data_df: pd.DataFrame) -> tp.Dict[str, float]:
    num_true_triplets = 0
    num_gen_triplets = 0
    num_all_triplets = 0
    true_triples = 0
    eps = 1e-5

    for _, row in tqdm(data_df.iterrows()):
        gold = row["target"]
        pred = row["model_preds"]
        if len(pred) == 0:
            pred = "<none>"
        gold_list = gold.split("; ")
        pred_list = pred.split("; ")
        gold_set = set(gold_list)
        pred_set = set(pred_list)
        true_triples += gold_set == pred_set
        # if gold != "<none>":
        num_true_triplets += len(gold_set.intersection(pred_set))
        num_all_triplets += len(gold_set)
        # if pred != "<none>":
        num_gen_triplets += len(pred_set)
    precision = num_true_triplets / (num_gen_triplets + eps)
    recall = num_true_triplets / (eps + num_all_triplets)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": true_triples / data_df.shape[0],
    }


def batch_tokenize(batch: tp.List[str]) -> tp.List[tp.List[str]]:
    return [word_tokenize(item) for item in batch]


def compute_bleu(data_df: pd.DataFrame) -> float:
    # raise NotImplementedError("BLEU-1 score is not implemented yet.")
    overall_score = 0.0
    num_generated_triples = 0.0
    for _, row in tqdm(data_df.iterrows()):
        gold = row["target"]
        pred = row["model_preds"]
        if len(pred) == 0:
            pred = "<none>"
        gold_list = list(set(gold.split("; <subj>")))
        pred_list = list(set(pred.split("; <subj>")))
        gold_list_tokenized = batch_tokenize(gold_list)
        pred_list_tokenized = batch_tokenize(pred_list)
        for pred_tokens in pred_list_tokenized:
            overall_score += sentence_bleu(
                references=gold_list_tokenized,
                hypothesis=pred_tokens,
                weights=(1, 0, 0, 0),
            )
            num_generated_triples += 1
    return overall_score / num_generated_triples