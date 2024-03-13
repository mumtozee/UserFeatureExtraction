import torch
import numpy as np
import re
import json
import typing as tp
import itertools
import argparse

from transformers import (
    AutoModelForSequenceClassification,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    T5Tokenizer,
)

from .utils.predicate_clf import labels, rel2desc, desc2rel

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

print("Loading tokenizers ...")

device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RANKER_SEP_TOKEN = "[SEP]"
EGEN_SEP_TOKEN = "<sep>"
EXTRA_ID_0 = "<extra_id_0>"
EXTRA_ID_1 = "<extra_id_1>"
predicates = labels


ranker_toker = None
egen_toker = None
mt5_toker = None


def gen_triplet_e2e(model, utt_batch: tp.List[str]) -> tp.List[str]:
    enc = mt5_toker(
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
    decoded = mt5_toker.batch_decode(out)
    result = []
    for sentence in decoded:
        found_res = re.findall(r"<pad>\s*(.*)</s>", sentence)
        if len(found_res) == 0:
            found_res = re.findall(r"<pad>\s*(.*)", sentence)
        result.append(found_res[0].strip())
    return result


def predict_rels(
    model: tp.Callable,
    batch: tp.List[str],
    all_rels: tp.Iterable,
    thres: float = 0.87,
    return_scores: bool = False,
    apply_sigmoid: bool = True,
) -> tp.List[tp.List[str]]:
    all_combs = []
    result = []
    scores = []
    for inpt, rel in itertools.product(batch, all_rels):
        all_combs.append(f"{inpt} {RANKER_SEP_TOKEN} {rel2desc[rel]}.")
    encoding = ranker_toker(
        all_combs,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits.squeeze().cpu()
    if apply_sigmoid:
        all_scores = logits.sigmoid()
    else:
        all_scores = logits
    all_scores = all_scores.numpy().reshape((len(batch), -1))
    all_rels = np.array(all_rels)
    for curr_scores in all_scores:
        idx = curr_scores >= thres
        true_rels = all_rels[idx].tolist()
        true_scores = curr_scores[idx].tolist()
        result.append([rel2desc[rel] for rel in true_rels])
        scores.append(true_scores)
    if return_scores:
        return result, scores
    return result


def gen_entities(model: tp.Callable, batch: tp.List[str]) -> tp.List[str]:
    if len(batch) == 0:
        return []
    enc = egen_toker(
        batch,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(enc["input_ids"], max_new_tokens=64).cpu()
    decoded = egen_toker.batch_decode(out)
    result = []
    for sentence in decoded:
        found_res = re.findall(r"<pad>\s*(.*)</s>", sentence)
        if len(found_res) == 0:
            found_res = re.findall(r"<pad>\s*(.*)", sentence)
        result.append(found_res[0])
    return result


def get_triplets(
    ranker: tp.Callable,
    egen: tp.Callable,
    batch: tp.List[str],
    all_rels: tp.Iterable,
    thres: float = 0.87,
) -> tp.List[str]:
    rels = predict_rels(ranker, batch, all_rels, thres=thres)
    all_combs = []
    for i, rel_set in enumerate(rels):
        for rel in rel_set:
            all_combs.append(
                f"{batch[i]} {EGEN_SEP_TOKEN} <subj> {EXTRA_ID_0} <rel> {rel} <obj> {EXTRA_ID_1}"
            )
    entities = gen_entities(egen, all_combs)
    result = []
    idx = 0
    for i in range(len(rels)):
        triples = []
        for j in range(len(rels[i])):
            subj = re.findall(r"<extra_id_0> (.*) <extra_id_1>", entities[idx])
            if len(subj) == 0:
                subj = "<none>"
            else:
                subj = subj[0].strip()
            obj = re.findall(r"<extra_id_1> (.*)", entities[idx])
            if len(obj) == 0:
                obj = "<none>"
            else:
                obj = obj[0].strip()
            triples.append(f"<subj> {subj} <rel> {desc2rel[rels[i][j]]} <obj> {obj}")
            idx += 1
        result.append("; ".join(triples))
    return result


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_type", default=1, type=int)
    parser.add_argument("--thres", type=float)
    parser.add_argument(
        "--ranker-path",
        type=str,
        default="./chekpoints/ranker_mdeberta_base/checkpoint-7336",
    )
    parser.add_argument(
        "--egen-path",
        type=str,
        default="./chekpoints/egen_small_adamw_desc/checkpoint-3500",
    )
    parser.add_argument(
        "--genae-path", type=str, default="./checkpoints/mT5_small/checkpoint-2000"
    )
    return parser.parse_args()


def main():
    args = get_args()

    ranker_toker = AutoTokenizer.from_pretrained(args.ranker_path)
    egen_toker = T5Tokenizer.from_pretrained("google/mt5-small")
    egen_toker.add_special_tokens(
        {"additional_special_tokens": ["<subj>", "<rel>", "<obj>", "<sep>", "<blank>"]}
    )

    mt5_toker = T5Tokenizer.from_pretrained("google/mt5-base")
    mt5_toker.add_special_tokens(
        {"additional_special_tokens": ["<subj>", "<rel>", "<obj>", "<none>"]}
    )

    thres = args.thres
    mt5 = None
    egen = None
    ranker = None
    print("Loading models ...")
    if args.method_type == 1:
        mt5 = (
            MT5ForConditionalGeneration.from_pretrained(args.genae_path)
            .to(device_0)
            .eval()
        )
    else:
        ranker = (
            AutoModelForSequenceClassification.from_pretrained(args.ranker_path)
            .to(device_0)
            .eval()
        )
        egen = (
            MT5ForConditionalGeneration.from_pretrained(args.egen_path)
            .to(device_1)
            .eval()
        )

    while True:
        print("Enter an utterance or <exit> to close the app: ")
        try:
            inp = input().strip()
        except Exception as e:
            print("Error in the input, please try again.")
            continue
        if inp == "<exit>":
            break
        result = None
        if args.method_type == 1:
            result = gen_triplet_e2e(mt5, [inp])[0]
        else:
            result = get_triplets(ranker, egen, [inp], predicates, thres)[0]
        if len(result) == 0:
            result = "<none>"
        print("Output: ")
        print(result)


if __name__ == "__main__":
    main()
