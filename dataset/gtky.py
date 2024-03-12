# !wget https://raw.githubusercontent.com/jasonwu0731/GettingToKnowYou/master/data/ConvAI2/train_both_original_final.txt
# !wget https://raw.githubusercontent.com/jasonwu0731/GettingToKnowYou/master/data/ConvAI2/test_both_original_final.txt
# !wget https://raw.githubusercontent.com/jasonwu0731/GettingToKnowYou/master/data/ConvAI2/valid_both_original_final.txt

import ast
import json
from pathlib import Path
import os
import typing as tp
import numpy as np
import pandas as pd

from tqdm import tqdm


def _validate_filename(
    path: str,
    split: str,
    persona_type: str = "both",
    is_original: bool = True,
    is_preprocessed: bool = True,
):
    if split not in ["train", "valid", "test"]:
        raise ValueError(
            'Invalid split type. Appropriate splits: ["train", "valid", "test"]'
        )

    if persona_type not in ["both", "self", "their", "none"]:
        raise ValueError(
            'Persona type must be one of ["both", "self", "their", "none"]'
        )

    split_prefix = "valid" if split == "val" else split
    persona_prefix = "other" if persona_type == "their" else persona_type
    file_suffix = "original" if is_original else "revised"
    extension = ".json" if is_preprocessed else ".txt"

    filename = "".join(
        [split_prefix, "_", persona_prefix, "_", file_suffix, "_final", extension]
    )

    data_dir = Path(path) / filename

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Given file: {data_dir} doesn't exist.")

    return filename, data_dir


def _process_dialogue(
    dialogue_id: int, dialogue: tp.List[str], persona_type: str = "both"
) -> tp.Dict[str, tp.Any]:
    result = {"dialogue_id": dialogue_id, "triplets": [], "utterances": []}
    for line in dialogue:
        splitted = line.split("\t")
        utt = str(splitted[1])
        result["triplets"].append(
            [ast.literal_eval(triplet) for triplet in splitted[2:]]
        )
        result["utterances"].append(utt)
    return result


def process_data(
    path: str, split: str, persona_type: str = "both", is_original: bool = True
):
    """
    Preprocess dataset and convert it to json.
    path - path to the dataset
    split - "test" / "valid" / "train"
    persona_type - "none" / "self" / "their" / "both". Default: "both"
    is_original - True if original persona is needed, otherwise revised persona is used. Default: True
    """
    filename, data_dir = _validate_filename(
        path, split, persona_type, is_original, False
    )
    new_data_dir = data_dir.parent / "processed" / Path(split).with_suffix(".json")
    data = []
    print(f"Preprocessing {data_dir}")
    is_first_line = True
    with data_dir.open() as f:
        lines = f.readlines()
        slice = []
        dialogue_idx = 0
        for line in tqdm(lines):
            line = line.strip()
            first_line_token = line.split()[0]
            if not first_line_token.isdigit():
                continue
            row_num = int(first_line_token)
            if row_num == 1 and not is_first_line:
                data.append(_process_dialogue(dialogue_idx, slice, persona_type="none"))
                slice.clear()
                dialogue_idx += 1
            elif row_num == 1:
                is_first_line = False
            slice.append(line)
    with new_data_dir.open("w+") as f:
        print(f"Writing to {new_data_dir}")
        f.write(json.dumps(data, indent=2))


def load_gtky(path: str, split: str) -> tp.List[dict]:
    """
    Load preprocessed GTKY Dataset
    path - path to the dataset
    split - "test" / "valid" / "train"
    """
    if split not in ["train", "valid", "test"]:
        raise ValueError(
            'Invalid split type. Appropriate splits: ["train", "valid", "test"]'
        )
    data_dir = Path(path) / f"{split}.json"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Given file: {data_dir} doesn't exist.")

    data = None
    print(f"Loading {data_dir}")
    with data_dir.open() as f:
        data = json.load(f)
    return data


def to_pandas(data: tp.List[dict]):
    df_dict = {"dialogue_id": [], "utterance": [], "target": []}
    for dialogue in data:
        for i in range(len(dialogue["utterances"])):
            df_dict["utterance"].append(dialogue["utterances"][i])
            df_dict["dialogue_id"].append(dialogue["dialogue_id"])
            proc_triplets = []
            for triplet in dialogue["triplets"][i]:
                proc_triplets.append(
                    f"<subj> {triplet[0]} <rel> {triplet[1]} <obj> {triplet[2]}"
                )
            if len(proc_triplets) > 0:
                df_dict["target"].append("; ".join(proc_triplets))
            else:
                df_dict["target"].append("<none>")
    return pd.DataFrame(df_dict)
