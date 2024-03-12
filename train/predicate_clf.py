import json
from dataclasses import dataclass, field

from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    HfArgumentParser,
)
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from ..utils.predicate_clf import *


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../chekpoints/ranker_mdeberta_base")
    evaluation_strategy: str = field(default="steps")
    evaluation_steps: int = field(default=4490)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    save_steps: int = field(default=4490)
    per_device_eval_batch_size: int = field(default=32)
    per_device_train_batch_size: int = field(default=16)
    optim: str = field(default="adamw_torch")
    num_train_epochs: int = field(default=2)
    logging_steps: int = field(default=100)
    logging_strategy: str = field(default="steps")


def single_label_metrics(predictions, labels):
    probs = torch.Tensor(predictions).sigmoid()
    y_pred = (probs >= 0.5).type(torch.int32).squeeze()
    y_true = labels.squeeze()
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    metrics = {
        "recall": recall,
        "precision": precision,
        "f1": f1_micro_average,
        "accuracy": accuracy,
    }
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = single_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def main() -> None:
    set_seed(42)
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()

    MODEL_PATH = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    toker = AutoTokenizer.from_pretrained(MODEL_PATH)
    gtky_enc = get_data(toker)

    mbert_base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        problem_type="multi_label_classification",
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    trainer = Trainer(
        model=mbert_base,
        args=training_args,
        train_dataset=gtky_enc["train"],
        eval_dataset=gtky_enc["test"],
        tokenizer=toker,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=False)
    with open(f"{training_args.output_dir}/training_args.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
