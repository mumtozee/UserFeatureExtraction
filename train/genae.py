from dataclasses import dataclass, field

from transformers import (
    set_seed,
    MT5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser,
)

from ..dataset.data_collator import T2TDataCollator
from utils.egen import *


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(default="../chekpoints/mT5_small")
    evaluation_strategy: str = field(default="epoch")
    learning_rate: float = field(default=3e-4)
    weight_decay: float = field(default=0.01)
    save_steps: int = field(default=500)
    per_device_eval_batch_size: int = field(default=32)
    per_device_train_batch_size: int = field(default=64)
    optim: str = field(default="adamw_torch")
    num_train_epochs: int = field(default=3)
    logging_steps: int = field(default=100)
    logging_strategy: str = field(default="steps")


def main() -> None:
    set_seed(42)
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()

    toker = T5Tokenizer.from_pretrained(TOKER_PATH)
    toker.add_special_tokens(
        {"additional_special_tokens": ["<subj>", "<rel>", "<obj>", "<sep>", "<blank>"]}
    )
    gtky_enc = get_data(toker)

    mt5 = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    mt5.resize_token_embeddings(toker.vocab_size + 4)

    trainer = Seq2SeqTrainer(
        model=mt5,
        args=training_args,
        train_dataset=gtky_enc["train"],
        eval_dataset=gtky_enc["test"],
        data_collator=T2TDataCollator(),
    )
    trainer.train(resume_from_checkpoint=False)
    with open(f"{training_args.output_dir}/training_args.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
