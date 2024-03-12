import typing as tp
from dataclasses import dataclass
import torch


@dataclass
class T2TDataCollator:
    def __call__(self, batch: tp.List) -> tp.Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["labels"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["decoder_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }


@dataclass
class NLLDataCollator:
    def __call__(self, batch: tp.List) -> tp.Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.concat([example["input_ids"] for example in batch])
        attention_mask = torch.concat([example["attention_mask"] for example in batch])
        labels = torch.zeros(size=(len(batch),), dtype=torch.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
