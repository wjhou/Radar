import transformers
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import torch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model: transformers.PreTrainedModel

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        pixel_values, input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("pixel_values", "input_ids", "labels")
        )
        padding_side = (
            "left"
            if "padding_side" not in instances[0]
            else instances[0]["padding_side"]
        )
        pixel_values = torch.stack(pixel_values, dim=0)
        input_ids, attention_mask = self.padding(
            input_ids, self.tokenizer.pad_token_id, padding_side
        )
        labels, _ = self.padding(labels, -100, padding_side)
        batch = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        prior_pixel_values = (
            [instance["prior_pixel_values"] for instance in instances]
            if "prior_pixel_values" in instances[0]
            else None
        )
        has_prior = (
            [instance["has_prior"] for instance in instances]
            if prior_pixel_values is not None
            else None
        )
        if (
            prior_pixel_values is not None
            and has_prior is not None
            and sum(has_prior) > 0
        ):
            has_prior = torch.tensor(has_prior)
            batch["has_prior"] = has_prior
            prior_pixel_values = torch.stack(prior_pixel_values, dim=0)
            batch["prior_pixel_values"] = prior_pixel_values

        return batch

    def padding(self, inputs, padding_value, padding_side, max_length=None):
        attention_masks = []
        if max_length is None:
            max_length = max(len(x) for x in inputs)
        padded_inputs = []
        for x in inputs:
            padding = torch.full(
                (max_length - len(x),), padding_value, dtype=torch.long
            )
            if padding_side == "left":
                padded_input = torch.cat([padding, x])
                attention_mask = torch.cat(
                    (torch.zeros_like(padding), torch.ones_like(x))
                )
            else:
                padded_input = torch.cat([x, padding])
                attention_mask = torch.cat(
                    (torch.ones_like(x), torch.zeros_like(padding))
                )
            padded_inputs.append(padded_input)
            attention_masks.append(attention_mask)
        input_ids = torch.stack(padded_inputs, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        return input_ids, attention_masks
