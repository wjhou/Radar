import logging
import transformers
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    Seq2SeqTrainer,
)
from arguments import ModelArguments, DataArguments, TrainingArguments
from blip3.modeling_blip3 import (
    XGenMMModelForConditionalGeneration as Blip3ModelForConditionalGeneration,
)
import copy
import os
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
import torch
from dataset import SelfAskDataset
from data_collator import DataCollatorForSupervisedDataset
import numpy as np
import json
import re


def clean_text(text: str) -> str:
    """
    from: https://github.com/X-iZhang/Libra/blob/main/libra/eval/radiology_report.py
    Perform basic cleanup of text by removing newlines, dashes, and some special patterns.
    """
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\(___, __, __\)", "", text)
    text = re.sub(r"---, ---, ---", "", text)
    text = re.sub(r"\(__, __, ___\)", "", text)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"[^\w\s.,:;()\-]", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def write_results(trainer, tokenizer, results, prefix="valid"):
    preds, labels = results.predictions, results.label_ids
    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = [p.split("<|end|>")[0].strip() for p in predictions]
    references = [r.split("@") for r in references]

    outputs = {}

    for pred, label in zip(predictions, references):
        idx, label = label
        outputs[idx] = {"hyp": pred, "ref": label}

    output_dir = trainer.args.output_dir
    output_path = f"{output_dir}/{prefix}_results.json"
    json.dump(outputs, open(output_path, "w"), indent=4)


def create_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    image_processor,
    model,
):
    train_dataset = SelfAskDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        split="train",
        image_processor=image_processor,
    )
    eval_dataset = SelfAskDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        split="valid",
        image_processor=image_processor,
    )
    test_dataset = SelfAskDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        split="test",
        image_processor=image_processor,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, model=model)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
    )


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        legacy=False,
    )
    model = Blip3ModelForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    logging.info(f"Training Arguments:")
    print(training_args)
    logging.info(f"Model")

    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer = model.update_special_tokens(tokenizer)
    tokenizer.eos_token = "<|end|>"

    stage = data_args.stage
    # Load LoRA for Phi-3
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
    )
    model.vlm.lang_model.enable_input_require_grads()
    lang_model = get_peft_model(
        model.vlm.lang_model, peft_config, adapter_name="lang_model"
    )
    model.vlm.lang_model = lang_model
    model.vlm.lang_model.print_trainable_parameters()

    peft_config = LoraConfig(
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
    )
    vision_encoder = get_peft_model(
        model.vlm.vision_encoder, peft_config, adapter_name="vision_encoder"
    )
    model.vlm.vision_encoder = vision_encoder
    model.vlm.vision_encoder.print_trainable_parameters()

    sft_checkpoint = model_args.sft_model_name_or_path
    state_dict_path = os.path.join(sft_checkpoint, "pytorch_model.bin")
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model from Stage {stage - 1}: {state_dict_path}")

    print(model)
    print(model.vlm.num_trainable_params_per_module)
    data_module = create_dataset(tokenizer, data_args, image_processor, model)
    train_dataset = data_module["train_dataset"]
    test_dataset = data_module["test_dataset"]
    data_module.pop("test_dataset")
    eval_dataset = data_module["eval_dataset"]
    data_module.pop("eval_dataset")

    def compute_metrics(eval_preds):
        results = {
            "BLEU_1": 0,
            "BLEU_4": 0,
            "ROUGE_L": 0,
            "METEOR": 0,
        }
        return results

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        **data_module,
    )

    model.vlm.global_step = trainer.state.global_step

    stage_generation_config = {
        0: {"max_new_tokens": 256, "length_penalty": 2.0, "num_beams": 1},  # base
    }

    stage_generation_config = stage_generation_config.get(
        stage, stage_generation_config[0]
    )

    print(f"===== Stage {stage} =====")
    for key, val in stage_generation_config.items():
        print(f"==== {key}: {val} ====")
    print(f"===== Stage {stage} =====")

    stage_generation_config["num_beams"] = 5
    trainer.stage_generation_config = stage_generation_config

    import types
    from trainer_fn import train
    from trainer import prediction_step

    trainer.prediction_step = types.MethodType(prediction_step, trainer)
    trainer.train = types.MethodType(train, trainer)
    # prepare model
    trainer.train()

    trainer.args.per_device_eval_batch_size //= stage_generation_config["num_beams"]
    trainer.stage_generation_config = stage_generation_config
    valid_results = trainer.predict(eval_dataset)
    if trainer.accelerator.is_main_process:
        write_results(trainer, tokenizer, valid_results, "valid")
    test_results = trainer.predict(test_dataset)
    if trainer.accelerator.is_main_process:
        write_results(trainer, tokenizer, test_results, "test")

    train_stage_generation_config = copy.deepcopy(stage_generation_config)
    train_stage_generation_config["num_beams"] = 1
    trainer.stage_generation_config = train_stage_generation_config
    train_dataset.split = "test"
    train_results = trainer.predict(train_dataset)
    if trainer.accelerator.is_main_process:
        write_results(trainer, tokenizer, train_results, "train")


if __name__ == "__main__":
    transformers.set_seed(42)
    main()
