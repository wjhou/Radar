import logging
import pathlib
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
import os
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
import torch
from dataset import RadarDataset
from data_collator import DataCollatorForSupervisedDataset
import numpy as np

import evaluate
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


def write_results(trainer, infos):
    refs, hyps, results = infos
    output_dir = trainer.args.output_dir
    global_step = trainer.state.global_step
    output_path = f"{output_dir}/eval_results_{global_step}.txt"
    count = 0
    while os.path.exists(output_path):
        output_path = f"{output_dir}/eval_results_{global_step}.{count}.txt"
        count += 1
    with open(output_path, "w") as f:
        for i in range(len(refs)):
            idx, ref = refs[i].split("@")
            line = {"id": idx, "ref": refs[i], "hyp": hyps[i]}
            line = json.dumps(line)
            f.write(line + "\n")
        f.write("============================\n")
        for score in results:
            f.write("[%s]\t\t%0.4f\n" % (score, results[score]))
        f.write("============================\n")


def create_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    image_processor,
    model,
    sft_checkpoint,
):
    train_dataset = RadarDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        split="train",
        image_processor=image_processor,
        sft_checkpoint=sft_checkpoint,
    )
    eval_dataset = RadarDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        split="valid",
        image_processor=image_processor,
        sft_checkpoint=sft_checkpoint,
    )
    test_dataset = RadarDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        split="test",
        image_processor=image_processor,
        sft_checkpoint=sft_checkpoint,
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

    # metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

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
    print(model)

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

    for param in model.vlm.vision_tokenizer.parameters():
        param.requires_grad = True

    if stage != 0:
        sft_checkpoint = model_args.sft_model_name_or_path
        state_dict_path = os.path.join(sft_checkpoint, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from Stage 0: {state_dict_path}")
        for param in model.vlm.vision_tokenizer.parameters():
            param.requires_grad = False
        for param in model.vlm.vision_encoder.parameters():
            param.requires_grad = False

    print(model)

    print(model.vlm.num_trainable_params_per_module)
    data_module = create_dataset(
        tokenizer, data_args, image_processor, model, model_args.sft_model_name_or_path
    )
    test_dataset = data_module["test_dataset"]
    data_module.pop("test_dataset")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # decode preds and labels
        preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
        labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions = [
            p.strip().split("Overall Findings:")[-1].strip() for p in predictions
        ]

        references = [r.split("@")[-1].strip() for r in references]
        predictions = [clean_text(p.lower()) for p in predictions]
        references = [clean_text(r.lower()) for r in references]

        references = [[r] for r in references]
        bleu_1 = bleu_metric.compute(
            predictions=predictions, references=references, max_order=1
        )["bleu"]
        bleu_4 = bleu_metric.compute(
            predictions=predictions, references=references, max_order=4
        )["bleu"]
        rouge_l = rouge_metric.compute(predictions=predictions, references=references)[
            "rougeL"
        ]
        meteor = meteor_metric.compute(predictions=predictions, references=references)[
            "meteor"
        ]
        results = {
            "BLEU_1": bleu_1,
            "BLEU_4": bleu_4,
            "ROUGE_L": rouge_l,
            "METEOR": meteor,
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
        1: {"max_new_tokens": 360, "length_penalty": 2.0, "num_beams": 1},
    }

    stage_generation_config = stage_generation_config.get(
        stage, stage_generation_config[0]
    )

    print(f"===== Stage {stage} =====")
    for key, val in stage_generation_config.items():
        print(f"==== {key}: {val} ====")
    print(f"===== Stage {stage} =====")
    import types
    from trainer import prediction_step, _save, _load_best_model

    trainer.stage_generation_config = stage_generation_config

    trainer.prediction_step = types.MethodType(prediction_step, trainer)
    trainer._save = types.MethodType(_save, trainer)
    trainer._load_best_model = types.MethodType(_load_best_model, trainer)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    trainer.stage_generation_config["num_beams"] = 5
    trainer.predict(test_dataset)


if __name__ == "__main__":
    transformers.set_seed(42)
    main()
