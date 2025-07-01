import torch
import torch.nn as nn
from typing import Dict, Union, Optional, List, Any, Tuple
from transformers.trainer_seq2seq import *
from transformers.trainer import *
from transformers import PreTrainedModel

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


def prediction_step(
    self,
    model: PreTrainedModel,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    prediction_loss_only: bool,
    ignore_keys: Optional[List[str]] = None,
    **gen_kwargs,
) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

    self.gather_function = self.accelerator.gather_for_metrics
    if not self.args.predict_with_generate or prediction_loss_only:
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    has_labels = "labels" in inputs
    inputs = self._prepare_inputs(inputs)

    # Priority (handled in generate):
    # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
    if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
        gen_kwargs = self._gen_kwargs.copy()
    if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
        gen_kwargs.pop("num_beams")
    if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
        gen_kwargs.pop("max_length")

    default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
    gen_kwargs["synced_gpus"] = (
        gen_kwargs["synced_gpus"]
        if gen_kwargs.get("synced_gpus") is not None
        else default_synced_gpus
    )

    generation_inputs = inputs.copy()
    with torch.cuda.amp.autocast():
        generation_inputs = {
            "pixel_values": generation_inputs["pixel_values"],
            "input_ids": generation_inputs["input_ids"],
            "attention_mask": generation_inputs["attention_mask"],
            "prior_pixel_values": generation_inputs.get("prior_pixel_values", None),
            "has_prior": generation_inputs.get("has_prior", None),
        }
        gen_kwargs.pop("max_length", None)
        gen_kwargs["min_new_tokens"] = 24
        gen_kwargs["max_new_tokens"] = (
            256
            if not hasattr(self, "stage_generation_config")
            else self.stage_generation_config.get("max_new_tokens", 256)
        )
        gen_kwargs["num_beams"] = (
            1
            if not hasattr(self, "stage_generation_config")
            else self.stage_generation_config.get("num_beams", 1)
        )
        if gen_kwargs["num_beams"] > 1:
            gen_kwargs["length_penalty"] = (
                2.0
                if not hasattr(self, "stage_generation_config")
                else self.stage_generation_config.get("length_penalty", 2.0)
            )
        gen_kwargs["do_sample"] = (
            False
            if not hasattr(self, "stage_generation_config")
            else self.stage_generation_config.get("do_sample", False)
        )
        gen_kwargs["temperature"] = (
            1.0
            if not hasattr(self, "stage_generation_config")
            else self.stage_generation_config.get("temperature", 1.0)
        )
        gen_kwargs["top_p"] = (
            1.0
            if not hasattr(self, "stage_generation_config")
            else self.stage_generation_config.get("top_p", 1.0)
        )
        generated_tokens = self.model.generate(
            **generation_inputs,
            **gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
    # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
    # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
    if self.model.generation_config._from_model_config:
        self.model.generation_config._from_model_config = False

    # Retrieves GenerationConfig from model.generation_config
    gen_config = self.model.generation_config
    # in case the batch is shorter than max length, the output should be padded
    if generated_tokens.shape[-1] < gen_config.max_length:
        generated_tokens = self._pad_tensors_to_max_len(
            generated_tokens, gen_config.max_length
        )
    elif (
        gen_config.max_new_tokens is not None
        and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1
    ):
        generated_tokens = self._pad_tensors_to_max_len(
            generated_tokens, gen_config.max_new_tokens + 1
        )

    loss = None

    if self.args.prediction_loss_only:
        return loss, None, None

    if has_labels:
        labels = inputs["labels"]
        if labels.shape[-1] < gen_config.max_length:
            labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
        elif (
            gen_config.max_new_tokens is not None
            and labels.shape[-1] < gen_config.max_new_tokens + 1
        ):
            labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
    else:
        labels = None

    return loss, generated_tokens, labels


def _save(self, output_dir: Optional[str] = None, state_dict=None):
    # If we are executing this function, we are the process zero, so we don't check for that.
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n[CUSTOM] Saving model checkpoint to {output_dir}")

    param_grad_dic = {
        k: v.requires_grad
        for (k, v) in self.accelerator.unwrap_model(self.model).named_parameters()
    }
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

    if self.tokenizer is not None:
        self.tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def _load_best_model(self):
    logger.info(
        f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
    )
    best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
    model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
    state_dict = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
