import transformers
from dataclasses import dataclass, field

from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    sft_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_type: Optional[str] = field(default="clip")
    num_beams: int = field(default=4)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    clinical_context_path: str = field(default=None)
    knowledge_path: str = field(default=None)
    sentence_knowledge_path: Optional[str] = field(default=None)
    image_annotations_path: Optional[str] = field(default=None)
    gt_observation_path: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    debug_model: bool = field(default=False)
    stage: int = field(default=0)
    topk: int = field(default=1)
    num_repeat: int = field(default=1)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
