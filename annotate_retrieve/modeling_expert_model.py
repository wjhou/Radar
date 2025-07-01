import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import SwinConfig
from transformers import AutoModel


class ExpertModel(PreTrainedModel):
    def __init__(self, config: SwinConfig, text_model=None):
        super().__init__(config)
        self.text_model = text_model
        hidden_size = config.hidden_size + text_model.config.hidden_size
        num_observation = config.num_observation
        self.observation_cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_observation),
        )
        self.model = AutoModel.from_pretrained(self.config.pretrained_visual_extractor)

    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
    ):
        image_embeds = self.model(
            input_pixels,
        ).pooler_output

        text_embeds = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        image_embeds = torch.cat((image_embeds, text_embeds), dim=-1)
        observation_cls_logits = self.observation_cls(image_embeds)
        return observation_cls_logits
