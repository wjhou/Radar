from transformers.trainer_seq2seq import *
from transformers.trainer import *
from torch.optim import AdamW


def train(self):
    self._memory_tracker.start()
    model = self._wrap_model(self.model_wrapped)
    optimizer = AdamW(model.parameters())
    model = self.accelerator.prepare(self.model, optimizer)
    if model is not self.model:
        self.model_wrapped = model
    if self.is_deepspeed_enabled:
        self.deepspeed = self.model_wrapped
