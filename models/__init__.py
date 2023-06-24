
from torch import nn
from .bloom import get_bloom_causal_lm_specs
from .llama import get_llama_causal_lm_specs



class LMCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ignore_index=-100)

    def forward(self, lm_logits, labels):
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        return super().forward(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length)
        )


