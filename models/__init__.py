
import torch
from torch import nn
from .bloom import get_bloom_causal_lm_specs
from .llama import get_llama_causal_lm_specs
from .baichuan2_7b import get_baichuan2_7b_causal_lm_specs



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



class MaxZLoss(nn.Module):

    def __init__(self, weight=2.e-4):
        super().__init__()
        self.weight = weight

    def forward(self, lm_logits):
        '''
        用于lm的，先给shift一下，再计算loss就好了
        '''
        lm_logits_max = lm_logits.log_softmax(dim=-1).max(dim=-1)[0]
        loss = lm_logits_max.pow(2.).mul(self.weight).mean()
        return loss



class LMLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.crit_ce = LMCrossEntropyLoss()

        self.crit_max_z = None
        if cfg.get('aux_max_z_loss', None):
            weight = cfg['aux_max_z_loss']['weight']
            self.crit_max_z = MaxZLoss(weight)

    @torch.compile
    def forward(self, lm_logits, labels):
        loss = self.crit_ce(lm_logits, labels)
        if self.crit_max_z:
            loss = loss + self.crit_max_z(lm_logits)
        return loss

