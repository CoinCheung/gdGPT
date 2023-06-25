
import math
import os.path as osp
from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import *
from transformers.models.llama.modeling_llama import _make_causal_mask, _expand_mask
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec




class LlamaDecoderLayerTupleIO(LlamaDecoderLayer):

    def __init__(self, config: LlamaConfig, load_path=None, gradient_checkpointing=False):
        super().__init__(config)
        if load_path: self.load_state_dict(torch.load(load_path))
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, inputs):
        """
        Args:
            inputs:
                hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                    `(batch, src_len)` where padding elements are 0
        """

        hidden_states, attention_mask = inputs
        batch_size, seq_length, _ = hidden_states.shape
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )
        # because original position_ids is None, we use this directly
        #  position_ids = torch.arange(
        #      0, seq_length, dtype=torch.long, device=hidden_states.device
        #  ).unsqueeze(0).view(-1, seq_length)
        position_ids = (attention_mask.cumsum(dim=-1) - 1) * attention_mask
        if self.gradient_checkpointing and self.training:
            outputs = torch.utils.checkpoint.checkpoint(
                super().forward,
                hidden_states,
                causal_mask,
                position_ids,
            )
        else:
            outputs = super().forward(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids)
        hidden_states = outputs[0]

        return hidden_states, attention_mask

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


class LlamaTerminal(nn.Module):
    """
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, is_first=True, load_path=None):
        super(LlamaTerminal, self).__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,
                config.hidden_size, config.pad_token_id)

        self.forward = self.forward_first
        if not is_first:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.forward = self.forward_last

        if load_path: self.load_state_dict(torch.load(load_path))

    def forward_last(self, inputs):
        hidden_states, attention_mask = inputs
        hidden_states = self.norm(hidden_states)
        logits = F.linear(hidden_states, self.embed_tokens.weight, None)
        return logits

    def forward_first(self, inputs):
        output_attentions = False
        output_hidden_states = False
        use_cache = False
        return_dict = False
        inputs_embeds = None
        past_key_values = None
        position_ids = None
        input_ids = inputs[..., 0]
        attention_mask = inputs[..., 1]

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool,
                device=inputs_embeds.device
            )
        attention_mask = attention_mask.clone()

        hidden_states = inputs_embeds
        return hidden_states, attention_mask

    @property
    def weight(self):
        return self.embed_tokens.weight



## llama does not tie weights
def get_llama_causal_lm_specs(config, load_path=None, grad_ckpt=False, tie_emb=False):
    specs = []
    ldpth = osp.join(load_path, 'layer_00-model_states.pt') if load_path else None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', LlamaTerminal,
                config, is_first=True, load_path=ldpth,
                tied_weight_attr='weight'))
    else:
        specs.append(LlamaTerminal(config, is_first=True, load_path=ldpth))

    for i in range(1, config.num_hidden_layers+1):
        ldpth = None
        if load_path: ldpth = osp.join(load_path, f'layer_{i:02d}-model_states.pt')
        specs.append(LayerSpec(LlamaDecoderLayerTupleIO, config,
            load_path=ldpth, gradient_checkpointing=grad_ckpt))

    ldpth = None
    ind = config.num_hidden_layers + 1
    if load_path: ldpth = osp.join(load_path, f'layer_{ind:02d}-model_states.pt')
    if tie_emb:
        specs.append(TiedLayerSpec('embed', LlamaTerminal,
                config, is_first=False, load_path=ldpth,
                tied_weight_attr='weight'))
    else:
        specs.append(LlamaTerminal(config, is_first=False, load_path=ldpth))
    return specs

