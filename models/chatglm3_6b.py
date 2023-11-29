

import math
import os.path as osp
from typing import Optional, Tuple, Union
import importlib

import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F

from transformers.dynamic_module_utils import get_class_from_dynamic_module

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec



def init_weights(model, std):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



@torch.compile
def get_chatglm3_6b_components(model_path):

    RotaryEmbedding = get_class_from_dynamic_module(
            'modeling_chatglm.RotaryEmbedding', model_path)
    GLMBlock = get_class_from_dynamic_module(
            'modeling_chatglm.GLMBlock', model_path)
    Embedding = get_class_from_dynamic_module(
            'modeling_chatglm.Embedding', model_path)
    RMSNorm = get_class_from_dynamic_module(
            'modeling_chatglm.RMSNorm', model_path)

    class GLMBlockTupleIO(GLMBlock):

        def __init__(self, config, layer_number, load_path=None,
                gradient_checkpointing=False):
            self.config = config
            super().__init__(config, layer_number)

            rotary_dim = (
                config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
            )
            self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, dtype=config.torch_dtype)

            init_weights(self, 0.02)
            if load_path:
                print('load checkpoint: ', load_path)
                state = torch.load(load_path, map_location='cpu')
                self.load_state_dict(state, strict=True)
            self.gradient_checkpointing = gradient_checkpointing


        def forward(self, inputs):
            hidden_states, attention_mask = inputs
            seq_length, batch_size, hidden_dim = hidden_states.shape

            causal_mask = self.get_masks(
                hidden_states, None, attention_mask
            )

            position_ids = torch.arange(seq_length, device=attention_mask.device)
            position_ids = position_ids.unsqueeze(0)
            rotary_pos_emb = self.rotary_pos_emb(seq_length)
            rotary_pos_emb = rotary_pos_emb[position_ids]

            if self.gradient_checkpointing and self.training:
                outputs = torch.utils.checkpoint.checkpoint(
                    super().forward, hidden_states,
                    causal_mask, rotary_pos_emb, None, False)
            else:
                outputs = super().forward(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        use_cache=False)

            hidden_states = outputs[0].contiguous()
            return hidden_states, attention_mask

        def get_masks(self, hidden_states, past_key_values=None, padding_mask=None):
            seq_length, batch_size, hidden_dim = hidden_states.shape
            device = hidden_states.device
            full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=device)
            full_attention_mask.tril_()
            past_length = 0
            if past_key_values:
                past_length = past_key_values[0][0].shape[0]
            if past_length:
                full_attention_mask = torch.cat(
                        (torch.ones(batch_size, seq_length, past_length, device=device),
                         full_attention_mask), dim=-1)
            if padding_mask is not None:
                full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
            if not past_length and padding_mask is not None:
                full_attention_mask -= padding_mask.unsqueeze(-1) - 1
            full_attention_mask = (full_attention_mask < 0.5).bool()
            full_attention_mask.unsqueeze_(1)
            return full_attention_mask


    class ChatGLM3Enter(Embedding):

        def __init__(self, config, load_path=None):
            self.config = config
            super().__init__(config)

            init_weights(self, 0.02)
            if load_path:
                print('load checkpoint: ', load_path)
                state = torch.load(load_path, map_location='cpu')
                self.load_state_dict(state, strict=True)


        def forward(self, inputs):
            input_ids = inputs[..., 0].contiguous()
            attention_mask = inputs[..., 1].contiguous()
            inputs_embeds = super().forward(input_ids)

            return inputs_embeds, attention_mask

        @property
        def weight(self):
            return self.weight

    return GLMBlockTupleIO, ChatGLM3Enter, RMSNorm


class ChatGLM3Exit(nn.Module):

    def __init__(self, config, norm_cls, load_path=None):
        super().__init__()
        self.config = config

        self.final_layernorm = nn.Identity()
        if config.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = norm_cls(config.hidden_size,
                    eps=config.layernorm_epsilon, dtype=config.torch_dtype)
        self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False)

        init_weights(self, 0.02)
        if load_path:
            print('load checkpoint: ', load_path)
            state = torch.load(load_path, map_location='cpu')
            self.load_state_dict(state, strict=True)


    @torch.compile
    def forward(self, inputs):
        hidden_states, attention_mask = inputs

        hidden_states = self.final_layernorm(hidden_states)
        lm_logits = self.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        return lm_logits

    @property
    def weight(self):
        return self.output_layer.weight




def get_chatglm3_6b_causal_lm_specs(config, load_path=None, grad_ckpt=False,
        tie_emb=False, use_flash_attn=False, from_scratch=False):

    DecoderLayerTupleIO, ChatGLM3Enter, RMSNorm = get_chatglm3_6b_components(load_path)
    LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm

    specs = []
    ldpth = osp.join(load_path, 'layer_00-model_states.pt')
    if from_scratch: ldpth = None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', ChatGLM3Enter,
                config, load_path=ldpth,
                tied_weight_attr='weight'))
    else:
        specs.append(ChatGLM3Enter(config, load_path=ldpth))

    for i in range(1, config.num_layers+1):
        ldpth = osp.join(load_path, f'layer_{i:02d}-model_states.pt')
        if from_scratch: ldpth = None
        specs.append(LayerSpec(DecoderLayerTupleIO, config, i,
            load_path=ldpth, gradient_checkpointing=grad_ckpt))

    ind = config.num_layers + 1
    ldpth = osp.join(load_path, f'layer_{ind:02d}-model_states.pt')
    if from_scratch: ldpth = None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', ChatGLM3Exit,
                config, LayerNormFunc, load_path=ldpth,
                tied_weight_attr='weight'))
    else:
        specs.append(ChatGLM3Exit(config, LayerNormFunc, load_path=ldpth))
    return specs

