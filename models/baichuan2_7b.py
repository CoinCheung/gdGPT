

import math
import os.path as osp
from typing import Optional, Tuple, Union
import importlib

import torch
from torch import nn
import torch.nn.functional as F

from transformers.dynamic_module_utils import get_class_from_dynamic_module

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


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



#  @torch.compile
def get_baichuan2_7b_components(model_path):

    DecoderLayer = get_class_from_dynamic_module(
            'modeling_baichuan.DecoderLayer', model_path)
    NormHead = get_class_from_dynamic_module(
            'modeling_baichuan.NormHead', model_path)
    RMSNorm = get_class_from_dynamic_module(
            'modeling_baichuan.RMSNorm', model_path)

    class DecoderLayerTupleIO(DecoderLayer):

        def __init__(self, config, load_path=None,
                gradient_checkpointing=False, use_flash_attn=False):
            self.config = config
            super().__init__(config)
            init_weights(self, config.initializer_range)
            if load_path:
                print('load checkpoint: ', load_path)
                state = torch.load(load_path, map_location='cpu')
                self.load_state_dict(state, strict=False)
            self.gradient_checkpointing = gradient_checkpointing


        def forward(self, inputs):
            hidden_states, attention_mask = inputs
            batch_size, seq_length, _ = hidden_states.shape

            causal_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, 0
            )
            position_ids = torch.arange(seq_length, device=attention_mask.device)
            position_ids = position_ids.unsqueeze(0)

            if self.gradient_checkpointing and self.training:
                outputs = torch.utils.checkpoint.checkpoint(
                    super().forward, hidden_states,
                    causal_mask, position_ids, None)
            else:
                outputs = super().forward(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask, position_ids=position_ids,
                        past_key_value=None)

            hidden_states = outputs[0].contiguous()
            return hidden_states, attention_mask

        def _make_causal_mask(
                self, input_ids_shape: torch.Size, dtype: torch.dtype,
                device: torch.device, past_key_values_length: int = 0
        ):
            """
            Make causal mask used for bi-directional self-attention.
            """
            bsz, tgt_len = input_ids_shape
            mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(dtype)

            if past_key_values_length > 0:
                mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
            return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

        def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype,
                tgt_len: Optional[int] = None):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            if len(mask.size()) == 3:
                bsz, src_len, _ = mask.size()
                tgt_len = tgt_len if tgt_len is not None else src_len
                expanded_mask = mask[:,None,:,:].expand(bsz, 1, tgt_len, src_len).to(dtype)
            else:
                bsz, src_len = mask.size()
                tgt_len = tgt_len if tgt_len is not None else src_len
                expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

        def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
            # create causal mask
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = None
            if input_shape[-1] > 1:
                combined_attention_mask = self._make_causal_mask(
                    input_shape,
                    inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    past_key_values_length=past_key_values_length,
                )

            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                    inputs_embeds.device
                )
                combined_attention_mask = (
                    expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
                )

            return combined_attention_mask

    return DecoderLayerTupleIO, NormHead, RMSNorm



class BaichuanEnter(nn.Module):
    """
    Args:
        config: BaichuanConfig
    """

    def __init__(self, config, load_path=None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,
                config.hidden_size, config.pad_token_id)

        init_weights(self, config.initializer_range)
        if load_path:
            print('load checkpoint: ', load_path)
            self.load_state_dict(torch.load(load_path))

    @torch.compile
    def forward(self, inputs):
        return_dict = False
        output_hidden_states = None
        output_attentions = None
        use_cache = None
        inputs_embeds = None
        past_key_values = None
        position_ids = None
        input_ids = inputs[..., 0].contiguous()
        attention_mask = inputs[..., 1].contiguous()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

        hidden_states = inputs_embeds
        return hidden_states, attention_mask

    @property
    def weight(self):
        return self.embed_tokens.weight


class BaichuanExit(nn.Module):
    """
    Args:
        config: BaichuanConfig
    """

    def __init__(self, config, head_cls, norm_cls, load_path=None):
        super(BaichuanExit, self).__init__()
        self.config = config

        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = head_cls(config.hidden_size, config.vocab_size, bias=False)

        init_weights(self, config.initializer_range)
        if load_path:
            print('load checkpoint: ', load_path)
            self.load_state_dict(torch.load(load_path))

    @torch.compile
    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @property
    def weight(self):
        return self.lm_head.weight


def get_baichuan2_7b_causal_lm_specs(config, load_path=None, grad_ckpt=False,
        tie_emb=False, use_flash_attn=False, from_scratch=False):

    DecoderLayerTupleIO, NormHead, RMSNorm = get_baichuan2_7b_components(load_path)

    specs = []
    ldpth = osp.join(load_path, 'layer_00-model_states.pt')
    if from_scratch: ldpth = None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', BaichuanEnter,
                config, load_path=ldpth,
                tied_weight_attr='weight'))
    else:
        specs.append(BaichuanEnter(config, load_path=ldpth))

    for i in range(1, config.num_hidden_layers+1):
        ldpth = osp.join(load_path, f'layer_{i:02d}-model_states.pt')
        if from_scratch: ldpth = None
        specs.append(LayerSpec(DecoderLayerTupleIO, config,
            load_path=ldpth, gradient_checkpointing=grad_ckpt,
            use_flash_attn=use_flash_attn))

    ind = config.num_hidden_layers + 1
    ldpth = osp.join(load_path, f'layer_{ind:02d}-model_states.pt')
    if from_scratch: ldpth = None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', BaichuanExit,
                config, head_cls=NormHead, norm_cls=RMSNorm, load_path=ldpth,
                tied_weight_attr='weight'))
    else:
        specs.append(BaichuanExit(config, head_cls=NormHead, norm_cls=RMSNorm,
                load_path=ldpth))
    return specs

