
import math
import os.path as osp
from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import *
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


class MixtralDecoderLayerTupleIO(MixtralDecoderLayer):

    def __init__(self, config: MixtralConfig, layer_idx, load_path=None,
            gradient_checkpointing=False, use_flash_attn=False):
        if use_flash_attn: config._attn_implementation = "flash_attention_2"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        super().__init__(config, layer_idx)
        init_weights(self, config.initializer_range)
        if load_path:
            print('load checkpoint: ', load_path)
            self.load_state_dict(torch.load(load_path), strict=False)
        self.gradient_checkpointing = gradient_checkpointing

    #  @torch.compile
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
                    position_ids=position_ids,
                    )
        hidden_states = outputs[0]

        return hidden_states, attention_mask

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        return causal_mask


class MixtralEnter(nn.Module):
    """
    Args:
        config: MixtralConfig
    """

    def __init__(self, config: MixtralConfig, load_path=None):
        super(MixtralEnter, self).__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,
                config.hidden_size, config.pad_token_id)

        init_weights(self, config.initializer_range)
        if load_path:
            print('load checkpoint: ', load_path)
            self.load_state_dict(torch.load(load_path))

    @torch.compile
    def forward(self, inputs):
        output_attentions = False
        output_hidden_states = False
        output_router_logits = None
        use_cache = False
        return_dict = False
        inputs_embeds = None
        past_key_values = None
        position_ids = None
        input_ids = inputs[..., 0].contiguous()
        attention_mask = inputs[..., 1].contiguous()

        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        return hidden_states, attention_mask

    @property
    def weight(self):
        return self.embed_tokens.weight


class MixtralExit(nn.Module):
    """
    Args:
        config: MixtralConfig
    """

    def __init__(self, config: MixtralConfig, load_path=None):
        super(MixtralExit, self).__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,
                config.hidden_size, config.pad_token_id)

        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        init_weights(self, config.initializer_range)
        if load_path:
            print('load checkpoint: ', load_path)
            self.load_state_dict(torch.load(load_path))

    @torch.compile
    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        hidden_states = self.norm(hidden_states)
        logits = F.linear(hidden_states, self.embed_tokens.weight, None)
        return logits

    @property
    def weight(self):
        return self.embed_tokens.weight



## mixtral does not tie weights
def get_mixtral_causal_lm_specs(config, load_path=None, grad_ckpt=False,
        tie_emb=False, use_flash_attn=False, from_scratch=False):
    specs = []
    ldpth = osp.join(load_path, 'layer_00-model_states.pt')
    if from_scratch: ldpth = None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', MixtralEnter,
                config, load_path=ldpth, tied_weight_attr='weight'))
    else:
        specs.append(MixtralEnter(config, load_path=ldpth))

    for i in range(1, config.num_hidden_layers+1):
        ldpth = osp.join(load_path, f'layer_{i:02d}-model_states.pt')
        if from_scratch: ldpth = None
        specs.append(LayerSpec(MixtralDecoderLayerTupleIO, config, i - 1,
            load_path=ldpth, gradient_checkpointing=grad_ckpt,
            use_flash_attn=use_flash_attn))

    ind = config.num_hidden_layers + 1
    ldpth = osp.join(load_path, f'layer_{ind:02d}-model_states.pt')
    if from_scratch: ldpth = None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', MixtralExit,
                config, load_path=ldpth, tied_weight_attr='weight'))
    else:
        specs.append(MixtralExit(config, load_path=ldpth))
    return specs

