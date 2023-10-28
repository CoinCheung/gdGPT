
import math
import os.path as osp
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers.models.bloom.modeling_bloom import *
from transformers.models.bloom.modeling_bloom import _make_causal_mask, _expand_mask

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


class BloomAlibiEmbedding(nn.Module):

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32
        )
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
                dtype=torch.float32
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        slopes = slopes.unsqueeze(-1)

        self.register_buffer("slopes", slopes, persistent=False)

    @torch.no_grad()
    def forward(self, attention_mask):
        _, seq_length = attention_mask.shape
        position_ids = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)
        alibi = self.slopes * position_ids[:, None, :]
        alibi = alibi.reshape(-1, 1, seq_length).detach()
        alibi.requires_grad = False
        return alibi


class BloomAttentionFast(BloomAttention):

    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.dropout_p = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None


        query_layer = query_layer.reshape(batch_size, self.num_heads, q_length, self.head_dim)
        key_layer   = key_layer.reshape(batch_size, self.num_heads, self.head_dim, kv_length)
        value_layer = value_layer.reshape(batch_size, self.num_heads, kv_length, self.head_dim)
        key_layer   = torch.einsum('bhdl->bhld', key_layer)

        alibi = alibi.reshape(batch_size, self.num_heads, 1, -1).detach()
        attn_mask = torch.zeros(batch_size, 1, q_length, kv_length).to(alibi)
        attn_mask.masked_fill_(attention_mask, torch.finfo(attn_mask.dtype).min)
        attn_mask = attn_mask + alibi
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            context_layer = F.scaled_dot_product_attention(query_layer, key_layer,
                    value_layer, dropout_p=self.dropout_p, attn_mask=attn_mask,
                    scale=self.inv_norm_factor)
        context_layer = torch.einsum('bhld->blhd', context_layer)
        context_layer = context_layer.flatten(2)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs



class BloomBlockTupleIO(BloomBlock):

    def __init__(self, config, load_path=None,
            gradient_checkpointing=False):
        self.config = config
        super(BloomBlockTupleIO, self).__init__(config)
        init_weights(self, config.initializer_range)
        if load_path: self.load_state_dict(torch.load(load_path), strict=False)
        self.alibi_emb = BloomAlibiEmbedding(self.config.n_head)
        self.gradient_checkpointing = gradient_checkpointing
        ## this slows down training with pytorch 2.1.0, let's see what it would be in future pytorch versions
        self.self_attention = BloomAttentionFast(config=config)

    @torch.compile
    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        batch_size, seq_length, _ = hidden_states.shape
        causal_mask = self._prepare_attn_mask(
            attention_mask, input_shape=(batch_size, seq_length),
            past_key_values_length=0,
        ).detach()
        alibi = self.alibi_emb(attention_mask)

        if self.gradient_checkpointing and self.training:
            outputs = torch.utils.checkpoint.checkpoint(
                super(BloomBlockTupleIO, self).forward,
                    hidden_states, alibi, causal_mask)
        else:
            outputs = super(BloomBlockTupleIO, self).forward(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask, alibi=alibi)

        hidden_states = outputs[0]
        return hidden_states, attention_mask

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask


class BloomTerminal(nn.Module):

    def __init__(self, config, is_first=True, load_path=None):
        super(BloomTerminal, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.is_first = is_first
        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = nn.LayerNorm(
                self.embed_dim, eps=config.layer_norm_epsilon)

        self.forward = self.forward_last
        if is_first: self.forward = self.forward_first

        init_weights(self, config.initializer_range)
        if load_path: self.load_state_dict(torch.load(load_path))

    @property
    def word_embeddings_weight(self):
        """Easy accessory for the pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight

    @torch.compile
    def forward_first(self, tp_inputs):

        input_ids = tp_inputs[..., 0].contiguous()
        attention_mask = tp_inputs[..., 1].contiguous()
        past_key_values = None
        head_mask = None
        inputs_embeds = None

        config = self.config

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * config.num_hidden_layers)

        head_mask = [None for _ in range(config.n_layer)]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)
        attention_mask = attention_mask.clone().detach()

        return hidden_states, attention_mask

    @torch.compile
    def forward_last(self, inputs):
        hidden_states, attention_mask = inputs
        lm_output = self.word_embeddings_layernorm(hidden_states)
        lm_output = F.linear(lm_output, self.word_embeddings.weight, None)
        return lm_output

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings


def get_bloom_causal_lm_specs(config, load_path=None, grad_ckpt=False,
        tie_emb=True, use_flash_attn=False, from_scratch=False):
    specs = []
    ldpth = osp.join(load_path, 'layer_00-model_states.pt')
    if from_scratch: ldpth = None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', BloomTerminal, config,
            is_first=True, load_path=ldpth,
            tied_weight_attr='word_embeddings_weight'))
    else:
        specs.append(BloomTerminal(config, is_first=True, load_path=ldpth))

    for i in range(1, config.num_hidden_layers + 1):
        ldpth = osp.join(load_path, f'layer_{i:02d}-model_states.pt')
        if from_scratch: ldpth = None
        specs.append(LayerSpec(BloomBlockTupleIO, config, load_path=ldpth,
                   gradient_checkpointing=grad_ckpt))

    ind = config.num_hidden_layers + 1
    ldpth = osp.join(load_path, f'layer_{ind:02d}-model_states.pt')
    if from_scratch: ldpth = None
    if tie_emb:
        specs.append(TiedLayerSpec('embed', BloomTerminal, config,
            is_first=False, load_path=ldpth,
            tied_weight_attr='word_embeddings_weight'))
    else:
        specs.append(BloomTerminal(config, is_first=False, load_path=ldpth))
    return specs

