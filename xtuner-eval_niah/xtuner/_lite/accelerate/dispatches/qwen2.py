# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from typing import Optional

import torch
from mmengine import MessageHub
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (apply_rotary_pos_emb,
                                                      repeat_kv)

from ._attention import flash_attn_wo_mask, varlen_flash_attn
from flash_attn import flash_attn_with_kvcache
import torch.distributed as dist
from xtuner._lite.parallel.setup import get_sp_group, get_ring_group, get_ring_world_size, get_sp_world_size, \
    get_ulysess_group
from xtuner._lite.yunchang import attention_sp_ulysses_ring, ring_flash_attn_inference_func
from mmengine.dist import is_distributed, all_gather_object, all_gather

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func

    _flash_supports_window_size = 'window_size' in list(
        inspect.signature(flash_attn_func).parameters)
    SUPPORT_FLASH2 = True
except ImportError:
    pass


def qwen2_attn_forward_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,  # will become mandatory in v4.46
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    kv_seq_len = key_states.shape[-2]

    is_prefilled = past_key_value.get_seq_length(self.layer_idx) == 0
    if is_distributed():
        # 单卡不进行修改
        rank = dist.get_rank(get_sp_group())
        world_size = dist.get_world_size(get_sp_group())
    else:
        rank = 0
        world_size = 1

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        kv_seq_len = key_states.shape[-2] + cache_position[0]

        if world_size > 1:
            if is_prefilled:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                if rank == world_size - 1:
                    # 在 decode 阶段，只有最后一个 rank 才需要更新
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx,
                                                                     cache_kwargs)
                else:
                    kv_seq_len = cache_position[0]
                    key_states, value_states = past_key_value.key_cache[self.layer_idx], past_key_value.value_cache[
                        self.layer_idx]
        else:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        print(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if world_size == 1:
        attn_output = flash_attn_with_kvcache(
            query_states,
            key_states,
            value_states,
            causal=True,
            cache_seqlens=kv_seq_len.item())
    else:
        if query_states.shape[1] > 1 and is_prefilled:
            key_states = key_states[:, :kv_seq_len, ...]
            value_states = value_states[:, :kv_seq_len, ...]
            assert key_states.shape[1] == query_states.shape[1]
            attn_output = attention_sp_ulysses_ring(query_states,
                                                    key_states,
                                                    value_states,
                                                    ulysses_pg=get_ulysess_group(),
                                                    ring_pg=get_ring_group(),
                                                    ring_impl_type='basic')
        else:
            attn_output = ring_flash_attn_inference_func(query_states,
                                                         key_states,
                                                         value_states,
                                                         causal=True,
                                                         group=get_ring_group(),
                                                         cache_seqlens=kv_seq_len.item())
    # ---------------- flash attention forward end ------------------- #
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def qwen2_varlen_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
):
    is_training = self.training
    if not is_training:
        attn_output, attn_weights, past_key_value = qwen2_attn_forward_inference(self,
                                                                                 hidden_states,
                                                                                 attention_mask,
                                                                                 position_ids,
                                                                                 past_key_value,
                                                                                 output_attentions,
                                                                                 use_cache, **kwargs)
        return attn_output, attn_weights, past_key_value

    assert is_training == (past_key_value is None)

    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37'
            ' Please make sure use `attention_mask` instead.`')

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')
    bsz, q_len, _ = hidden_states.size()

    attn_context = MessageHub.get_instance('packed_sequence')
    position_ids = attn_context.get_info('position_ids')
    assert position_ids.size(1) == q_len, f'{position_ids.size(1)} {q_len}'

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                'The cache structure has changed since version v4.36. '
                f'If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, '
                'please make sure to initialize the attention class '
                'with a layer index.')
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    assert position_ids is not None
    rotary_seq_len = max(kv_seq_len, position_ids.max().item() + 1)
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value
        # `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (getattr(self.config, 'sliding_window', None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    'past key must have a shape of (`batch_size, num_heads, '
                    'self.config.sliding_window-1, head_dim`), got'
                    f' {past_key.shape}')

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat(
                    [attention_mask,
                     torch.ones_like(attention_mask[:, -1:])],
                    dim=-1)

        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads for sequence parallel
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for
    # training stability reasons, therefore the input hidden states gets
    # silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # ----------------- flash attention forward ------------------------#

    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and q_len != 1

    use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, 'sliding_window', None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.layer_idx < self.config.max_window_layers
            and self.config.use_sliding_window)

    window_size = (self.config.sliding_window,
                   self.config.sliding_window) if use_sliding_windows else (-1,
                                                                            -1)

    assert SUPPORT_FLASH2
    cumulative_lengths = attn_context.get_info('cumulative_lengths')
    if cumulative_lengths is not None and SUPPORT_FLASH2 and bsz == 1:
        max_seqlen = attn_context.get_info('max_seqlen')
        attn_output = varlen_flash_attn(
            query_states,
            key_states,
            value_states,
            cumulative_lengths,
            max_seqlen,
            causal=causal,
            dropout_p=dropout_rate,
            window_size=window_size,
            training=self.training)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=causal,
            dropout_p=dropout_rate,
            window_size=window_size,
            training=self.training)

    # ---------------- flash attention forward end ------------------- #

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
