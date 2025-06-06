from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from mmengine.logging import MessageHub
import copy
from xtuner._lite.parallel.new_setup import get_sp_group
import math
import os
from xtuner._lite.parallel.sequence import split_for_sequence_parallel


def rescale_sp_loss(loss_per_sp_rank,
                    labels_per_sp_rank,
                    sp_group: dist.ProcessGroup = None,
                    ignore_index=-100):
    if sp_group is None:
        sp_group = get_sp_group()

    if (sp_group is None) or (dist.get_world_size(sp_group) == 1):
        return loss_per_sp_rank

    shift_labels = labels_per_sp_rank
    active_tokens = (shift_labels != ignore_index).long().sum()
    global_active_tokens = copy.deepcopy(active_tokens)
    dist.all_reduce(global_active_tokens, group=sp_group)
    loss_weight = active_tokens / global_active_tokens * dist.get_world_size(
        group=sp_group)

    if active_tokens == 0:
        # convert nan to 0 just for logging
        loss_per_sp_rank = torch.nan_to_num(loss_per_sp_rank)

    return loss_per_sp_rank * loss_weight


def internvl2_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    sp_size = dist.get_world_size(get_sp_group())
    if sp_size > 1:
        sp_group = get_sp_group()
        sp_rank = dist.get_rank(sp_group)

        no_split_input_ids = os.environ.get('NO_SPLIT_INPUT_IDS')
        split_input_ids = not no_split_input_ids
        if split_input_ids:
            pad_id = 0
            orig_len_input_ids = input_ids.shape[1]
            image_flags = image_flags.squeeze(-1)
            assert input_ids.shape[0] == 1, 'batch size must be 1 for sequence parallel'
            # input_ids 均匀切分
            if orig_len_input_ids % sp_size != 0:  # 确保能均匀切
                max_inputs_len = math.ceil(orig_len_input_ids / sp_size) * sp_size
                _temp = input_ids.new_full((1, max_inputs_len - orig_len_input_ids), pad_id)
                input_ids_new = torch.cat([input_ids, _temp], dim=-1)
            else:
                input_ids_new = input_ids
            input_ids_list = torch.split(input_ids_new, input_ids_new.shape[1] // sp_size, dim=-1)
            input_ids_rank_pre = input_ids_list[sp_rank].contiguous()
            input_embeds_rank_pre = self.language_model.get_input_embeddings()(input_ids_rank_pre).clone()

            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            input_embeds = all_gather(input_embeds_rank_pre, group=sp_group)
            # torch.cuda.synchronize()
            # elapsed = time.perf_counter() - start_time
            # print(elapsed,'xxxx',flush=True)
            input_embeds = torch.cat(input_embeds, dim=1)
            input_embeds = input_embeds[:, :orig_len_input_ids]
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        no_split_pixel_values = os.environ.get('NO_SPLIT_PIXEL_VALUES')
        split_pixel_values = not no_split_pixel_values
        # print(split_input_ids, split_pixel_values, os.environ.get('USE_CUSTOM_LOSS'), flush=True)
        if split_pixel_values:
            # pixel_values 均匀切分
            orig_img_batch = pixel_values.shape[0]
            if orig_img_batch % sp_size != 0:  # 确保能均匀切
                max_inputs_len = math.ceil(orig_img_batch / sp_size) * sp_size
                pad_img_batch = max_inputs_len - orig_img_batch
                pad_pixel_values_ = pixel_values.new_zeros(pad_img_batch, 3,
                                                           pixel_values.shape[2],
                                                           pixel_values.shape[3])
                pixel_values = torch.cat([pixel_values, pad_pixel_values_], dim=0)
            pixel_values = torch.split(pixel_values, len(pixel_values) // sp_size, dim=0)
            pixel_values = pixel_values[sp_rank].contiguous()

            vit_embeds = self.extract_feature(pixel_values)

            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            vit_embeds = all_gather(vit_embeds, group=sp_group)
            # torch.cuda.synchronize()
            # elapsed = time.perf_counter() - start_time
            # print(elapsed,'qqqqqxx',flush=True)

            vit_embeds = torch.cat(vit_embeds, dim=0)[:orig_img_batch]
        else:
            vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
    else:
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

    # vit_batch_size = pixel_values.shape[0]

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    # if torch.distributed.get_rank() == 0:
    #     print(
    #         f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == self.img_context_token_id)
    try:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
    except Exception as e:
        vit_embeds = vit_embeds.reshape(-1, C)
        print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
              f'vit_embeds.shape={vit_embeds.shape}')
        n_token = selected.sum()
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

    input_embeds = input_embeds.reshape(B, N, C)

    if sp_size > 1:
        # 此处开始进行切分处理
        # 只需要处理 inputs_embeds 和 position_ids，其余用不到
        attn_context = MessageHub.get_instance('packed_sequence')
        position_ids = attn_context.get_info('position_ids')
        # phi3 attention 计算时候有特殊用途
        attn_context.update_info('global_position_ids', position_ids)

        assert position_ids.size(1) == input_embeds.shape[1] == labels.shape[1], \
            f'{position_ids.size(1)} {input_embeds.shape[1]} {labels.shape[1]}'
        assert position_ids.size(1) % sp_size == 0
        # `dim` is 1 as the shape of tensor is (bs, seq_len)
        position_ids = split_for_sequence_parallel(
            position_ids, dim=1, sp_group=sp_group)
        input_embeds = split_for_sequence_parallel(
            input_embeds, dim=1, sp_group=sp_group)
        labels = split_for_sequence_parallel(
            labels, dim=1, sp_group=sp_group)
        attention_mask = None  # 不需要
        attn_context.update_info('position_ids', position_ids)

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    logits = outputs.logits

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        if sp_size > 1:
            # sp 间均衡
            loss = rescale_sp_loss(loss, shift_labels, sp_group=sp_group)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )



def internvl2_hico_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    sp_size = dist.get_world_size(get_sp_group())
    if sp_size > 1:
        sp_group = get_sp_group()
        sp_rank = dist.get_rank(sp_group)

        no_split_input_ids = os.environ.get('NO_SPLIT_INPUT_IDS')
        split_input_ids = not no_split_input_ids
        if split_input_ids:
            pad_id = 0
            orig_len_input_ids = input_ids.shape[1]
            image_flags = image_flags.squeeze(-1)
            assert input_ids.shape[0] == 1, 'batch size must be 1 for sequence parallel'
            # input_ids 均匀切分
            if orig_len_input_ids % sp_size != 0:  # 确保能均匀切
                max_inputs_len = math.ceil(orig_len_input_ids / sp_size) * sp_size
                _temp = input_ids.new_full((1, max_inputs_len - orig_len_input_ids), pad_id)
                input_ids_new = torch.cat([input_ids, _temp], dim=-1)
            else:
                input_ids_new = input_ids
            input_ids_list = torch.split(input_ids_new, input_ids_new.shape[1] // sp_size, dim=-1)
            input_ids_rank_pre = input_ids_list[sp_rank].contiguous()
            input_embeds_rank_pre = self.language_model.get_input_embeddings()(input_ids_rank_pre).clone()

            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            input_embeds = all_gather(input_embeds_rank_pre, group=sp_group)
            # torch.cuda.synchronize()
            # elapsed = time.perf_counter() - start_time
            # print(elapsed,'xxxx',flush=True)
            input_embeds = torch.cat(input_embeds, dim=1)
            input_embeds = input_embeds[:, :orig_len_input_ids]
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        split_pixel_values = True
        # print(split_input_ids, split_pixel_values, os.environ.get('USE_CUSTOM_LOSS'), flush=True)
        if split_pixel_values:
            # pixel_values 均匀切分
            orig_img_batch = pixel_values.shape[0]
            if orig_img_batch % sp_size != 0:  # 确保能均匀切
                max_inputs_len = math.ceil(orig_img_batch / sp_size) * sp_size
                pad_img_batch = max_inputs_len - orig_img_batch
                pad_pixel_values_ = pixel_values.new_zeros(pad_img_batch, 3,
                                                           pixel_values.shape[2],
                                                           pixel_values.shape[3])
                pixel_values = torch.cat([pixel_values, pad_pixel_values_], dim=0)
            pixel_values = torch.split(pixel_values, len(pixel_values) // sp_size, dim=0)
            pixel_values = pixel_values[sp_rank].contiguous()

            vit_embeds = self.extract_feature_vit(pixel_values)

            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            vit_embeds = all_gather(vit_embeds, group=sp_group)
            # torch.cuda.synchronize()
            # elapsed = time.perf_counter() - start_time
            # print(elapsed,'qqqqqxx',flush=True)

            vit_embeds = torch.cat(vit_embeds, dim=0)[:orig_img_batch]
        else:
            vit_embeds = self.extract_feature_vit(pixel_values)
        
        vit_embeds = self.extract_feature_connector(vit_embeds, image_flags=image_flags)
    else:
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = self.extract_feature_vit(pixel_values)
        vit_embeds = self.extract_feature_connector(vit_embeds, image_flags=image_flags)

    # vit_batch_size = pixel_values.shape[0]

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    # if torch.distributed.get_rank() == 0:
    #     print(
    #         f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == self.img_context_token_id)
    try:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
    except Exception as e:
        vit_embeds = vit_embeds.reshape(-1, C)
        print(f'warning (in /mnt/petrelfs/lixinhao/lxh_exp/LongVideo/xtuner-hha_1028/xtuner/_lite/accelerate/dispatches/internvl2.py): {e}, input_embeds[selected].shape={input_embeds[selected].shape}, pixel_values.shape: {pixel_values.shape} sp_size={sp_size}, vit_embeds.shape={vit_embeds.shape}')
        n_token = selected.sum()
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

    input_embeds = input_embeds.reshape(B, N, C)

    if sp_size > 1:
        # 此处开始进行切分处理
        # 只需要处理 inputs_embeds 和 position_ids，其余用不到
        attn_context = MessageHub.get_instance('packed_sequence')
        position_ids = attn_context.get_info('position_ids')
        # phi3 attention 计算时候有特殊用途
        attn_context.update_info('global_position_ids', position_ids)

        assert position_ids.size(1) == input_embeds.shape[1] == labels.shape[1], \
            f'{position_ids.size(1)} {input_embeds.shape[1]} {labels.shape[1]}'
        assert position_ids.size(1) % sp_size == 0
        # `dim` is 1 as the shape of tensor is (bs, seq_len)
        position_ids = split_for_sequence_parallel(
            position_ids, dim=1, sp_group=sp_group)
        input_embeds = split_for_sequence_parallel(
            input_embeds, dim=1, sp_group=sp_group)
        labels = split_for_sequence_parallel(
            labels, dim=1, sp_group=sp_group)
        attention_mask = None  # 不需要
        attn_context.update_info('position_ids', position_ids)

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    logits = outputs.logits

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        if sp_size > 1:
            # sp 间均衡
            loss = rescale_sp_loss(loss, shift_labels, sp_group=sp_group)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
