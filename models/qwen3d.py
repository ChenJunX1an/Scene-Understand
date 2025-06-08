import random
import logging
from abc import ABC

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
import sys

# 添加库的绝对路径到 sys.path
# package_path = '/home/u2120220610/anaconda3/envs/chat-scene/lib/python3.9/site-packages'
# sys.path.append(package_path)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLPreTrainedModel, Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLConfig, Qwen2_5_VisionTransformerPretrainedModel, add_start_docstrings_to_model_forward, replace_return_docstrings, QWEN2_5_VL_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
#from transformers import Qwen2_5_VLPreTrainedModel, GenerationMixin, Qwen2_5_VLConfig, Qwen2_5_VLCausalLMOutputWithPast,Qwen2_5_VisionTransformerPretrainedModel, add_start_docstrings_to_model_forward, replace_return_docstrings, QWEN2_5_VL_INPUTS_DOCSTRING, _CONFIG_FOR_DOC



logger = logging.getLogger(__name__)


def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))


class qwen3d(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # 获取vision encoder输出的特征维度
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        print("config.vocab_size:",config.vocab_size) #152064
        # temp=aaa
        self.vision_output_dim = config.vision_config.out_hidden_size
        # 定义2层MLP
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.vision_output_dim + 1024, 512),  # 假设输入的3D position特征维度为3
        #     nn.ReLU(),
        #     nn.Linear(512, self.vision_output_dim)
        # )
        # --- Freezing Logic ---
        logger.info("Freezing all parameters initially...")
        # 1. Freeze ALL parameters in the entire model first.
        # This includes the base Qwen2_5_VLForConditionalGeneration parameters
        # AND the parameters of self.mlp initially.
        for param in self.parameters():
            param.requires_grad = False

        # 2. Unfreeze ONLY the parameters belonging to self.mlp.
        # logger.info("Unfreezing parameters of self.mlp...")
        # for param in self.mlp.parameters():
        #     param.requires_grad = True
        # --- End Freezing Logic ---

        logger.info("Parameter trainable status after freezing:")

    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)



    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            position_3d_features: Optional[torch.Tensor] = None,  # 新增3D position特征输入
            instruction_end_pos: Optional[int] = None,
            train_emb: Optional[bool] = False,
            ori_vocab_size: Optional[int] = None
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """
               这是forward方法的文档字符串，用于描述该方法的功能、参数和返回值。

               参数:
                   input_ids (torch.LongTensor): 输入的token ID。
                   attention_mask (Optional[torch.Tensor]): 注意力掩码。
                   ... 其他参数描述 ...
                   position_3d_features (Optional[torch.Tensor]): 3D位置特征。

               Return:

               """
        #print("starting forward")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids.unsqueeze(0))
            if train_emb:
                # 生成索引
                indices = input_ids >= ori_vocab_size
                indices = (indices * 1).unsqueeze(-1)
                # 处理词嵌入
                inputs_embeds = (1 - indices) * inputs_embeds.detach() + indices * inputs_embeds
            # else:
            #     inputs_embeds = inputs_embeds.detach()

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                #print("n_image_tokens:",n_image_tokens) #648
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if position_3d_features is not None:

                vison_pad_id = 151654
                mask = input_ids == vison_pad_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                vision_mask = mask_expanded.to(inputs_embeds.device)

                position_3d_features = position_3d_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(vision_mask, position_3d_features)
                '''
                    # 生成额外tokens的嵌入向量
                    #print(inputs_embeds.size(),image_embeds.size(), position_3d_features.size())
                    additional_embeds = position_3d_features.unsqueeze(0)
                    # 找到image_embeds在inputs_embeds中的位置，假设image_embeds是连续的
                    image_embeds_end_index = torch.nonzero(mask).max().item() + 1
                    # 在image_embeds后面插入额外的嵌入向量
                    new_inputs_embeds = torch.cat([
                        inputs_embeds[:,:, :instruction_end_pos, :],
                        additional_embeds,
                        inputs_embeds[:,:, instruction_end_pos:, :]
                    ], dim=2)
                    inputs_embeds = new_inputs_embeds
                    
                # 更新位置标记
                if position_ids is not None:
                    # 假设position_ids是从0开始连续的
                    new_position_ids = torch.cat([
                        position_ids[:, :image_embeds_end_index],
                        torch.arange(image_embeds_end_index, image_embeds_end_index + additional_embeds.size(1), device=position_ids.device).unsqueeze(0).expand(position_ids.size(0), -1),
                        position_ids[:, image_embeds_end_index:] + additional_embeds.size(1)
                    ], dim=1)
                    position_ids = new_position_ids

                '''
                # if position_3d_features is not None:
                #     # 拼接3D position特征和视觉特征
                #     position_3d_features = position_3d_features.squeeze()
                #     print(image_embeds.size(), position_3d_features.size())
                #     if position_3d_features.shape[0] > image_embeds.shape[0]:
                #         position_3d_features = position_3d_features[:image_embeds.shape[0]]
                #     elif position_3d_features.shape[0] < image_embeds.shape[0]:
                #         pad_size = image_embeds.shape[0] - position_3d_features.shape[0]
                #         padding = torch.zeros(pad_size, position_3d_features.shape[1],
                #                               device=position_3d_features.device)
                #         position_3d_features = torch.cat([position_3d_features, padding], dim=0)
                #     combined_features = torch.cat([image_embeds, position_3d_features], dim=-1)
                #     # 通过MLP处理拼接后的特征
                #     processed_features = self.mlp(combined_features)
                #     inputs_embeds = inputs_embeds.masked_scatter(image_mask, processed_features)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                if position_3d_features is not None:
                    # 拼接3D position特征和视频特征
                    combined_features = torch.cat([video_embeds, position_3d_features], dim=-1)
                    # 通过MLP处理拼接后的特征
                    processed_features = self.mlp(combined_features)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, processed_features)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        inputs_embeds = inputs_embeds.squeeze(0)
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                #print("input_ids:",input_ids.size())
                '''
                if position_3d_features is not None:
                    virtual_token_id=1000
                    num_extra_tokens = additional_embeds.size(2)
                    batch_size = input_ids.shape[0]

                    # 创建虚拟 input_ids
                    virtual_input_ids = torch.full((batch_size, num_extra_tokens), virtual_token_id, dtype=torch.long).to(input_ids.device)

                    # 在第二个位置（索引为 1）拆分 input_ids
                    prefix = input_ids[:, :instruction_end_pos]
                    suffix = input_ids[:, instruction_end_pos:]

                    # 拼接张量
                    input_ids = torch.cat([prefix, virtual_input_ids, suffix], dim=1)
                '''
                # position_ids, rope_deltas = self.get_rope_index1(
                #     input_ids,
                #     image_grid_thw,
                #     video_grid_thw,
                #     second_per_grid_ts,
                #     attention_mask,
                #     additional_embeds.size(2) if position_3d_features is not None else 0
                # )
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask
                )
                #print("place1")
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                #print("place2")
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        #print("inputs_embeds shape:", inputs_embeds.size())
        
        #print("inputs_embeds:",inputs_embeds.size())
        #print("attention_mask:",attention_mask.size())
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            # print("logits:",logits.size())
            # print("logits:",torch.softmax(logits, dim=-1))
            # print("labels:",labels.size())
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # print("self.config.vocab_size:",self.config.vocab_size) #151765
            # print("ori_vocab_size:",ori_vocab_size) #151665
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )



if __name__ == "__main__":
    model_name_or_path = "/home/u2120220610/Chat-Scene-dev/Qwen2.5-VL-7B-Instruct"  # 替换为实际的预训练模型路径
    custom_model = qwen3d.from_pretrained(model_name_or_path)

    from PIL import Image
    import requests
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("/home/u2120220610/Chat-Scene-dev/Qwen2.5-VL-7B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ]
        },
        {"role": "assistant", "content": "this is a test answer."}
    ]
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("text:",text)
    inputs = processor(text=[text], images=[image], vision_infos=None)
    #position_3d_features = torch.randn(inputs.pixel_values.shape[0], 3)

    outputs = custom_model(input_ids=torch.tensor(inputs.input_ids), pixel_values=torch.tensor(inputs.pixel_values), position_3d_features=None, image_grid_thw=torch.tensor(inputs.image_grid_thw))