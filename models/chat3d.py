import random
import logging
from abc import ABC
from PIL import Image
import requests
from .qwen3d import qwen3d
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoProcessor
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from models.position_embedding import PositionEmbeddingCoordsSine
from peft import LoraConfig, get_peft_model
# from models.load_llama import init_llama_model
from torch.nn.utils.rnn import pad_sequence

import contextlib
from dataset.base_dataset import update_caption, recover_caption

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


class Chat3D(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config


        model_name_or_path = config.model.qwen_model_path #"/home/u2120220610/Chat-Scene-dev/Qwen2.5-VL-7B-Instruct"  # 替换为实际的预训练模型路径
        self.custom_model = qwen3d.from_pretrained(model_name_or_path,
                                                   torch_dtype=torch.bfloat16#,
                                                   #attn_implementation="flash_attention_2"
                                                   )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.resolution = config.resolution

        for param in  self.custom_model.parameters():
            param.requires_grad = False

        if hasattr(self.custom_model, 'lm_head'):
            for param in self.custom_model.lm_head.parameters():
                param.requires_grad = True
                print(f"Unfroze parameter: {param.shape}")
        else:
            print("Warning: Model does not have 'lm_head' module")
        #llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.end_sym = config.model.end_sym
        self.end_sym = "<|im_end|>"
        self.system_path = config.model.system_path
        self.instruction_path = config.model.instruction_path
        self.role = config.model.role
        self.no_obj = config.model.no_obj
        self.add_scene_token = config.model.add_scene_token
        self.add_img_token = config.model.add_img_token
        self.train_emb = config.model.train_emb
        self.train_img_proj = config.model.train_img_proj
        self.input_dim = config.model.input_dim
        self.img_input_dim = config.model.img_input_dim
        self.attr_dim = config.model.attr_dim
        self.scene_dim = config.model.scene_dim
        self.pos_dim = config.model.pos_dim
        self.max_obj_num = config.model.max_obj_num
        self.bidirection = config.model.bidirection
        self.add_pos_emb = config.model.add_pos_emb
        self.feat_fusion = config.model.feat_fusion
        self.fuse_with_id = config.model.fuse_with_id
        self.use_location_token = config.model.use_location_token

        self.debug = config.debug
        self.llama_dim = self.custom_model.vision_output_dim

        if config.model.use_lora:
            def find_linear_layers(model, lora_target_modules):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    if (
                        isinstance(module, cls)
                        and all(
                            [
                                x not in name
                                for x in [
                                    "instance2embed",
                                    "hidden_state2query"
                                ]
                            ]
                        )
                        and any([x in name for x in lora_target_modules])
                    ):
                        lora_module_names.add(name)
                return sorted(list(lora_module_names))
        
            lora_target_modules = find_linear_layers(self.custom_model, config.lora.lora_target_modules)

            lora_config = LoraConfig(
                r=config.lora.lora_r//2,
                lora_alpha=config.lora.lora_alpha//2,
                target_modules=lora_target_modules,
                lora_dropout=config.lora.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.custom_model = get_peft_model(self.custom_model, lora_config)
            self.custom_model.print_trainable_parameters()
            self.custom_model.model.lm_head.weight.requires_grad = True
            self.custom_model.model.lm_head.weight.data = self.custom_model.model.lm_head.weight.data.float()
            self.custom_model.print_trainable_parameters()
            self.custom_model.model.model.embed_tokens.weight.requires_grad = True
            self.custom_model.model.model.embed_tokens.weight.data = self.custom_model.model.model.embed_tokens.weight.data.float()
            self.custom_model.print_trainable_parameters()
            #self.custom_model.gradient_checkpointing_enable()
        else:
            self.custom_model.lm_head.weight.requires_grad = True
            self.custom_model.lm_head.weight.data = self.custom_model.lm_head.weight.data.float()
            self.custom_model.model.embed_tokens.weight.requires_grad = True
            self.custom_model.model.embed_tokens.weight.data = self.custom_model.model.embed_tokens.weight.data.float()



        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.pos_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.pos_dim, self.llama_dim)
        )
        self.objid_tokens = []
        for i in range(self.max_obj_num):
            self.objid_tokens.append(f"<OBJ{i:03}>")
        self.objid_start_idx = self.ori_vocab_size = len(self.processor.tokenizer)
        # print("objid_start_idx:",self.objid_start_idx)s
        self.processor.tokenizer.add_tokens(self.objid_tokens, special_tokens=False)
        self.objid_end_idx = len(self.processor.tokenizer)
        # print("objid_end_idx:",self.objid_end_idx)
        self.custom_model.resize_token_embeddings(len(self.processor.tokenizer))


                

        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        with open(self.instruction_path, "r") as f:
            self.instruction = "\n".join([x.strip() for x in f.readlines()])

        # if not self.debug:
        #     self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()
        self.last_embed = None
        

    def encode_object_feat(self, feat, img_feat, locs):
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        #print("img_feat:", img_feat.shape)
        return feat, img_feat
    
    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn


    def prepare_fixed_embed(self):
        prompt = self.system + " " + self.instruction + " " + self.role[0] + ": " 
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed

    def get_object_list_embed(self, embed_obj, embed_img, embed_scene, scene_mask, obj_id, assigned_ids):
        valid_ids = torch.where(scene_mask)[0].tolist()
        #print("valid_ids:",valid_ids)
        if not self.config.model.use_lora:
            objid_embeds = self.custom_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.custom_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]


        assigned_ids = assigned_ids[valid_ids]
        #print("assigned_ids:",assigned_ids)
        if not self.train_emb:
            objid_embeds = objid_embeds.detach()
        selected_objid_embeds = objid_embeds[valid_ids]
        if self.use_location_token:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] += embed_obj[assigned_ids]
            object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        if self.fuse_with_id:
            object_list_embed = selected_objid_embeds
            if not self.no_obj:
                object_list_embed += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed += embed_img[assigned_ids]
            return object_list_embed
        if self.feat_fusion:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            if not self.no_obj:
                object_list_embed[1::2, :] += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        if self.no_obj:
            # if embed_img is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_img[assigned_ids]
            # else:
            #     object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            #     object_list_embed[0::3, :] = selected_objid_embeds
            #     object_list_embed[1::3, :] = embed_scene[assigned_ids]
            #     object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_obj[assigned_ids]
            return object_list_embed
            # object_list_embed = selected_objid_embeds + embed_obj[assigned_ids]
        if embed_img is None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_scene[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0], selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            #object_list_embed[0::2, :] = selected_objid_embeds
            # print("selected_objid_embeds:",selected_objid_embeds)
            #object_list_embed[1::2, :] = embed_obj[assigned_ids]
            #object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 4, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::4, :] = selected_objid_embeds
            object_list_embed[1::4, :] = embed_obj[assigned_ids]
            object_list_embed[2::4, :] = embed_scene[assigned_ids]
            object_list_embed[3::4, :] = embed_img[assigned_ids]
            return object_list_embed
        return object_list_embed
    
    def get_objid_embeds(self):
        if self.config.model.use_lora:
            objid_embeds = self.custom_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.custom_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        return objid_embeds

    def get_min_max_coord(self, xyz, scene_mask):
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def forward_train(self,image_paths, scene_feat, scene_img_feat, scene_locs, scene_mask, obj_ids, assigned_ids, questions, answers, is_eval=False, **kwargs):




        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed) #BX100X3584
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        proj_scene_embed = None
        if self.add_scene_token:  # remember to change the evaluate 
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)
            #print('proj_scene_embed', proj_scene_embed.shape)
        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        # p_0_embed = self.p_0_embed.to(device)
        # p_1_embed = self.p_1_embed.to(device)
        object_list_intervals = []
        images_for_processor = []
        prompts_for_tokenizer = []
        inputs_list = []

        inputs_id_list = []
        messages = []
        answerlist = []
        attention_mask_list = []
        pixel_values_list = None
        image_grid_thw_list = None
        proj_object_embed_list = None
        target_list = []
        max_seq_len = 0
        

        for i, question in enumerate(questions):
            valid_ids = torch.where(scene_mask[i])[0].tolist()
    
            assigned_ids_new = assigned_ids[i][valid_ids]
            id_str = ""
            # for obj_id in self.objid_tokens:
            #     id_str = id_str + obj_id + '<|vision_start|><|vision_pad|><|vision_end|>'
            
            # new_object_embed = self.get_object_list_embed(
            #     proj_object_embed[i], 
            #     proj_object_img_embed[i] if self.add_img_token else None, 
            #     proj_scene_embed[i] if self.add_scene_token else None, 
            #     scene_mask[i],
            #     obj_ids[i],
            #     assigned_ids[i]
            # )
            new_object_embed = torch.zeros((proj_object_embed.shape[1], proj_object_embed.shape[2]), dtype=proj_object_embed.dtype, device=proj_object_embed.device)
            for k,assigned_id in enumerate(assigned_ids_new):
                new_object_embed[k,:] = proj_object_embed[i][assigned_id]
                id_str = id_str + self.objid_tokens[assigned_id] + '<|vision_start|><|vision_pad|><|vision_end|>'
            # print("proj_object_embed:",proj_object_embed.size())
            if proj_object_embed_list is None:
                proj_object_embed_list = new_object_embed.unsqueeze(0)
            else:
                proj_object_embed_list = torch.cat([proj_object_embed_list,new_object_embed.unsqueeze(0)],dim=0)

            current_image_paths = image_paths[i]
            question_text = questions[i]

            content = []

            # 循环添加图像项
            for current_image_path in current_image_paths:
                content.append({"type": "image","image":current_image_path,"resized_height": self.resolution, "resized_width": self.resolution})

            # 添加文本项
            content.append({"type": "text", "text": question_text})
            
            # 构造 message 列表
            message = [
                {"role": "system", "content": self.system + " " + self.instruction + id_str},
                {
                    "role": "user",
                    "content": content
                },
                {"role": "assistant", "content": answers[i]}
            ]
            messages.append(message)
            #print("messages:",messages)

            answer = [
                {"role": "assistant", "content": answers[i] + self.end_sym}
            ]
            # answerlist.append(answer)
            #print("answer:",answer)
            #prompt_answer = self.processor.apply_chat_template(answer, tokenize=False, add_generation_prompt=False)
            # # 获取<im_start>的token_id
            # im_start_id = self.processor.tokenizer.convert_tokens_to_ids(["<|im_start|>"])[0]
            # #print('im_start_id:',im_start_id)
            # im_start_positions = (input_ids[0] == im_start_id).nonzero(as_tuple=True)[0]
            '''
            prompt_answer = answers[i] + self.end_sym
            answer_token = self.processor(text=[prompt_answer])
            answer_id = torch.tensor(answer_token['input_ids']).to(device)
            #print("answer_id:",answer_id)
            answerlist.append(answer_id)
            '''
        #4.23
        '''
        instruction_text = [{"role": "system", "content": self.system + " " + self.instruction}]
        instruction_text = self.processor.apply_chat_template(instruction_text, tokenize=False, add_generation_prompt=False)
        #print("instruction_text:",instruction_text)
        instruction_tokens = self.processor(text=[instruction_text])
        instruction_end_pos = len(instruction_tokens) - 1
        '''
        #print("instruction_end_pos",instruction_end_pos)
        #4.23
        #add in 4.22
        image_inputs, video_inputs = process_vision_info(messages)
        texts = [
        self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
        ]
        #print("texts:",texts)
        inputs = self.processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        )
        inputs = inputs.to(device)

        input_ids = inputs.get('input_ids')
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')
        attention_mask = inputs.get('attention_mask')
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids(["<|im_start|>"])[0]
        batch_size = input_ids.size(0)
        labels = torch.full_like(input_ids, -100)  # 初始化为-100（忽略位置）
        third_position_list =[]
        for i in range(batch_size):
            # 查找第三个<im_start>的位置
            positions = (input_ids[i] == im_start_id).nonzero(as_tuple=True)[0]
            if len(positions) >= 3:
                third_position = positions[-1].item()+3
                third_position_list.append(third_position)
                # 从第三个<im_start>开始到序列结束的token作为labels
                labels[i, third_position:] = input_ids[i, third_position:]
        targets = labels.to(device)
        '''
        pos_3d_mask = torch.ones((batch_size, 200), dtype=torch.long, device=device)
        new_attention_mask = torch.cat([
            attention_mask[:, :instruction_end_pos],
            pos_3d_mask,
            attention_mask[:, instruction_end_pos:],
        ], dim=1)
        '''
        '''
        target_list = None
        for answer_id in answerlist:
            padding_length = input_ids.size(1)  - answer_id.size(1) #+ 200

            # 生成填充张量
            # 形状为 (batch_size, padding_length)，值全为 -100
            padding_tensor = torch.full((1, padding_length), -100, dtype=answer_id.dtype, device=answer_id.device)

            # 在第二个维度上拼接填充张量和 target_id
            target_id = torch.cat([padding_tensor, answer_id], dim=1)
            #print("target_id:",target_id.size())
            if target_list == None:
                target_list = target_id
            else:
                target_list = torch.cat([target_list, target_id], dim=0)
            #target_list.append(torch.squeeze(target_id))
        targets = target_list.to(device)
        #add in 4.22
        '''




        with self.maybe_autocast(dtype=self.custom_model.dtype):

            outputs = self.custom_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_3d_features=proj_object_embed_list,
                labels=targets, # Standard for Causal LM loss
                return_dict=True,
                #instruction_end_pos = instruction_end_pos,
                train_emb=self.train_emb,
                ori_vocab_size = self.ori_vocab_size
            )
        '''
        logits= outputs.logits

        # print("logits:",logits.size())
        # print("self.objid_end_idx:",self.objid_end_idx)
        probs = torch.softmax(logits, dim=-1)

        predicted_token_ids = torch.argmax(probs, dim=-1)
        # print("predicted_token_ids:",predicted_token_ids.size())
        # print("answer_id.size(1):",answer_id.size(1))
        #predicted_token_ids = [predicted_token_id[-(answer_id.size(1)+1):] for predicted_token_id in predicted_token_ids]
        # predicted_token_id_pre = [predicted_token_id[:-(answer_id.size(1)+1)] for predicted_token_id in predicted_token_ids]
        # print("predicted_token_ids_pre:",predicted_token_id_pre)
        # predict=self.processor.batch_decode(
        #         predicted_token_id_pre, skip_special_tokens=False, clean_up_tokenization_spaces=False
        #     )
        # print("train_predict_pre:",predict)
        predicted_token_back = [predicted_token_id[third_position-1:] for predicted_token_id,third_position in zip(predicted_token_ids,third_position_list)]
        print("predicted_token_ids:",predicted_token_back)
        predict=self.processor.batch_decode(
                predicted_token_back, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        print("train_predict:",predict)
        target = [predicted_token_id[third_position:] for predicted_token_id,third_position in zip(targets,third_position_list)]
        target_text = self.processor.batch_decode(
                target, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        print("target_token_ids:",target)
        print("target_text:",target_text)
        #print("gt_answers:",answers)
        '''
        return dict(
            loss=outputs.loss,
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
            obj_img_norm=proj_object_img_embed.norm(dim=-1).mean().detach().cpu(),
            objid_norm=self.get_objid_embeds().norm(dim=-1).mean().detach().cpu(),
            scene_norm=proj_scene_embed.norm(dim=-1).mean().detach().cpu() if proj_scene_embed is not None else 0.,
            max_seq_len=max_seq_len
        )

    def evaluate(self, image_paths, scene_feat, scene_img_feat, scene_locs, scene_mask, custom_prompt, obj_ids, assigned_ids, is_eval=True, **kwargs):
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        proj_scene_embed = None
        if self.add_scene_token:  # remember to change the evaluate 
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)
            #print('proj_scene_embed', proj_scene_embed.shape)

        output_texts = []
        # p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        # p_1_embed = self.p_1_embed.to(device).unsqueeze(0)
        for i in range(batch_size):



            valid_ids = torch.where(scene_mask[i])[0].tolist()
    
            assigned_ids_new = assigned_ids[i][valid_ids]
            id_str = ""
            # for obj_id in self.objid_tokens:
            #     id_str = id_str + obj_id + '<|vision_start|><|vision_pad|><|vision_end|>'
            #print('id_str:',id_str)
            # new_object_embed = self.get_object_list_embed(
            #     proj_object_embed[i], 
            #     proj_object_img_embed[i] if self.add_img_token else None, 
            #     proj_scene_embed[i] if self.add_scene_token else None, 
            #     scene_mask[i],
            #     obj_ids[i],
            #     assigned_ids[i]
            # )
            new_object_embed = torch.zeros((proj_object_embed.shape[1], proj_object_embed.shape[2]), dtype=proj_object_embed.dtype, device=proj_object_embed.device)
            for k,assigned_id in enumerate(assigned_ids_new):
                new_object_embed[k,:] = proj_object_embed[i][assigned_id]
                id_str = id_str + self.objid_tokens[assigned_id] + '<|vision_start|><|vision_pad|><|vision_end|>'
            # print("proj_object_embed:",proj_object_embed.size())
            proj_object_embed_list = new_object_embed.unsqueeze(0)
            # if proj_object_embed_list is None:
            #     proj_object_embed_list = new_object_embed.unsqueeze(0)
            # else:
            #     proj_object_embed_list = torch.cat([proj_object_embed_list,new_object_embed.unsqueeze(0)],dim=0)




            current_image_paths = image_paths[i]
            question_text = custom_prompt[i]
            # print("question_text:",question_text)
            content = []

            # 循环添加图像项
            for current_image_path in current_image_paths:
                content.append({"type": "image","image":current_image_path,"resized_height": self.resolution, "resized_width": self.resolution})

            # 添加文本项
            content.append({"type": "text", "text": question_text})
            
            # 构造 message 列表
            message = [
                {"role": "system", "content": self.system + " " + self.instruction + id_str},
                {
                    "role": "user",
                    "content": content
                }
            ]
            image_inputs, video_inputs = process_vision_info(message)
            texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
            #print("texts:",texts)
            inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            )
            inputs = inputs.to(device)
            proj_object_embedi = proj_object_embed[i].unsqueeze(0)
            #print("proj_object_embed:",proj_object_embedi.size())
            input_ids = inputs.get('input_ids')
            pixel_values = inputs.get('pixel_values')
            image_grid_thw = inputs.get('image_grid_thw')
            attention_mask = inputs.get('attention_mask')
            
            '''
            instruction_text = [{"role": "system", "content": self.system + " " + self.instruction}]
            instruction_text = self.processor.apply_chat_template(instruction_text, tokenize=False, add_generation_prompt=False)
            instruction_tokens = self.processor(text=[instruction_text])
            instruction_end_pos = len(instruction_tokens)
            pos_3d_mask = torch.ones((1, 200), dtype=torch.long, device=device)
            new_attention_mask = torch.cat([
                attention_mask[:, :instruction_end_pos],
                pos_3d_mask,
                attention_mask[:, instruction_end_pos:],
            ], dim=1)
            '''

            
            
            #print(wrapped_embed.shape)
            with self.maybe_autocast():
                outputs = self.custom_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_3d_features=proj_object_embed_list,
                #labels=targets, # Standard for Causal LM loss
                #return_dict=False,
                #instruction_end_pos = instruction_end_pos
            )
            # print("outputs:",outputs.size())
            #output_token = outputs[0]
            # for in_ids, out_ids in zip(input_ids, outputs):
            #     print("in_ids:",in_ids.size())
            #     print("out_ids:",out_ids.size())
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, outputs)
            ]
            # generated_ids_trimmed = [
            #     out_ids for in_ids, out_ids in zip(input_ids, outputs)
            # ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            #output_text = self.processor.batch_decode(outputs)
            #print("output_text:",output_text)
            for out_text in output_text:
                print("output_text:",out_text)
                out_text = out_text.split(self.end_sym)[0]
                out_text = out_text.replace('  ', ' ').replace(' .', '.').strip()
                out_text = recover_caption(out_text, assigned_ids[i].tolist())
                #print("final_output_text:",out_text)
                output_texts.append(out_text)
        return output_texts

    def forward(self, **kwargs):
        if "answers" in kwargs:
            return self.forward_train(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
