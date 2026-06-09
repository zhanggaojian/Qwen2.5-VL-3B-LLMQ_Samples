# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""  utility method to evaluate perplexity score on WikiText """
from itertools import chain
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from datasets import IterableDataset, load_dataset
from transformers import default_data_collator
import json
import torch
from PIL import Image
import os
from .vl_utils import get_vl_inputs, get_grid_thw
from qwen_vl_utils.vision_process import fetch_image 
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

class Qwen_MROPE_Index:
    def __init__(self, image_grid_thw, generation:Qwen2VLForConditionalGeneration):
        self.image_grid_thw = image_grid_thw
        self.generation = generation
    
    def get_rope_idx(self,  input_ids, attention_mask):
        position_ids, rope_deltas = self.generation.get_rope_index(input_ids, self.image_grid_thw, None, attention_mask)
        return position_ids


class VisualEmbeddingGenerator(torch.nn.Module):

    def __init__(self, visual_patch_embed, visual_blocks, visual_merger, visual_rot_pos_emb, grid_thw, device):
        super().__init__()
        self.patch_embed = visual_patch_embed
        self.blocks = visual_blocks
        self.merger = visual_merger

        self.device = device
        self.rotary_pos_emb_ = visual_rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        self.cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # this forwrad gets the image pixel values that we get from the AutoProcessor when we pass the image and text (text -> input ids, and image-> pixel values)
    def forward(self, hidden_states):
        hidden_states = self.patch_embed(hidden_states.to(self.device))
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=self.cu_seqlens, rotary_pos_emb=self.rotary_pos_emb_)
        return self.merger(hidden_states)


class QwenDataset(Dataset):

    def __init__(self, model, visual, processor, dataset_path, json_file_path, emb_length, calibration, img_h, img_w, R1_path=None, num_test_batches=300, mrope_position:Qwen_MROPE_Index=None):
        self.img_h = img_h
        self.img_w = img_w
        self.model = model
        self.visual = visual
        self.R1_path = R1_path
        self.visual.eval()
        self.processor = processor
        self.calibration = calibration
        self.dataset_path = dataset_path
        self.json_file_path = json_file_path
        self.emb_length = emb_length
        self.mrope_position = mrope_position
        with open(json_file_path) as file:
            self.json_file = json.load(file)
        self.num_test_batches = num_test_batches

    def __len__(self):
        if self.calibration:
            return len(self.json_file)
        else:
            return self.num_test_batches

    def __getitem__(self, idx):
        json_data = self.json_file[idx] if idx < len(self.json_file) else self.json_file[-1]
        batch = {}
        batch['query'] = json_data
        # Add 640 image here
        batch['image_file'] = os.path.join(self.dataset_path, "coco/train2017/000000000025.jpg")
        if self.calibration:
            try:
                batch['image_file'] = os.path.join(self.dataset_path, json_data['image'])
                batch['query'] = json_data['conversations'][2]['value']
            except:
                batch['query'] = "Please create a recipe for me with these ingredients."

        # Append the system prompt first

        inputs = get_vl_inputs(self.processor, batch['image_file'], batch['query'], self.model.device, self.img_w, self.img_h)

        # Extract image features from VEG
        # ref:/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py
        inputs_embeds = self.model.embed_tokens(inputs['input_ids'].to(self.model.device))
        pixel_values = inputs['pixel_values']
        with torch.no_grad():
            image_embeds = self.visual(pixel_values)

        # adapt to SpinQuant
        if self.R1_path is not None:
            image_embeds = image_embeds.to(device="cuda", dtype=torch.float64)
            image_embeds = image_embeds - image_embeds.mean(dim=-1, keepdim=True)
            R1 = torch.load(self.R1_path)["R1"].cuda().to(torch.float64)
            image_embeds = torch.matmul(image_embeds, R1).to(dtype=torch.float32)

        image_mask = ((inputs['input_ids'] == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device))
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        labels = inputs['input_ids']
        # ref:/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py

        emb_length = self.emb_length  # ARN
        #'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|im_start|>assistant\n'
        system_start_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        system_start = self.processor.tokenizer(system_start_prompt, return_tensors="pt", add_special_tokens=False)
        system_start_prompt_length = len(system_start["input_ids"][0])  #14

        #'<|im_end|><|im_start|>assistant\n'
        system_end_prompt = '<|im_end|>\n<|im_start|>assistant\n'
        system_end = self.processor.tokenizer(system_end_prompt, return_tensors="pt", add_special_tokens=False)
        system_end_prompt_length = len(system_end["input_ids"][0])  #5

        #'<|vision_start|><|image_pad|>*529<|vision_end|>' 529=image_pad_num
        # image_pad_num = (inputs['input_ids'] == torch.tensor(151655)).sum().item()
        image_pad_num = (inputs['input_ids'] == torch.tensor(self.processor.tokenizer('<|image_pad|>')['input_ids'][0])).sum().item()
        image_prompt = f"{'<|vision_start|>'}{'<|image_pad|>' * image_pad_num}{'<|vision_end|>'}"
        image_encoder = self.processor.tokenizer(image_prompt, return_tensors="pt", add_special_tokens=False)
        image_prompt_length = len(image_encoder["input_ids"][0])  #529 +2

        if self.calibration:
            repeated_emb = inputs_embeds
            repeated_labels = labels
            while repeated_emb.shape[1] < emb_length:
                repeated_emb = torch.cat([repeated_emb, inputs_embeds], dim=1)
                repeated_labels = torch.cat([repeated_labels, labels], dim=1)
            inputs_embeds = repeated_emb[:, :emb_length, :].contiguous()
            labels = repeated_labels[:, :emb_length].contiguous()
        else:
            # for ppl, we remove the sys prompt + image embeddings, and use whatever actual wikitext is after that
            cut_embeds = inputs_embeds[:, (system_start_prompt_length + image_prompt_length):, :].contiguous()
            cut_labels = labels[:, (system_start_prompt_length + image_prompt_length):].contiguous()
            if cut_embeds.shape[1] > emb_length:
                # for llava llm, the begiinig 37 is the fixed prompt embedding, the following 576 is the image embedding.
                inputs_embeds = torch.cat([cut_embeds[:, :(emb_length - system_end_prompt_length), :], cut_embeds[:, -system_end_prompt_length:, :]], dim=1)
                labels = torch.cat([cut_labels[:, :(emb_length - system_end_prompt_length)], cut_labels[:, -system_end_prompt_length:]], dim=1)
            else:
                repeated_emb = cut_embeds
                repeated_labels = cut_labels
                while repeated_emb.shape[1] < emb_length:
                    repeated_emb = torch.cat([repeated_emb, cut_embeds], dim=1)
                    repeated_labels = torch.cat([repeated_labels, cut_labels], dim=1)
                inputs_embeds = repeated_emb[:, :emb_length, :].contiguous()
                labels = repeated_labels[:, :emb_length].contiguous()

        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)
        position_ids = None
        if self.mrope_position is not None:
            position_ids = self.mrope_position.get_rope_idx(labels, attention_mask)
        return {'input_embeddings': inputs_embeds.squeeze(dim=0), 'attention_mask': attention_mask.squeeze(dim=0), 'labels': labels.squeeze(dim=0), 'position_ids': position_ids}



def get_vl_dataset(llm_model, dataset_setting, num_test_batches, is_qwen): 
    emb_length = dataset_setting['emb_length']
    device = dataset_setting['device']
    qwen2vl_model_path = dataset_setting['qwen2vl_model_path']
    calibration_dataset_path = dataset_setting['calibration_dataset_path']
    ppl_evaluation_dataset_path = dataset_setting['ppl_evaluation_dataset_path']
    image_dataset_path = dataset_setting['image_dataset_path']
    R1_path = dataset_setting["R1_path"]
    img_w, img_h = dataset_setting["img_w"], dataset_setting["img_h"]
    use_mrope = dataset_setting.get("use_mrope", False)
    Qwen2VLmodel = Qwen2VLForConditionalGeneration.from_pretrained(qwen2vl_model_path, ignore_mismatched_sizes=True).to(device)
    

    visual = Qwen2VLmodel.visual
    # llm_model = Qwen2VLmodel.model
    processor = AutoProcessor.from_pretrained(qwen2vl_model_path)

    grid_thw = get_grid_thw(processor, img_h, img_w)  #torch.tensor([[1, 46, 46]])

    mrope_position:Qwen_MROPE_Index=None
    if use_mrope:
        mrope_position = Qwen_MROPE_Index(grid_thw, Qwen2VLmodel)

    veg = VisualEmbeddingGenerator(visual.patch_embed, visual.blocks, visual.merger, visual.rot_pos_emb, grid_thw, device)

    DatasetCls = QwenDataset
    # 构建数据集
    dataset = {}
    dataset_train = DatasetCls(llm_model,
                               veg,
                               processor,
                               image_dataset_path,
                               calibration_dataset_path,
                               emb_length,
                               calibration=True,
                               img_h=img_h,
                               img_w=img_w,
                               R1_path=R1_path,
                               num_test_batches=num_test_batches,
                               mrope_position=mrope_position)

    dataset_test = DatasetCls(llm_model,
                              veg,
                              processor,
                              image_dataset_path,
                              ppl_evaluation_dataset_path,
                              emb_length,
                              calibration=False,
                              img_h=img_h,
                              img_w=img_w,
                              R1_path=R1_path,
                              num_test_batches=num_test_batches,
                              mrope_position=mrope_position)
    dataset['train'] = dataset_train
    dataset['test'] = dataset_test

    def custom_collate_fn(batch):
        return batch

    train_dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=1, collate_fn=default_data_collator)
    test_dataloader = DataLoader(dataset['test'], shuffle=False, batch_size=1, collate_fn=default_data_collator)
    del llm_model, veg, Qwen2VLmodel
    return train_dataloader, test_dataloader, dataset


def get_qwen_dataset(llm_model, dataset_setting, num_test_batches):
    return get_vl_dataset(llm_model, dataset_setting, num_test_batches, True)
    
