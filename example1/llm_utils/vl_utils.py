import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
import torch.nn.functional as F
import numpy as np
from qwen_vl_utils import process_vision_info
from PIL import Image


def load_vl_model(model_id, device, config=None):
    if config is None:
        config = AutoConfig.from_pretrained(model_id, cache_dir="cache_dir", trust_remote_code=True)
        setattr(config, '_attn_implementation', 'eager')
    vl_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, config=config).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return vl_model.visual, vl_model.model, vl_model, processor

def load_vl_2_5_model(model_id, device, config=None):
    if config is None:
        config = AutoConfig.from_pretrained(model_id, cache_dir="cache_dir", trust_remote_code=True)
        setattr(config, '_attn_implementation', 'eager')
    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, config=config).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return vl_model.visual, vl_model.model, vl_model, processor

def get_vl_inputs(processor, img_path=None, text="", device="cuda:0", img_w=640, img_h=640):
    messages = [{"role": "user", "content": [{"type": "text", "text": text},]}]
    # messages = [{"role": "user", "content": [{"type": "text", "text": "用中文描述这张图片"},]}]
    if img_path is not None:
        messages[0]["content"].insert(0, {"type": "image", "image": img_path, "resized_height": img_h, "resized_width": img_w})

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        # videos=video_inputs,
        # padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def get_vl_embeddings(visual, llm, inputs):
    input_ids = inputs["input_ids"]
    inputs_embeds = llm.embed_tokens(input_ids)

    pixel_values, image_grid_thw = inputs.get("pixel_values", None), inputs.get("image_grid_thw", None)

    if pixel_values is not None:
        pixel_values = pixel_values.type(visual.get_dtype())
        image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids == llm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return inputs_embeds



def get_grid_thw(processor, inp_h, inp_w):
    img = np.random.uniform(0, 255, (inp_h, inp_w, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    inputs = get_vl_inputs(processor, img, img_h=inp_h, img_w=inp_w)

    return inputs["image_grid_thw"]


def show_pixel_shape_and_grid_thw(model_id, img_shape):
    processor = AutoProcessor.from_pretrained(model_id)
    image_inputs = (np.random.random(img_shape) * 255).astype(np.uint8)
    # Preparation for inference
    inputs = processor(
        text=[""],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    print("pixel_values.shape =", pixel_values.shape)
    print("image_grid_thw =", image_grid_thw)


class VisualEmbeddingGenerator(torch.nn.Module):
    def __init__(self, visual_patch_embed, visual_blocks, visual_merger, visual_rot_pos_emb, grid_thw):
        super().__init__()
        self.patch_embed = visual_patch_embed
        self.blocks = visual_blocks
        self.merger = visual_merger
        self.rotary_pos_emb_ = visual_rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        self.cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # this forwrad gets the image pixel values that we get from the AutoProcessor when we pass the image and text (text -> input ids, and image-> pixel values)
    def forward(self, hidden_states):
        hidden_states = self.patch_embed(hidden_states)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=self.cu_seqlens, rotary_pos_emb=self.rotary_pos_emb_)
        return self.merger(hidden_states)

def get_veg(qwen2vl_model_path, img_h, img_w):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    Qwen2VLmodel = Qwen2VLForConditionalGeneration.from_pretrained(qwen2vl_model_path, ignore_mismatched_sizes=True)
    visual = Qwen2VLmodel.visual
    processor = AutoProcessor.from_pretrained(qwen2vl_model_path)

    grid_thw = get_grid_thw(processor, img_h, img_w)
    veg = VisualEmbeddingGenerator(visual.patch_embed, visual.blocks, visual.merger, visual.rot_pos_emb, grid_thw)
    return veg, processor

def get_veg_2_5(qwen2vl_model_path, img_h, img_w):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    Qwen2VLmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen2vl_model_path, ignore_mismatched_sizes=True)
    visual = Qwen2VLmodel.visual
    processor = AutoProcessor.from_pretrained(qwen2vl_model_path)

    grid_thw = get_grid_thw(processor, img_h, img_w)
    veg = VisualEmbeddingGenerator(visual.patch_embed, visual.blocks, visual.merger, visual.rot_pos_emb, grid_thw)
    return veg, processor

def get_embeddings(processor, llm_model, veg, prompt, img_path, img_h, img_w, R1=None):
    batch = {}
    batch['query'] = prompt
    batch['image_file'] = img_path
    # Append the system prompt first
    inputs = get_vl_inputs(processor, batch['image_file'], batch['query'], llm_model.device, img_w, img_h)

    # Extract image features from VEG
    # ref:/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py
    inputs_embeds = llm_model.embed_tokens(inputs['input_ids'].to(llm_model.device))
    pixel_values = inputs['pixel_values']
    with torch.no_grad():
        image_embeds = veg(pixel_values)

    # adapt to SpinQuant
    if R1 is not None:
        image_embeds = image_embeds.to(device="cuda", dtype=torch.float64)
        image_embeds = image_embeds - image_embeds.mean(dim=-1, keepdim=True)
        image_embeds = torch.matmul(image_embeds, R1).to(dtype=torch.float32)

    image_mask = ((inputs['input_ids'] == llm_model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device))
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)

    return {'input_embeddings': inputs_embeds.squeeze(dim=0), 'attention_mask': attention_mask.squeeze(dim=0),   'position_ids': None}

