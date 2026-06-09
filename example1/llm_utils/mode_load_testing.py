import sys
import os
import copy
from tqdm import tqdm

workfolder = os.getcwd()
sys.path.append(workfolder+'/../../../example1/verify_script')

from infer_defender_videos import Infrencer,read_jsonl_file
from builder import build_mllm_model
from transformers import GenerationConfig
from PIL import Image
import torch
# chatbot = Infrencer("/prj/qct/aicechina_scratch/ruzhongl/llm/vlm/NIO-Qwen2.5VL/example1/verify_script/configs/model_config_1_5B_175v3.yaml", batch_size=1)
# data_list = read_jsonl_file("/prj/qct/aicechina_scratch/ruzhongl/llm/vlm/NIO-Qwen2.5VL/example1/verify_script/defender_video_test.jsonl")
# correct = chatbot.infer_list(data_list)

def parse_to_batch(data_list):
    batch_size = 1
    batch_list = []
    counter = 0
    temp_batch_questions = []
    temp_batch_images = []
    temp_batch_ids = []
    for sample in data_list:
        temp_batch_questions.append("<image>"+sample["prompt"])
        print(temp_batch_questions)
        temp_sample_images = []
        for frame in sorted(sample["frames"], key=lambda x: x["index"]):
            temp_sample_images.append(frame["url"])
        temp_batch_images.append(temp_sample_images)
        temp_batch_ids.append(sample["id"])
        counter += 1
        if counter >= batch_size:
            batch_list.append({
                "prompts": copy.deepcopy(temp_batch_questions),
                "frames": copy.deepcopy(temp_batch_images),
                "ids": copy.deepcopy(temp_batch_ids)
                })
            counter = 0
            temp_batch_questions = []
            temp_batch_images = []
            temp_batch_ids = []
    print(batch_list)
    return batch_list

def infer_list(model,data_list):
    batch_list = self.parse_to_batch(data_list)
    answer_list = []
    id_list = []
    for item in tqdm(batch_list):
        answers = model.batch_chat_video(item["frames"], item["prompts"])
        print(answers)
        answer_list += answers
        id_list += item["ids"]

def get_embeddings(model,sample_paths,sample_questions):
    batch_video_embeddings = []
    device = 'cuda' 
    dtype = torch.bfloat16
    # import ipdb
    # ipdb.set_trace()  
    for sample_frames in sample_paths:
        frames_paths = sample_frames if type(sample_frames) is list else [sample_frames]
        frames = [Image.open(frame_path).convert('RGB') for frame_path in frames_paths]
        frame_tensors = [model.process_image(frame, data_type='frame') for frame in frames]
        
        frame_tensors = torch.stack(frame_tensors).to(device).to(dtype)
        # [num_frames, 3, 384, 384]
        frame_features = model.encode_images(frame_tensors) # [num_frames, 144, llm_dim]
        batch_video_embeddings.append(frame_features)
            
    inputs_embeds, attention_masks, max_length = [], [], 0
    
    for i in range(len(sample_questions)):
        input_id = []
        question = sample_questions[i]
        prefix_id = torch.tensor(model.tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n")).to(device) # torch.tensor([151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198]).to(self.device)
        input_id.append(prefix_id)
        question_embed_chunks = [model.language_model.get_input_embeddings()(prefix_id).to(device)]
        
        question_chunks = question.split("<image>")
        assert len(question_chunks) == 2 # Only 1 video in one conversation.
        question_part1_id = torch.tensor(model.tokenizer.encode(question_chunks[0]+"<img>")).to(device)
        question_embed_chunks.append(model.language_model.get_input_embeddings()(question_part1_id).to(device))
        frame_num = batch_video_embeddings[i].shape[0]
        
        for frame_idx in range(frame_num):
            frame_idx_id = torch.tensor(model.tokenizer.encode(f'<frame-{frame_idx+1}>')).to(device)
            question_embed_chunks.append(model.language_model.get_input_embeddings()(frame_idx_id).to(device))
            question_embed_chunks.append(batch_video_embeddings[i][frame_idx])
        
        suffix_id = torch.tensor(model.tokenizer.encode("</img>"+question_chunks[1]+"<|im_end|>\n<|im_start|>assistant\n")).to(device) # torch.tensor([151645, 198, 151644, 77091, 198]).to(self.device) # <|im_end|>\n<|im_start|>assistant\n
        input_id.append(suffix_id)
        input_id = torch.cat(input_id, dim=0)
        
        question_embed_chunks.append(model.language_model.get_input_embeddings()(suffix_id).to(device))
        inputs_embed = torch.cat(question_embed_chunks, dim=0)
            
        inputs_embeds.append(inputs_embed)
        attention_masks.append(torch.ones(inputs_embed.shape[0], dtype=torch.bool))
        max_length = max(inputs_embed.shape[0], max_length)
    
    pad_embed = model.language_model.get_input_embeddings()(torch.tensor(model.tokenizer.pad_token_id).to(device))
    # make sure same sequence length in batch
    inputs_embeds_list = [torch.cat([pad_embed.repeat((max_length - inputs_embed.shape[0]),1), inputs_embed]) for inputs_embed in inputs_embeds]
    attention_masks_list = [torch.cat([torch.tensor([False] * (max_length - attention_mask.shape[0]), dtype=torch.bool), attention_mask]) for attention_mask in attention_masks]
    inputs_embeds = torch.stack(inputs_embeds_list).to(device)
    attention_masks = torch.stack(attention_masks_list).to(device)

    return {"input_embeds":inputs_embeds,
            "attention_masks":attention_masks}

model = build_mllm_model('/prj/qct/aicechina_scratch/ruzhongl/llm/vlm/NIO-Qwen2.5VL/example1/verify_script/configs/model_config_1_5B_175v3.yaml').eval()
# model.tokenizer.save_pretrained("./tokenizer")

# tokenizer_json = model.tokenizer.to_json_string()
# with open("./tokenizer/tokenizer.json", "w", encoding="utf-8") as f:
#     f.write(tokenizer_json)

# assert 0
data_list = read_jsonl_file("/prj/qct/aicechina_scratch/ruzhongl/llm/vlm/NIO-Qwen2.5VL/example1/verify_script/defender_video_test.jsonl")
generation_config = GenerationConfig.from_pretrained("/prj/qct/aicechina_scratch/ruzhongl/llm/vlm/NIO-Qwen2.5VL/example1/verify_script/configs", "qwen2_generation_config.json")

batch_list = parse_to_batch(data_list)
for item in tqdm(batch_list):
    # answers = model.batch_result_video(item["frames"], item["prompts"],generation_config = generation_config)
    inputs = get_embeddings(model,item["frames"],item["prompts"])
    input_embeds = inputs['input_embeds'][0]
    input_mask =  inputs["attention_masks"][0]
    print(input_embeds.shape)
    pass



