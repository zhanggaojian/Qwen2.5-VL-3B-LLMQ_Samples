
import os 
import sys
import torch

workfolder = os.getcwd()

sys.path.append(workfolder+'/../../example1')
sys.path.append(workfolder+'/../../example1/llm_utils')
from example1.llm_utils.vl_utils import load_vl_2_5_model, get_vl_inputs

model_id = '/data/huggingface/hf_model/Qwen_Qwen2.5-VL-3B-Instruct'
device = 'cuda:1'
visual, llm, vl_model, processor = load_vl_2_5_model(model_id, device)
print(llm)
print(visual)
import numpy as np
embeddings = llm.get_input_embeddings().weight
print(embeddings.shape)
embeddings.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join('.', 'embedding_weights_151936x2048.raw'))
img_path ="/local/qwen2.5-VL-3B-os/vit/qwen2_vl/example1_vit/images/demo512x342.jpeg"
prompt = "详细地介绍图片,用中文回答"
# prompt = "图片里面有什么,用中文回答"
# prompt = "Describe the image"
inputs = get_vl_inputs(processor, img_path, prompt, device,img_w=384, img_h=384) #h,w keep same as vit part
inputs_embeds = None
input_ids_qwen2vl = inputs['input_ids']
attention_mask_qwen2vl = inputs['attention_mask']
pixel_values_qwen2vl = inputs['pixel_values']  #
image_grid_thw_qwen2vl = inputs['image_grid_thw']  #
print(prompt)
inputs_embeds = llm.embed_tokens(input_ids_qwen2vl)
if pixel_values_qwen2vl is not None:
    pixel_values = pixel_values_qwen2vl.type(visual.parameters().__next__().dtype)
    print(pixel_values.shape)
    #run on device or using fp32 to run, just example here
    #get embeddings from fp32
    image_embeds = visual(pixel_values, grid_thw=image_grid_thw_qwen2vl)
    
    #get embeddings from device
    #run vit on device 
    # 1 push pixel_values_qwen2vl.reshpae(-1,1,1,embeddings.shape[-1])
    # 2 push position_cos/sin generated in vit part
    # 3 push window_attention_mask and full_attention_mask generated from vit part
    # 4 run vit model on device with "qnn-net-run --backend libQnnHtp.so --retrieve_context veg.serialized.bin --input_list input_list.txt" normally, or adding more optimiation
    # 5 get result from device like vision_embedding.raw
    # 6 read vision_embedding.raw with torch.from_numpy(np.fromfile('vision_embedding.raw', dtype=np.float32).reshape(-1,embeddings.shape[-1]))
    # 7 Now, we could do next steps as normal image_embeds

    use_on_device = False
    if use_on_device:
        #hardcode here
        pixel_values_qwen2vl.reshape(-1,1,1,1176).detach().cpu().float().numpy().astype(np.float32).tofile('pix_value.raw')
        #push pix_value.raw into device
        #push VEG/example1/output_dir/veg_exports cos/sin/attention mask into device
        #run qnn-net-run and pull ouptut 
        image_embeds = torch.from_numpy(np.fromfile('vision_embedding.raw', dtype=np.float32).reshape(-1,embeddings.shape[-1])).to(device)
        print(image_embeds.shape)

    # !!!!! For the adaptation for SpinQuant, if the VEG does not fuse the R1 Matrix, you should set fuse_R1=True; otherwise, you should set fuse_R1=False.
    fuse_R1 = False
    if fuse_R1:
        #I do not think doing this ops is necessary, but from spinquant source code https://github.com/facebookresearch/SpinQuant/blob/main/utils/fuse_norm_utils.py#L45
        # W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        #if remove this case, accuracy seems do not affect
        image_embeds =image_embeds.to(device="cuda", dtype=torch.float64)
        image_embeds =  image_embeds - image_embeds.mean(dim=-1, keepdim=True)
        R1 = torch.load(nb_cfg.calib.R1_path)["R1"].cuda().to(torch.float64)
        image_embeds = torch.matmul(image_embeds, R1).to(device="cpu", dtype=torch.float32)


    n_image_tokens = (input_ids_qwen2vl == vl_model.config.image_token_id).sum().item()
    n_image_features = image_embeds.shape[0]
    print(n_image_features)
    if n_image_tokens != n_image_features:
        raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
    image_mask = ((input_ids_qwen2vl == vl_model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device))
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

inputs_embeds_np = inputs_embeds.detach().cpu().numpy()
inputs_embeds_np.tofile(os.path.join('.', 'inputs_embeds.bin'))
