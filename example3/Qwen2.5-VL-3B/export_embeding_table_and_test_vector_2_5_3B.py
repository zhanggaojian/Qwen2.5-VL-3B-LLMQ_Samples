import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from example1.huggingface.baseline_models.qwen2 import modeling_qwen2


# Text-only Example3: load only the language model, without visual/processor.
model_id = os.getenv(
    "QWEN_MODEL_DIR",
    str(REPO_ROOT / "models" / "Qwen2.5-VL-3B-Instruct"),
)
device = os.getenv(
    "QWEN_DEVICE",
    "cuda:0" if torch.cuda.is_available() else "cpu",
)

llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=True,
)
llm = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
    model_id,
    config=llm_config,
).eval().to(device)

embedding_layer = llm.get_input_embeddings()
embedding_weights = embedding_layer.weight
embedding_weights.detach().cpu().float().numpy().astype(np.float32).tofile(
    "embedding_weights_151936x2048.raw"
)

prompt = "请用中文简单介绍一下你自己。"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
input_ids = tokenizer(
    text,
    add_special_tokens=False,
    return_tensors="pt",
)["input_ids"].to(device)

with torch.no_grad():
    inputs_embeds = embedding_layer(input_ids)

print("prompt:", prompt)
print("input_ids shape:", tuple(input_ids.shape))
print("inputs_embeds shape:", tuple(inputs_embeds.shape))

inputs_embeds.detach().cpu().float().numpy().astype(np.float32).tofile(
    "inputs_embeds.bin"
)
