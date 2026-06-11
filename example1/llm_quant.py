import sys
import os

from llm_utils.config_io import get_required_section, load_yaml_config

quant_config = load_yaml_config(os.environ.get("LLM_QUANT_CONFIG") or None)
environment_config = get_required_section(quant_config, "environment")
model_config = get_required_section(quant_config, "model")
dataset_config = get_required_section(quant_config, "dataset")
evaluation_config = get_required_section(quant_config, "evaluation")
prepare_config = get_required_section(quant_config, "prepare")
quantization_config = get_required_section(quant_config, "quantization")
seq_mse_config = get_required_section(quant_config, "seq_mse")
encoding_config = get_required_section(quant_config, "encoding")
test_vectors_config = get_required_section(quant_config, "test_vectors")
export_config = get_required_section(quant_config, "export")

QNN_SDK_ROOT = environment_config["qnn_sdk_root"]
compute_device = environment_config.get("compute_device", "cuda")
cpu_device = environment_config.get("cpu_device", "cpu")
model_name = model_config["name"]
model_id = model_config["model_id"]
cache_dir = model_config["cache_dir"]
output_dir = model_config["output_dir"]


lib_clang_path = os.path.join(QNN_SDK_ROOT, 'lib', 'x86_64-linux-clang')
sys.path.insert(0, QNN_SDK_ROOT + '/lib/python')
LD_LIBRARY_PATH = os.getenv('LD_LIBRARY_PATH', None)
os.environ['LD_LIBRARY_PATH'] = lib_clang_path + ':' + LD_LIBRARY_PATH if LD_LIBRARY_PATH is not None else lib_clang_path
enable_fp16 = model_config.get("enable_fp16", False)
sys.path.append('../')

htp_config_file = quantization_config["htp_config_file"]
# 8gen3  htp_v73
# SA8295P htp_v68
# SA8797  htp_v81

from huggingface.baseline_models.qwen2 import modeling_qwen2
from transformers import cache_utils
from llm_utils.qcqwen2_adaptation import (
    QcAttention,
    bypass_update_causal_mask,
    MLP_prepare_conv,
    ForCausalLM_prepare_conv,
    MLP_forward_conv,
    DynamicCache_update,
    DynamicCache_get_seq_length,
    update_attr
)

# ————————————————Model Adaptation————————————————
modeling_qwen2.QWEN2_ATTENTION_CLASSES['eager'] = QcAttention
assert update_attr(modeling_qwen2.Qwen2Model, '_update_causal_mask', bypass_update_causal_mask) or \
       update_attr(modeling_qwen2.Qwen2Model, '_prepare_decoder_attention_mask', bypass_update_causal_mask), \
    f"neither _prepare_decoder_attention_mask(..) nor _update_causal_mask(..) found, Unknown Qwen2Model definition in {modeling_qwen2.__file__}"
setattr(modeling_qwen2.Qwen2MLP, 'prepare_conv', MLP_prepare_conv)
setattr(modeling_qwen2.Qwen2MLP, 'forward_conv', MLP_forward_conv)
setattr(modeling_qwen2.Qwen2ForCausalLM, 'prepare_conv', ForCausalLM_prepare_conv)
assert update_attr(cache_utils.DynamicCache, 'update',
                   DynamicCache_update), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
assert update_attr(cache_utils.DynamicCache, 'get_seq_length',
                   DynamicCache_get_seq_length), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"

from tqdm import tqdm
import torch
os.makedirs(output_dir, exist_ok=True)

from transformers import AutoConfig, AutoTokenizer

trust_remote_code = model_config.get("trust_remote_code", True)
llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
context_length = model_config["context_length"]
print(f'num_layer: {llm_config.num_hidden_layers}, context_length : {context_length},'
      f'num_hidden_size :{llm_config.num_attention_heads},  num_kv_heads: {llm_config.num_key_value_heads}')

ARN = model_config["arn"]

for attr_name, attr_value in model_config.get("config_overrides", {}).items():
    setattr(llm_config, attr_name, attr_value)
mask_neg_config = model_config.get("mask_neg", {})
setattr(llm_config, 'mask_neg', mask_neg_config.get("fp16" if enable_fp16 else "fp32", -50 if enable_fp16 else -100))

os.environ['TOKENIZERS_PARALLELISM'] = str(environment_config.get("tokenizers_parallelism", "0"))
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          cache_dir=cache_dir,
                                          use_fast=model_config.get("tokenizer_use_fast", True),
                                          trust_remote_code=trust_remote_code)
tokenizer.model_max_length = context_length

model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(model_id, config=llm_config)
for name, module in model.named_modules():
    if hasattr(module, "prepare_conv"):
        module.prepare_conv()


# Cast to fp 32/16 if needed
from aimet_torch import elementwise_ops
class PreCast(torch.nn.Module):
    def __init__(self, module, dtype):
        super(PreCast, self).__init__()
        self.module = module
        self.upcast = elementwise_ops.Cast(dtype)

    def forward(self, *inputs):
        casted_inputs = [self.upcast(input) for input in inputs]
        return self.module(*casted_inputs)


class PostCast(torch.nn.Module):
    def __init__(self, module, dtype):
        super(PostCast, self).__init__()
        self.module = module
        self.downcast = elementwise_ops.Cast(dtype)

    def forward(self, *inputs):
        output = self.module(*inputs)
        casted_output = self.downcast(output)
        return casted_output

def convert_model_to_fp16(model):
    model.half()
    for name, module in model.named_modules():
        if name.endswith("norm_Pow"):
            setattr(model, name, PreCast(module, torch.float32))
        if name.endswith("norm_Mul_1"):
            setattr(model, name, PostCast(module, torch.float16))

def convert_model_to_fp32(model):
    model.float()

    for name, module in model.named_modules():
        if name.endswith("norm_Pow"):
            setattr(model, name, module.module)
        if name.endswith("norm_Mul_1"):
            setattr(model, name, module.module)

if (enable_fp16):
    convert_model_to_fp16(model)


# ————————————Eval fp model with ARN(BERT) MODE——————————————
from torch.nn import CrossEntropyLoss
from llm_utils.forward_pass_wrapper import slice_inputs_and_run_successive_kvcache_inference


def ppl_eval(data_loader, forward_pass_manager, num_batches=10):
    if num_batches == 0:
        num_batches = len(data_loader)
    loss = 0

    for batch_id, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Evaluating")):
        if batch_id >= num_batches:
            break

        outputs = slice_inputs_and_run_successive_kvcache_inference(forward_pass_manager, input_ids=batch['input_ids'])
        lm_logits = outputs["lm_logits"].cpu()

        if 'input_ids' not in batch:
            batch['input_ids'] = batch['labels']

        lm_logits = lm_logits.reshape(batch['input_ids'].shape[0], -1, lm_logits.shape[-1])
        shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
        shift_labels = batch['input_ids'][..., 1:].contiguous().to(shift_logits.device)
        loss_fct = CrossEntropyLoss()
        loss += loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    loss = loss / num_batches
    ppl = loss.exp()
    return ppl


def ppl_eval_embedding(data_loader, forward_pass_manager, num_batches=10):
    if num_batches == 0:
        num_batches = len(data_loader)
    loss = 0
    for batch_id, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Evaluating")):
        if batch_id >= num_batches:
            break
        outputs = slice_inputs_and_run_successive_kvcache_inference(forward_pass_manager, input_embeds=batch['input_embeddings'])
        lm_logits = outputs["lm_logits"].cpu()
        # TODO
        # lm_logits = lm_logits[:,637:,:]
        # we can either pass input_ids or input_embeds in our fpm, hence with input_embeds we pass the labels.
        if 'input_ids' not in batch:
            # batch['input_ids'] = batch['labels'][:,637:]
            batch['input_ids'] = batch['labels']

        lm_logits = lm_logits.reshape(batch['input_ids'].shape[0], -1, lm_logits.shape[-1])
        shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
        shift_labels = batch['input_ids'][..., 1:].contiguous().to(shift_logits.device)
        loss_fct = CrossEntropyLoss()
        loss += loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    loss = loss / num_batches
    ppl = loss.exp()
    return ppl

if not llm_config.use_input_embeddings:
    from llm_utils.wikitext_dataloader import get_wiki_dataset

    wiki_dataset_config = get_required_section(dataset_config, "wiki")
    train_dataloader, test_dataloader, _ = get_wiki_dataset(context_length,
                                                            tokenizer,
                                                            cache_dir,
                                                            dataset_path=wiki_dataset_config.get("path", "wikitext"),
                                                            dataset_name=wiki_dataset_config.get("name", "wikitext-2-raw-v1"),
                                                            train_split=wiki_dataset_config.get("train_split", "train"),
                                                            test_split=wiki_dataset_config.get("test_split", "test"),
                                                            batch_size=wiki_dataset_config.get("batch_size", 1),
                                                            shuffle=wiki_dataset_config.get("shuffle", False))
else:
    from llm_utils.qwen2_5_vl_dataloader import get_qwen_dataset

    vl_dataset_config = get_required_section(dataset_config, "vl")
    llava_dataset_setting = {
        "img_h": vl_dataset_config["img_h"],
        "img_w": vl_dataset_config["img_w"],
        "emb_length": vl_dataset_config.get("emb_length") or ARN,
        "device": vl_dataset_config.get("device") or compute_device,
        "qwen2vl_model_path": vl_dataset_config.get("qwen2vl_model_path") or model_id,
        "calibration_dataset_path": vl_dataset_config["calibration_dataset_path"],
        "ppl_evaluation_dataset_path": vl_dataset_config["ppl_evaluation_dataset_path"],
        "image_dataset_path": vl_dataset_config["image_dataset_path"],
        "default_image_path": vl_dataset_config.get("default_image_path", "coco/train2017/000000000025.jpg"),
        "R1_path": vl_dataset_config.get("R1_path"),
        "batch_size": vl_dataset_config.get("batch_size", 1),
        "shuffle": vl_dataset_config.get("shuffle", False),
        "use_mrope": llm_config.use_mrope
    }
    train_dataloader, test_dataloader, dataset = get_qwen_dataset(model.model,
                                                                  llava_dataset_setting,
                                                                  num_test_batches=vl_dataset_config.get("num_test_batches", 100))

from llm_utils.forward_pass_wrapper import LLMForwardPassManager

orig_fpm = LLMForwardPassManager(cfg=llm_config,
                                 model=model,
                                 tokenizer=tokenizer,
                                 separate_tuple_input_output=False,
                                 num_tokens=ARN)

if not llm_config.use_input_embeddings:
    ppl_eval_embedding = ppl_eval
with torch.no_grad():
    with orig_fpm.place_on_device(compute_device):
        orig_ppl = ppl_eval_embedding(test_dataloader, orig_fpm, num_batches=evaluation_config.get("original_ppl_batches", 10))

print(f"ppl score of original fp model: {orig_ppl}")

# ———————————Prepare dummy input for ARN(BERT) mode———————————
from llm_utils.forward_pass_wrapper import get_position_embeddings_from_position_ids, prepare_combined_attention_mask, get_padded_kv_values, flatten_tensors


def get_dummy_data(config, tokenizer, device, separate_tuple_input_output, num_tokens=None, dtype=torch.float32):
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    rope_theta = config.rope_theta

    max_tokens = tokenizer.model_max_length
    attention_mask = torch.ones((1, max_tokens), dtype=torch.long, device=device)

    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    position_ids = position_ids.clip(0, max_tokens - 1)
    position_ids = position_ids[..., :num_tokens]
    position_ids = position_ids.to(device=device)
    if config.use_combined_mask_input:
        past_kv_length = max_tokens - num_tokens
        attention_mask = prepare_combined_attention_mask(attention_mask, input_shape=(1, num_tokens),
                                                         past_key_values_length=past_kv_length, device=device,
                                                         mask_neg=config.mask_neg, dtype=dtype)

    if config.use_position_embedding_input:
        position_ids = get_position_embeddings_from_position_ids(position_ids,
                                                                 head_dim=hidden_size // num_attention_heads,
                                                                 max_length=max_tokens,
                                                                 device=device, dtype=dtype,
                                                                 config=config)
    if not config.use_input_embeddings:
        inputs = {
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'input_ids': torch.randint(0, len(tokenizer), (1, num_tokens), device=device),
        }

    else:
        inputs = {
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'inputs_embeds': torch.rand((1, num_tokens, hidden_size), device=device),
        }

    inputs['past_key_values'] = get_padded_kv_values(past_size=max_tokens - num_tokens,
                                                     num_layers=num_layers,
                                                     hidden_size=hidden_size,
                                                     num_attention_heads=num_attention_heads,
                                                     num_kv_heads=num_kv_heads,
                                                     transposed_key_cache=config.transposed_key_cache,
                                                     device=device,
                                                     dtype=dtype)

    if not config.use_input_embeddings:
        if separate_tuple_input_output:
            flattened_kvcache = tuple(flatten_tensors(inputs['past_key_values']))
            if isinstance(inputs['position_ids'], tuple):
                inputs = inputs['input_ids'], inputs['attention_mask'], inputs['position_ids'][0], \
                inputs['position_ids'][1]
            else:
                inputs = inputs['input_ids'], inputs['attention_mask'], inputs['position_ids']
            inputs = inputs + flattened_kvcache
    else:
        if separate_tuple_input_output:
            flattened_kvcache = tuple(flatten_tensors(inputs['past_key_values']))
            if isinstance(inputs['position_ids'], tuple):
                inputs_ = inputs['inputs_embeds'], inputs['attention_mask'], inputs['position_ids'][0], inputs['position_ids'][1]
                inputs = inputs_ + flattened_kvcache
            else:
                inputs = inputs['inputs_embeds'], inputs['attention_mask'], inputs['position_ids']
                inputs = inputs + flattened_kvcache

    return inputs


# ————————————————Prepare model by QAIRT(QNN)——————————————————
# torch graph → ONNX → QuIR → QNNIR → 重建torch图
import time
from utilities.aimet_patch import load_pytorch_model
from aimet_torch import onnx_utils

onnx_utils.EXPORT_TO_ONNX_DIRECT = prepare_config.get("export_to_onnx_direct", True)
import qti.aisw.emitter.ir_graph_op_handler as ir_graph_op_handler

ir_graph_op_handler.KEEP_ORIGINAL_MODEL_STRUCTURE = prepare_config.get("keep_original_model_structure", False)
from qti.aisw.preparer_api import model_preparer


def _get_past_key_values_names(sfx, n_layers):
    all_kvs = []
    for i in range(n_layers):
        all_kvs.append(f'past_key_{i}_{sfx}')
        all_kvs.append(f'past_value_{i}_{sfx}')
    return all_kvs


dummy_input = get_dummy_data(llm_config, tokenizer, cpu_device, separate_tuple_input_output=False, num_tokens=ARN, dtype=model.dtype)
if not llm_config.use_input_embeddings:
    input_names = ['input_ids', 'attention_mask']
    input_names += ['position_ids_cos', 'position_ids_sin'] if llm_config.use_position_embedding_input else ['position_ids']
    input_names += _get_past_key_values_names('in', llm_config.num_hidden_layers)
    output_names = ['logits'] + _get_past_key_values_names('out', llm_config.num_hidden_layers)
else:
    input_names = ['inputs_embeds', 'attention_mask']
    input_names += ['position_ids_cos', 'position_ids_sin'] if llm_config.use_position_embedding_input else ['position_ids']
    input_names += _get_past_key_values_names('in', llm_config.num_hidden_layers)
    output_names = ['logits'] + _get_past_key_values_names('out', llm_config.num_hidden_layers)

# Build converter args
converter_args = []
converter_args_value = prepare_config.get("converter_args_value", "NONTRIVIAL")
for input_name in input_names:
    converter_args.append({"name": input_name, "source_model_input_layout": converter_args_value})
converter_args = {"input_tensors": converter_args}

skip_prepare = prepare_config.get("skip_prepare", True)
prepare_path = f"{output_dir}/prepare"
os.makedirs(prepare_path, exist_ok=True)
prepare_filename = f'{model_name}_kvcache_{llm_config.num_hidden_layers}_layer'

if skip_prepare:
    prepared_model_path = os.path.join(prepare_path, f'{prepare_filename}.py')
    if not os.path.exists(prepared_model_path):
        raise ValueError(f"prepared artifacts not found in {prepare_path}")
    else:
        print(
            f'WARNING: preparation skipped for model={prepare_filename}, prepared at {time.ctime(os.path.getmtime(prepared_model_path))}')
        from qti.aisw.emitter.utils.torch_utils import load_torch_model_using_safetensors
        prepared_model = load_torch_model_using_safetensors(prepare_filename, prepare_path, prepare_filename).eval()

else:
    if enable_fp16:
        convert_model_to_fp32(model)
    model.num_logits_to_return = ARN  # configuring the model for KVCache mode
    prepared_model = model_preparer.prepare_model(model,
                                                  dummy_input,
                                                  model_name=prepare_filename,
                                                  filename=prepare_filename,
                                                  path=prepare_path,
                                                  input_names=input_names,
                                                  output_names=output_names,
                                                  onnx_export_args={"opset_version": prepare_config.get("onnx_export_opset", 20)},
                                                  # converter_args=converter_args,
                                                  return_prepare_model=True,
                                                  keep_original_model_structure=prepare_config.get("keep_original_model_structure", False))
print("————Prepare done————")
del model

if enable_fp16:
    convert_model_to_fp16(prepared_model)

# ————————Eval prepared model with ARN(BERT) mode——————————
fp_prepared_fpm = LLMForwardPassManager(cfg=llm_config,
                                        model=prepared_model,
                                        tokenizer=tokenizer,
                                        separate_tuple_input_output=True,
                                        num_tokens=ARN)

with torch.no_grad():
    with fp_prepared_fpm.place_on_device(compute_device):
        prepared_kvcache_ppl = ppl_eval_embedding(test_dataloader,
                                                  fp_prepared_fpm,
                                                  num_batches=evaluation_config.get("prepared_ppl_batches", 10))

# This should be very close (<1e-4 delta) to original model's perplexity
# If the perplexity score goes further up, it indicates the AIMET/QNN pair is producing a faulty prepared model
print(f"ppl score of KVCACHE prepared fp model: {prepared_kvcache_ppl}\n"
      f"orig ppl - prepared ppl = {orig_ppl - prepared_kvcache_ppl}")


# ————————————Quantization——————————————

from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel
import copy

sim_fpm = LLMForwardPassManager(cfg=llm_config,
                                model=copy.deepcopy(prepared_model),
                                # to avoid creating the sim in_place on the original model
                                tokenizer=tokenizer,
                                separate_tuple_input_output=True,
                                num_tokens=ARN)

dummy_input = get_dummy_data(llm_config,
                             tokenizer, compute_device, separate_tuple_input_output=True,
                             num_tokens=ARN, dtype=sim_fpm.dtype)

with sim_fpm.place_on_device(compute_device):
    quantsim = QuantizationSimModel(model=sim_fpm.model,
                                    quant_scheme=getattr(QuantScheme, quantization_config.get("quant_scheme", "post_training_tf")),
                                    dummy_input=dummy_input,
                                    default_output_bw=quantization_config.get("default_output_bw", 16),
                                    default_param_bw=quantization_config.get("default_param_bw", 4),
                                    # default_param_bw=8,
                                    in_place=quantization_config.get("in_place", True),
                                    config_file=htp_config_file)

# Setting 16bit x 8bit Matmul. To keep key and value tensors as 8 bits, reducing data I/O costs associated with KV-cache orchestration.
from aimet_torch.v2.experimental.quantsim_utils import set_matmul_second_input_producer_to_8bit_symmetric
set_matmul_second_input_producer_to_8bit_symmetric(quantsim)

# concat have shared encoding on input and output activations.
from aimet_torch.v2.experimental import propagate_output_encodings
import aimet_torch.elementwise_ops as aimet_ops
propagate_output_encodings(quantsim, aimet_ops.Concat)
# Manual mixed precision config
from llm_utils.mixed_precision_overrides import ManualQuantsimMixedPrecisionConfig
quantsim_adjuster = ManualQuantsimMixedPrecisionConfig(mixed_precision_config_file=quantization_config.get("mixed_precision_config_file", "./config/mixed_precision_config/exceptions.json"))
quantsim_adjuster.apply_exceptions(quantsim)

from aimet_torch.v2.seq_mse import apply_seq_mse
from aimet_torch.seq_mse import SeqMseParams


def _forward_fn_inputs_id(model, inputs):
    if model == fp_prepared_fpm.model:
        fpm = fp_prepared_fpm
    else:
        fpm = sim_fpm
    # slice inputs so that we only end up doing inference using first n tokens
    input_length = inputs["input_ids"].shape[1]
    prepared_inputs, _ = fpm.prepare_inputs(input_ids=inputs["input_ids"][:, :min(input_length, fpm.num_tokens), ...])
    # prepared_inputs = {name: t.to(torch.half) if t.is_floating_point() else t for name, t in prepared_inputs.items()}
    # prepared_inputs = {name: t.to(torch.half) if t.is_floating_point() else t for name, t in prepared_inputs.items()}
    fpm.model(**prepared_inputs)


def _forward_fn(model, inputs):
    if model == fp_prepared_fpm.model:
        fpm = fp_prepared_fpm
    else:
        fpm = sim_fpm

    # slice inputs so that we only end up doing inference using first n tokens
    input_length = inputs["input_embeddings"].shape[1]
    # input_length = inputs["input_ids"].shape[1]
    prepared_inputs, _ = fpm.prepare_inputs(
        input_embeddings=inputs["input_embeddings"][:, :min(input_length, fpm.num_tokens), ...])
    # prepared_inputs = {name: t.to(torch.half) if t.is_floating_point() else t for name, t in prepared_inputs.items()}
    # prepared_inputs = {name: t.to(torch.half) if t.is_floating_point() else t for name, t in prepared_inputs.items()}
    fpm.model(**prepared_inputs)


if not llm_config.use_input_embeddings:
    _forward_fn = _forward_fn_inputs_id

params = SeqMseParams(num_batches=seq_mse_config.get("num_batches", 20),
                      inp_symmetry=seq_mse_config.get("inp_symmetry", "symqt"),
                      num_candidates=seq_mse_config.get("num_candidates", 20),
                      loss_fn=seq_mse_config.get("loss_fn", "mse"),
                      forward_fn=_forward_fn)

# fp_prepared_fpm.model.to(torch.half)
# sim_fpm.model.to(torch.half)
print(train_dataloader)

with fp_prepared_fpm.place_on_device(compute_device), sim_fpm.place_on_device(compute_device):
    apply_seq_mse(fp_prepared_fpm.model, quantsim, train_dataloader, params)

del fp_prepared_fpm
del prepared_model
sim_fpm.model.to(torch.float32)


# compute activation encodings using AIMET
def _forward_fn_inputs_id(model, kwargs):
    data_loader = kwargs['data_loader']
    fpm = kwargs['fpm']
    max_iterations = kwargs['num_batches']
    for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
        if batch_id < max_iterations:
            slice_inputs_and_run_successive_kvcache_inference(fpm, input_ids=batch['input_ids'])
        else:
            break


def _forward_fn(model, kwargs):
    data_loader = kwargs['data_loader']
    fpm = kwargs['fpm']
    max_iterations = kwargs['num_batches']
    for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
        if batch_id < max_iterations:
            slice_inputs_and_run_successive_kvcache_inference(fpm, input_embeds=batch['input_embeddings'])
        else:
            break


if not llm_config.use_input_embeddings:
    _forward_fn = _forward_fn_inputs_id

kwargs = {
    'data_loader': train_dataloader,
    'fpm': sim_fpm,
    'num_batches': encoding_config.get("num_batches", 20)

}

with sim_fpm.place_on_device(compute_device):
    quantsim.compute_encodings(_forward_fn, kwargs)

# ————Eval KV Cache QuantSimModel————
with torch.no_grad():
    with sim_fpm.place_on_device(compute_device):
        # sim_ppl = ppl_eval(test_dataloader, sim_fpm,num_batches=2)
        sim_ppl = ppl_eval_embedding(test_dataloader, sim_fpm, num_batches=evaluation_config.get("sim_ppl_batches", 10))

print(f"ppl score of KVCACHE sim fp model: {sim_ppl}\n"
      f"orig ppl - kvcache sim ppl = {orig_ppl - sim_ppl}")

# ————————————Export static graph with quantization encodings——————————————
# generate test tensor for inference on edge with QNN
from llm_utils.test_vectors import generate_test_vectors
test_vector_layers = test_vectors_config.get("layers", [
    "model_layers_\\d+_input_layernorm_Pow",
    "model_layers_\\d+_Add_1",
    "rms_norm_\\d+"
])
with sim_fpm.place_on_device(compute_device):
    # generate_test_vectors(quantsim, sim_fpm, train_dataloader, output_dir, num_batches=1, test_vector_layers=test_vector_layers, input_names=input_names)
    generate_test_vectors(quantsim,
                          sim_fpm,
                          train_dataloader,
                          output_dir,
                          num_batches=test_vectors_config.get("num_batches", 1),
                          test_vector_layers=test_vector_layers,
                          input_names=input_names)
    # use more inputs if we choose fp16 at downproj instead of w4fp16
    # generate_test_vectors(quantsim, sim_fpm, train_dataloader, output_dir, num_batches=20, test_vector_layers=test_vector_layers, input_names=input_names)
# Export KVCache Model
dummy_input = get_dummy_data(llm_config, tokenizer, compute_device, separate_tuple_input_output=True, num_tokens=ARN)

from aimet_torch.utils import change_tensor_device_placement
from aimet_torch.onnx_utils import OnnxExportApiArgs

onnx_dir = os.path.join(output_dir, export_config.get("onnx_dir_name", "onnx"))
os.makedirs(onnx_dir, exist_ok=True)

if (enable_fp16):
    # Convert FP16 model back to FP32 for ONNX export
    convert_model_to_fp32(quantsim.model)

onnx_api_args = OnnxExportApiArgs(input_names=input_names,
                                  output_names=output_names,
                                  opset_version=export_config.get("onnx_opset_version", 14))
sample_inputs = change_tensor_device_placement(dummy_input, torch.device(cpu_device))
quantsim.export(onnx_dir, model_name, sample_inputs, onnx_export_args=onnx_api_args)
