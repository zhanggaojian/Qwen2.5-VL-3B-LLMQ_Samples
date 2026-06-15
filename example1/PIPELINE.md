# Qwen2.5-VL-3B 量化（example1）执行流程详解

> 本文梳理 `example1/llm_quant.py` 的完整执行流程，说明每个阶段做了什么、为什么这么做、产物是什么。
> 一句话概括：**加载 FP 模型 → 评估基线 → QNN 友好化改造（prepare）→ AIMET 量化（SeqMSE + 校准）→ 评估量化精度 → 导出 ONNX + encodings 给 QNN 编译**。

---

## 全流程总览

```
① 读配置 & 环境设置
        ↓
② 模型结构适配 (Model Adaptation)
        ↓
③ 加载 FP 模型 + tokenizer + 数据集
        ↓
④ 评估原始 FP 模型 PPL（基线）
        ↓
⑤ Prepare：QNN 友好化（torch→ONNX→QuIR→QNNIR→重建 torch 图）
        ↓
⑥ 评估 prepared 模型 PPL（验证 prepare 无损）
        ↓
⑦ 量化：建 QuantSim + 混合精度 + SeqMSE + compute_encodings
        ↓
⑧ 评估量化（KVCache sim）模型 PPL（验证量化精度）
        ↓
⑨ 生成测试向量 (test vectors)
        ↓
⑩ 导出 ONNX 静态图 + 量化 encodings
```

---

## 各阶段详解

### ① 读配置 & 环境设置（L1–L35）

- 从 `config.yaml` 读取全部配置（环境、量化、模型覆盖项、数据集、SeqMSE、评估、导出等）。
- 设置 QAIRT SDK 的 `lib/python` 到 `sys.path`，拼接 `LD_LIBRARY_PATH`。
- 读取 `htp_config_file`（HTP 后端配置，对应芯片：`htp_v73`=8gen3 / `htp_v68`=SA8295P / `htp_v81`=SA8797）。

### ② 模型结构适配（Model Adaptation，L37–L61）

把原生 Qwen2 的若干模块替换成 **QNN 友好的实现**，这是上板的关键改造：

- `QcAttention` 替换 eager 注意力实现；
- `bypass_update_causal_mask`：绕过动态 causal mask 生成（改用外部传入的 mask）；
- `MLP/ForCausalLM.prepare_conv`：把 Linear 改造成 Conv 形式（QNN 对 Conv 支持更好）；
- `DynamicCache_update / get_seq_length`：改造 KV-cache 行为，适配定长 KV-cache。

### ③ 加载 FP 模型 + tokenizer + 数据集（L67–L142）

- `AutoConfig` 加载配置，并用 `config.yaml` 的 `model_overrides` **覆盖关键开关**：
  - `return_new_key_value_only`、`transposed_key_cache`、`use_combined_mask_input`、
    `use_position_embedding_input`、`use_cache`、`use_input_embeddings`、`use_mrope` 等。
  - 这些开关决定模型 I/O 形态（是否走 embedding 输入、是否用 mrope 旋转位置编码、KV-cache 排布等）。
- 加载 `Qwen2ForCausalLM` 权重，对每个模块调用 `prepare_conv()` 完成 Linear→Conv 改造。
- 若 `enable_fp16`，把模型转 FP16（RMSNorm 的 Pow/Mul 用 PreCast/PostCast 保持 FP32 精度）。

### ④ 评估原始 FP 模型 PPL（基线，L145–L242）

- 根据 `use_input_embeddings` 选数据集：
  - 否 → WikiText 文本数据集；
  - 是 → Qwen2.5-VL 多模态数据集（图像 + R1），本工程走这条。
- 用 `LLMForwardPassManager` 以 **KVCache（ARN/BERT）模式**前向，计算 **PPL（困惑度）** 作为基线。
- 输出：`ppl score of original fp model: 1.42...`

> **ARN（num_tokens）**：一次前向处理的 token 数（本工程 1073），配合定长 context_length（2048）做 KV-cache 推理。

### ⑤ Prepare：QNN 友好化（L244–L407）

这是把 PyTorch 模型变成 QNN 可消费形式的核心步骤：

- 构造 **dummy input**（attention_mask、position_ids/旋转位置编码、past_key_values 等），并定义 `input_names`/`output_names`（含每层 `past_key_i/past_value_i`）。
- 调 `model_preparer.prepare_model`，内部链路：
  **torch 图 → ONNX → QuIR → QNNIR → 重建 torch 图**。
- 导出前的三个关键修复（见 TROUBLESHOOTING）：
  - `return_dict=False`：让模型返回 tuple，规避 tracing 时 `outputs[0]` 的 `KeyError`；
  - `model.eval()` + `requires_grad_(False)`：规避 "Tensor requires grad as a constant"；
  - patch 掉 ONNX shape inference pass：规避 protobuf 2GB 限制。
- **`skip_prepare` 开关**：
  - `false`：真正执行 prepare，产物写入 `output/prepare/`（`.py`/`.safetensors`/`.json` 等）；
  - `true`：跳过，直接从 `output/prepare/` 加载已有产物（省时间省磁盘）。
- 产物：`output/prepare/qwen25llm_kvcache_36_layer.{py,safetensors,json,...}`

### ⑥ 评估 prepared 模型 PPL（L413–L427）

- 用 prepared 模型再算一次 PPL。
- **校验目的**：prepared PPL 应与原始 FP PPL **几乎相等（delta < 1e-4）**。若明显变差，说明 AIMET/QNN 的 prepare 产出了错误模型。
- 输出：`ppl score of KVCACHE prepared fp model: ...` + `orig - prepared` 差值。

### ⑦ 量化（L430–L561）

AIMET 量化的核心阶段：

1. **建 QuantSimModel**（L447–L455）：
   - `quant_scheme`（如 `post_training_tf`）、`default_output_bw=16`、`default_param_bw=4`；
   - `config_file=htp_config_file`（HTP 后端约束）。
2. **MatMul 第二输入设 8bit 对称**（L458–L459）：让 key/value 张量保持 8bit，降低 KV-cache 的 I/O 开销。
3. **Concat 输入输出共享 encoding**（L462–L464）。
4. **手动混合精度覆盖**（L466–L471）：按 `exceptions.json` 对特定层做精度例外。
5. **SeqMSE**（L473–L520）：逐层用序列 MSE 搜索最优权重量化参数（`num_candidates` 个候选），比单纯 min-max 更准。
6. **compute_encodings**（L527–L561）：跑校准数据，统计并确定**激活量化的 encodings**（scale/offset）。

### ⑧ 评估量化（KVCache sim）模型 PPL（L563–L570）

- 对量化后的 sim 模型再算 PPL，衡量量化掉点。
- 输出：`ppl score of KVCACHE sim fp model: 1.52...` + `orig - sim` 差值（本次 ≈ -0.1，量化后略升属正常）。

### ⑨ 生成测试向量（L572–L580）

- `generate_test_vectors`：导出指定层（`config.yaml` 的 `test_vector_layers`）的中间输入/输出张量。
- **用途**：上板（QNN）推理时做**逐层数值比对**，验证端侧推理与训练侧一致。
- 产物：`output/test_vectors/`

### ⑩ 导出 ONNX 静态图 + 量化 encodings（L582–L596）

- 重建 dummy input（放 CPU），用 `OnnxExportApiArgs` 指定 input/output names、opset。
- `quantsim.export(onnx_dir, model_name, sample_inputs, ...)` 导出：
  - `qwen25llm.onnx`（图结构，权重外置）
  - 大量 `*.weight`/`*.bias`（外置权重，模型 >2GB）
  - `qwen25llm.encodings`（**QNN 用的 ONNX 量化 encodings**）
  - `qwen25llm.pth` / `qwen25llm_torch.encoding`（torch 侧产物，QNN 不用）
- 产物目录：`output/onnx/`

---

## 关键概念速查

| 概念 | 含义 |
|---|---|
| **PPL（困惑度）** | 语言模型评估指标，越低越好；用来对比量化前后精度 |
| **Prepare** | 把 torch 模型转成 QNN 友好结构（Linear→Conv、定长 KV-cache、外部 mask 等） |
| **ARN / num_tokens** | 单次前向的 token 数（1073），配合定长 KV-cache 推理 |
| **KVCache 模式** | 定长 KV-cache 的推理形态，便于端侧固定 shape 编译 |
| **SeqMSE** | 逐层序列 MSE 搜索最优权重量化参数 |
| **compute_encodings** | 跑校准数据确定激活量化的 scale/offset |
| **encodings** | 量化参数文件，QNN 编译时套用 |
| **test vectors** | 逐层中间张量，用于端侧数值比对验证 |
| **skip_prepare** | 复用已有 prepare 产物的开关，省时间省磁盘 |

---

## 输入 / 输出一览

**输入**：
- 原始 HF 模型：`Qwen2.5-VL-3B-Instruct`
- 校准/评估数据集：Qwen2.5-VL 多模态数据（图像 + R1 文本）
- 配置：`config.yaml`

**最终输出（`output/onnx/`，交给 example2 做 QNN 编译）**：
- `qwen25llm.onnx` + 所有 `*.weight`/`*.bias`（外置权重，必须同目录）
- `qwen25llm.encodings`（ONNX 量化 encodings）

**中间产物**：
- `output/prepare/`：prepared 模型（可用 `skip_prepare` 复用）
- `output/test_vectors/`：端侧比对用测试向量
