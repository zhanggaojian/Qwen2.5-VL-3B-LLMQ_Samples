# 01 · 配置文件 config.yaml 详解

> **流程位置**：整个流程的第一步（`llm_quant.py` 第 5-29 行读取它）。
> **一句话本质**：`config.yaml` 把所有"可调参数"集中管理，逻辑代码不写死任何路径/超参；改行为只改这一个文件。

---

## 一、它是怎么被读进来的

```python
# llm_quant.py 第 5-16 行
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(_CONFIG_PATH, 'r', encoding='utf-8') as _f:
    CONFIG = yaml.safe_load(_f)

_env      = CONFIG['environment']
_quant    = CONFIG['quantization']
_overrides= CONFIG['model_overrides']
_dataset_cfg = CONFIG['dataset']
_seq_mse_cfg = CONFIG['seq_mse']
_eval_cfg = CONFIG['evaluation']
_export_cfg= CONFIG['export']
```

- 读取的是**脚本同级目录**的 `config.yaml`（不是 `config/` 子目录里的那个，那个已废弃删掉了）。
- `yaml.safe_load` 把 YAML 解析成 Python 字典，之后按段(section)取用。

---

## 二、八个配置段总览

| 段 | 作用 | 关键消费位置 |
|----|------|--------------|
| `environment` | 环境与路径（SDK、模型、输出目录） | 第 18-28 行设环境变量、`LD_LIBRARY_PATH` |
| `quantization` | 量化基础参数（位宽、方案、HTP配置） | QuantSim 创建处 |
| `model_overrides` | 覆盖模型 config 的开关 | 第 80-88 行 `setattr(llm_config, ...)` |
| `dataset` | 校准/评估数据集路径与尺寸 | dataloader 构建处 |
| `seq_mse` | SeqMSE 优化参数 | `apply_seq_mse(...)` |
| `evaluation` | 各阶段 PPL 评估的 batch 数 | `ppl_eval(...)` |
| `export` | ONNX 导出 opset 版本 | prepare / quantsim.export |
| `test_vector_layers` | 要导出中间张量的层（正则） | `generate_test_vectors` |

---

## 三、逐段详解

### 1. environment（环境与路径）

```yaml
environment:
  qnn_sdk_root: /root/.../qairt/2.42.0.251225   # 高通 QAIRT(QNN) SDK 根目录
  model_name: qwen25llm                          # 输出文件命名用
  model_id: /root/.../Qwen2.5-VL-3B-Instruct     # 模型权重目录(本地路径或HF名)
  cache_dir: /root/.../outputs/cache             # HF 缓存目录
  output_dir: /root/.../outputs/output           # 所有产物输出目录
```
- `qnn_sdk_root` 最关键：脚本据此设置 `LD_LIBRARY_PATH` 和 `sys.path`，找不到会直接影响运行（见 `run.sh`、`TROUBLESHOOTING.md`）。

### 2. quantization（量化基础）

```yaml
quantization:
  enable_fp16: false              # 是否用 FP16 半精度跑(见笔记里 enable_fp16 说明)
  htp_config_file: htp_v73        # 目标芯片: 8gen3->v73, SA8295P->v68, SA8797->v81
  context_length: 2048            # 上下文长度
  arn: 1073                       # KVCache 模式下一次返回的 logits 数(num_logits_to_return)
  quant_scheme: post_training_tf  # 量化方案(训练后量化, tf=TensorFlow风格)
  default_output_bw: 16           # 激活默认位宽 = 16bit
  default_param_bw: 4             # 权重默认位宽 = 4bit
  skip_prepare: true              # 是否跳过 prepare 阶段(复用已有产物)
  mixed_precision_config_file: ./config/mixed_precision_config/exceptions.json  # 混合精度例外
```
- **核心量化设定**：权重 4bit、激活 16bit（即常说的 "W4A16"）。
- `htp_config_file` 决定生成的产物面向哪颗芯片。
- `skip_prepare`：true 时直接加载之前 prepare 好的模型，省时间（前提是产物已存在，否则报错）。

### 3. model_overrides（模型 config 覆盖）

```yaml
model_overrides:
  return_new_key_value_only: true    # KV Cache 只返回新算的 K/V
  transposed_key_cache: true         # K 转置存储
  use_combined_mask_input: true      # 因果掩码由外部输入
  use_position_embedding_input: true # RoPE 的 cos/sin 由外部输入
  use_cache: true
  attn_implementation: eager         # 选用 eager -> 即被换成 QcAttention
  pretraining_tp: 1
  use_input_embeddings: true
  use_mrope: false                   # 是否用多模态 RoPE
  mask_neg_fp16: -50                 # 掩码负值(FP16)
  mask_neg_fp32: -100                # 掩码负值(FP32)
```
- 这些在第 80-88 行被 `setattr` 到 `llm_config` 上，**直接驱动"模型适配"那一块的行为**（详见笔记 02）。
- 它们是连接"配置"与"适配代码"的桥梁。

### 4. dataset（数据集）

```yaml
dataset:
  img_h: 672
  img_w: 336
  device: cuda
  calibration_dataset_path: /root/.../llava_v1_5_mix665k.json  # 量化校准数据
  ppl_evaluation_dataset_path: /root/.../llava_v1_5_mix665k.json # PPL评估数据
  image_dataset_path: /root/.../data
  r1_path: null
  num_test_batches: 100
```
- **校准集**：量化时用来统计激活分布(compute_encodings)。
- **评估集**：算 PPL 衡量精度。

### 5. seq_mse（SeqMSE 优化）

```yaml
seq_mse:
  num_batches: 20       # 用多少 batch 做 SeqMSE
  inp_symmetry: symqt   # 输入对称性
  num_candidates: 20    # 每层搜索的候选量化范围数
  loss_fn: mse          # 损失函数
```
- SeqMSE：逐层搜索最优量化 scale，降低低位宽(4bit)带来的精度损失。

### 6. evaluation（评估 batch 数）

```yaml
evaluation:
  ppl_num_batches: 10                 # PPL 评估用的 batch 数
  compute_encodings_num_batches: 20   # 统计激活分布用的 batch 数
```

### 7. export（ONNX 导出）

```yaml
export:
  prepare_opset_version: 20   # prepare 阶段导出 onnx 的 opset
  onnx_opset_version: 14      # 最终导出 onnx 的 opset
```

### 8. test_vector_layers（测试向量层）

```yaml
test_vector_layers:
  - "model_layers_\\d+_input_layernorm_Pow"
  - "model_layers_\\d+_Add_1"
  - "rms_norm_\\d+"
```
- 正则匹配层名，导出这些层的中间输入/输出张量，用于端侧比对调试。

---

## 四、记忆要点

- **W4A16**：`default_param_bw: 4` + `default_output_bw: 16`，是本项目量化的核心规格。
- **改芯片**：只动 `htp_config_file`（v73/v68/v81）。
- **config 与适配代码的关系**：`model_overrides` → `setattr(llm_config)` → 驱动 QcAttention / KVCache 等行为（接下篇 02）。
- **路径全是 Linux 绝对路径**：换环境时这些都要改。

---

## 五、待确认 / 疑问（自己往下填）

- [ ] `quant_scheme: post_training_tf` 还有哪些可选值？区别是什么？
- [ ] `mixed_precision_config/exceptions.json` 里具体配了哪些层用不同位宽？
- [ ] `arn=1073` 这个数字是怎么算出来的？（和 context_length 有关吗）
