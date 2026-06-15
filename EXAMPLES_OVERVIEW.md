# Qwen2.5-VL-3B 端侧部署三阶段（example1 / 2 / 3）职责总览

> 本工程把「从训练侧模型到高通端侧部署」拆成三个 example，构成一条完整流水线。
> 一句话：**example1 量化 → example2 主机编译 → example3 端侧运行**。

---

## 流水线总览

```
example1  ──→  example2  ──→  example3
  量化          主机编译         端侧运行
 (AIMET)      (QNN compile)     (Genie)
 x86 Linux     x86 Linux 主机    高通设备
 (需 GPU)      (纯 CPU 工具链)   (NPU/HTP)
```

---

## 职责对照表

| | 阶段 | 在哪跑 | 是否需要 GPU | 做什么 | 主要产物 |
|---|---|---|---|---|---|
| **example1** | 模型量化 | x86 Linux | ✅ 需要（PyTorch forward / SeqMSE / compute_encodings 在 cuda） | AIMET 量化（prepare + SeqMSE + compute_encodings） | `qwen25llm.onnx` + `*.weight/*.bias` + `qwen25llm.encodings` |
| **example2** | 主机编译 | x86 Linux 主机（可与 example1 同一台 GPU 服务器） | ❌ 不需要（QNN x86 主机工具链，纯 CPU） | QNN 编译（split onnx、MHA2SHA 转换、ONNX→DLC→量化 DLC→context binary） | 设备可执行的模型 binary（context 序列化文件） |
| **example3** | 端侧运行 | 高通 SnapDragon 设备 | ❌（端侧 NPU/HTP） | 准备 embedding/输入、推送库和模型、用 Genie 推理 | 实际推理输出（文本结果） |

> 说明：example2 虽然常常**就在那台 GPU 服务器上执行**（GPU 服务器本身是 x86 Linux 机器），但它跑的是 QNN 的 `x86_64-linux-clang` 主机工具（`qairt-converter` / `qairt-quantizer` / `qnn-context-binary-generator` / `mha2sha-onnx-converter`），**全在 CPU 上，不使用 GPU**。它只要求 x86 Linux + QNN SDK。

---

## 各 example 详解

### example1 — 模型量化

- **目标**：把原始 Qwen2.5-VL-3B FP 模型量化成 QNN 可用的 ONNX + encodings。
- **核心步骤**：
  1. 模型结构适配（QcAttention、Linear→Conv、定长 KV-cache、外部 mask）；
  2. Prepare（torch → ONNX → QuIR → QNNIR → 重建 torch 图）；
  3. 量化（QuantSim：权重 4bit / 激活 16bit / KV 8bit + 混合精度 + SeqMSE + compute_encodings）；
  4. 三次 PPL 评估（原始 FP → prepared → 量化后），验证精度不崩；
  5. 生成 test vectors；
  6. 导出 ONNX + encodings。
- **关键文件**：`example1/llm_quant.py`、`config.yaml`、`run.sh`
- **产物（给 example2）**：`output/onnx/` 下的 `.onnx` + 所有 `.weight/.bias` + `.encodings`

> 详见 `example1/PIPELINE.md`（流程详解）和 `example1/TROUBLESHOOTING.md`（问题与解决）。

### example2 — 主机编译（Generate model artifacts for SnapDragon）

- **目标**：在 x86 Linux 主机上，把 example1 的量化 ONNX 编译成高通设备可执行的模型 binary。
- **硬件**：**不需要 GPU**，跑的是 QNN 的 `x86_64-linux-clang` 主机工具链（纯 CPU）。可以就在那台 GPU 服务器上执行（它本身是 x86 Linux），只是不会用到显卡。
- **核心步骤**（`qnn_compile_deploy.py` 串起整条流水线）：
  1. **change_hardcoding**：按 AR（如 ar1 / ar128）和 context length 生成不同配置的导出；
  2. **split onnx**：按 split 数切分模型，生成各 split 的 onnx / 输入向量 / golden 输出；
  3. **MHA→SHA**（`mha2sha-onnx-converter`）：把多头注意力转成单头（SHA）形式，适配 HTP，输入 AIMET 的 `.encodings`；
  4. **ONNX→DLC**（`qairt-converter`）：转成 QNN DLC 表示，套用 `--quantization_overrides`（encodings）；
  5. **量化 DLC**（`qairt-quantizer`）：act 16bit / bias 32bit，生成 `*_quantized.dlc`；
  6. **生成 context binary**（`qnn-context-binary-generator`）：生成 HTP weight-sharing context 序列化文件（可部署到 8 Gen4 Android 设备）。
  - HTP 后端配置：`HtpConfigFile_*.json`、`htp_backend_ext_config*.json`（按目标芯片改 soc_id / dsp_arch）。
- **环境**：Ubuntu 22.04 + Python 3.10 + QNN SDK，按 `host_linux/README.md` 配置。
- **关键文件**：`example2/host_linux/qnn_compile_deploy.py`、`example2/G2G/MHA2SHA/`、`example2/G2G/split_onnx_utils/`
- **产物（给 example3）**：设备可执行的 context 序列化模型 binary

### example3 — 端侧运行（run on device with Genie）

- **目标**：在高通 SnapDragon 设备上，用 Genie 框架真正跑起来做推理。
- **核心步骤**：
  1. **生成 embedding 表 + 测试输入**（`export_embeding_table_and_test_vector_2_5_3B.py`，依赖 example1 的 `vl_utils`）：
     - 导出 LLM embedding 权重表 `embedding_weights_151936x2048.raw`；
     - 跑视觉模型（ViT）得到 vision embedding；
     - 拼接 vision + text embedding 成最终输入 `input_embeds.bin`。
  2. **推送运行时库到设备**：QNN libs（`libQnnHtp.so` 等）、`libGenie.so`、`genie-t2t-run`；
  3. **推送模型 + 配置到设备**：模型 binary、`qwen2.5vl.json`（Genie 配置）、`tokenizer.json`、`htp_backend_ext_config.json`（按目标芯片改 soc_id/dsp_arch）；
  4. **设备上运行**：
     ```bash
     ./genie-t2t-run -c qwen2.5vl.json -e input_embeds.bin -t embedding_weights_151936x2048.raw
     ```
- **关键文件**：`example3/Qwen2.5-VL-3B/export_embeding_table_and_test_vector_2_5_3B.py`、`qwen25vl3B_os.json`、`tokenizer.json`
- **已知 TODO**：Genie 目前不支持 mrope（旋转位置编码），后续在 Genie 中补充可提升精度。

---

## 阶段间数据流

```
原始 HF 模型 (Qwen2.5-VL-3B)
        │  example1：量化
        ▼
.onnx + .weight/.bias + .encodings
        │  example2：QNN 编译（MHA2SHA / split / compile）
        ▼
设备可执行模型 binary
        │  example3：推送设备 + Genie 运行
        ▼
端侧推理输出
```

---

## 芯片配置对照（HTP）

| 芯片 | HTP 配置 |
|---|---|
| 8gen3 | `htp_v73` |
| SA8295P | `htp_v68` |
| SA8797 | `htp_v81` |

> example2/example3 的 `htp_backend_ext_config.json` 需按实际目标芯片设置 `soc_id` / `dsp_arch`。
