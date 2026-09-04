# Qwen2.5-VL-3B LLMQ Samples · 运行指南

> 本仓库用于把 Qwen2.5-VL-3B 的语言模型和视觉编码器分别量化、编译，并最终部署到高通 HTP/NPU。
>
> **先记结论**：只验证 LLM 量化与编译时，不需要运行根目录 `vit/`；只有要完成“设备输入图片 → 设备输出文字”的完整多模态部署，才需要再跑独立的视觉编码器分支。

详细原理和逐步检查见 [09 · 工程运行指南：LLM 与 ViT 双链路](./docs/学习笔记/09-工程运行指南-LLM与ViT双链路.md)。

---

## 一、工程整体结构

这不是一条单线流程，而是 LLM 与视觉编码器两条分支最终汇合：

```text
LLM / KV Cache 分支
example1
  → AIMET 量化 LLM
  → ONNX + External Weight + Encoding + Test Vector
  → example2
  → AR1 / AR128 Quantized DLC
  → LLM Context Binary（当前脚本需要另行启用生成步骤）

视觉编码器分支
vit/qwen2_5_vl/example1
  → AIMET 量化 VEG / ViT
  → veg.onnx + veg.encodings + RAW
  → vit/qwen2_5_vl/example2
  → VEG Quantized DLC / Context Binary

最终在 example3 汇合
图片 → VEG → vision embedding → LLM → 文本
```

| 目标 | 需要运行 |
|---|---|
| 学习或验证 LLM 量化 | 主 `example1` |
| 得到 LLM 量化 DLC | 主 `example1 → example2` |
| 文本或外部预生成 Embedding 推理 | 主 LLM 分支即可 |
| 设备直接接收图片并生成文字 | 主 LLM 分支 + `vit/qwen2_5_vl` 视觉分支 + `example3` |

---

## 二、运行环境

- Ubuntu 22.04 x86_64；不要直接在 Windows PowerShell 中运行 QAIRT Linux 工具。
- Python 3.10，与当前 QAIRT 原生 Python 扩展 ABI 保持一致。
- QAIRT/QNN SDK；当前示例路径和已验证版本为 `2.42.0.251225`。
- 主 example1 需要 NVIDIA GPU；SeqMSE、Calibration 和 PPL 会执行 CUDA 前向。
- 主 example2 不使用 GPU，但 MHA2SHA/QNN 转换需要较大的 CPU RAM 和磁盘空间。
- 当前历史日志中 example2 的 MHA2SHA 峰值 RAM 超过 50GB；运行前务必检查 `free -h`。

建议给 example1 和 example2 使用两个独立 Python 3.10 环境：根目录 `req.txt` 使用 `torch==2.6.0+cu118`，而 `example2/host_linux/requirements.txt` 使用 `torch==1.13.1`。另外，当前根依赖中的 `transformers==5.2.0` 与这份模型适配代码不兼容；安装后需按 [example1/README.md](./example1/README.md) 固定回 `transformers==4.49.0`。

主 example1 的依赖安装、版本冲突和常见问题见 [example1/README.md](./example1/README.md) 与 [example1/TROUBLESHOOTING.md](./example1/TROUBLESHOOTING.md)。

---

## 三、先跑主 example1：量化 LLM

### 3.1 检查配置

修改 [example1/config.yaml](./example1/config.yaml)，至少确认：

```yaml
environment:
  qnn_sdk_root: <QAIRT SDK 路径>
  model_id: <Qwen2.5-VL-3B-Instruct 路径>
  cache_dir: <缓存目录>
  output_dir: <输出目录>

dataset:
  calibration_dataset_path: <校准 JSON>
  ppl_evaluation_dataset_path: <PPL JSON>
  image_dataset_path: <图片根目录>
```

如果 `<output_dir>/prepare` 还没有 prepared artifact，首次运行设置：

```yaml
quantization:
  skip_prepare: false
```

Prepare 成功后，后续复用产物时可以改回 `true`。

### 3.2 执行

在仓库根目录运行：

```bash
source /你的虚拟环境/env310/bin/activate

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
bash example1/run.sh 2>&1 | tee llm_quant.log
```

优先使用 `run.sh`，不要直接裸跑 `python example1/llm_quant.py`。启动脚本会在 Python 加载 QAIRT 原生库以前设置 `LD_LIBRARY_PATH`，并把临时目录放到大容量输出盘。

### 3.3 验收 example1

当前配置的输出目录可以这样读取：

```bash
OUT=$(python -c "import yaml; print(yaml.safe_load(open('example1/config.yaml'))['environment']['output_dir'])")
echo "$OUT"

test -s "$OUT/onnx/qwen25llm.onnx" && echo "ONNX OK"
test -s "$OUT/onnx/qwen25llm.encodings" && echo "Encoding OK"
test -s "$OUT/test_vectors/qt_0.pkl" && echo "QT Test Vector OK"

find "$OUT/onnx" -maxdepth 1 -type f \
  \( -name '*.weight' -o -name '*.bias' -o -name '*.data' \) \
  | head

du -sh "$OUT/onnx" "$OUT/test_vectors"
```

进入 example2 以前，至少要确认：

```text
qwen25llm.onnx
qwen25llm.encodings
全部 External Weight/Bias/Data
test_vectors/qt_0.pkl
```

---

## 四、再跑主 example2：编译 LLM

### 4.1 先核对硬编码

[example2/host_linux/qnn_compile_deploy.py](./example2/host_linux/qnn_compile_deploy.py) 当前硬编码了：

```python
LLAMA_MODELS = "/root/autodl-tmp/zgj/Qwen25/outputs/output"
QNN_SDK_ROOT = "/root/autodl-tmp/zgj/tools/qairt/2.42.0.251225"
ARNs = [1, 128]
EXPORT_AR = 1073
EXPORT_CONTEXT_LENGTH = 2048
```

这些值必须与 example1 的真实输出和目标部署配置一致。

目标芯片参数也必须统一。当前示例里 example1 使用 `htp_v73`，而 example2 的 `GEN4`/JSON 示例使用另一套 SoC、DSP 参数；它们不能不加检查地混用，需按实际设备统一 `soc_id`、`dsp_arch` 和 HTP 配置。

### 4.2 运行前检查

```bash
cd ~/autodl-tmp/zgj/code/qwen25/example2/host_linux

test -f /root/autodl-tmp/zgj/Qwen25/outputs/output/onnx/qwen25llm.onnx
test -f /root/autodl-tmp/zgj/Qwen25/outputs/output/onnx/qwen25llm.encodings
test -f /root/autodl-tmp/zgj/Qwen25/outputs/output/test_vectors/qt_0.pkl

test -x /root/autodl-tmp/zgj/tools/qairt/2.42.0.251225/bin/x86_64-linux-clang/qairt-converter
test -x /root/autodl-tmp/zgj/tools/qairt/2.42.0.251225/bin/x86_64-linux-clang/qairt-quantizer
test -x ../G2G/MHA2SHA/bin/mha2sha-onnx-converter

python -c "import onnx, torch, numpy, psutil; print('Python dependencies OK')"
df -h .
free -h
```

必须先 `cd example2/host_linux`。脚本用 `os.getcwd()` 作为工作目录，从仓库根目录直接运行会导致模块路径和输出目录错误。

### 4.3 执行

```bash
PYTHONUNBUFFERED=1 \
python qnn_compile_deploy.py 2>&1 | tee qnn_compile.log
```

首次排错可先把 `ARNs` 临时改为 `[1]`，确认 AR1 全阶段跑通后再单独验证 `[128]`，最后恢复 `[1, 128]`。这样能更快定位失败阶段并降低并行峰值资源占用。

当前脚本依次执行：

```text
AR1073 → AR1 / AR128
  → Split ONNX + RAW / Golden
  → MHA2SHA
  → qairt-converter：ONNX → DLC
  → qairt-quantizer：普通 DLC → Quantized DLC
```

### 4.4 分阶段验收

不要只相信日志中的 `All ... done`，必须检查文件：

```bash
# 1. AR 模型
du -sh assets/models_ar_n/ar1-cl2048 assets/models_ar_n/ar128-cl2048

# 2. Split ONNX
ls -lh assets/artifacts/ar{1,128}-cl2048/split_onnx/

# 3. MHA2SHA
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/sha_output/

# 4. 普通 DLC
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/converted_model/*.dlc

# 5. 当前脚本的最终产物：量化 DLC
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/compiled_model/*_quantized.dlc
```

### 4.5 当前代码边界

- 当前 `qnn-context-binary-generator` 示例命令位于三引号字符串中，自动生成代码也被注释。
- 因此直接运行当前脚本，真实终点是 AR1、AR128 两份 `*_quantized.dlc`，不是 Context Binary。
- `executor.map()` 的结果没有被遍历，子进程异常可能没有向父进程传播。
- MHA2SHA、Converter 和 Quantizer 没有严格检查 `returncode`；即使失败，后面仍可能打印 `done`。
- 当前会生成约几十 GB 中间产物；重复运行前要先确认旧结果是否需要保留。

---

## 五、`vit/` 视觉编码器分支什么时候运行

主 `example1/llm_quant.py` 不导入根目录 `vit/`，但当前 `use_input_embeddings=true` 时，数据加载器会调用 Hugging Face 模型内置的浮点视觉编码器，为校准样本生成 image embedding。

这和“量化并部署视觉编码器本身”是两件事：

```text
主 example1：使用浮点 visual 生成校准 embedding，只量化 LLM
独立 vit：量化 visual/VEG 本身，生成端侧视觉模型
```

完整多模态部署应使用：

```text
vit/qwen2_5_vl/example1/veg.ipynb
  → vit/qwen2_5_vl/example2/qnn_model_prepare_for_veg.ipynb
```

不要误用旧版本：

```text
vit/qwen2_vl/
```

当前视觉分支注意事项：

- 优先以 `.ipynb` 为准。
- `qnn_model_prepare_for_veg.py` 带有旧机器绝对路径、无条件 `exit()` 和 Notebook 专用 `get_ipython()`，不能直接作为完整脚本运行。
- `veg_config.json` 中的模型、输出、COCO 和 QAIRT 路径都需要按当前机器修改。
- ViT 固定图与输入图片分辨率绑定；修改分辨率后需要重新生成辅助输入和模型。

如果只做文本推理，或者 vision embedding 在 GPU/服务器上预生成，可以暂时不运行这一分支。

---

## 六、example3：设备端运行

example3 负责：

1. 导出 LLM Embedding Table；
2. 获取 vision embedding（GPU 预计算或端侧 ViT）；
3. 拼接 text/vision embedding；
4. 推送 QNN/Genie 库、模型、Tokenizer 和配置；
5. 在设备上运行 `genie-t2t-run`。

示例命令：

```bash
./genie-t2t-run \
  -c qwen2.5vl.json \
  -e input_embeds.bin \
  -t embedding_weights_151936x2048.raw
```

详见 [example3/README.md](./example3/README.md)。

---

## 七、example2 报错时怎么查

先找第一个真实错误，不要只看最后一行：

```bash
grep -nEi \
  'error|exception|traceback|failed|killed|no space|not found|out of memory' \
  qnn_compile.log | head -100

tail -n 200 qnn_compile.log
```

检查是否被 OOM Killer 终止：

```bash
dmesg -T 2>/dev/null \
  | grep -Ei 'killed process|out of memory|oom' \
  | tail -30
```

高频原因：

- 没有从 `example2/host_linux` 启动；
- `LLAMA_MODELS` 或 `QNN_SDK_ROOT` 路径错误；
- ONNX 外置权重没有完整复制；
- 缺少 `qt_0.pkl`、`psutil` 或 MHA2SHA 可执行文件；
- example1 的 ARN/Context/模型名与 example2 不一致；
- Python、ONNX、NumPy 版本冲突；
- CPU RAM 不足、磁盘写满或进程被杀死。

---

## 八、文档入口

- [example1 运行与依赖](./example1/README.md)
- [example1 完整流程](./docs/PIPELINE.md)
- [example1 排错记录](./example1/TROUBLESHOOTING.md)
- [三个 Example 的职责](./docs/EXAMPLES_OVERVIEW.md)
- [学习笔记总索引](./docs/学习笔记/README.md)
- [09 · 工程运行指南：LLM 与 ViT 双链路](./docs/学习笔记/09-工程运行指南-LLM与ViT双链路.md)

---

## 九、一句话总结

> **先跑主 example1 得到 ONNX/Encoding/Test Vector，再从 `example2/host_linux` 编译出 AR1/AR128 Quantized DLC；只有完整端侧图片推理才额外运行 `vit/qwen2_5_vl`，最后由 example3 把视觉结果和 LLM 串起来。**
