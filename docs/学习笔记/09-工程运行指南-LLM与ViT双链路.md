# 09 · 工程运行指南：LLM 与 ViT 双链路

> 上游：[08 · ONNX 导出与测试向量](./08-ONNX导出与测试向量.md)
>
> 配套入口：[仓库根 README](../../README.md)
> 本篇范围：从 `example1` 的 LLM 量化，一直到 `example2` 主机编译、视觉编码器分支和 `example3` 端侧汇合。

## 一、先记住六个结论

1. 主目录的 `example1/llm_quant.py` 量化的是 **LLM/KV Cache 分支**，它不会导入根目录的 `vit/`。
2. 当前 `use_input_embeddings: true` 时，example1 的校准数据处理仍会调用 Hugging Face 模型内置的浮点 `visual`，把图片先变成 vision embedding；这不等于量化了视觉模型。
3. 只验证 LLM，或者在 GPU/服务器上预先生成 vision embedding，可以暂时不跑独立 `vit/`。
4. 要实现“设备直接输入图片 → 设备输出文字”，必须补齐 `vit/qwen2_5_vl` 的 VEG/ViT 分支。
5. 主 example2 必须进入 `example2/host_linux` 后运行 `qnn_compile_deploy.py`。
6. 当前主 example2 脚本实际结束于 AR1/AR128 的 `*_quantized.dlc`；Context Binary 生成代码尚未启用。

一句话本质：**这个工程不是一条直线，而是 LLM 和视觉编码器两条编译链，最后在端侧推理阶段汇合。**

---

## 二、工程为什么分成两条链

Qwen2.5-VL 的一次图片问答包含两个核心模型：

```text
图片
  │
  ▼
视觉编码器 VEG / ViT
  │  输出 vision embedding
  ▼
语言模型 LLM
  │  自回归生成 token
  ▼
文本答案
```

因此，端侧部署也被拆成两条链：

```text
LLM / KV Cache 分支
主 example1
  → AIMET 量化 LLM
  → ONNX + External Weight + Encoding + Test Vector
主 example2
  → AR1 / AR128 Quantized DLC
  → LLM Context Binary（当前需另行启用）

视觉编码器分支
vit/qwen2_5_vl/example1
  → AIMET 量化 VEG
  → veg.onnx + veg.encodings + RAW
vit/qwen2_5_vl/example2
  → VEG Quantized DLC + Context Binary

两条链在 example3 汇合
  → vision embedding 与 text embedding 拼接
  → Genie 调用 LLM 生成文本
```

### 2.1 三种容易混淆的“视觉模型”

| 名称 | 在哪里 | 用来做什么 | 是否是端侧视觉模型 |
|---|---|---|---|
| Hugging Face 内置 `model.visual` | 主 example1 的数据加载流程 | 为 LLM 校准样本生成 vision embedding | 否，仍是浮点校准工具 |
| 根目录 `vit/qwen2_5_vl` | 独立视觉量化与编译流程 | 把 VEG/ViT 变成 QNN 可部署模型 | 是 |
| example3 使用的视觉结果 | GPU 预计算或设备端 VEG | 提供最终送入 LLM 的 vision embedding | 取决于运行方式 |

所以“example1 用到了 visual”和“example1 依赖根目录 `vit/`”不是一回事。

---

## 三、推荐运行顺序

### 3.1 只学习或验证 LLM 量化

```text
主 example1
  → 检查量化后 PPL
  → 检查 ONNX / Encoding / Test Vector
```

此时不用跑根目录 `vit/`，也不用跑 example2、example3。

### 3.2 得到 LLM 的 QNN 量化 DLC

```text
主 example1
  → 主 example2
  → AR1 / AR128 Quantized DLC
```

### 3.3 完整端侧多模态部署

```text
主 example1 → 主 example2 → LLM 端侧产物
vit/qwen2_5_vl/example1 → example2 → VEG 端侧产物
两者 → example3 → 图片问答
```

---

## 四、第一步：运行主 example1

example1 和 example2 建议使用两个独立的 Python 3.10 环境：前者当前使用 CUDA 版 PyTorch 2.6，后者的 requirements 固定 PyTorch 1.13.1。根 `req.txt` 安装完成后，还要按 `example1/README.md` 把不兼容的 Transformers 5.x 固定为 `transformers==4.49.0`。

### 4.1 example1 做什么

主脚本是 `example1/llm_quant.py`，主要完成：

```text
加载模型与数据
  → 浮点 PPL 基线
  → Prepare 固定图模型
  → QuantSim
  → SeqMSE 优化权重 Encoding
  → compute_encodings 标定激活和 KV Cache
  → 量化后 PPL
  → 导出 Test Vector、ONNX、Encoding 和外置权重
```

### 4.2 修改配置

先检查 `example1/config.yaml`：

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

如果 `<output_dir>/prepare` 还不存在，首次运行应设置：

```yaml
quantization:
  skip_prepare: false
```

Prepare 成功且产物没有变化时，后续才可以改成 `true` 复用。

### 4.3 运行命令

在仓库根目录执行：

```bash
source /你的虚拟环境/env310/bin/activate

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
bash example1/run.sh 2>&1 | tee llm_quant.log
```

优先使用 `run.sh`。它会先设置 QAIRT 原生库路径和临时目录，再启动 Python；直接裸跑 `python example1/llm_quant.py` 容易在导入阶段遇到动态库问题。

### 4.4 验收产物

```bash
OUT=$(python -c "import yaml; print(yaml.safe_load(open('example1/config.yaml'))['environment']['output_dir'])")

test -s "$OUT/onnx/qwen25llm.onnx" && echo "ONNX OK"
test -s "$OUT/onnx/qwen25llm.encodings" && echo "Encoding OK"
test -s "$OUT/test_vectors/qt_0.pkl" && echo "QT Test Vector OK"

find "$OUT/onnx" -maxdepth 1 -type f \
  \( -name '*.weight' -o -name '*.bias' -o -name '*.data' \) \
  | head

du -sh "$OUT/onnx" "$OUT/test_vectors"
```

进入 example2 前，至少保证以下四类数据完整：

```text
qwen25llm.onnx
qwen25llm.encodings
全部 ONNX External Weight / Bias / Data
test_vectors/qt_0.pkl
```

---

## 五、第二步：运行主 example2

### 5.1 example2 做什么

主入口为 `example2/host_linux/qnn_compile_deploy.py`。当前脚本依次执行：

```text
AR1073 原始导出图
  → 生成 AR1 / AR128 定长图
  → Split ONNX + RAW / Golden Output
  → MHA2SHA
  → qairt-converter：ONNX → 普通 DLC
  → qairt-quantizer：普通 DLC → Quantized DLC
```

其中：

- AR1 通常服务于逐 token Decode；
- AR128 通常服务于多 token Prefill；
- Context Length 当前固定为 2048；
- ARN、Context Length 和模型名必须和 example1 导出配置一致。

### 5.2 先修改硬编码

当前脚本中至少有以下机器相关配置：

```python
LLAMA_MODELS = "/root/autodl-tmp/zgj/Qwen25/outputs/output"
QNN_SDK_ROOT = "/root/autodl-tmp/zgj/tools/qairt/2.42.0.251225"
ARNs = [1, 128]
EXPORT_AR = 1073
EXPORT_CONTEXT_LENGTH = 2048
```

不要照抄路径；应把前两项改成当前机器的真实目录，并核对后面三个 Shape 参数。

还要按实际目标设备统一芯片配置。当前示例中的 example1 HTP 配置与 example2 的 `GEN4`/JSON SoC、DSP 示例并非同一套参数，不能直接混用。

### 5.3 运行前检查

必须先进入工作目录：

```bash
cd ~/autodl-tmp/zgj/code/qwen25/example2/host_linux
```

脚本以 `os.getcwd()` 作为 `workfolder`。如果从仓库根目录运行，模块搜索路径和 `assets/` 输出位置都会不正确。

然后检查输入、工具和资源：

```bash
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

注意：`utilities/profiler.py` 会导入 `psutil`，但当前 `host_linux/requirements.txt` 没有列出它。两份 requirements 中的 NumPy/ONNX 版本也不一致，不建议在已跑通的环境里连续无脑安装两份依赖文件。

### 5.4 正式执行

```bash
PYTHONUNBUFFERED=1 \
python qnn_compile_deploy.py 2>&1 | tee qnn_compile.log
```

首次排错可以先把 `ARNs` 临时设为 `[1]`，跑通 AR1 后再单独设为 `[128]`，最后恢复 `[1, 128]`。这样更容易定位错误，也能降低并行阶段的资源峰值。

### 5.5 每一阶段都要验收

脚本中的 `All ... done` 不能作为唯一成功依据。应检查目标文件是否真实存在且非空：

```bash
# 1. AR1 / AR128 模型
du -sh assets/models_ar_n/ar1-cl2048 assets/models_ar_n/ar128-cl2048

# 2. Split ONNX
ls -lh assets/artifacts/ar{1,128}-cl2048/split_onnx/
ls -lh assets/artifacts/ar{1,128}-cl2048/input_list_*.txt

# 3. MHA2SHA
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/sha_output/

# 4. 普通 DLC
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/converted_model/*.dlc

# 5. 当前脚本最终产物
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/compiled_model/*_quantized.dlc
```

### 5.6 为什么日志说 done 仍可能失败

当前脚本存在几处需要特别留意的控制流：

- `executor.map()` 的返回结果没有被遍历，子进程异常可能不会在父进程中重新抛出；
- 部分异常分支使用成功退出码；
- MHA2SHA、Converter 和 Quantizer 没有统一严格检查子进程 `returncode`；
- 因而后续仍可能打印 `done`。

判断是否成功应以“日志没有真实错误 + 目标文件存在且非空”为准。

### 5.7 当前 Context Binary 边界

主 example2 中，`qnn-context-binary-generator` 的示例命令位于三引号字符串中，自动调用代码也被注释。因此：

```text
直接运行当前 qnn_compile_deploy.py
  → 得到 AR1 / AR128 Quantized DLC
  → 不会自动得到 Context Binary
```

Context Binary 是后续部署目标，但在启用它之前，还要按目标芯片核对 SoC ID、DSP 架构、HTP 配置和 weight sharing 组合。

---

## 六、第三步：什么时候运行独立 ViT/VEG

### 6.1 不需要运行的情况

- 只学习或验证 LLM 量化；
- 只编译 LLM 的 Quantized DLC；
- 纯文本推理；
- vision embedding 已在 GPU/服务器上预先计算。

### 6.2 必须运行的情况

要让高通设备直接接收图片，并在端侧执行完整 Qwen2.5-VL 推理，就需要运行：

```text
vit/qwen2_5_vl/example1/veg.ipynb
  → vit/qwen2_5_vl/example2/qnn_model_prepare_for_veg.ipynb
```

不要误用旧的 `vit/qwen2_vl/`。旧目录对应 Qwen2-VL，不是本工程主模型 Qwen2.5-VL-3B。

### 6.3 视觉 example1 的输入输出

`veg.ipynb` 会提取并封装模型的 `visual`，其固定图输入包括：

```text
pixel_values
RoPE cos
RoPE sin
window attention mask
full attention mask
```

输出是：

```text
vision_embedding
```

主要导出内容包括：

```text
veg.onnx
veg.encodings
外置权重
五种输入 RAW
golden_output.raw
```

### 6.4 为什么优先运行 Notebook

当前视觉分支的 `.py` 是 Notebook 导出的中间版本，并未完全整理成稳定 CLI：

- `qnn_model_prepare_for_veg.py` 仍有旧机器路径；
- 它读取了不匹配的旧配置路径；
- 文件中有无条件 `exit()`，后续编译代码不可达；
- 后面还残留 `get_ipython()` 等 Notebook 语句。

因此当前应以两个 `.ipynb` 为准，并在执行前逐项修改模型、COCO、输出目录、QAIRT 和目标芯片配置。

---

## 七、第四步：example3 如何把两条链汇合

example3 的职责不是重新量化模型，而是准备和组织端侧运行：

1. 导出 LLM Embedding Table；
2. 从 GPU 预计算或设备端 VEG 获取 vision embedding；
3. 用 vision embedding 替换序列中的图片 token embedding；
4. 拼接视觉和文本输入；
5. 推送 QNN/Genie 库、模型、Tokenizer 和配置；
6. 调用 `genie-t2t-run` 生成文本。

示例：

```bash
./genie-t2t-run \
  -c qwen2.5vl.json \
  -e input_embeds.bin \
  -t embedding_weights_151936x2048.raw
```

因此，是否必须部署 ViT 取决于 vision embedding 在哪里生成：

```text
服务器/GPU 生成 → 设备只运行 LLM
设备端生成      → 设备同时运行 VEG 与 LLM
```

---

## 八、example2 报错时的排查顺序

### 8.1 先找第一个真实错误

```bash
grep -nEi \
  'error|exception|traceback|failed|killed|no space|not found|out of memory' \
  qnn_compile.log | head -100

tail -n 200 qnn_compile.log
```

### 8.2 检查 OOM

```bash
dmesg -T 2>/dev/null \
  | grep -Ei 'killed process|out of memory|oom' \
  | tail -30
```

历史日志中 MHA2SHA 峰值内存超过 50GB，量化 DLC 阶段也使用过约 40GB，因此进程突然消失时要优先检查 Linux OOM Killer。

### 8.3 高频原因

- 从错误目录启动；
- `LLAMA_MODELS` 或 `QNN_SDK_ROOT` 仍是旧机器路径；
- 外置权重没有和 ONNX 一起复制；
- 缺少 `.encodings`、`qt_0.pkl`、`psutil` 或 MHA2SHA 可执行文件；
- example1 和 example2 的模型名、ARN、Context Length 不一致；
- Python、ONNX 或 NumPy 版本冲突；
- QAIRT 动态库路径不完整；
- CPU RAM 或磁盘不足。

---

## 九、为什么 `assets/` 会达到几十 GB

主 example2 的 `assets/` 不是源码依赖，而是运行生成的模型和中间产物：

```text
assets/models_ar_n
  → 从原始 AR1073 改成 AR1 / AR128 后的完整模型副本

assets/artifacts
  → Split ONNX、RAW、Golden、SHA ONNX、普通 DLC、量化 DLC
```

同一套大模型在多个阶段、多个 ARN 下被复制或重新导出，所以出现几十 GB 很正常。它只说明 example2 至少启动并生成过产物，**不能反推出 example1 在当前目录成功完整运行过**；example2 也可能使用了其他位置复制来的 example1 输出。

清理前应先判断：

- 任务是否仍在运行；
- 哪一阶段需要断点复用；
- 最终 `*_quantized.dlc` 或 Context Binary 是否已经另行备份；
- example1 的原始 ONNX、Encoding 和 Test Vector 是否仍安全保留。

---

## 十、完整执行检查表

### LLM example1

- [ ] QAIRT、模型、数据集和输出路径已修改
- [ ] 首次运行时 `skip_prepare: false`
- [ ] 从仓库根目录执行 `bash example1/run.sh`
- [ ] PPL 没有明显崩坏
- [ ] ONNX、Encoding、外置权重和 `qt_0.pkl` 均存在

### LLM example2

- [ ] 已修改 `LLAMA_MODELS` 和 `QNN_SDK_ROOT`
- [ ] ARN、Context Length、模型名和 example1 一致
- [ ] 已进入 `example2/host_linux`
- [ ] `qairt-converter`、`qairt-quantizer`、MHA2SHA 可执行
- [ ] CPU RAM 和磁盘足够
- [ ] AR1/AR128、Split、SHA、普通 DLC、量化 DLC 分阶段验收
- [ ] 没把日志里的 `done` 当成唯一成功依据
- [ ] 知道当前脚本不会自动生成 Context Binary

### 完整多模态

- [ ] 明确 vision embedding 在 GPU 还是设备端生成
- [ ] 若设备端生成，运行的是 `vit/qwen2_5_vl` 而非旧 `vit/qwen2_vl`
- [ ] 视觉分支优先使用 Notebook
- [ ] VEG 与 LLM 端侧产物的芯片配置一致
- [ ] example3 的 Genie、Tokenizer、Embedding Table 和输入准备完成

---

## 十一、一句话总结

> **先跑主 example1 得到 LLM 的 ONNX/Encoding/Test Vector，再从主 example2 编译出 AR1/AR128 Quantized DLC；只有设备要直接处理图片时，才额外运行 `vit/qwen2_5_vl`，最后由 example3 把 VEG 输出的 vision embedding 送进 LLM。**
