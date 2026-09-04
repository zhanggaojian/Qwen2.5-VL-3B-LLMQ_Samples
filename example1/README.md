# Quantization recipe of LLM(from Qwen2.5-VL)

## 系统依赖

- Computer with NVGPU(VRAM>64GB is fine、RAM > 32GB is fine)
- Ubuntu 22.04
- Python 3.10（当前 QAIRT 原生库按 Python 3.10 构建；不要直接使用 3.12）
  - 已验证 virtualenv/virtualenvwrapper
  - Anaconda 环境尚未验证
- Download the QAIRT SDK from Qualcomm® Software Center
  - https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community?osArch=Any&osType=All&version=2.42.0.251225
  - QAIRT version 2.36~2.46 is fine

## Calibration Sets & pip requirements

- Download llava_v1_5_mix665k.json from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- Download coco/train2017 from http://images.cocodataset.org/zips/train2017.zip
- Put the above files like this:
```shell
# tree -L 2 /data/huggingface/hf_dataset/
/data/huggingface/hf_dataset/
|-- coco
|   `-- train2017
|-- llava_v1_5_mix665k.json
```
- 安装 Python 依赖：

```bash
python -m pip install -r req.txt

# 当前 req.txt 中的 transformers 5.x 与项目代码不兼容，安装后必须覆盖为 4.49.0
python -m pip install "transformers==4.49.0" datasets
python -c "import transformers; print(transformers.__version__)"
```

最后一条命令应输出 `4.49.0`。不要在降级后再次执行 `pip install -r req.txt`，否则会重新装回 5.x。

---

## 运行 `llm_quant.py`

### 1. 先检查配置

编辑 [config.yaml](./config.yaml)，至少确认：

```yaml
environment:
  qnn_sdk_root: <QAIRT SDK 路径>
  model_id: <Qwen2.5-VL-3B 模型路径>
  cache_dir: <缓存目录>
  output_dir: <量化输出目录，预留足够磁盘空间>

dataset:
  calibration_dataset_path: <校准 JSON>
  ppl_evaluation_dataset_path: <PPL 评估 JSON>
  image_dataset_path: <图片根目录>
```

如果 `<output_dir>/prepare` 中还没有 prepared artifact，第一次运行设置：

```yaml
quantization:
  skip_prepare: false
```

Prepare 成功后，后续复用已有产物时再改为 `true`。

### 2. 推荐执行命令

在 Ubuntu 服务器的仓库根目录执行：

```bash
source /你的虚拟环境/env310/bin/activate

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
bash example1/run.sh 2>&1 | tee llm_quant.log
```

如果已经进入 `example1` 目录：

```bash
bash run.sh 2>&1 | tee llm_quant.log
```

优先使用 `run.sh`，不要直接裸跑 `python llm_quant.py`。启动脚本会在 Python 启动前设置 QAIRT 所需的 `LD_LIBRARY_PATH`，并把 `TMPDIR` 指向大容量输出盘，降低 ONNX 导出时系统盘写满的风险。

### 3. 后台运行

```bash
nohup env CUDA_VISIBLE_DEVICES=0 PYTHON=python \
bash example1/run.sh > llm_quant.log 2>&1 &

tail -f llm_quant.log
```

### 4. 主要输出

```text
<output_dir>/
├─ prepare/       # prepared PyTorch 模型；skip_prepare=true 时从这里加载
├─ test_vectors/  # fp_0.pkl、qt_0.pkl
└─ onnx/          # ONNX、外置 weight/bias、encodings
```

### 5. 常见启动问题

- `libc++.so.1` 缺失：安装 `libc++1 libc++abi1`，并通过 `run.sh` 启动。
- Python ABI mismatch：使用 Python 3.10，并确认虚拟环境中的 `python` 被实际调用。
- `num_hidden_layers` 不存在：确认 `transformers==4.49.0`，不要使用 5.x。
- 缺少 `datasets`：执行 `python -m pip install datasets`。
- `prepared artifacts not found`：首次运行把 `skip_prepare` 改为 `false`。
- `No space left on device`：检查 `output_dir` 空间，并确认日志中打印的 `TMPDIR` 位于大容量磁盘。

更完整的历史问题和解决方案见 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)。


