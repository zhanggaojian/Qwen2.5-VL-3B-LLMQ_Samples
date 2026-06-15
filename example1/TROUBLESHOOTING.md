# Qwen2.5-VL-3B 量化（example1）问题与解决方案汇总

> 记录 example1 从环境搭建到成功导出 ONNX 全过程遇到的问题与解决方案，便于换机 / 换人时快速排查。
> 运行平台：AutoDL 容器，Python 3.10，QAIRT SDK 2.42.0.251225，AIMET。

---

## 一、环境搭建类

### 1. `ModuleNotFoundError: No module named 'utilities'`

- **原因**：`llm_quant.py` 用了 `sys.path.append('../')`，这是**相对当前工作目录（CWD）**的路径。在父目录或用 `run.sh` 启动时 CWD 变了，找不到上层模块。
- **解决**：改成基于脚本自身位置的绝对路径，同时让 `mixed_precision_config_file` 也走绝对路径。

```python
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

_mp_cfg_file = _quant['mixed_precision_config_file']
if not os.path.isabs(_mp_cfg_file):
    _mp_cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), _mp_cfg_file)
```

### 2. `ImportError: libc++.so.1: cannot open shared object file`

- **原因**：
  1. QAIRT SDK 原生库依赖 `libc++.so.1`，系统里缺失；
  2. 原来把 `LD_LIBRARY_PATH` 设在 Python 脚本**内部**——对动态链接器来说太晚，必须在 Python 启动**之前**设好。
- **解决**：
  1. `apt-get install -y libc++1 libc++abi1` 安装系统库；
  2. 新建 `run.sh`，在启动 Python 前 `export LD_LIBRARY_PATH`，并自动在 SDK 内搜索 `libc++.so*` 把其目录加进去。

```bash
QNN_SDK_ROOT="$(...从 config.yaml 读取...)"
export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
LIBCXX_FILE="$(find "${QNN_SDK_ROOT}" -name "libc++.so*" -print -quit)"
[[ -n "${LIBCXX_FILE}" ]] && export LD_LIBRARY_PATH="$(dirname "${LIBCXX_FILE}"):${LD_LIBRARY_PATH}"
```

> ⚠️ AutoDL 系统盘会被重置，重置后 `libc++` 需要重新 `apt-get install`。

### 3. `ImportError: Python version mismatch: module was compiled for Python 3.10, but interpreter is 3.12.3`

- **原因**：QAIRT SDK 的原生二进制 `libPyIrGraph.so` 是**针对 Python 3.10 编译**的（`ldd` 可见它链接 `libpython3.10.so.1.0`）。C 扩展的 ABI 与具体小版本绑定，所以"3.10–3.12 都行"不适用于这种预编译 `.so`。
- **解决**：建原生 Python 3.10 的 `venv`（命名 `env310`），在其中安装全部依赖。

### 4. `AttributeError: 'Qwen2_5_VLConfig' object has no attribute 'num_hidden_layers'`

- **原因**：`req.txt` 里 `transformers==5.2.0` 改了 Qwen2.5-VL 的 config 结构，和工程里 vendored 代码的预期不符。
- **解决**：降级到 `transformers==4.49.0`。

### 5. `ModuleNotFoundError: No module named 'datasets'`

- **原因**：`req.txt` 漏了 `datasets`。
- **解决**：`pip install datasets`。

---

## 二、Prepare / ONNX 导出类

### 6. `ValueError: prepared artifacts not found in .../prepare`

- **原因**：`config.yaml` 里 `skip_prepare: true`，但 prepare 产物还没生成过。
- **解决**：先设 `skip_prepare: false` 生成一次，之后再设回 `true` 复用。

### 7. `KeyError:`（空字符串）— tracing 时 `hidden_states = outputs[0]`

- **原因**：JIT trace 时 `transformers.ModelOutput` 对象的整数索引（`outputs[0]`）会失败。
- **解决**：prepare 前强制模型返回 tuple。

```python
setattr(llm_config, 'return_dict', False)
model.config.return_dict = False
```

### 8. `RuntimeError: Cannot insert a Tensor that requires grad as a constant.`

- **原因**：ONNX tracing 时把 `requires_grad=True` 的权重当常量插入。
- **解决**：prepare 前 `model.eval()` + `model.requires_grad_(False)`。

### 9. `RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library.`

- **原因**：`torch.onnx.export` 的 shape inference pass 会把整个大模型图序列化进内存，超过 protobuf 2GB 上限。
- **解决**：在任何 export 调用**之前**，把内部 shape inference pass 打补丁成空操作（并移到无条件执行的位置，使 `skip_prepare=true` 时最终 export 也生效）。

```python
import torch.onnx
torch.onnx._globals.GLOBALS.onnx_shape_inference = False
for _pass_name in ('_jit_pass_onnx_node_shape_type_inference',
                   '_jit_pass_onnx_graph_shape_type_inference'):
    if hasattr(torch._C, _pass_name):
        setattr(torch._C, _pass_name, (lambda *a, **k: None))
```

---

## 三、磁盘空间类（最反复的一类）

> 3B 模型 fp32 全流程峰值需 ~40–50GB，100G 数据盘几乎刚好够，期间多次 `OSError: [Errno 28] No space left on device`。

### 10. 临时文件写满系统盘

- **原因**：ONNX 导出的大临时文件默认写到小的系统盘 `/tmp`。
- **解决**：`run.sh` 里把 `TMPDIR/TEMP/TMP` 指到数据盘 `output_dir/.tmp`。

```bash
OUTPUT_DIR="$(...从 config.yaml 读取...)"
TMPDIR_PATH="${OUTPUT_DIR}/.tmp"
mkdir -p "${TMPDIR_PATH}"
export TMPDIR="${TMPDIR_PATH}"
export TEMP="${TMPDIR_PATH}"
export TMP="${TMPDIR_PATH}"
```

### 11. 数据盘 100G 被占满

- **原因**：模型产物 + 历次中间文件累积（prepare/ 47G、onnx/ 31G）。
- **解决（组合拳）**：
  1. **数据集裁剪**：COCO train2017 原 19G / 118287 张，分析 dataloader 发现实际只用 ~100 张校准/评估，裁到 101 张（几十 MB）；
  2. **prepare/ 清理**：删 UUID 临时目录、中间 `.onnx`、`_Constant_*`、`*.weight`/`*.bias` 等，保留 `*.py`/`*.safetensors`/`*.json`；
  3. **onnx/ 清理**：删历次失败的残留产物；
  4. **`skip_prepare: true`**：复用已生成的 prepare 产物，避免每次重生成 ~35G 中间文件，显著降低磁盘峰值。

---

## 四、最终导出阶段的"警告"（非错误）

成功收尾时出现两条 WARNING，**都正常**：

- **"layers not found in exported onnx model"**（q/k/v proj 的激活量化点）：encodings 只进了 torch encodings 文件、没进 onnx encodings，日志注明"若该层本就不运行则无影响"，KV-cache LLM 导出常见，不影响主干。
- **"Quantsim export will stop exporting encodings ... 用 save_encodings_to_json()"**：未来版本弃用提示，当前照常导出。

---

## 五、最终产物（确认成功）

| 文件 | 大小 | 是什么 | example2（QNN 编译）是否需要 |
|---|---|---|---|
| `qwen25llm.onnx` | 932K | ONNX 图结构（小是因为权重外置，正常） | ✅ 需要 |
| 大量 `*.weight` / `*.bias` | 各几 M | ONNX 的外置权重数据（模型 >2GB，权重单独存盘） | ✅ 需要（必须与 `.onnx` 同目录） |
| `qwen25llm.encodings` | 66M | ONNX 格式量化 encodings（QNN 用的就是这个） | ✅ 需要 |
| `qwen25llm.pth` | 12G | torch 权重 checkpoint | ❌ 不需要，可删 |
| `qwen25llm_torch.encoding` | 336M | torch 格式 encodings | ❌ QNN 不用，可删 |

**交给 example2 的输入（三类，必须同目录）**：

```
qwen25llm.onnx
qwen25llm.encodings
所有 *.weight / *.bias 文件   ← 不能漏，是 .onnx 的外置权重
```

**想省空间可删（约 12.3G）**：

```bash
rm .../onnx/qwen25llm.pth
rm .../onnx/qwen25llm_torch.encoding
```

---

## 六、涉及改动 / 新增的文件（部署到远程拷这几个）

| 文件 | 类型 | 说明 |
|---|---|---|
| `example1/config.yaml` | 新建 | 集中管理所有配置 |
| `example1/llm_quant.py` | 修改 | 读 YAML 配置 + 第 6/7/8/9 处导出修复 |
| `example1/run.sh` | 新建 | 启动前设 `LD_LIBRARY_PATH` 和 `TMPDIR`，再拉起 Python |

---

## 附：依赖版本要点

- Python：**3.10**（必须，匹配 QAIRT 原生库）
- `transformers==4.49.0`（不能用 5.x）
- 额外安装：`datasets`、系统库 `libc++1 libc++abi1`
- PyTorch：`+cu118` 对应 wheel
