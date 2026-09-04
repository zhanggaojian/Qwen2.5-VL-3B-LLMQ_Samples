# 08 · ONNX 导出与测试向量

> **上游主篇**：[07 · 量化主流程：QuantSim 到 Encoding](./07-量化主流程-QuantSim到Encoding.md)
>
> **流程位置**：SeqMSE、`compute_encodings()` 和量化后 PPL 评估完成以后，`example1` 的最后阶段。
>
> **对应代码**：`example1/llm_quant.py` 约 L579～603；测试向量实现见 `example1/llm_utils/test_vectors.py` 约 L235～284。
>
> **本篇范围**：解释 Test Vector、最终 QuantSim Export、产物组成和 example2 交接；Prepare 与 Encoding 的计算原理只做必要承接。
>
> **一句话本质**：先生成 PyTorch 浮点路径与 QuantSim 路径的端侧对拍样本，再用固定 Shape Dummy Input 导出 ONNX、外置 Weight 和量化 Encoding，作为 `example2` QNN 主机编译阶段的输入。

---

## 零、先抓住七个重点

1. 这段代码包含两件不同的事：**生成 Test Vector** 和 **导出 ONNX + Encoding**。
2. `fp_0.pkl` 是同一个 prepared QuantSim 模型临时关闭 Quantizer 的结果，不是最初的 Hugging Face 原始模型。
3. `qt_0.pkl` 是 Quantizer 已启用、使用现有 Encoding 执行 QDQ 模拟后的结果。
4. Dummy Input 只负责描述输入顺序、Shape、dtype 和静态图，不参与 SeqMSE、Activation Calibration 或 PPL。
5. 当前固定接口是 76 个输入、73 个输出；其中 36 层 Past Key/Value 被展开成 72 个独立输入端口。
6. `quantsim.export()` 产出的 ONNX、外置 Weight 和 `.encodings` 还不是设备可执行的 INT4 Binary，后面仍要进入 `example2` 的 QNN 编译流程。
7. ONNX 主文件很小不代表模型没有权重；当前大模型 Weight 被外置为大量 `.weight/.bias` 文件，移动模型时一个都不能漏。

---

## 一、这段代码整体在做什么

对应代码：

```python
# generate test tensor for inference on edge with QNN
from llm_utils.test_vectors import generate_test_vectors

test_vector_layers = CONFIG['test_vector_layers']
with sim_fpm.place_on_device("cuda"):
    generate_test_vectors(
        quantsim,
        sim_fpm,
        train_dataloader,
        output_dir,
        num_batches=1,
        test_vector_layers=test_vector_layers,
        input_names=input_names,
    )

dummy_input = get_dummy_data(
    llm_config,
    tokenizer,
    'cuda',
    separate_tuple_input_output=True,
    num_tokens=ARN,
)

onnx_dir = os.path.join(output_dir, 'onnx')
os.makedirs(onnx_dir, exist_ok=True)

if enable_fp16:
    convert_model_to_fp32(quantsim.model)

onnx_api_args = OnnxExportApiArgs(
    input_names=input_names,
    output_names=output_names,
    opset_version=_export_cfg['onnx_opset_version'],
)

sample_inputs = change_tensor_device_placement(
    dummy_input,
    torch.device('cpu'),
)

quantsim.export(
    onnx_dir,
    model_name,
    sample_inputs,
    onnx_export_args=onnx_api_args,
)
```

完整流程：

```text
已经完成 Encoding 的 QuantSim 模型
                 │
                 ├─→ 用真实样本生成 fp / qt Test Vector
                 │
                 └─→ 用 Dummy Input 描述固定接口和计算图
                                      ↓
                             quantsim.export()
                                      ↓
                       ONNX + 外置 Weight + Encoding
                                      ↓
                              交给 example2
```

这一步不会再次搜索 Weight Encoding，不会重新统计 Activation Encoding，也不会计算 PPL。

---

## 二、Test Vector 和导出文件不是一回事

| 对比项 | Test Vector | ONNX + Encoding |
|---|---|---|
| 目的 | 数值对拍和逐层排错 | 描述部署计算图和量化规则 |
| 本次前向／Trace 输入 | `train_dataloader` 的真实样本 | 固定 Shape Dummy Input；图和权重仍来自 QuantSim 模型 |
| 是否保存模型 | 否 | 是 |
| 是否保存中间层数值 | 是，只记录配置指定的层 | 通常不保存某次运行的具体中间数值 |
| 是否决定 Encoding | 否，直接使用已有 Encoding | 否，只把已有 Encoding 导出 |
| 主要产物 | `fp_0.pkl`、`qt_0.pkl` | `.onnx`、`.weight/.bias`、`.encodings` |
| 下游用途 | Golden Output、拆图和端侧对拍 | QNN 转换、量化覆盖和编译 |

可以把两者理解为：

```text
ONNX + Encoding = 要部署的“模型说明书”
Test Vector      = 验证说明书是否被正确执行的“标准试题和参考答案”
```

---

## 三、`generate_test_vectors()` 在做什么

### 3.1 选择需要记录的中间层

当前配置：

```yaml
test_vector_layers:
  - "model_layers_\\d+_input_layernorm_Pow"
  - "model_layers_\\d+_Add_1"
  - "rms_norm_\\d+"
```

这些字符串是正则表达式，用来匹配 prepared 模型中的模块名称。工具会在匹配模块上注册 Forward Hook，记录这些模块的输入和输出。

它不会默认保存模型所有层，否则 3B 模型的中间 Tensor 会占用非常大的磁盘空间。

对当前 `output/prepare/qwen25llm_kvcache_36_layer.py` 做静态名称检查后，实际结果是：

| 配置正则 | 当前命中数量 |
|---|---:|
| `model_layers_\d+_input_layernorm_Pow` | 0 |
| `model_layers_\d+_Add_1` | 36 |
| `rms_norm_\d+` | 0 |

也就是说，当前预计只记录 36 层 `model_layers_i_Add_1`，另外两组旧命名没有命中。代码不会对“零命中”主动报警，所以必须检查生成文件中的实际 Layer Key。

### 3.2 每个 Batch 跑两条路径

核心循环：

```python
for vector_type in ['fp', 'qt']:
    with quantizers_state(sim, disabled=(vector_type == 'fp')):
        ...
        outputs = recorder.generate_layer_outputs(prepared_inputs, idx)
```

| `vector_type` | Quantizer 状态 | 含义 |
|---|---|---|
| `fp` | 临时移除所有 Quantizer | prepared QuantSim 结构的浮点参考路径 |
| `qt` | Quantizer 正常启用 | 使用已确定 Encoding 的 QDQ 模拟量化路径 |

因此，`num_batches=1` 不是只执行一次模型前向，而是对第一个 Batch 至少执行：

```text
第 1 次：FP 路径
第 2 次：QuantSim/QDQ 路径
```

`quantizers_state()` 是上下文管理器；退出 `fp` 上下文以后，Quantizer 会恢复，不会把量化模型永久改成浮点模型。

### 3.3 `fp` 为什么不是最初的原始模型

这里使用的是：

```python
forward_pass_manager.model
```

也就是 `sim_fpm.model`。它已经经过：

```text
Monkey Patch
→ Prepare
→ QuantSim 插桩
```

`fp` 只是临时关闭这个模型中的 Quantizer。因此它适合回答：

> 同一份端侧友好计算图，打开和关闭量化噪声后相差多少？

它不直接回答：

> 最初 Hugging Face Qwen2.5-VL 模型与端侧模型的全部改造误差是多少？

后一个问题由前面的原始 PPL、prepared PPL 和 QuantSim PPL 三阶段评估共同检查。

### 3.4 每个 `.pkl` 大致保存什么

`LLMLayerOutputUtil` 先保存 prepared 输入和匹配层的输入／输出，然后 `_sanitize_and_update_test_vectors()` 补充最终输出并整理 KV：

```text
fp_0.pkl / qt_0.pkl
└─ "0"
   ├─ inputs_embeds
   ├─ attention_mask
   ├─ position_ids_cos
   ├─ position_ids_sin
   ├─ past_key_values
   ├─ 匹配层名称
   │  ├─ input
   │  └─ output
   ├─ logits
   └─ output_key_values
```

实际 Key 会随 prepared 模块命名和 AIMET 版本略有变化，但总体分成四类：

1. 模型输入；
2. 指定中间层的输入和输出；
3. 最终 logits；
4. 输出的新 KV Cache。

所有 Tensor 在保存前都被移动到 CPU，避免 Pickle 中残留 CUDA Tensor 对加载环境造成不必要依赖。

### 3.5 为什么只生成一个 Batch

当前调用：

```python
num_batches=1
```

Test Vector 的主要目标是确定性对拍，不是估计数据分布，所以一个覆盖主路径的样本通常可以作为第一轮 Smoke Test：

```text
PyTorch FP
    ↓
PyTorch QuantSim
    ↓
ONNX Host
    ↓
QNN/HTP
```

如果需要覆盖不同输入内容、异常激活或多种 KV 状态，可以增加 Batch 数，但文件大小和生成时间也会明显增加。

---

## 四、Test Vector 怎样被下游使用

`example2` 会读取：

```text
output/test_vectors/*.pkl
```

当前 Split 主路径在 `split_onnx_utils/utils.py` 中只遍历 `source in ['qt']`，因此自动生成 Split 输入和 Golden Output 时实际消费的是 `qt_*.pkl`。`fp_*.pkl` 仍保留为浮点参考，可用于人工比较或扩展验证流程，但不能写成当前主路径已经同时消费 FP/QT 两套文件。

当前 `qt` Test Vector 主要用于：

- 拆分 ONNX 时确定 Split 边界输入和 Golden Output；
- 生成 Host/QNN 输入文件；
- 在端侧结果异常时逐层定位第一处明显偏差。

典型判断顺序：

```text
FP 与 QT 已经差很多
→ 量化误差或 Encoding 问题

QT 正常，但 ONNX/QNN 与 QT 差很多
→ 导出、算子转换、布局、Encoding 映射或后端实现问题

Host QNN 正常，但设备 HTP 异常
→ Context、设备配置、输入打包或 Runtime 问题
```

所以 Test Vector 不是部署模型本身。当前代码链中它从 `example1` 进入 `example2`；`example3` 没有直接读取这些 Pickle 文件。若要在设备执行阶段继续对拍，需要额外把 example3／端侧输出取回，并与 Test Vector 中的参考值进行比较。

---

## 五、导出 Dummy Input 是什么

### 5.1 它只是一套“接口模具”

```python
dummy_input = get_dummy_data(
    llm_config,
    tokenizer,
    'cuda',
    separate_tuple_input_output=True,
    num_tokens=ARN,
)
```

它负责告诉 ONNX Exporter：

- 一共有多少个输入；
- Tuple 中的输入顺序；
- 每个输入的 Shape 和 dtype；
- 固定 Mask、RoPE 和 KV Cache 布局；
- 用这套输入执行时会经过哪些算子。

它的数值内容是人为构造的：

- `inputs_embeds` 使用随机值；
- Past KV 使用零 Padding；
- 原始 Attention Mask 使用全 1，再构造成 combined causal mask；
- Position cos/sin 根据位置生成。

因此 Dummy Input 不应被拿去评价模型语义正确性。

Dummy Input 的完整构造细节见 [06-附录A · Prepare Dummy Input 输入模具](./06-附录A-Prepare-Dummy-Input输入模具.md)。

### 5.2 当前固定输入接口

当前配置：

```text
use_input_embeddings = true
use_position_embedding_input = true
num_hidden_layers = 36
ARN / Current Length = 1073
Context Length = 2048
Past KV Length = 975
```

输入数量：

```text
1 个 inputs_embeds
+ 1 个 attention_mask
+ 2 个 position_ids_cos / position_ids_sin
+ 36 × 2 个 Past Key / Past Value
= 76 个输入
```

关键 Shape：

| 输入 | 当前 Shape／含义 |
|---|---|
| `inputs_embeds` | `[1, 1073, 2048]` |
| `attention_mask` | `[1, 1, 1073, 2048]` |
| `position_ids_cos/sin` | `[1, 1, 1073, 64]`，RoPE 半维表示 |
| 每层 Past Key | `[1, 2, 128, 975]`，Key 已转置 |
| 每层 Past Value | `[1, 2, 975, 128]` |

输出数量：

```text
1 个 logits
+ 36 × 2 个新 Key / Value
= 73 个输出
```

当前主要输出 Shape：

| 输出 | 当前 Shape |
|---|---|
| `logits` | `[1, 1073, 151936]` |
| 每层 New Key | `[1, 2, 128, 1073]` |
| 每层 New Value | `[1, 2, 1073, 128]` |

### 5.3 为什么要把嵌套 KV 展平

PyTorch 内部可以使用：

```text
past_key_values[layer][key_or_value]
```

但 ONNX 对外需要一组明确、扁平的 Tensor 端口，因此会变成：

```text
past_key_0_in
past_value_0_in
...
past_key_35_in
past_value_35_in
```

输出同理使用 `_out` 后缀。

### 5.4 Dummy、Calibration 和 Test Vector 的区别

| 数据 | 是否真实 | 用途 | 是否影响 Encoding |
|---|---|---|---|
| Dummy Input | 否 | 建图、确认静态接口、ONNX Trace | 否 |
| SeqMSE Calibration | 是 | 搜索低位宽 Weight Encoding | 是 |
| `compute_encodings` Calibration | 是 | 观察 Activation，并处理可覆盖 Parameter Encoding | 是 |
| PPL Test Data | 是 | 评价端到端精度 | 否 |
| Test Vector Sample | 是 | 保存 Golden 输入和输出用于对拍 | 否 |

---

## 六、输入输出名称为什么很重要

### 6.1 当前输入输出名称

```python
onnx_api_args = OnnxExportApiArgs(
    input_names=input_names,
    output_names=output_names,
    opset_version=14,
)
```

当前输入名称从：

```text
inputs_embeds
attention_mask
position_ids_cos
position_ids_sin
```

开始，然后跟随 72 个 Past KV 名称。

输出名称从：

```text
logits
```

开始，然后跟随 72 个新 KV 名称。

### 6.2 名字和 Tuple 顺序必须同时正确

ONNX Exporter 会把：

```text
sample_inputs[0] ↔ input_names[0]
sample_inputs[1] ↔ input_names[1]
...
```

逐项绑定。因此只保证名称列表“看起来正确”还不够；`get_dummy_data()` 返回 Tuple 的顺序也必须完全一致。

如果顺序错误，最危险的情况不是立即报错，而是：

> Shape 恰好兼容，导出成功，但某个名称实际绑定了错误 Tensor。

这会在 example2 拆图、QNN 输入打包或端侧推理时变成非常难定位的问题。

### 6.3 为什么是静态图

这里没有传递 `dynamic_axes`，而 Dummy Input 又给出了全部具体 Shape，所以最终导出采用固定 Batch、Current Length 和 Past KV Length。

```text
Batch = 1
Current = 1073
Past = 975
Context = 2048
```

静态 Shape 更容易被 HTP 编译器做内存规划、Kernel 选择和图优化，但也意味着不同 Token 长度通常需要 Padding、切块或另一套 AR 图。

---

## 七、ONNX Opset 是什么

当前最终导出配置：

```yaml
export:
  onnx_opset_version: 14
```

Opset 不是：

- ONNX Python 包版本；
- 模型版本；
- Weight/Activation 位宽；
- QNN SDK 版本。

它描述 ONNX 图中标准算子的接口和语义版本。例如同一个 `Add`、`Slice` 或 `Resize` 在不同 Opset 中可能有不同输入形式或属性定义。

本项目配置中还有：

```yaml
prepare_opset_version: 20
onnx_opset_version: 14
```

二者服务于不同阶段：

| 配置 | 用途 |
|---|---|
| `prepare_opset_version=20` | 实际执行 QAIRT Prepare 时的内部转换与重建配置 |
| `onnx_opset_version=14` | 最终 QuantSim 模型交付 ONNX |

版本不同不必然冲突，但最终 Opset 必须同时满足：

1. PyTorch/AIMET 能成功表达当前图；
2. ONNX Checker 能解析；
3. example2 使用的 ONNX 和 QNN Converter 能支持；
4. 模型中的每个实际算子在 Opset 14 下都有正确表达。

当前 `skip_prepare=true`，本次运行会加载已有 prepared artifact，不会重新执行 Prepare。因此不能只根据当前 YAML 断言磁盘上的旧 prepared artifact 一定由 Opset 20 生成；这取决于它最初生成时的配置。

---

## 八、为什么 Dummy 先在 CUDA，随后又移到 CPU

代码先构造：

```python
dummy_input = get_dummy_data(..., 'cuda', ...)
```

随后执行：

```python
sample_inputs = change_tensor_device_placement(
    dummy_input,
    torch.device('cpu'),
)
```

前者复用了项目原有的 CUDA Shape/RoPE/KV 构造路径；后者满足传统 AIMET `quantsim.export()` 对 CPU Dummy Input 的要求。

这不是把整个 QuantSim 模型重新移到 CPU，也不是一次数据集推理，只是递归移动用于 Trace 的样例 Tensor。

---

## 九、`enable_fp16` 分支在做什么

```python
if enable_fp16:
    convert_model_to_fp32(quantsim.model)
```

如果前面的模型启用了 FP16，SeqMSE 以后代码已经无条件执行：

```python
sim_fpm.model.to(torch.float32)
```

因此生成 Test Vector 时主体 Parameter 已经转回 FP32，但部分 Norm 的 Cast Wrapper 仍可能保留。最终导出前的 `convert_model_to_fp32()` 还会：

- 再调用 `model.float()`；
- 去掉为部分 Norm 算子添加的 FP16/FP32 Cast Wrapper；
- 让最终 ONNX Exporter 面对 FP32 模型。

这不会重新计算 SeqMSE，也不会删除 Quantizer 中已有的 Encoding。

当前配置：

```yaml
enable_fp16: false
```

所以当前分支不会执行。

需要注意：如果将来打开 FP16，Test Vector 在最终 Cast Wrapper 清理**以前**生成，而 ONNX 在清理**以后**导出。若要求逐元素严格对拍，应确认两条路径的实际 Cast 和容差一致。

---

## 十、`quantsim.export()` 真正导出了什么

核心调用：

```python
quantsim.export(
    onnx_dir,
    model_name,
    sample_inputs,
    onnx_export_args=onnx_api_args,
)
```

参数含义：

| 参数 | 当前值／作用 |
|---|---|
| `onnx_dir` | `<output_dir>/onnx` |
| `model_name` | `qwen25llm`，作为文件名前缀 |
| `sample_inputs` | CPU 上的固定 Shape Dummy Tuple |
| `onnx_export_args` | 输入名、输出名和 Opset 14 |

传统 AIMET Export 可以概括为：

```text
内存中的 QuantSim 模型
        ↓ 移除仅用于模拟的 Quantizer/QDQ 包装
普通部署计算图
        +
从 Quantizer 读取已确定的 Encoding
        ↓
ONNX / PyTorch 产物 + 独立 Encoding 文件
```

它不是再次量化，也不会因为 Dummy 使用随机值而改变已经确定的 scale/offset。

### 10.1 当前成功运行记录中的产物

当前本地 `output/` 目录只有 Prepare 产物，尚未生成 `output/onnx` 和 `output/test_vectors`。所以下表不是当前工作区的实时文件扫描结果，而是本项目 `TROUBLESHOOTING.md` 记录的一次远程成功运行结果：

| 文件 | 记录中的大小 | 含义 | example2 是否需要 |
|---|---:|---|---|
| `qwen25llm.onnx` | 约 932 KB | ONNX 图结构和外置 Weight 引用 | **需要** |
| 大量 `*.weight/*.bias` | 每个若干 MB | 大模型外置参数数据 | **全部需要** |
| `qwen25llm.encodings` | 约 66 MB | 与 ONNX Tensor 对齐的量化规则 | **需要** |
| `qwen25llm.pth` | 约 12 GB | PyTorch Checkpoint | QNN 不需要 |
| `qwen25llm_torch.encodings` | 约 336 MB | Torch 侧 Encoding | QNN 不需要 |

文件大小来自一次已有运行，不是格式保证；模型、AIMET 版本和配置变化后可能不同。

#### 最终产物按用途分组

整个 example1 最后阶段的产物，可以分成三组：

```text
一、数值验证组
generate_test_vectors()
  ├─ fp_0.pkl
  └─ qt_0.pkl

二、PyTorch 侧产物
quantsim.export()
  ├─ qwen25llm.pth
  └─ qwen25llm_torch.encodings

三、ONNX / QNN 组
quantsim.export()
  ├─ qwen25llm.onnx
  ├─ *.weight / *.bias
  └─ qwen25llm.encodings
```

| 分组 | 每个文件负责什么 | 主要用途 |
|---|---|---|
| Test Vector | `fp_0.pkl` 保存浮点参考结果；`qt_0.pkl` 保存 QuantSim 结果 | 检查 PyTorch、ONNX、QNN 各阶段从哪里开始出现数值误差 |
| PyTorch 侧 | `.pth` 保存 PyTorch 模型；`_torch.encodings` 保存与 PyTorch 模块名称对应的量化参数 | 恢复、复现或调试 QuantSim；当前 example2 不使用 |
| ONNX/QNN | `.onnx` 保存计算图；外置文件保存 Weight/Bias；`.encodings` 保存与 ONNX Tensor 对应的量化参数 | 交给 example2 的 Split、MHA2SHA 和 QAIRT/QNN 编译 |

为什么既有 `.pth` 又有 `.onnx`：

```text
.pth   = PyTorch/AIMET 可重新加载的模型版本
.onnx  = 跨框架部署、交给 QNN 编译的模型版本
```

#### `.pth` 到底是什么阶段的什么产物

先记结论：

> `qwen25llm.pth` 是最后 `quantsim.export()` 自动保存的 **Prepare 后 PyTorch 浮点模型副本**；它不是打包后的 INT4 权重，也不是 example2 的输入。

##### 1. `.pth` 从哪里来

当前代码先把 `prepared_model` 复制给 `sim_fpm`，再在这个模型上建立 QuantSim：

```python
sim_fpm = LLMForwardPassManager(
    model=copy.deepcopy(prepared_model),
    ...
)

quantsim = QuantizationSimModel(
    model=sim_fpm.model,
    in_place=True,
    ...
)
```

因此，QuantSim 的模型底座就是 Prepare 后的 PyTorch 模型：

```text
prepared_model
  → copy.deepcopy()
  → sim_fpm.model
  → QuantSim 包装和量化模拟
  → quantsim.export()
  → qwen25llm.pth
```

##### 2. `.pth` 里面保存什么

当前 `enable_fp16=false`，并且导出前代码又执行了：

```python
sim_fpm.model.to(torch.float32)
```

所以 `.pth` 主要保存：

- Prepare 后的 PyTorch 模型结构；
- 该模型的浮点 Weight 和 Bias；
- 供同类 PyTorch/AIMET 环境重新加载的模型状态。

它不等于下面这些内容：

```text
不是打包后的 INT4 Weight
不是 ONNX 模型
不是 Quantized DLC
不是 HTP Context Binary
```

文件大小也能辅助判断。Qwen 3B 如果按 FP32 保存：

```text
约 30 亿参数 × 4 Byte ≈ 12 GB
```

已有运行中的 `qwen25llm.pth` 正好约 12 GB；如果是真正紧密打包的 INT4 权重，理论主体应接近 `30 亿 × 0.5 Byte ≈ 1.5 GB`，不会仍是约 12 GB。

##### 3. 它和量化是什么关系

SeqMSE 和 `compute_encodings()` 主要确定 Quantizer 使用的量化参数，并不把底层 PyTorch Parameter 永久改写成紧密打包的 INT4：

```text
SeqMSE
  → 确定并冻结 Weight Encoding

compute_encodings()
  → 确定 Activation/KV Encoding

QuantSim 前向
  → 用 QDQ 模拟低比特误差
  → 底层模型仍保留浮点 Parameter
```

最终 PyTorch 侧拆成两个文件：

```text
qwen25llm.pth
  → Prepare 后的浮点模型与权重

qwen25llm_torch.encodings
  → Weight/Activation/KV 的量化规则
```

所以要分清“生成阶段”和“文件内容”：

```text
生成阶段：最终 quantsim.export()
文件内容：Prepare 后的 PyTorch 浮点模型副本
量化信息：单独放在 qwen25llm_torch.encodings
```

##### 4. 为什么还要生成它，example2 需要吗

AIMET 的传统 `quantsim.export()` 是通用导出接口，同时照顾两种后续用途：

```text
PyTorch/AIMET 恢复与排错
  → qwen25llm.pth
  → qwen25llm_torch.encodings

ONNX/QNN 部署
  → qwen25llm.onnx
  → 外置 *.weight / *.bias
  → qwen25llm.encodings
```

`.pth` 的价值是以后可以回到相同 PyTorch/AIMET 环境做 PPL、层输出对比或导出排错，避免从原始 Hugging Face 模型重新经历完整 Prepare 流程。

但当前 example2 不读取 `.pth`。如果已经确认 ONNX 产物正确，并且以后不再恢复或排查 PyTorch 模型，可以不把它交付给 example2；删除前仍建议保留一份可恢复备份。

##### 5. `.pth` 和 ONNX 外置 Weight/Bias 的区别

二者可能保存同一套模型参数的不同序列化版本，所以会重复占用磁盘：

| 文件 | 参数属于哪套格式 | 给谁读取 | example2 是否需要 |
|---|---|---|---|
| `qwen25llm.pth` | PyTorch 模型格式 | PyTorch/AIMET | 否 |
| `*.weight/*.bias` | ONNX External Data | ONNX、MHA2SHA、QAIRT/QNN | **是** |
| `qwen25llm.encodings` | ONNX Tensor 对应的量化规则 | MHA2SHA、QAIRT/QNN | **是** |

最简记忆：

```text
.pth                  = PyTorch 浮点模型备份
ONNX + weight/bias    = QNN 编译所需的模型和参数
.encodings            = 如何量化这些参数和激活
Quantized DLC         = example2 真正生成的低比特部署产物
```

为什么 ONNX 旁边还有大量 Weight/Bias：

```text
qwen25llm.onnx  = 主要保存算子、连接关系和外置参数引用
*.weight/*.bias = 保存真正的大体积参数数据
```

因此，交给 example2 的最小必要集合是：

```text
qwen25llm.onnx
qwen25llm.encodings
全部 *.weight / *.bias
test_vectors/qt_0.pkl
```

其中 `qt_0.pkl` 不是 `quantsim.export()` 生成的，而是前面的 `generate_test_vectors()` 生成，供 Split 和数值验证使用。

### 10.2 两个 Encoding 分别是哪个阶段的产物

先记结论：**两个文件都是最后 `quantsim.export()` 阶段同时生成的，不是 SeqMSE 和 `compute_encodings()` 各生成一个。**

```text
SeqMSE
  → 只把 Weight Encoding 保存到内存中的 QuantSim

compute_encodings()
  → 只把 Activation Encoding 保存到内存中的 QuantSim

quantsim.export()
  → qwen25llm_torch.encodings   PyTorch/QuantSim 名称版本
  → qwen25llm.encodings         ONNX Tensor 名称版本
```

| 文件 | 生成阶段 | 面向谁 | 后面是否使用 |
|---|---|---|---|
| `qwen25llm_torch.encodings` | 最终 `quantsim.export()` | AIMET/PyTorch QuantSim，用 PyTorch 模块名称记录 | 当前 example2 不使用；主要用于恢复或调试 QuantSim |
| `qwen25llm.encodings` | 同一次最终 `quantsim.export()` | ONNX/QNN，用 ONNX Tensor 名称记录 | **example2 使用这个文件** |

两份文件保存的是同一套量化结果的两种名称视角，里面都可能同时包含：

```text
param_encodings       ← 包含 SeqMSE 优化后的 Weight Encoding
activation_encodings  ← 包含 compute_encodings() 标定的 Activation/KV Encoding
```

因此不能理解成：

```text
qwen25llm_torch.encodings = SeqMSE 产物       ×
qwen25llm.encodings       = Activation 产物   ×
```

正确理解是：

```text
两个文件 = 最终 Export 产物
区别     = 一个按 PyTorch 名称保存，一个按 ONNX 名称保存
```

### 10.3 两个真实 Encoding 文件的逐项解析与对比

本节不是概念示例，而是对下面两个实际文件进行完整 JSON 解析后的结果：

```text
qwen25llm_torch.encodings  351,280,691 Byte
qwen25llm.encodings         68,808,697 Byte
```

两者都能完整解析，公共元数据完全一致：

```json
{
  "version": "1.0.0",
  "producer": {
    "package": "aimet-torch",
    "version": "2.29.0"
  },
  "quantizer_args": {
    "activation_bitwidth": 16,
    "dtype": "int",
    "is_symmetric": true,
    "param_bitwidth": 4,
    "per_channel_quantization": true,
    "quant_scheme": "min_max"
  }
}
```

`quantizer_args` 表示 QuantSim 的全局默认设置；判断某个具体 Tensor 时，仍应以该 Tensor 自己的 `bitwidth/bw`、对称性和粒度为准。

#### 10.3.1 总体数量

| 对比项 | Torch 文件 | ONNX 文件 |
|---|---:|---:|
| 文件大小 | 351,280,691 Byte | 68,808,697 Byte |
| Activation 顶层记录 | 1,300 个模块键 | 1,522 个 ONNX Tensor |
| Activation 实际量化点 | 1,768 个模块 Input/Output 端口 | 1,522 个 ONNX Tensor |
| Weight 名称 | 326 | 326 |
| Weight Encoding 数值总数 | 1,184,201 | 1,184,201 |

Activation 分布：

| 量化类型 | Torch 文件 | ONNX 文件 |
|---|---:|---:|
| 16-bit、INT、非对称 | 1,696 | 1,450 |
| 8-bit、INT、对称 | 72 | 72 |
| ONNX `PER_TENSOR` | 不单独写 `enc_type` | 1,522 |

Weight 分布：

| 量化类型 | 数量 |
|---|---:|
| W4 Weight 总数 | 326 |
| `PER_CHANNEL` | 253 |
| `PER_TENSOR` | 73 |
| Bias Encoding | 0 |

这里不能直接用 `1768 - 1522` 判断“ONNX 丢失了多少 Encoding”，因为两份文件的统计单位不同：

```text
Torch：模块名 + input/output + 端口号
ONNX ：导出图中的 Tensor 名
```

Encoding 传播、ONNX 重命名、常量折叠和图输入输出展开都会造成多对一、一对多或一组对一组的映射。

#### 10.3.2 普通 Activation：Q Projection 输出

选择第 0 层 Q Projection 输出作为普通 Activation 样例。

Torch 文件按“模块输出端口”保存：

```text
模块：model_layers_0_self_attn_q_proj_conv
端口：output[0]
```

```json
{
  "bitwidth": 16,
  "dtype": "int",
  "is_symmetric": "False",
  "max": 27.65810203552246,
  "min": -23.869138717651367,
  "offset": -30358,
  "scale": 0.0007862552884034812
}
```

ONNX 文件按“最终 Tensor 名”保存：

```json
{
  "bw": 16,
  "dtype": "INT",
  "enc_type": "PER_TENSOR",
  "is_sym": false,
  "name": "/model_layers_0_self_attn_q_proj_conv/Conv_output_0",
  "offset": [-30358],
  "scale": [0.0007862552884034812]
}
```

两边的核心量化参数完全相同：

```text
bitwidth = 16
scale    = 0.0007862552884034812
offset   = -30358
对称性   = 非对称
```

区别只是表达方式：Torch 文件直接保存 `min/max`，ONNX 文件省略它们，因为可以根据 `bw + scale + offset` 还原相同的可表示范围。

#### 10.3.3 KV Cache：外部缓存与内部 K/V 工作张量

KV Cache 本质上仍属于 Activation，所以它位于 `activation_encodings`，不存在单独的 `kv_cache_encodings` 顶层字段。

这份实际文件中需要区分两层量化边界：

```text
外部 Past-KV 图输入与拼接结果   → 16-bit
Attention MatMul 使用的展开 K/V → 8-bit
```

##### A. 第 0 层外部 Past Key/Value：16-bit

Torch 文件没有直接使用 `past_key_0_in` 作为模块键，而是记录接收并拼接 Past Key 的 `Concat_9` 端口：

```text
model_layers_0_self_attn_Concat_9.input[0]
model_layers_0_self_attn_Concat_9.output[0]
```

两处共享：

```json
{
  "bitwidth": 16,
  "dtype": "int",
  "is_symmetric": "False",
  "max": 94.86463165283203,
  "min": -78.63336181640625,
  "offset": -29702,
  "scale": 0.0026474096812307835
}
```

ONNX 文件把同一规则绑定到明确的图输入：

```json
{
  "bw": 16,
  "dtype": "INT",
  "enc_type": "PER_TENSOR",
  "is_sym": false,
  "name": "past_key_0_in",
  "offset": [-29702],
  "scale": [0.0026474096812307835]
}
```

并且 `/model_layers_0_self_attn_Concat_9/Concat_output_0` 使用完全相同的 Encoding。

第 0 层 Value 的对应关系是：

| Torch 模块端口 | ONNX Tensor | bw | scale | offset |
|---|---|---:|---:|---:|
| `Concat_10.input[0]` / `output[0]` | `past_value_0_in` 及 `Concat_10` 输出 | 16 | `8.426007843809202e-05` | `-27115` |

全模型明确命名的 Past-KV 图输入一共有：

```text
36 个 past_key_i_in
36 个 past_value_i_in
合计 72 个，全部是 16-bit INT 非对称 Encoding
```

##### B. 第 0 层 Attention 使用的完整 K/V：8-bit

Prepare 后的真实数据流是：

```text
past_key_0_in + 当前 token Key
  → Concat_9
  → Unsqueeze / Expand
  → Attention MatMul

past_value_0_in + 当前 token Value
  → Concat_10
  → Unsqueeze / Expand_1
  → Attention MatMul_1
```

Torch 文件中第 0 层完整 Key 工作张量为：

```text
model_layers_0_self_attn_Expand.output[0]
```

```json
{
  "bitwidth": 8,
  "dtype": "int",
  "is_symmetric": "True",
  "max": 94.86351013183594,
  "min": -95.61046600341797,
  "offset": -128,
  "scale": 0.7469567656517029
}
```

对应 ONNX Tensor：

```json
{
  "bw": 8,
  "dtype": "INT",
  "enc_type": "PER_TENSOR",
  "is_sym": true,
  "name": "/model_layers_0_self_attn_Expand/Mul_output_0",
  "offset": [-128],
  "scale": [0.7469567656517029]
}
```

第 0 层完整 Value 工作张量为：

| Torch 模块端口 | ONNX Tensor | bw | scale | offset |
|---|---|---:|---:|---:|
| `Expand_1.output[0]` | `/model_layers_0_self_attn_Expand_1/Mul_output_0` | 8 | `0.025490080937743187` | `-128` |

全模型恰好有 72 个 A8 Tensor：

```text
36 层 × 每层一个完整 Key + 一个完整 Value = 72
```

因此，对本次真实产物最准确的表述是：

> **外部 `past_key/value_i_in` 缓存输入是 A16；拼接并展开后、真正送入 Attention MatMul 的内部 K/V 工作张量是 A8。不能把整条 KV Cache 路径笼统地全部称为 KV8。**

#### 10.3.4 Weight：第 0 层 Q Projection

选择：

```text
model_layers_0_self_attn_q_proj_conv.weight
```

Torch 文件把每个输出通道展开成一个完整字典，共 2,048 个：

```json
{
  "bitwidth": 4,
  "dtype": "int",
  "is_symmetric": "True",
  "max": 0.08710937947034836,
  "min": -0.09955357760190964,
  "offset": -8,
  "scale": 0.012444197200238705
}
```

上面是第 0 个输出通道。其余 2,047 个通道各有自己的 `scale`。

ONNX 文件把公共字段只写一次，再把 2,048 个通道的值压成数组：

```text
bw        = 4
dtype     = INT
enc_type  = PER_CHANNEL
is_sym    = true
name      = model_layers_0_self_attn_q_proj_conv.weight
offset    = 长度 2048，全部为 -8
scale     = 长度 2048
```

Scale 抽样：

```text
前 4 个：
0.012444197200238705
0.01100027933716774
0.007324220146983862
0.01461181603372097

后 4 个：
0.02526855655014515
0.0214843787252903
0.018861612305045128
0.016113286837935448
```

统计：

```text
scale 最小值 = 0.003069196594879031
scale 最大值 = 0.11272323876619339
offset        = 2048 个 -8
```

对全部 Weight 进行逐项核对后的结果：

```text
Torch Weight 名称 = 326
ONNX  Weight 名称 = 326
名称集合完全一致  = 326 / 326
完整 bw/scale/offset 序列完全一致 = 326 / 326
不一致项 = 0
```

这说明 ONNX 文件没有重新计算另一套 Weight Encoding，只是把同一套最终 Parameter Encoding 换成更紧凑、与 ONNX Tensor 对齐的表达方式。

Encoding 文件本身不会写 `generated_by: SeqMSE`。结合当前执行流程，这些是 SeqMSE 和后续 `compute_encodings()` 完成后的最终 Weight Encoding；但仅凭单份 JSON，不能区分某一个具体 Weight Encoding 是 SeqMSE 搜索得到的，还是其他未冻结 Parameter Quantizer 按 Min-Max 补齐的。

#### 10.3.5 两个文件的区别与联系

| 对比项 | `qwen25llm_torch.encodings` | `qwen25llm.encodings` |
|---|---|---|
| 面向对象 | PyTorch/AIMET | ONNX、example2、QNN |
| Activation 名称 | 模块名 + Input/Output 端口 | ONNX Tensor 名 |
| Activation 结构 | 嵌套字典 | 一条 Tensor 一个扁平记录 |
| Weight 名称 | PyTorch Parameter 名 | 当前 326 个名称保持一致 |
| Per-Channel Weight | 每个通道重复一个完整字典 | 一个 Tensor 记录 + `scale/offset` 数组 |
| `min/max` | 直接保存 | 省略，可由量化参数还原 |
| 当前下游 | 恢复或调试 QuantSim | example2 实际读取 |

两者的联系：

```text
同一个 QuantSim
  → 同一套最终 Encoding
  → Torch 模块/端口视角保存一份
  → ONNX Tensor 视角再保存一份
```

为什么 Torch 文件大约是 ONNX 文件的 5.1 倍：

- 两边都包含 1,184,201 个 Weight Scale/Offset 数值；
- Torch 文件为每个通道重复保存 `bitwidth/dtype/is_symmetric/min/max/offset/scale` 整个字典；
- ONNX 文件只保存一次公共字段，把所有通道的 `scale/offset` 放进数组；
- Torch 文件还直接保存 `min/max`。

所以文件体积差异主要来自 **JSON 表达方式**，不代表 ONNX 文件少了约 80% 的 Weight 量化参数。

### 10.4 为什么 ONNX 主文件只有约 932 KB

Qwen2.5-VL-3B 的参数远大于 Protobuf 单文件适合承载的规模，因此当前导出把 Weight 外置：

```text
qwen25llm.onnx
  └─ 保存图结构和 External Data 引用

*.weight / *.bias
  └─ 保存真正的大体积参数字节
```

所以“ONNX 文件很小”不是模型丢了，而是模型参数分散在外部文件中。

移动、压缩或交付时必须保留相对路径关系，并把 ONNX 与所有外置参数放在同一目录结构中。

### 10.5 `.encodings` 保存什么

Encoding 文件保存的不是另一个模型，而是 Tensor 的量化规则，例如：

- Tensor 名称；
- bitwidth / dtype；
- `scale`；
- zero-point 或 legacy `offset`；
- 对称性；
- Per-Tensor、Per-Channel 或 Per-Block 粒度；
- Activation 与 Parameter Encoding。

具体格式见 [07-附录B · Encoding 量化参数基础](./07-附录B-Encoding量化参数基础.md)。

### 10.6 为什么 ONNX 和 Encoding 必须一起交给 QNN

```text
ONNX      = 哪些 Tensor、哪些算子、如何连接
Encoding  = 每个被量化 Tensor 用什么量化尺子
```

只有 ONNX，没有 Encoding：后端不知道本项目已经优化好的 W4/A16，以及 Attention 内部 K/V 工作张量 A8 等规则。

只有 Encoding，没有 ONNX：后端不知道这些量化规则应该绑定到哪条边或哪个 Parameter。

---

## 十一、导出的还不是真正端侧 INT4 模型

`example1` 的结束产物是：

```text
ONNX + External Weight + AIMET Encoding + Test Vector
```

它不是：

- 已打包的 QNN Context Binary；
- 已在 HTP 上选择好 Kernel 的设备图；
- 可以直接推送手机运行的最终文件；
- 已经证明端侧输出与 QuantSim 完全一致的结果。

设计上的后续 `example2` 流程是：

```text
change_hardcoding / 不同 AR-Context 配置
                ↓
Split ONNX
                ↓
MHA → SHA 转换
                ↓
qairt-converter：ONNX → DLC
                ↓
qairt-quantizer：结合 Encoding 生成量化 DLC
                ↓
qnn-context-binary-generator
                ↓
设备可执行 Context Binary
```

再由 `example3` 把模型、Embedding、Tokenizer 和 Runtime 配置推送到设备执行。

不过当前 `example2/host_linux/qnn_compile_deploy.py` 中，`qnn-context-binary-generator` 调用块处于注释状态。因此“Context Binary”是目标流程产物，不是当前脚本无条件执行后一定得到的文件；真正部署前还需要按目标芯片配置并启用、验证该步骤。

---

## 十二、Prepare 阶段导出与最终导出有什么区别

本项目至少出现两次 ONNX 相关转换，不能混为一谈：

| 对比项 | 06 Prepare 阶段 | 08 最终 QuantSim Export |
|---|---|---|
| 主要目的 | 把 PyTorch 图改造成 QNN 友好结构并重建 prepared PyTorch 模型 | 生成交给 example2 的部署交接物 |
| 输入模型 | Monkey Patch 后的浮点模型 | 已完成 SeqMSE、Calibration，随后已做 PPL 评估的 QuantSim 模型 |
| 是否包含最终量化数值 | 否 | 是，导出独立 `.encodings` |
| Opset | Prepare 配置目标为 20；既有 artifact 的实际版本需核查 | 最终导出配置为 14 |
| ONNX 是否是最终交付物 | 主要是内部转换中间物 | 是 example2 的核心输入 |
| 后续去向 | QuIR/QNNIR → Emitter → prepared PyTorch | Split/MHA2SHA/QNN Converter/Quantizer |

一句话：

```text
Prepare ONNX：为了“改模型”
最终 ONNX：为了“交模型”
```

Prepare 的内部流程见 [06-附录C · QAIRT model_preparer 内部流程](./06-附录C-QAIRT-model_preparer内部流程.md)。

---

## 十三、当前配置汇总

| 配置 | 当前值 | 影响 |
|---|---:|---|
| `model_name` | `qwen25llm` | 导出文件名前缀 |
| `output_dir` | `/root/autodl-tmp/zgj/Qwen25/outputs/output` | Test Vector 和 ONNX 根目录 |
| `enable_fp16` | `false` | 不执行导出前 FP16→FP32 分支 |
| `context_length` | 2048 | 固定总 Attention 长度 |
| `arn` | 1073 | Current Input 固定长度 |
| Past KV Length | 975 | `2048 - 1073` |
| `use_input_embeddings` | `true` | ONNX 第一输入是 `inputs_embeds` |
| `use_position_embedding_input` | `true` | cos/sin 作为两个显式输入 |
| `onnx_opset_version` | 14 | 最终 ONNX 算子集版本 |
| Test Vector Batch | 1 | 生成 `fp_0.pkl` 和 `qt_0.pkl` |

---

## 十四、代码审查时值得注意的细节

### 14.1 当前 Test Vector 没有覆盖非零真实 Past KV

当前 calibration sample 被固定为 ARN=1073，`generate_test_vectors()` 又只取：

```python
batch['input_embeddings'][:, :forward_pass_manager.num_tokens, :]
```

然后直接调用一次 `prepare_inputs()`。因此当前 Test Vector 的 Past KV 输入是 975 个零槽位，没有第二块把真实历史 KV 回喂进模型。

它可以验证固定图 Prefill/单块路径和新 KV 输出，但不能证明带非零历史 KV 的 Decode 路径已经对拍通过。

### 14.2 `num_batches=1` 只适合基础 Smoke Test

一个样本可以快速发现：

- 输入顺序错误；
- 某层输出突然发散；
- Encoding 没有正确应用；
- ONNX/QNN 节点映射错误。

但它不能代表整个数据分布，也不能替代 PPL 或任务指标。

另外，循环是在从 Dataloader 取出 Batch 后才判断 `idx >= num_batches`；当前可能额外加载和预处理第二个 Batch，只是不会对它执行模型前向。

### 14.3 使用的是 `train_dataloader`

Test Vector 来自 calibration dataloader，而不是 `test_dataloader`。这本身没有问题，因为它不计算泛化指标；但要确保该样本确实执行到想观察的算子路径。

### 14.4 当前三组正则中有两组零命中

`test_vector_layers` 依赖 prepared 模块名称。重新 Prepare、升级 QAIRT 或改变 `KEEP_ORIGINAL_MODEL_STRUCTURE` 后，名称可能变化。

当前静态检查结果是 `0 / 36 / 0`，只有 `model_layers_\d+_Add_1` 命中。因此生成后应检查 `.pkl` 是否真的包含预期 Layer Key，不能只看到文件存在就认为 Hook 已命中。

### 14.5 Test Vector 可能很大

选中的层越多、Batch 越多、序列越长，Pickle 体积越大。尤其 logits 和大 Hidden Tensor 会快速放大磁盘占用。

按当前 FP32 Shape 粗略估算，每个 `fp/qt` 文件的原始 Tensor 下限约 1.35 GiB，两份至少约 2.7 GiB，尚未计入 Pickle 开销和临时复制。实际大小应以生成结果为准。

生成策略应以“能定位问题的关键边界层”为主，而不是无差别记录全部算子。

文件名固定为 `fp_0.pkl`、`qt_0.pkl`，重复运行会直接覆盖；当前文件内也没有自动写入 Git Commit、配置摘要、随机种子或 Encoding 版本，归档时应额外记录运行上下文。

### 14.6 FP16 模式存在生成时机差异

Test Vector 在 `convert_model_to_fp32()` 以前生成，最终 ONNX 在转换以后导出。当前 `enable_fp16=false` 不受影响；打开 FP16 后应重新定义对拍容差并确认参考路径一致。

### 14.7 导出目录可能残留失败文件

代码只执行：

```python
os.makedirs(onnx_dir, exist_ok=True)
```

不会自动清理旧文件。导出失败后重新运行时，目录里可能同时存在新旧外置 Weight，导致人工交付时混入残留文件。

应在运行前明确核对目标目录，并在不误删其他结果的前提下管理旧产物。

### 14.8 PPL 没有自动精度门禁

前面代码只是打印 QuantSim PPL，没有判断阈值；即使精度不满足要求，程序仍会继续来到 Export。

因此“成功生成文件”不等于“模型精度验收通过”。

### 14.9 `quantsim.export()` 是版本敏感接口

当前项目使用传统：

```python
quantsim.export(...)
```

最新版 AIMET 文档已经将其标为 Deprecated，并推荐新的 ONNX Export API。但本仓库下游依赖当前独立 Encoding、外置 Weight 和命名布局，升级时不能只机械替换 API，必须重新验证：

- 输出文件格式；
- Tensor 名称；
- Encoding Schema；
- example2 MHA2SHA/Split/QNN 工具兼容性；
- Test Vector 对拍结果。

### 14.10 “layers not found” 警告不能一概忽略

项目已有记录显示，部分 q/k/v Projection Encoding 在最终 ONNX 中找不到对应 Layer 时可能出现 Warning。如果相关分支确实不会执行，可能不影响主路径；否则也可能意味着 Encoding 没有正确映射。

正确处理方式是结合：

1. ONNX 实际节点和 Tensor 名称；
2. 导出的 Encoding 条目；
3. Test Vector 层输出；
4. example2 编译日志；
5. 端侧对拍结果；

共同判断，而不是只根据 Warning 文案决定忽略。

### 14.11 Hook 只记录多输入算子的第一个输入

当前 Hook：

```python
self.layer_name_to_layer_output_dict[layer_name] = {
    "input": to_cpu(input[0]),
    "output": to_cpu(output),
}
```

而当前实际命中的 `Add_1` 是二输入加法。记录中只保存 `input[0]`，第二个分支输入没有保存，因此不能只依赖这份记录完整重放 Add。若要检查残差两路输入，需要修改记录格式以保存整个 `input` Tuple。

### 14.12 `finally: return output` 可能掩盖原始异常

`run_hook_for_layers_with_given_input_get_output()` 在 `finally` 中执行 `return output`。Python 中在 `finally` 里返回可能吞掉前向或 Hook 抛出的原始异常；如果异常发生在 `output` 赋值以前，还可能表现为新的 `UnboundLocalError`。

更安全的结构是只在正常路径返回，并让 `finally` 专门负责移除 Hook。

### 14.13 Pickle 读取方式和可信边界

代码用 `pickle.dump()` 写文件，却用：

```python
np.load(filename, allow_pickle=True)
```

重新读取。这种接口混用比较脆弱，并且 Pickle/`allow_pickle=True` 可能执行序列化对象中携带的代码，因此只能加载本项目自己生成、来源可信的文件。

### 14.14 当前没有自动 ONNX 数值验证

`quantsim.export()` 以后脚本没有自动执行：

- ONNX Checker；
- ONNX Runtime 前向；
- ONNX 输出与 `qt_0.pkl` 的误差比较；
- External Data 完整性检查；
- Encoding 名称覆盖率检查。

这些都需要作为 export 后的独立验证步骤补充，不能只以“API 没抛异常”作为成功标准。

---

## 十五、导出完成后的检查清单

### 文件完整性

- [ ] `qwen25llm.onnx` 存在；
- [ ] `qwen25llm.encodings` 存在且不是空文件；
- [ ] `qwen25llm_torch.encodings` 存在；若只交付 example2，它不是必需文件；
- [ ] 所有 ONNX 引用的 `.weight/.bias` 都在同一目录；
- [ ] `fp_0.pkl`、`qt_0.pkl` 可以在 CPU 环境加载；
- [ ] Test Vector 中实际存在期望的 Layer Key，而不是正则零命中；
- [ ] 没有把上一次失败导出的残留文件混入交付目录。

### 接口完整性

- [ ] ONNX 有 76 个预期输入和 73 个预期输出；
- [ ] 输入输出名称和顺序正确；
- [ ] `inputs_embeds`、Mask、RoPE、Past KV Shape 正确；
- [ ] ONNX Opset 是预期的 14；
- [ ] External Data 相对路径可以解析。

### 数值完整性

- [ ] FP 与 QT 差异符合量化预期；
- [ ] ONNX Host 输出与 QT Golden Output 在容差内；
- [ ] Split 前后关键边界层输出一致；
- [ ] QNN Host/HTP 输出没有从某一层开始明显发散；
- [ ] PPL 已经人工确认满足项目精度要求。

---

## 十六、常见误解

### 16.1 “Dummy Input 是最后一个校准 Batch”

不是。Dummy 是随机／零值组成的 Shape 模具；Calibration 使用真实 `train_dataloader`。

### 16.2 “生成 `qt_0.pkl` 会重新计算 Encoding”

不会。它只是使用已经确定的 Encoding 执行普通 QuantSim 前向。

### 16.3 “`fp_0.pkl` 来自原始 Hugging Face 模型”

不是。它来自同一个 prepared QuantSim 模型临时关闭 Quantizer 后的路径。

### 16.4 “ONNX 只有 932 KB，所以 Weight 没导出来”

不是。Weight 被存放在外置 `.weight/.bias` 文件中。

### 16.5 “有了 `.onnx + .encodings` 就能直接在手机运行”

不能。还需要 example2 的 QNN 转换、量化和 Context Binary 生成，以及 example3 的端侧 Runtime 配置。

### 16.6 “`.encodings` 就是 INT4 Weight 文件”

不是。它保存量化规则；真正的参数数据仍在 ONNX External Data 中，后续由 QNN 工具根据规则处理。

### 16.7 “Opset 越高，端侧性能一定越好”

不是。Opset 主要决定算子语义版本。性能取决于 QNN Converter 支持、图优化、HTP Kernel、Shape 和数据布局；过高的 Opset 反而可能超出下游工具支持范围。

### 16.8 “Test Vector 和 PPL 都是在测精度，所以可以二选一”

不能。PPL 检查端到端任务质量；Test Vector 用来做确定性的逐层数值对拍和定位。

---

## 十七、面试速答

### Q：最后这段代码主要做什么？

> 它先为同一个 prepared QuantSim 模型生成关闭 Quantizer 的 FP Test Vector 和启用 QDQ 的 QT Test Vector，然后使用固定 Shape CPU Dummy Input 导出 ONNX、外置 Weight 和量化 Encoding，作为后续 QNN 编译输入。

### Q：为什么既要 Test Vector，又要 PPL？

> PPL 衡量量化后语言模型的整体任务精度；Test Vector 保存固定输入下的中间层和最终输出，用于定位 PyTorch、ONNX、QNN Host 与 HTP 之间从哪一层开始出现数值偏差。

### Q：Dummy Input 为什么不能用于 Calibration？

> Dummy 的数值是随机 Embedding、零 Past KV 和人造 Mask，只代表接口 Shape；它不代表真实 Activation 分布，用它标定会得到错误 Encoding。

### Q：为什么 ONNX 很小，却还有大量 Weight 文件？

> 大模型参数采用 ONNX External Data：`.onnx` 保存图和外部引用，实际 Weight 字节位于同目录的 `.weight/.bias` 文件。

### Q：最终导出为什么需要 `.encodings`？

> ONNX 描述计算图和 Tensor，Encoding 描述这些 Tensor 的 bitwidth、scale、zero-point/offset 和粒度；QNN 需要两者才能复现 AIMET 确定的量化方案。

### Q：为什么同时生成两个 Encoding 文件？

> 两个文件都由最后一次 `quantsim.export()` 同时生成：`qwen25llm_torch.encodings` 使用 PyTorch/QuantSim 名称，供 AIMET 恢复和调试；`qwen25llm.encodings` 使用 ONNX Tensor 名称，交给 example2 和 QNN。它们不是 SeqMSE 与 Activation Calibration 各自独立生成的文件。

### Q：这个阶段是否已经生成 HTP Context Binary？

> 没有。`example1` 只完成 ONNX、External Weight、Encoding 和 Test Vector 交付；设计上应在 `example2` 的 QNN 主机编译流程中生成 Context Binary，但当前脚本中的生成调用仍处于注释状态，需要启用后再验证。

---

## 十八、参考资料

- [AIMET · QuantizationSimModel Export API](https://quic.github.io/aimet-pages/releases/latest/apiref/torch/quantsim.html)
- [AIMET · Quantization Simulation Guide](https://quic.github.io/aimet-pages/releases/latest/tutorials/quantsim.html)
- [AIMET · Encoding Format Specification](https://quic.github.io/aimet-pages/releases/latest/techniques/encoding_spec.html)
- [ONNX · External Data](https://onnx.ai/onnx/repo-docs/ExternalData.html)
- [ONNX · Versioning / Opset](https://onnx.ai/onnx/repo-docs/Versioning.html)
- [项目 · example1 最终产物与警告](../../example1/TROUBLESHOOTING.md)
- [项目 · example1/example2/example3 职责](../EXAMPLES_OVERVIEW.md)

---

## 十九、一句话总结

> **`example1` 最后用真实样本生成 FP/QT Test Vector 作为数值对拍基准，再用固定 76 输入、73 输出的 CPU Dummy Tuple 导出 ONNX、外置 Weight 和量化 Encoding；这些文件是 AIMET 到 QNN 的交接物，不是最终 HTP Binary，必须继续经过 `example2` 编译并用 Test Vector 验证。**
