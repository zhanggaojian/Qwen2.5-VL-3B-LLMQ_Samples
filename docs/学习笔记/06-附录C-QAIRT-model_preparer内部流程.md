# 06 · 附录C · QAIRT `model_preparer.prepare_model()` 内部流程

> **关联主篇**：[06 · Prepare 模型准备阶段](./06-Prepare模型准备阶段.md)。主篇负责建立 Prepare 主线，本附录专门向下展开 `qti.aisw.preparer_api.model_preparer.prepare_model()` 的阶段、产物、风险与检查重点。
>
> **前置附录**：[附录A · Prepare Dummy Input](./06-附录A-Prepare-Dummy-Input输入模具.md) 解释输入模具；[附录B · QAIRT / QNN / AIMET / QuantSim](./06-附录B-QAIRT-QNN-AIMET-QuantSim概念与关系.md) 解释工具边界。
>
> **一句话本质**：`prepare_model()` 用固定 dummy input 把已经完成端侧适配的 PyTorch 浮点模型追踪成 ONNX，经 QAIRT 内部图表示转换后，再由 Emitter 重建成输入输出扁平、shape 固定、算子显式、便于 AIMET QuantSim 继续处理的新 PyTorch 模型。
>
> **证据边界**：当前仓库能直接验证调用参数、ONNX 入口、重建结果、映射文件和 PPL 验证，但不包含 QAIRT 2.42 的 `model_preparer` 源码。因此 QuIR/QNNIR 的精确 schema、pass 名称和逐节点归属不能当作已知事实。
>
> **版本范围**：本文针对本仓库当前组合——QAIRT `2.42.0.251225`、Python `3.10`、Prepare 中间 ONNX opset `20`。QAIRT 的原生扩展与 Python ABI、内部 flag 和 pass 行为都可能随版本变化；升级 SDK 后应重新核验，不能把本文的内部观察无条件外推到其他版本。

---

## 〇、30 秒速记版

### 0.1 一张图看完整流程

```text
已经完成端侧适配的 PyTorch 浮点模型
  Attention / MLP / lm_head 已改造
  mask / RoPE / KV Cache 接口已外部化或定长化
                    │
阶段 0：调用前固定
  eval / no grad / tuple 输出 / FP32 导出环境
                    │
                    ├─ dummy_input：确定执行路径、shape、dtype
                    ├─ input_names / output_names：确定外部端口
                    ▼
阶段 1：dummy tracing 与接口定型
                    │
                    ▼
阶段 2：Torch 图导出为 ONNX
                    │
                    ▼
阶段 3：ONNX → QuIR
  QAIRT 内部前一级图表示
                    │
                    ▼
阶段 4：QuIR → QNNIR
  更靠近 QNN 算子与 Tensor 契约的内部图表示
                    │
                    ▼
阶段 5：QAIRT Emitter 重建 PyTorch
  扁平 Module + 显式算子 + 固定 forward 签名
                    │
                    ▼
阶段 6：保存 .py / 权重 / 名称映射并返回实例
                    ▼
prepared_model（仍然是浮点模型）
                    │
阶段 7：重新计算 PPL，验证转换基本无损
                    │
                    └─ 后续才进入 AIMET QuantSim / SeqMSE / encodings
```

### 0.2 三句话抓住重点

1. Prepare 的输入不是未经处理的 Hugging Face 原模型，而是**已经完成端侧算子与接口适配的浮点模型**。
2. Prepare 的核心不是量化，而是**追踪、规范图、重建图、保存映射**。
3. Prepare 的验收不是“模型变小”，而是**接口静态化后仍能跑通，且数值与原模型足够接近**。

### 0.3 每个阶段最该盯什么

| 阶段 | 主要输入 | 主要工作 | 主要输出 | 最该关注 |
|---|---|---|---|---|
| 调用前固定 | 适配后 Torch 模型 | 固定推理模式、dtype、输出类型和导出环境 | 可稳定追踪的模型 | 是否仍有训练分支、ModelOutput、带梯度常量 |
| Torch → ONNX | 模型 + dummy | 跑通 forward、建立静态图、贴 I/O 名称 | ONNX 中间图 | shape、顺序、opset、大模型外置权重 |
| ONNX → QuIR | ONNX 图 | 导入 QAIRT 内部前一级图表示 | QuIR | 算子/属性能否被识别，Tensor 元数据是否完整 |
| QuIR → QNNIR | QuIR | 向 QNN 图语义继续 lowering/规范化 | QNNIR | QNN 算子契约、layout、dtype、显式转换节点 |
| QNNIR → Torch | QNNIR | Emitter 生成新 `nn.Module` | prepared model | 原模块树会改变，权重与名称映射必须正确 |
| 保存与验证 | prepared model | 落盘、加载、PPL 回归 | 可复用产物 | 产物是否完整、PPL 是否明显变化 |

> 表中的 QuIR/QNNIR 职责是基于链路位置、命名和最终产物建立的**工作性理解**；对应 QAIRT 2.42 的精确内部 pass 仍需 SDK 源码或版本文档确认。

---

## 一、先定位：这到底是谁的工具

### 1.1 它来自 QAIRT SDK，不是仓库内函数

主脚本先把 QAIRT SDK 的 Python 目录加入搜索路径：

```python
QNN_SDK_ROOT = _env['qnn_sdk_root']
sys.path.insert(0, QNN_SDK_ROOT + '/lib/python')
```

随后导入：

```python
from qti.aisw.preparer_api import model_preparer
```

当前配置指向：

```yaml
qnn_sdk_root: /root/.../qairt/2.42.0.251225
```

因此准确表述是：

```text
工具归属：Qualcomm QAIRT SDK
Python API：qti.aisw.preparer_api.model_preparer
执行位置：开发机 / 服务器主机侧
当前版本：QAIRT 2.42.0.251225
```

它不是：

- AIMET 公开的 FX `prepare_model`；
- 手机上的 QNN Runtime；
- `qnn-net-run`；
- 最终的 `qairt-converter → DLC` 编译步骤；
- HTP 上真正执行算子的阶段。

### 1.2 它接收的是“已适配模型”

进入 `model_preparer` 前，项目已经完成：

- `QcAttention` 替换；
- Attention / MLP / lm_head 的 Linear → 1×1 Conv；
- causal mask 外部化；
- RoPE cos/sin 外部输入；
- KV Cache 定长、转置与只返回新 K/V；
- 原始适配后浮点模型 PPL 基线评估。

所以不能把这些改造都归功于 `prepare_model()`：

```text
prepare_conv / Monkey Patch：先改“模型会怎么计算”
model_preparer：再把“已经改好的计算”转换并重建成静态友好图
```

### 1.3 当前配置默认不会重新执行 Prepare

当前 `config.yaml` 是：

```yaml
skip_prepare: true
```

因此日常运行走的是已有产物加载分支：

```python
prepared_model = load_torch_model_using_safetensors(...).eval()
```

只有将 `skip_prepare` 改成 `false`，才会重新进入本文描述的转换链并产生完整日志与中间文件。

---

## 二、阅读这条黑盒链路时，先分清三类结论

由于 SDK 源码不在当前仓库，本文统一使用以下证据等级。

| 标记 | 含义 | 本文示例 |
|---|---|---|
| **已确认** | 能从代码、配置或真实生成物直接验证 | `opset=20`、76 输入、73 输出、Emitter 生成 `.py` |
| **合理推断** | 符合编译器常规和现有前后产物，但无法定位到具体 SDK pass | QuIR 更通用、QNNIR 更靠近 QNN 算子契约 |
| **未知** | 当前仓库和可访问公开资料不足以确认 | QuIR/QNNIR 正式 schema、每个节点由哪个 pass 插入 |

以下事实是确定的：

```text
Torch → ONNX → QuIR → QNNIR → 重建 Torch
```

但不要把它误读成：

```text
每一级一定会落盘成一个同名文件
每一级一定只执行一个 pass
QNNIR 已经等于 HTP 机器指令
所有 Reshape/Permute 都能精确归因到某一级
```

当前 `output/prepare/` 没有 `.quir` 或 `.qnnir` 文件；它们可能是内存对象、临时对象或临时目录内容，但仅凭仓库快照无法判断具体保存形式。

---

## 三、核心调用与参数契约

### 3.1 调用代码

```python
prepared_model = model_preparer.prepare_model(
    model,
    dummy_input,
    model_name=prepare_filename,
    filename=prepare_filename,
    path=prepare_path,
    input_names=input_names,
    output_names=output_names,
    onnx_export_args={"opset_version": _export_cfg['prepare_opset_version']},
    # converter_args=converter_args,
    return_prepare_model=True,
    keep_original_model_structure=False,
)
```

### 3.2 参数逐项理解

| 参数 | 当前值/来源 | 作用 | 重点风险 |
|---|---|---|---|
| `model` | 适配后的 Qwen2ForCausalLM | 被追踪和重建的浮点模型 | 不能仍含无法导出的 Python 分支或对象 |
| `dummy_input` | `get_dummy_data(..., ARN, cpu)` | 选择路径并确定 shape/dtype | 内容可假，结构、顺序、shape 必须真 |
| `model_name` | `qwen25llm_kvcache_36_layer` | 重建模型的逻辑名；当前产物中也是类名 | 名称必须可作为 Python 类/模块标识 |
| `filename` | 同上 | 落盘文件前缀 | 加载时必须与保存名称一致 |
| `path` | `${output_dir}/prepare` | Prepare 产物目录 | 大模型会产生大量临时文件和外置权重 |
| `input_names` | 76 个名字 | 静态图输入端口标签 | 顺序必须和 exporter 展平后的 Tensor 一致 |
| `output_names` | 73 个名字 | 静态图输出端口标签 | logits 与每层 K/V 不能错位 |
| `onnx_export_args` | `opset_version=20` | 控制中间 ONNX 导出 | opset 与算子支持必须匹配 |
| `return_prepare_model` | `True` | 落盘后直接返回重建模型实例 | 后续代码依赖返回值立即可运行 |
| `keep_original_model_structure` | `False` | 允许按转换图重新组织模块 | 原 Hugging Face 模块路径不再可靠 |

### 3.3 一个“看起来配置了、实际没传入”的参数

代码构造了：

```python
converter_args = {
    "input_tensors": [
        {"name": input_name, "source_model_input_layout": "NONTRIVIAL"}
        for input_name in input_names
    ]
}
```

但调用处是：

```python
    # converter_args=converter_args,
```

因此当前 Prepare **没有实际收到这组 converter layout 参数**。阅读日志或分析布局变化时，不能说它由这里的 `NONTRIVIAL` 配置触发。

---

## 四、阶段 0：调用前固定模型和导出环境

这一阶段严格说发生在 `prepare_model()` 外部，但它决定内部转换能否成功。

### 4.1 FP16 模式下临时恢复 FP32

```python
if enable_fp16:
    convert_model_to_fp32(model)
```

本项目的 FP16 模型还为部分 Norm 算子插入了 PreCast/PostCast。恢复 FP32 时不仅执行 `model.float()`，也会移除这些包装层，避免把临时混合精度包装固化进 Prepare 图。

当前配置 `enable_fp16: false`，所以本次配置下不会执行这次往返转换。

### 4.2 把 Transformers 输出固定为 tuple

```python
setattr(llm_config, 'return_dict', False)
model.config.return_dict = False
```

目的：避免追踪期间继续处理 `ModelOutput` 这类 Python 映射对象，使输出成为稳定的位置 tuple。

重点关注：

- 第 0 个输出是否始终是 logits；
- 后续输出是否按层严格排列 K/V；
- 改成 tuple 后是否仍保留需要的 cache 输出；
- `output_names` 数量是否与实际 Tensor 输出数量一致。

### 4.3 固定推理模式并关闭梯度

```python
model.eval()
model.requires_grad_(False)
```

分别解决：

- 关闭 dropout 和训练专用分支；
- 避免导出时把带梯度 Tensor 当作常量插入而报错；
- 让 dummy tracing 更接近真实推理路径。

注意：这里能确认的是**原始待 Prepare 模型**被关闭梯度；不能仅凭这两行推断重建后所有参数的 `requires_grad` 默认值。

### 4.4 为大模型关闭 PyTorch ONNX 内存内 shape inference

```python
torch.onnx._globals.GLOBALS.onnx_shape_inference = False

for _pass_name in (
    '_jit_pass_onnx_node_shape_type_inference',
    '_jit_pass_onnx_graph_shape_type_inference',
):
    setattr(torch._C, _pass_name, lambda *a, **k: None)
```

项目给出的原因是：3B 模型超过 protobuf 2 GiB 限制，PyTorch 的内存内 shape inference 可能尝试序列化整图并失败。

重点关注：

- 这是**进程级 monkey patch**，不仅影响 Prepare，也影响后续最终 ONNX 导出；
- 禁用 PyTorch 的这一步不等于整个工具链再也不做 shape 推导；
- 如果后续报 shape 缺失，需要区分是 PyTorch pass 被禁用，还是 QAIRT 前端无法恢复形状。

### 4.5 允许 Emitter 不保留原模块树

```python
onnx_utils.EXPORT_TO_ONNX_DIRECT = True
ir_graph_op_handler.KEEP_ORIGINAL_MODEL_STRUCTURE = False
```

调用处又传入：

```python
keep_original_model_structure=False
```

其直接可观察结果是：prepared model 不再保持 `model.layers.0.self_attn...` 的原始 Python 层级，而是按图节点生成大量平铺模块。

### 4.6 `num_logits_to_return` 的代码审阅提醒

```python
model.num_logits_to_return = ARN
```

注释称它用于配置 KVCache 模式。但在当前仓库可见的 Qwen2 `forward` 中，没有找到对 `num_logits_to_return` 的读取；全仓库的业务代码也只有这一处赋值。

因此更严谨的结论是：

- **已确认**：ARN=1073 的 dummy input 和最终生成代码把 logits 的序列维固定成 1073；
- **不能确认**：这一赋值本身是否被当前实际运行环境的某个外部补丁读取；
- 调试输出长度时，应先看 dummy shape 和真实 `forward`，不要只看这句注释。

---

## 五、阶段 1：dummy tracing 与外部接口定型

### 5.1 Prepare 使用的 dummy 仍是嵌套 PyTorch 接口

调用是：

```python
dummy_input = get_dummy_data(
    llm_config,
    tokenizer,
    'cpu',
    separate_tuple_input_output=False,
    num_tokens=ARN,
    dtype=model.dtype,
)
```

因此 Python 表面输入仍类似：

```python
{
    "inputs_embeds": Tensor,
    "attention_mask": Tensor,
    "position_ids": (cos, sin),
    "past_key_values": ((K0, V0), (K1, V1), ...),
}
```

但 `input_names` 已按最终静态端口准备成扁平序列：

```text
inputs_embeds
attention_mask
position_ids_cos
position_ids_sin
past_key_0_in
past_value_0_in
...
past_key_35_in
past_value_35_in
```

这意味着 exporter / Prepare 链需要把嵌套 pytree 中的 Tensor 叶子按稳定顺序展开，并把 76 个名字逐位贴上去。

### 5.2 当前配置的形状契约

| 输入 | 数量 | 当前 shape |
|---|---:|---|
| `inputs_embeds` | 1 | `[1, 1073, 2048]` |
| `attention_mask` | 1 | `[1, 1, 1073, 2048]` |
| `position_ids_cos` | 1 | `[1, 1, 1073, 64]` |
| `position_ids_sin` | 1 | `[1, 1, 1073, 64]` |
| 每层 Past K | 36 | `[1, 2, 128, 975]`，转置存储 |
| 每层 Past V | 36 | `[1, 2, 975, 128]` |

长度关系：

```text
Current = ARN = 1073
Past    = 2048 - 1073 = 975
Total   = 2048
```

输入总数：

```text
4 个基础输入 + 36 × 2 个 KV = 76
```

输出总数：

```text
1 个 logits + 36 × 2 个新 KV = 73
```

### 5.3 tracing 实际固定了什么

从最终生成物可以确认，dummy 至少使以下内容被静态化：

- 当前 token 长度 1073；
- Past KV 长度 975；
- context 2048；
- batch size 1；
- hidden size、head 数、KV head 数；
- RoPE 输入形式；
- K/V 转置布局；
- 当前配置下实际执行到的 forward 分支；
- 扁平输入输出数量和顺序。

因此 dummy 不是“随便造一个 Tensor”——它是整个 prepared graph 的**接口 ABI 模具**。

当前调用没有传入 `dynamic_axes` 或 `dynamic_shapes`，最终生成代码又直接出现 1073、975、2048 等常量。因此这里讨论的是**由当前示例输入建立的固定 shape 图**，不是可以仅凭 tracing 自动覆盖任意长度的动态图。数据相关的 Python 控制流也只记录 dummy 实际走过的分支。

### 5.4 这一阶段重点关注什么

- 名字数量是否等于 Tensor 叶子数量；
- 名字顺序是否和 pytree 展平顺序一致；
- cos/sin 是否颠倒；
- K/V 是否颠倒；
- K 是否使用转置布局；
- Past 长度是否恰好为 `context - ARN`；
- mask 的 dtype、负值和广播 shape 是否正确；
- dummy 是否误走了训练、无 cache 或其他不想保留的分支。

---

## 六、阶段 2：Torch 图导出为 ONNX

### 6.1 ONNX 在这里的角色

ONNX 不是本阶段的最终交付物，而是 PyTorch 与 QAIRT 内部图之间的通用交换层：

```text
Python 模块和 forward 语义
        ↓ tracing/export
ONNX 节点、Tensor、属性、常量和 I/O
        ↓ QAIRT 前端导入
内部 IR
```

调用明确传入：

```python
onnx_export_args={"opset_version": 20}
```

所以可以确认：Prepare 链会以 opset 20 建立中间 ONNX 表达。

这里要和最终量化模型导出分开：当前配置中 Prepare 的中间 ONNX 使用 opset `20`，后续 `quantsim.export(...)` 使用 `onnx_opset_version: 14`。两个数字服务于不同阶段，不应混写成“本项目 ONNX opset”。

### 6.2 能直接确认的 ONNX 相关行为

- 使用 dummy 执行导出；
- 输入输出有明确名称；
- opset 为 20；
- 模型输出被改成 tuple；
- PyTorch 内存内 shape inference 被主动禁用；
- 项目记录显示 Prepare 过程可能产生中间 `.onnx`、外置权重和 `_Constant_*` 文件；
- 当前仓库快照没有保留这份 Prepare 中间 ONNX。

### 6.3 不应未经证据直接写死的内部行为

下面这些属于常见 converter 工作，但无法从当前仓库精确定位到 `prepare_model` 的哪个 pass：

- 常量折叠具体发生几次；
- 哪些算子被融合或拆分；
- 哪个阶段插入了某个 `Reshape` / `Permute`；
- layout 推导具体在 ONNX 前端还是后续 IR 完成；
- 中间 ONNX 是纯内存对象还是临时文件后被清理；
- 每个 unsupported op 检查发生在哪一级。

### 6.4 这一阶段重点关注什么

| 关注点 | 典型问题 |
|---|---|
| opset | 导出成功但 QAIRT 前端不支持某个新属性 |
| 静态 shape | dummy 长度不对，最终所有 Reshape 都被写错 |
| I/O 数量与顺序 | 名字贴错后，模型仍可能运行但语义已经错位 |
| 外置权重 | 只复制 `.onnx`，漏掉 `.weight/.bias` 等数据 |
| 大模型限制 | protobuf 2 GiB、内存峰值、磁盘临时文件爆满 |
| Python 对象 | `ModelOutput`、嵌套非 Tensor 对象导致 tracing 失败 |

---

## 七、阶段 3：ONNX → QuIR

### 7.1 先给一个严谨定义

本仓库只能确认 QuIR 位于：

```text
ONNX → QuIR → QNNIR
```

当前可访问的项目资料没有给出 QuIR 的正式全称、公开 schema 或 API。因此本文采用以下**工作性理解**：

> QuIR 是 QAIRT Converter 内部较靠前、较通用的图表示，用于把不同来源框架的节点、Tensor、属性和常量导入为 QAIRT 可统一处理的图对象。

不要在没有对应版本 SDK 文档的情况下，强行把 QuIR 展开成某个确定英文全称。

### 7.2 为什么 ONNX 后面还需要一层内部 IR

ONNX 仍然带有自己的算子 schema、opset 和前端表达习惯。工具内部使用统一 IR，可以把：

```text
ONNX / TensorFlow / TFLite / PyTorch
```

转换到同一种图数据结构，再复用后续检查、规范化和 backend 映射逻辑。

在概念上，QuIR 至少需要能够承载：

- op 节点和属性；
- Tensor 边；
- 输入输出；
- shape / dtype / layout 元数据；
- 常量与权重引用；
- 原始名称或名称映射。

这里的“至少需要”是从一个可完成图转换的 IR 必备信息推导出的抽象，不代表已经获得 QAIRT 2.42 的结构定义。

### 7.3 可以怎样类比

```text
ONNX：外部提交的标准交换图
QuIR：QAIRT 内部统一使用的 CAD 图
```

它的价值是把“来源框架怎么写”与“后面怎么映射到 QNN”解耦。

### 7.4 这一阶段重点关注什么

- ONNX op 与属性是否能被完整导入；
- shape/dtype 是否在禁用 PyTorch shape inference 后仍然足够；
- 常量和外置权重是否能正确解析；
- 原始 I/O 名称和顺序是否保留；
- 自定义/不支持算子是否在这里暴露；
- 导入前后的节点语义是否等价。

### 7.5 不要把 QuIR 理解成什么

QuIR 不是：

- 量化后的模型；
- HTP 指令；
- 可直接交给 Genie 的设备文件；
- 当前仓库中稳定存在的 `.quir` 文件；
- 用户需要手写的模型格式。

---

## 八、阶段 4：QuIR → QNNIR

### 8.1 工作性理解

同样，当前仓库没有 QNNIR 的正式 schema。根据名称、链路位置和后续 Emitter 产物，可以把它理解为：

> QNNIR 是比 QuIR 更靠近 QNN 图、算子定义和 Tensor 契约的内部表示。

`QNNIR` 可直观读作 “QNN IR”，但如果要写正式全称、字段或 pass 名称，仍应以 QAIRT 2.42 对应版本的 SDK 资料为准。

### 8.2 从“通用图”向“QNN 图”靠近意味着什么

从编译器角度，通常需要逐步明确：

- 高层算子对应哪些 QNN op；
- 算子参数用什么形式表达；
- Tensor rank、shape、dtype、layout；
- 必要的显式 Reshape / Transpose / Cast；
- 常量和权重如何绑定；
- 输入输出和中间 Tensor 的生命周期；
- 一个来源 op 是否映射为多个目标节点。

但必须注意：当前仓库无法证明上述每项具体在哪个 QAIRT pass 完成，也无法把最终每个节点精确归因到 QuIR 或 QNNIR。

### 8.3 QuIR 与 QNNIR 的对照

| 对比 | QuIR | QNNIR |
|---|---|---|
| 链路位置 | ONNX 之后 | QuIR 之后、Emitter 之前 |
| 工作性定位 | QAIRT 内部较通用的图 | 更贴近 QNN 图契约的图 |
| 更关注 | 来源模型语义和统一表示 | QNN op / Tensor / layout / dtype 契约 |
| 是否等于 HTP binary | 否 | 否 |
| 当前是否独立落盘 | 仓库未发现 | 仓库未发现 |
| 精确 schema | 当前未知 | 当前未知 |

### 8.4 QNNIR 仍不等于具体 HTP 编译结果

QNN 是一套图 API/runtime/backend 体系，可以面向 CPU、GPU、HTP 等 backend。即使图已经“QNN 化”，也不代表已经完成：

- HTP 专属调度；
- VTCM/内存规划；
- 最终算子 kernel 选择；
- INT4 权重打包；
- context binary 生成；
- 目标 SoC 的离线编译。

本文的 Prepare 甚至会从 QNNIR 再反向生成 PyTorch，供后面的 AIMET 浮点量化模拟使用。

### 8.5 这一阶段重点关注什么

- QNN 是否支持当前算子及属性组合；
- layout 转换是否保持 K/V 和 attention 语义；
- 高层 op 拆分后是否改变数值顺序；
- dtype/Cast 是否引入额外精度变化；
- 输入输出名称和原模型节点名称能否建立映射；
- 转换结果是否仍然保持浮点语义。

---

## 九、阶段 5：QAIRT Emitter 重建 PyTorch 模型

### 9.1 为什么又从内部图生成回 Torch

本项目下一站是 AIMET `QuantizationSimModel`。AIMET 需要一个可执行的 PyTorch `nn.Module`，并希望 Add、MatMul、Concat、Reshape 等关键操作能够作为明确模块被识别和插桩。

因此这里不是“绕了一圈又回到原点”，而是：

```text
原始 Torch：Hugging Face 层级 + Python 对象 + 嵌套 KV
重建 Torch：按静态图生成 + 显式算子 + 扁平 Tensor I/O
```

> **“算子显式”一句话**：把原来藏在 Python 表达式或 Tensor 方法里的 `x + residual`、`reshape`、`transpose`、`matmul` 等计算，变成有明确名称、类型、属性和输入输出的独立模块/图节点。

例如：

```python
# Prepare 前：Add 隐藏在 Python 表达式中
x = x + residual

# Emitter 重建后：Add 成为可识别、可插桩的命名模块
self.Add = elementwise_ops.Add()
x = self.Add(x, residual)
```

这样 AIMET 可以在算子边界插入量化模拟器，ONNX/QNN 转换器也能稳定识别和映射节点，并支持逐 Tensor 排错；**算子显式不代表已经量化或已经生成 HTP 指令**。

### 9.2 真实生成文件证明了什么

生成模型开头：

```python
import torch
from qti.aisw.emitter import emitter_ops

try:
    import aimet_torch.nn.modules.custom as elementwise_ops
except ImportError:
    from qti.aisw.emitter import elementwise_ops

class qwen25llm_kvcache_36_layer(torch.nn.Module):
    ...
```

这直接确认新模型由 QAIRT Emitter 生成，并优先使用 AIMET 可识别的自定义算子模块。

### 9.3 模块树被重新组织

原模型类似：

```text
Qwen2ForCausalLM
  model.layers.0
    self_attn
    mlp
```

重建模型则是单一大类下的大量平铺属性：

```python
self.rms_norm_model_layers_0_input_layernorm = ...
self.model_layers_0_self_attn_Reshape = ...
self.model_layers_0_self_attn_Transpose = ...
self.model_layers_0_self_attn_q_proj_conv = torch.nn.Conv2d(...)
self.model_layers_0_self_attn_MatMul = ...
```

当前产物可统计到约 2495 个 `self.*` 模块定义。这个数字只描述当前 36 层、ARN=1073 的生成物，不是 `prepare_model` 的固定规律。

### 9.4 forward 变成逐算子静态程序

当前生成文件的 `forward` 从约 L2509 开始，持续到约 L7207：

```python
def forward(
    self,
    inputs_embeds,
    attention_mask,
    position_ids_cos,
    position_ids_sin,
    past_key_0_in,
    past_value_0_in,
    ...,
    past_key_35_in,
    past_value_35_in,
):
    ...
```

中间计算按图顺序显式执行：

```python
x = self.Reshape(...)
y = self.Permute(x, ...)
x = None
z = self.Conv(y)
...
```

将不再使用的中间变量设为 `None`，说明生成代码还显式表达了 Tensor 生命周期释放机会。

### 9.5 输出也彻底展平

最终返回：

```text
logits,
past_key_0_out, past_value_0_out,
...
past_key_35_out, past_value_35_out
```

共 73 个 Tensor。原始 Transformers 的嵌套 `past_key_values` 和 `ModelOutput` 已不再是模型图外部接口。

### 9.6 Emitter 阶段重点关注什么

- 原模块权重是否正确复制到新模块；
- 一个原模块映射到多个新模块时是否遗漏；
- 新模块命名是否稳定且可供 AIMET/ONNX encodings 对齐；
- 76 个输入与 73 个输出顺序是否一致；
- 固定 Reshape 中的 1073、975、2048 是否正确；
- `keep_original_model_structure=False` 后，下游代码是否仍错误依赖原模块路径；
- generated model 是否能够单独加载并执行。

---

## 十、阶段 6：保存产物并返回 `prepared_model`

### 10.1 路径和名称

```python
prepare_path = f"{output_dir}/prepare"
prepare_filename = f"{model_name}_kvcache_{num_layers}_layer"
```

当前得到：

```text
qwen25llm_kvcache_36_layer
```

### 10.2 当前仓库实际保留的文件

| 文件 | 当前大小 | 可确认用途 |
|---|---:|---|
| `qwen25llm_kvcache_36_layer.py` | 约 745 KB | 重建后的模型结构和 forward |
| `qwen25llm_kvcache_36_layer.json` | 约 41 KB | 原模块到重建模块的一对多映射 |
| `qwen25llm_kvcache_36_layer_io_map.json` | 约 868 KB | 参数名称与激活 I/O 名称映射 |

四类完整产物可以先这样记：

```text
.py          = 新模型的结构和执行代码
.json        = 旧模块展开成哪些新模块/辅助节点
_io_map.json = 新旧参数名，以及新模块端口对应的图 Tensor 名
.safetensors = 新模型的实际权重数值（当前仓库未保留）
```

当前 JSON 可统计到：

```text
原模块映射：322 项
参数映射：650 项
激活映射：2385 项
```

这些数字只对应当前生成物版本。

`_io_map.json` 中虽然出现 `param_encodings`、`activation_encodings` 这样的 key，但当前内容是**名称映射关系**，不是带有 `scale`、`offset`、`min`、`max` 的量化 encoding 数值。真正的量化参数要到后续 QuantSim calibration/export 阶段才产生。

### 10.3 完整运行中还可能出现的文件

项目加载代码和故障记录表明，完整 Prepare 通常还涉及：

- `.safetensors` 权重；
- Prepare 中间 ONNX；
- ONNX 外置 `.weight/.bias`；
- `_Constant_*` 常量文件；
- UUID 临时目录。

当前仓库快照没有 `.safetensors` 和中间 ONNX，因此不能把当前三个文件误认为一个可在所有环境独立恢复的完整产物集合。

同样，生成的 `.py` 不是一个完全自包含的模型文件：它依赖匹配版本的 `qti.aisw.emitter`，通常也依赖 AIMET 自定义模块和单独保存的权重。当前快照缺少 `.safetensors`，所以仅拿这份 `.py` 不能恢复出可直接推理的完整模型。

### 10.4 三类映射分别解决什么问题

#### 原模块 → 新模块映射

例如：

```json
"model.layers.0.self_attn.q_proj_conv": [
  "model_layers_0_self_attn_q_proj_conv",
  "model_layers_layers_0_self_attn_q_proj_conv_Conv_output_0_0123"
]
```

说明一个原始模块可能对应重建图中的多个模块或辅助节点。

#### 参数名称映射

把：

```text
prepared 参数名 ↔ 原模型参数名
```

连接起来，便于权重加载和 encoding 对齐。

#### 激活 I/O 映射

记录图节点的输入输出 Tensor 名称，便于：

- AIMET encodings 映射；
- ONNX 导出名称对齐；
- 中间层测试向量；
- 转换前后逐张量对拍。

#### 一个具体例子：第 0 层 `q_proj_conv`

Prepare 前的模块名是：

```text
model.layers.0.self_attn.q_proj_conv
```

三个当前文件与完整权重文件从不同角度描述它：

1. 生成的 `.py` 定义新模块并在 `forward` 中执行：

```python
# 为便于阅读，省略生成代码中的长变量名和部分 Conv2d 参数
self.model_layers_0_self_attn_q_proj_conv = torch.nn.Conv2d(
    in_channels=2048,
    out_channels=2048,
    kernel_size=(1, 1),
    bias=True,
)

q_output = self.model_layers_0_self_attn_q_proj_conv(transpose_output)
```

2. 普通 `.json` 记录“旧模块 → 新模块/辅助节点”：

```json
"model.layers.0.self_attn.q_proj_conv": [
  "model_layers_0_self_attn_q_proj_conv",
  "model_layers_layers_0_self_attn_q_proj_conv_Conv_output_0_0123"
]
```

这里第一个新模块是真正的 `Conv2d`，第二个是处理输出布局的 `Permute`。这说明一个旧模块经过图展开后可能对应多个新节点。

3. `_io_map.json` 的参数部分记录“新参数名 → Prepare 前参数名”：

```json
"model_layers_0_self_attn_q_proj_conv.weight":
  "model.layers.0.self_attn.q_proj_conv.weight",
"model_layers_0_self_attn_q_proj_conv.bias":
  "model.layers.0.self_attn.q_proj_conv.bias"
```

激活部分则记录新模块第 0 个输入/输出端口对应的图 Tensor 名：

```text
input[0]  → /model/layers.0/self_attn/Transpose_output_0_0231
output[0] → /model/layers/layers.0/self_attn/q_proj_conv/Conv_output_0
```

4. 完整产物中的 `.safetensors` 使用新参数名保存上面 `Conv2d` 的实际 weight/bias 数值；它不是差分包，而是完整 prepared 权重。

#### 两个易混点

- **Permute 是什么**：Permute 只重新排列 Tensor 的维度顺序。例如 Transformer 的 `[B,S,1,H]` 经过 `Permute(0,3,2,1)` 变成 Conv2d 需要的 `[B,H,1,S]`，元素语义不变，只是各维位置和 layout 发生变化；它表示逻辑换序，不等于此处一定立即复制物理内存。
- **为什么叫激活 Tensor 连线映射**：计算图中算子是节点，激活 Tensor 是连接节点的数据边。相同 Tensor 名同时出现在 `A.output[0]` 和 `B.input[0]`，就表示 A 的第 0 个输出接到 B 的第 0 个输入：

```text
Transpose.output[0] ── T1 ──→ q_proj_conv.input[0]
q_proj_conv.output[0] ── T2 ──→ Permute.input[0]
```

只写“Transpose 连接 q_proj_conv”无法说明具体输出/输入端口、分支复用以及应该给哪条数据设置 activation encoding；因此 `_io_map.json` 按模块端口记录 Tensor 名，而不是只保存算子之间的邻接关系。这里的 activation 是运行 `forward` 时才产生的输入、中间结果和输出，不包含单独映射在 `param_encodings` 中的 weight/bias。

#### 三个文件与权重文件如何串起来

```text
Prepare 前模块：model.layers.0.self_attn.q_proj_conv
                         │
             普通 .json │ 记录模块血缘
                         ▼
新模块：model_layers_0_self_attn_q_proj_conv ──┐
                                               │
     .py：定义 Conv2d 和 forward 执行顺序      │
     _io_map.json：对齐参数名和激活 Tensor 名  │
     .safetensors：提供实际 weight/bias 数值   │
                                               ▼
                                可执行的 prepared_model
```

一句话总结：`.py` 管“怎么算”，普通 `.json` 管“从哪来”，`_io_map.json` 管“名字和连线怎么对应”，`.safetensors` 管“参数实际是多少”。

### 10.5 prepared 权重与 Prepare 前权重有什么不同

> **核心结论**：Prepare 不训练、不量化权重；核心权重的数学语义应保持，主要变化是参数名称、所属模块、存储对象，以及可能的 shape/layout 表达。

| 对比项 | Prepare 前 | prepared model |
|---|---|---|
| 数值语义 | 已适配的浮点权重 | 通常保持等价，不是重新训练的权重 |
| 参数名称 | `model.layers.0...` | `model_layers_0...` 等扁平名称 |
| 所属模块 | Hugging Face 嵌套模块 | Emitter 新建的扁平模块 |
| shape/layout | 适配模型所需布局 | 可能 reshape、transpose、重新绑定或复制 |
| dtype | FP32/可选 FP16 | 启用 FP16 时先转 FP32 做 Prepare，完成后再转回 FP16 |
| 附加 Tensor | 原模型参数和 buffer | 可能增加图展开需要的常量与 buffer |
| 保存方式 | 原模型权重文件/state dict | 使用新参数名保存完整 `.safetensors` |

以当前 `q_proj_conv` 为例，Prepare 前和生成 `.py` 中都是 `2048 → 2048` 的 1×1 Conv，因此可观察到的逻辑权重 shape 都是 `[2048, 2048, 1, 1]`，名称和所属模块发生了变化。更早的原始 HF Linear `[2048, 2048] → Conv [2048, 2048, 1, 1]` 是 `prepare_conv()` 完成的，不属于 `model_preparer`。

#### 权重内存排布是否发生变化

- **存储对象会变化**：Emitter 新建模块，加载后是新的 Tensor/storage 和内存地址；
- **权重逻辑布局不能一概而论**：图转换理论上可能通过 reshape/transpose 等方式满足目标图契约；当前缺少 prepared `.safetensors`，不能逐字节证明每个权重是否重排；
- **当前 `q_proj_conv` 没有直接证据表明权重被转置**：前后 Conv shape 相同，`_io_map.json` 也是直接参数映射；
- **激活布局明确会变化**：生成图显式执行 `[B,S,H] → [B,H,1,S]` 等 Reshape/Permute；
- **HTP 最终物理排布尚未确定**：VTCM、权重打包和设备内存规划属于后续 QNN/backend 编译，而不是 Prepare。

一句话总结：核心权重“学到的数值语义”基本不变，但会换名字、模块和存储对象；激活布局明显重排，权重是否物理重排要用完整 `.safetensors` 对拍，最终 HTP 内存布局则在后续编译阶段决定。

### 10.6 `return_prepare_model=True`

调用不仅保存产物，还直接返回新模型：

```python
prepared_model = model_preparer.prepare_model(...)
```

后续无需再次从文件加载，就能立刻进行 PPL 验证和 QuantSim 构造。

---

## 十一、阶段 7：用 PPL 验证 Prepare 是否破坏数值

### 11.1 prepared model 使用扁平接口 FPM

```python
fp_prepared_fpm = LLMForwardPassManager(
    cfg=llm_config,
    model=prepared_model,
    tokenizer=tokenizer,
    separate_tuple_input_output=True,
    num_tokens=ARN,
)
```

原模型使用嵌套输入输出，prepared model 使用扁平 Tensor 接口，所以这里必须改成 `True`。

### 11.2 再计算一次 PPL

```python
prepared_kvcache_ppl = ppl_eval_embedding(...)

print(
    f"ppl score of KVCACHE prepared fp model: {prepared_kvcache_ppl}\n"
    f"orig ppl - prepared ppl = {orig_ppl - prepared_kvcache_ppl}"
)
```

代码注释给出的预期是：

```text
PPL delta < 1e-4
```

但当前代码只有打印，没有 `assert`，所以这是人工验收点，不是自动硬门禁。

### 11.3 PPL 明显变化时优先检查

1. input/output 顺序；
2. cos/sin 顺序；
3. 每层 K/V 顺序；
4. K 的转置布局；
5. mask shape、dtype 和负值；
6. logits 序列长度；
7. FP32/FP16 Cast；
8. 原权重到新权重的映射；
9. 高层算子拆分后的数值顺序；
10. prepared FPM 是否错误使用了嵌套接口。

### 11.4 PPL 不是唯一可做的验证

更强的调试流程可以增加：

- 固定随机种子和同一输入；
- 比较最终 logits 的 max abs / mean abs / cosine similarity；
- 按层保存输出并逐张量对拍；
- 单独比较每层 new K/V；
- 检查 NaN/Inf；
- 找到第一个误差突然放大的节点。

当前仓库已经生成激活 I/O 映射和测试向量工具，为这种逐层验证提供了基础，但主流程尚未把它做成 Prepare 的自动门禁。

---

## 十二、从输入到输出：各阶段“做什么 / 不做什么”总表

| 阶段 | 做什么 | 不做什么 | 验收重点 |
|---|---|---|---|
| 调用前适配 | Linear→Conv、外部 mask/RoPE、KV 改写 | 不属于 `prepare_model` 内部 | 原始适配 FP PPL 正常 |
| 调用前固定 | eval、no grad、tuple 输出、导出 patch | 不改变算法目标 | dummy forward 稳定可运行 |
| Torch tracing | 用具体输入固化路径和 shape | 不探索所有可能分支 | 路径、shape、dtype 正确 |
| ONNX | 建立通用静态交换图 | 不是最终交付 ONNX | opset、I/O、外置权重完整 |
| QuIR | 导入 QAIRT 内部前一级图 | 不是量化、不是设备指令 | 图语义和元数据完整 |
| QNNIR | 向 QNN 图契约继续规范化 | 不是 HTP binary | QNN op/Tensor/layout 语义正确 |
| Emitter | 生成新 PyTorch `nn.Module` | 不保留原 HF 模块树 | 权重、名称、扁平接口一致 |
| 保存 | 写模型结构、权重和映射 | 不保证当前快照文件完整 | 可重新加载、版本匹配 |
| PPL 验证 | 检查转换基本无损 | 不做量化 calibration | 与原始浮点 PPL 接近 |
| 后续 QuantSim | 插量化模拟器、算 encodings | 不属于 Prepare | 量化后精度和 encoding 正确 |

---

## 十三、按报错位置快速判断属于哪一阶段

| 现象 | 最可能阶段 | 优先检查 |
|---|---|---|
| `ModelOutput` / `KeyError` | tracing 前/ONNX 导出 | `return_dict=False` |
| `Tensor that requires grad as a constant` | tracing 前/ONNX 导出 | `eval()`、`requires_grad_(False)` |
| protobuf 2 GiB / shape inference | ONNX 导出 | 大模型 shape inference patch、外置权重 |
| unsupported op/attribute | ONNX → QuIR 或 QuIR → QNNIR | opset、模型适配、算子实现 |
| 输入数量不匹配 | tracing/Emitter 接口 | pytree 展平与 `input_names` |
| K/V shape 错误 | dummy/接口定型 | 975、1073、转置 K、head 数 |
| generated `.py` 能导入但权重缺失 | 保存/加载 | `.safetensors`、文件名前缀、路径 |
| `skip_prepare=true` 报产物不存在 | 复用分支 | 先运行一次 `skip_prepare=false` |
| prepared PPL 明显变差 | 图语义/映射 | I/O 顺序、layout、Cast、权重映射 |
| QuantSim 创建时找不到原模块名 | Emitter 后结构变化 | 使用映射文件或 prepared 模块名 |
| 磁盘突然占满 | ONNX/保存 | 临时目录、外置权重、历史失败残留 |

---

## 十四、Prepare 和量化、QNN 编译的边界

### 14.1 Prepare 后仍是浮点模型

```text
Prepare：重建浮点图
QuantSim：在浮点图中插入 fake quant
SeqMSE：优化权重量化参数
compute_encodings：统计激活范围
qairt-converter / quantizer：生成 Qualcomm 图并真正处理设备量化
context generator：面向 backend / SoC 生成设备上下文
QNN Runtime + HTP：端侧执行
```

### 14.2 Prepare 没有做的事情

`model_preparer.prepare_model()` 本身没有：

- 用真实校准数据统计范围；
- 计算 scale / offset；
- 执行 SeqMSE；
- 把权重打包成 INT4；
- 生成 `.encodings`；
- 生成最终 DLC；
- 生成 context binary；
- 在 HTP 上运行模型。

### 14.3 为什么名字里带 QNNIR，却仍然回到 Torch

QNNIR 在这里充当“QNN 友好图语义”的中间层；Emitter 把这种图重新表达为 AIMET 可处理的 PyTorch 模块。它说明 Prepare 借用了 QNN/QAIRT 图转换能力，不说明此时已经完成端侧编译。

---

## 十五、代码审阅时最值得留意的副作用

### 15.1 dummy 决定的分支会被固化

如果 forward 中存在依赖 Python 值或数据内容的控制流，tracing 通常只保留 dummy 实际走到的路径。换一套运行条件时，不保证未追踪分支仍存在。

### 15.2 shape 基本被写死

当前生成代码大量包含：

```text
1073 / 975 / 2048 / 36 层 / batch=1
```

因此这个 prepared model 不是随意接受任意序列长度的通用动态图。

### 15.3 原模块路径会失效

`keep_original_model_structure=False` 后：

```text
model.layers.17.self_attn.q_proj_conv
```

未必仍能作为 prepared model 的直接属性路径。下游混合精度覆盖、测试向量和 encoding 配置需要使用新名称或映射文件。

### 15.4 ONNX shape inference patch 是全局的

替换 `torch._C` pass 后，同一进程中的后续 export 也受影响。升级 PyTorch 或切换小模型时要重新评估是否仍需要该 patch。

### 15.5 `converter_args` 当前无效

不要因为前面构造了 `NONTRIVIAL` layout 参数，就假设 Prepare 已采用；调用处仍是注释。

### 15.6 `num_logits_to_return` 需要以实际 forward 为准

当前可见 Qwen2 实现没有读取该属性。输出长度由 dummy 和生成图固定是确定事实，这个属性是否在别的版本生效则需要单独核验。

### 15.7 Prepare 的资源峰值可能很高

项目故障记录曾出现：

```text
prepare/ 约 47 GB
每次重生成约 35 GB 中间文件
```

原因包括大模型外置权重、常量、临时 ONNX 和失败残留。应把临时目录放在空间足够的数据盘，并在成功后按明确白名单清理。

---

## 十六、QuIR / QNNIR 的“施工图”类比

```text
PyTorch：设计师用 Python 写的功能需求和结构
ONNX：跨公司的标准交换图纸
QuIR：QAIRT 内部统一 CAD 图
QNNIR：按 QNN 生产线接口整理的施工图
prepared Torch：根据施工图重新画出的可测试样机
DLC/context binary：后续真正交付生产线/设备的成品描述
HTP：最终干活的机器
```

这个类比里最重要的是：

- QuIR/QNNIR 都是**中间图**；
- QNNIR 比 QuIR 更靠近目标图契约，但仍不是机器指令；
- 本项目为了 AIMET 量化模拟，把内部图重新“发射”为 Torch；
- 不能从最终某个 Reshape 反推它一定由哪一级插入。

---

## 十七、执行 Prepare 前后的检查清单

### 17.1 运行前

- [ ] 模型已经完成 Attention/MLP/lm_head 的端侧适配；
- [ ] 原始适配后浮点 PPL 正常；
- [ ] `return_dict=False`；
- [ ] `model.eval()`；
- [ ] 参数已关闭梯度；
- [ ] dummy 的 ARN/context/KV/mask/RoPE shape 正确；
- [ ] `input_names` 与 Tensor 叶子顺序一致；
- [ ] `output_names` 与 logits/KV 顺序一致；
- [ ] opset 与 QAIRT 版本兼容；
- [ ] 输出盘有足够空间；
- [ ] `skip_prepare=false`；
- [ ] Python 版本与 `libPyIrGraph.so` ABI 匹配。

### 17.2 运行中

- [ ] 记录 QAIRT/ONNX 日志；
- [ ] 观察 unsupported op / attribute；
- [ ] 观察内存和磁盘峰值；
- [ ] 保留第一次失败现场，不立即删除全部临时文件；
- [ ] 确认没有悄悄切换到不期望的 dtype/layout。

### 17.3 运行后

- [ ] `.py` 能导入；
- [ ] 权重文件存在且能加载；
- [ ] JSON 映射文件存在；
- [ ] forward 输入为预期的 76 个 Tensor；
- [ ] 输出为 logits + 72 个 KV；
- [ ] 输出 shape/dtype 正确且无 NaN/Inf；
- [ ] prepared PPL 与原始 PPL 接近；
- [ ] 产物复制时没有漏外置权重；
- [ ] 验证成功后再启用 `skip_prepare=true`；
- [ ] 清理临时文件前先明确保留白名单。

---

## 十八、常见问答

### 18.1 Prepare 是不是 QNN 编译

不是最终 QNN 编译。它使用 QAIRT/QNN 图工具链整理并重建浮点模型，最终 DLC/context binary 仍在后续阶段。

### 18.2 Prepare 会不会把 Linear 变成 Conv

本项目的 Linear → Conv 主要在更早的 `prepare_conv()` / Monkey Patch 阶段完成。`model_preparer` 接收的是已经改造过的模型。

### 18.3 为什么要经过 ONNX

ONNX 提供 PyTorch 到 QAIRT Converter 的稳定静态图边界，使 Python 模块和嵌套对象先变成节点、Tensor、属性与常量。

### 18.4 为什么经过 QNNIR 后还生成回 PyTorch

因为下一步 AIMET QuantSim 需要可执行、可插桩的 `nn.Module`。重建模型保留浮点语义，但图结构与接口已经静态、显式。

### 18.5 dummy input 是不是校准数据

不是。dummy 决定图和接口；SeqMSE、`compute_encodings()` 才使用真实校准数据确定量化参数。

### 18.6 可以一直 `skip_prepare=true` 吗

只有 prepared 结构、权重、映射与当前代码/SDK/配置完全匹配时才可以。第一次必须成功生成；改变 ARN、context、层数、接口、模型权重或 SDK 后应重新评估。

### 18.7 QuIR/QNNIR 会不会各生成一个文件

当前仓库没有这种文件，不能这样假设。它们更适合先理解为 SDK 内部的图表示阶段。

### 18.8 Prepare 后为什么还要测 PPL

因为即使没有量化，tracing、算子拆分、layout/Cast、权重映射和 I/O 展平都可能引入错误。PPL 是进入量化前的隔离门。

---

## 十九、面试版回答

### 19.1 30 秒版本

> 这个项目的 Prepare 使用 QAIRT 的 `model_preparer.prepare_model()`。它先用固定 dummy input 将已完成端侧适配的 Qwen2 PyTorch 模型追踪为 ONNX，再经过 QAIRT 内部 QuIR、QNNIR 图表示，最后由 Emitter 重建成输入输出扁平、shape 固定、算子显式的新 PyTorch 模型。这个模型仍是浮点模型，主要为了让 AIMET QuantSim 能稳定插桩和导出；Prepare 后会重新测 PPL，确认图转换没有提前破坏精度。

### 19.2 深挖追问版本

如果继续追问“最难的点”，可以回答：

1. 大模型 ONNX 超过 protobuf 2 GiB，需要处理 shape inference 与外置权重；
2. Transformers 的 `ModelOutput` 和嵌套 KV 必须变成稳定 tuple/扁平 Tensor 接口；
3. 76 个输入和 73 个输出的名称与顺序必须严格一致；
4. `keep_original_model_structure=False` 后模块路径发生变化，需要 JSON/I/O map 对齐权重和 encodings；
5. Prepare 本身不量化，因此必须在 QuantSim 前用 PPL 或逐层对拍隔离图转换误差。

---

## 二十、记忆锚点

```text
model_preparer = 图编译往返，不是量化

已适配 Torch
  → dummy 定路径和 shape
  → ONNX 建静态交换图
  → QuIR 做内部统一表示
  → QNNIR 靠近 QNN 图契约
  → Emitter 重建扁平 Torch
  → PPL 验证
  → 后续才 QuantSim
```

再记住四条边界：

- `prepare_conv` 改算子，`model_preparer` 重建图；
- dummy 定图，真实 calibration data 定量化范围；
- QNNIR 是中间图，不是 HTP binary；
- PPL delta 是检查点，但当前代码没有自动 assert。

---

## 二十一、源码、产物与资料入口

### 21.1 主代码

- `example1/llm_quant.py`
  - QAIRT SDK 路径：约 L18～28
  - 模型适配与 `prepare_conv()`：约 L37～104
  - dummy 构造：约 L255～323
  - Prepare 环境与调用：约 L326～418
  - prepared PPL：约 L420～434
  - QuantSim 起点：约 L437～462
- `example1/config.yaml`
  - QAIRT 版本路径：L8
  - ARN/context/skip_prepare：约 L16～24
  - Prepare opset：约 L64～67
- `example1/TROUBLESHOOTING.md`
  - Python ABI、ModelOutput、requires-grad、2 GiB、磁盘清理问题

### 21.2 当前真实生成物

- `output/prepare/qwen25llm_kvcache_36_layer.py`
  - Emitter import：L1～7
  - 新模型类：L9
  - 显式模块：L13 起
  - 扁平 forward：约 L2509
  - 73 个输出：约 L7207
- `output/prepare/qwen25llm_kvcache_36_layer.json`
  - 原模块 → 新模块映射
- `output/prepare/qwen25llm_kvcache_36_layer_io_map.json`
  - 参数映射与激活 I/O 映射

### 21.3 关联笔记

- [06 · Prepare 模型准备阶段](./06-Prepare模型准备阶段.md)
- [06-附录A · Prepare Dummy Input](./06-附录A-Prepare-Dummy-Input输入模具.md)
- [06-附录B · QAIRT / QNN / AIMET / QuantSim](./06-附录B-QAIRT-QNN-AIMET-QuantSim概念与关系.md)
- [06-附录D · Prepare 面试速答](./06-附录D-Prepare面试速答.md)
- [05 · 通用前向处理流程](./05-通用前向处理流程.md)
- [02-附录E · 端侧定长与计算图导出](./02-附录E-端侧定长与计算图导出.md)
- [02-附录K · KV Cache](./02-附录K-KV%20Cache(键值缓存).md)

### 21.4 官方资料

- [PyTorch ONNX Export 文档](https://docs.pytorch.org/docs/main/onnx_export.html)：ONNX 导出 API、版本与动态 shape 入口。
- [PyTorch `torch.jit.trace` 文档](https://docs.pytorch.org/docs/2.9/generated/torch.jit.trace.html)：tracing 只记录示例输入实际执行的 Tensor 运算及其控制流限制。
- [Qualcomm `qairt-converter`](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/qairt_converter.html?product=1601111740010412)：来源模型、转换图、I/O layout/dtype、DLC 等公开说明。
- [Qualcomm AI Runtime SDK 文档入口](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/index_SNPE.html?product=1601111740010412)：QAIRT/QNN、backend、converter 与 API 文档树。
- [Qualcomm QNN Source-Op Mapping 类型](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/enum_QnnTypes_8h_1a18a7d4fb246fa988581a01ed79bf6f77.html)：来源框架 op 到 QNN op 映射概念。
- [AIMET FX Model Preparer](https://quic.github.io/aimet-pages/releases/2.26.0/apiref/torch/model_preparer.html)：用于对照 AIMET 公开 FX `prepare_model`；它不是本文的 QAIRT `qti.aisw.preparer_api.model_preparer`。
- [AIMET Quantization Simulation Guide](https://quic.github.io/aimet-pages/releases/latest/tutorials/quantsim.html)：prepared model 后续 QuantSim 工作流。

### 21.5 仍需 QAIRT 2.42 源码/文档确认的问题

- [ ] QuIR 的官方全称、schema 与 Python/C++ 类型；
- [ ] QNNIR 的官方全称、schema 与 Python/C++ 类型；
- [ ] ONNX → QuIR → QNNIR 的精确 pass 列表和顺序；
- [ ] 每个 Reshape/Permute/Cast 的具体插入阶段；
- [ ] Prepare 中间 ONNX、QuIR、QNNIR 的默认保存策略；
- [ ] `.safetensors` 的完整命名和版本兼容规则；
- [ ] 如何开启 SDK 级逐 pass dump / graph dump；
- [ ] 如何把逐张量误差验证做成 Prepare 自动门禁。
