# 06 · Prepare 模型准备阶段

> **流程位置**：承接 [05 · 通用前向处理流程](./05-通用前向处理流程.md)，位于“浮点模型 PPL 基线”之后、“QuantSim 量化”之前。
>
> **一句话本质**：Prepare 用一套固定形状的 dummy input 跑通已经完成端侧适配的 PyTorch 模型，经 QAIRT/QNN 的 `model_preparer` 转换并重建为输入输出签名明确、适合后续 QuantSim 和 ONNX 导出的 prepared model。
>
> **当前学习进度**：Prepare 阶段已完成。本篇负责主线地图；dummy input 的构造见 [附录A · Prepare Dummy Input 输入模具](./06-附录A-Prepare-Dummy-Input输入模具.md)，QAIRT `model_preparer.prepare_model()` 的内部阶段、产物和检查重点见 [附录C · model_preparer 内部流程](./06-附录C-QAIRT-model_preparer内部流程.md)，面试复习见 [附录D · Prepare 面试速答](./06-附录D-Prepare面试速答.md)。

---

## 一、介绍：这里的 Prepare 到底是什么

主线脚本在原始浮点 PPL 评估结束后进入：

```python
# Prepare model by QAIRT(QNN)
# torch graph → ONNX → QuIR → QNNIR → 重建torch图
```

这一步不是训练，也还不是正式量化。它主要完成：

- 用样例输入确定模型的固定计算路径与输入输出 shape；
- 把 PyTorch 模型转换到 QAIRT/QNN 能处理的中间表示；
- 将嵌套 KV Cache 接口整理成静态图友好的独立 Tensor 接口；
- 重建出后续可交给 AIMET `QuantizationSimModel` 的 prepared model；
- 用 PPL 再验证一次转换前后是否数值一致。

### 1.1 本项目有多个名字相似的 “prepare”

必须区分下面四件事：

| 名称 | 出现位置 | 作用 |
|---|---|---|
| `module.prepare_conv()` | 模型加载后 | 把 Attention/MLP/lm_head 的 Linear 权重搬到 1×1 Conv |
| `FPM.prepare_inputs()` | 每次真实前向前 | 把真实变长数据整理成 mask、RoPE、定长输入和 KV |
| `get_dummy_data()` | Prepare/QuantSim/导出前 | 伪造一套形状正确的固定输入 |
| `model_preparer.prepare_model()` | 本章主阶段 | 通过 QAIRT/QNN 流程转换并重建模型 |

它们的关系是：

```text
prepare_conv
  改模型里的算子实现
        ↓
get_dummy_data
  提供静态输入模具
        ↓
model_preparer.prepare_model
  转换/重建模型图
        ↓
FPM.prepare_inputs
  后续用真实数据运行 prepared model
```

所以本章的 “Prepare 模型” 和上一章的 `prepare_inputs()` 不是一回事。

---

## 二、Prepare 阶段全景

相关代码位于 `example1/llm_quant.py` 约 L251～429，可以先记住这条主线：

```text
已经适配过的浮点 Qwen2 模型
        │
        ├─ 生成 dummy_input
        ├─ 定义 input_names / output_names
        ├─ 配置静态图导出环境
        │
        ▼
model_preparer.prepare_model(...)
        │
        ├─ PyTorch graph
        ├─ ONNX
        ├─ QuIR
        ├─ QNNIR
        └─ 重建 PyTorch model
        │
        ▼
prepared_model（扁平、定长接口）
        │
        ├─ LLMForwardPassManager
        └─ 再测一次 PPL
        │
        ▼
确认 Prepare 没有破坏精度
        │
        ▼
进入 QuantSim
```

Prepare 阶段的输入是已经完成以下适配的浮点模型：

- Attention 已换成 `QcAttention`；
- Q/K/V/O Linear 已换成 1×1 Conv；
- MLP Linear 已换成 1×1 Conv；
- lm_head 已换成 1×1 Conv；
- causal mask 已外部化；
- RoPE cos/sin 可作为外部输入；
- KV Cache 已支持转置存储和只返回新 K/V。

因此 Prepare 不是重新做这些改造，而是让 QAIRT/QNN 识别并重建这张已经改造好的图。

---

## 三、第一步：生成 dummy input

```python
dummy_input = get_dummy_data(
    llm_config,
    tokenizer,
    "cpu",
    separate_tuple_input_output=False,
    num_tokens=ARN,
    dtype=model.dtype,
)
```

dummy input 不表达真实文本或图像语义，只负责给工具一个具体样例，以确定：

- 输入个数；
- 输入顺序；
- 每个 Tensor 的 shape；
- dtype 和 device；
- Attention mask 与 RoPE 的接口形态；
- 36 层 KV Cache 的布局。

当前核心长度为：

```text
Current = ARN = 1073
Past KV = 2048 - 1073 = 975
Total   = 2048
```

当前 embedding 路径的 dummy 输入包含：

```text
inputs_embeds       [1, 1073, 2048]
attention_mask      [1, 1, 1073, 2048]
position_ids_cos    [1, 1, 1073, 64]
position_ids_sin    [1, 1, 1073, 64]
36 层 × (K, V)      72 个 KV Tensor
```

完整逐行分析见 [06-附录A · Prepare Dummy Input 输入模具](./06-附录A-Prepare-Dummy-Input输入模具.md)。

---

## 四、第二步：给所有输入输出命名

QAIRT/ONNX 静态图需要稳定、独立的输入输出名称。

### 4.1 输入名称

当前 `use_input_embeddings=true`：

```python
input_names = ["inputs_embeds", "attention_mask"]
```

当前 `use_position_embedding_input=true`：

```python
input_names += ["position_ids_cos", "position_ids_sin"]
```

再追加每一层的 K/V：

```python
past_key_0_in
past_value_0_in
past_key_1_in
past_value_1_in
...
past_key_35_in
past_value_35_in
```

总输入数量：

```text
1 embedding + 1 mask + 2 RoPE + 36×2 KV
= 76 个 Tensor 输入
```

### 4.2 输出名称

输出由 logits 和每层新 K/V 组成：

```python
output_names = ["logits"] + KV_output_names
```

即：

```text
logits
past_key_0_out
past_value_0_out
...
past_key_35_out
past_value_35_out
```

总输出数量：

```text
1 logits + 36×2 KV
= 73 个 Tensor 输出
```

这正是“把 PyTorch 嵌套 tuple 变成端侧平面接口”的落地。

### 4.3 为什么必须命名（目的）

**一句话**：名字是这张静态图对外的**“接线端子标签”**。图导出后就脱离了 Python，外部（端侧运行时、编译工具、你自己的代码）只能**靠名字找到该往哪个口插数据**。本项目 76 进 / 73 出，没有名字根本没法管。

**不命名会怎样**：PyTorch 若不指定 `input_names`，会自动生成机器名，且模型一改就变：

```text
❌ onnx::Gather_1847   ← 这是第几层的 key 还是 value？
✅ past_key_17_in      ← 一眼看懂
```

**四个具体目的（按重要性）**：

| # | 目的 | 说明 |
|---|------|------|
| 1 | **端侧运行时按名字绑定内存** | 图编译成 context binary 后，**名字是唯一对外接口**：`runtime.setInput("past_key_17_in", buffer)`。没有名字就无法把数据递进去 |
| 2 | **KV 的 in/out 配对回环**（本项目最关键）| 历史 KV 由外部管理（[附录K](./02-附录K-KV%20Cache(键值缓存).md)），每步要把上次的 `past_key_i_out` 喂回下次的 `past_key_i_in`。命名**有规则**，一个循环就能自动配对 72 个张量，否则要手工维护映射 |
| 3 | **编译工具逐个配置输入** | `qairt-converter` 按名字为每个输入指定布局：`converter_args` 里 `{"name": input_name, "source_model_input_layout": ...}`（`llm_quant.py` L360-364）。⚠️ 当前该参数在调用处被注释（L411），属**预留能力**，但正说明“配置是按名字下发的” |
| 4 | **调试 / 可视化 / 测试向量** | Netron 看图能读懂；端侧对拍的测试向量也按名字存盘（`generate_test_vectors(..., input_names=input_names)`）|

**代码里已经在按名字寻址**——FPM 取回旧 KV 就是靠命名规则拼出来的（`forward_pass_wrapper.py` L595-599）：

```python
return tuple(
    (prepared_inputs[f"past_key_{i}_in"], prepared_inputs[f"past_value_{i}_in"])
    for i in range(self.num_layers)
)
```

**⚠️ 名字和顺序是“双重契约”**：`input_names` 的顺序必须和 dummy input 的扁平 tuple **严格逐位对应**，因为 tracing 是**按位置**把名字贴到张量上的：

```text
dummy tuple: (inputs_embeds, attention_mask, cos, sin, K0, V0, K1, V1, ...)
input_names: [ inputs_embeds, attention_mask, position_ids_cos, position_ids_sin, past_key_0_in, past_value_0_in, ...]
                    ↑ 第 k 个张量就贴第 k 个名字，顺序错位 = 名字全贴错
```

> 一句话收口：命名把图从“Python 里的一次函数调用”变成**可被外部按名字接线的独立部件**（详见 [06-附录A](./06-附录A-Prepare-Dummy-Input输入模具.md) 十一节的扁平 tuple 对应关系）。

---

## 五、第三步：配置 Prepare/ONNX 环境

### 5.1 让 QAIRT 保留重建所需的图信息

```python
onnx_utils.EXPORT_TO_ONNX_DIRECT = True
ir_graph_op_handler.KEEP_ORIGINAL_MODEL_STRUCTURE = False
```

从代码注释可知，本项目希望通过 QAIRT 的中间表示重新建立适合后续处理的 Torch 模型，而不是要求完全保留原始 Python 模块层级。

### 5.2 关闭大模型 ONNX 内存内 shape inference

Qwen2.5-VL-3B 模型超过 2 GB，PyTorch ONNX shape inference 可能触发 protobuf 的 2 GiB 限制。因此代码将相关底层 shape inference pass 替换为空操作：

```python
torch.onnx._globals.GLOBALS.onnx_shape_inference = False
```

并处理：

```python
_jit_pass_onnx_node_shape_type_inference
_jit_pass_onnx_graph_shape_type_inference
```

这属于大型 ONNX 导出的工程兼容处理，不是模型算法的一部分。

---

## 六、第四步：运行或跳过 Prepare

配置项：

```yaml
skip_prepare: true
```

决定当前运行是“重新 Prepare”还是“加载以前生成的结果”。

### 6.1 `skip_prepare=true`

代码检查 prepared Python 文件是否存在：

```python
prepared_model_path = ...
```

存在时通过 safetensors 相关加载函数恢复：

```python
prepared_model = load_torch_model_using_safetensors(...).eval()
```

这样可以避免每次都重复耗时的 PyTorch → ONNX → QAIRT 中间图 → PyTorch 转换。

### 6.2 `skip_prepare=false`

真正转换前先固定模型行为：

```python
model.num_logits_to_return = ARN
model.config.return_dict = False
model.eval()
model.requires_grad_(False)
```

分别表示：

- 固定本图需要返回的 logits 长度；
- 让模型返回 tuple，避免 ONNX tracing 对 `ModelOutput` 整数索引不兼容；
- 切换推理模式；
- 关闭参数梯度，避免 tracing 将带梯度 Tensor 当常量时报错。

然后执行核心调用：

```python
prepared_model = model_preparer.prepare_model(
    model,
    dummy_input,
    model_name=prepare_filename,
    filename=prepare_filename,
    path=prepare_path,
    input_names=input_names,
    output_names=output_names,
    onnx_export_args={"opset_version": prepare_opset_version},
    return_prepare_model=True,
    keep_original_model_structure=False,
)
```

关键输入可以理解为：

| 参数 | 作用 |
|---|---|
| `model` | 已完成端侧算子适配的浮点模型 |
| `dummy_input` | 用来跑图、确定 shape 的输入模具 |
| `input_names` | 76 个静态图输入名称 |
| `output_names` | 73 个静态图输出名称 |
| `opset_version` | Prepare 阶段 ONNX 算子集版本 |
| `return_prepare_model=True` | 返回重建后的 Torch 模型 |
| `keep_original_model_structure=False` | 允许按转换图重建，而非强行保留原模块结构 |

完整的 `Torch → ONNX → QuIR → QNNIR → Emitter 重建 Torch` 阶段拆解、证据边界与排错地图见 [附录C · QAIRT model_preparer 内部流程](./06-附录C-QAIRT-model_preparer内部流程.md)。

---

## 七、第五步：验证 prepared model

Prepare 完成后，原始 `model` 被释放：

```python
del model
```

然后给 `prepared_model` 创建新的 FPM：

```python
fp_prepared_fpm = LLMForwardPassManager(
    cfg=llm_config,
    model=prepared_model,
    tokenizer=tokenizer,
    separate_tuple_input_output=True,
    num_tokens=ARN,
)
```

这里和原始模型 FPM 的关键区别是：

```text
原始模型：   separate_tuple_input_output=False
prepared：   separate_tuple_input_output=True
```

原因是 prepared model 的 K/V 已经变成：

```text
K0, V0, K1, V1, ...
```

而不是：

```python
((K0, V0), (K1, V1), ...)
```

最后重新计算 PPL：

```python
prepared_kvcache_ppl = ppl_eval_embedding(
    test_dataloader,
    fp_prepared_fpm,
)
```

应该比较：

```text
orig_ppl              原始适配后浮点模型
prepared_kvcache_ppl  QAIRT Prepare 后浮点模型
```

两者应非常接近。若 Prepare 后 PPL 明显恶化，说明问题发生在量化之前，应优先检查：

- 输入输出顺序是否一致；
- KV Cache K/V 是否错位；
- K 的转置布局是否一致；
- cos/sin 顺序是否一致；
- mask shape/dtype 是否一致；
- prepared model 重建是否改变了算子语义。

---

## 八、Prepare 和量化的边界

Prepare 后的模型仍然是浮点模型：

```text
Prepare：转换/规范图与接口
QuantSim：插入量化模拟器
compute_encodings：用校准数据统计量化范围
SeqMSE：优化权重量化误差
```

因此本阶段的验收目标不是“模型变小了多少”，而是：

```text
接口是否适合静态图
计算是否能跑通
Prepare 前后数值是否一致
```

---

## 九、记忆锚点

```text
Prepare = 用 dummy input 把“已适配的浮点模型”转换成
          QAIRT/QuantSim/ONNX 能继续处理的固定图模型
```

- dummy input 是输入模具，不是校准数据。
- `input_names/output_names` 把嵌套 KV 展平成独立 Tensor 接口。
- 名字是静态图对外的“接线端子标签”：运行时按名字绑内存、KV 靠命名规则 `_out → _in` 自动回环（见 4.3）。
- 名字与 dummy tuple 的**顺序**是双重契约，tracing 按位置贴名字，错位即全错。
- `model_preparer.prepare_model()` 才是本章真正的 Prepare 核心。
- `skip_prepare=true` 表示复用以前准备好的产物。
- Prepare 后仍是浮点模型，还没有计算量化 encodings。
- Prepare 后必须重新测 PPL，先确认图转换没有破坏精度。
- `FPM.prepare_inputs()` 是运行时数据整理，与本章模型 Prepare 不同。

---

## 十、源码入口与后续问题

- `example1/llm_quant.py`
  - `get_dummy_data`：约 L255～323
  - `_get_past_key_values_names`：约 L339～344
  - Prepare 配置与调用：约 L347～414
  - prepared model PPL：约 L420～429
- `example1/llm_utils/forward_pass_wrapper.py`
  - `flatten_tensors`：约 L23
  - `get_padded_kv_values`：约 L30
  - `prepare_combined_attention_mask`：约 L218
- 关联笔记：
  - [05 · 通用前向处理流程](./05-通用前向处理流程.md)
  - [06-附录A · Prepare Dummy Input](./06-附录A-Prepare-Dummy-Input输入模具.md)
  - [06-附录B · QAIRT / QNN / AIMET / QuantSim 概念关系](./06-附录B-QAIRT-QNN-AIMET-QuantSim概念与关系.md)
  - [06-附录C · QAIRT model_preparer 内部流程](./06-附录C-QAIRT-model_preparer内部流程.md)
  - [06-附录D · Prepare 面试速答](./06-附录D-Prepare面试速答.md)
  - [07 · 量化主流程：QuantSim 到 Encoding](./07-量化主流程-QuantSim到Encoding.md)
  - [02-附录E · 端侧定长与计算图导出](./02-附录E-端侧定长与计算图导出.md)

完成情况与可选进阶：

- [x] dummy input 为什么需要 mask、RoPE 和 36 层 KV？→ 见附录A。
- [x] `model_preparer.prepare_model()` 内部各阶段具体生成什么产物？→ 见附录C。
- [x] QuIR 与 QNNIR 分别是什么？→ 见附录C（按证据边界给出工作性理解）。
- [x] prepared model 的扁平 forward 签名长什么样？→ 见附录C。
- [ ] 可选进阶：把 Prepare 前后逐张量误差验证做成自动门禁。
