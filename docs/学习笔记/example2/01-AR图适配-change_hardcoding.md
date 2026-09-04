# 01 · AR 图适配：从 AR1073 到 AR1 / AR128

> **学习位置**：Example2 五阶段中的第一阶段。
>
> **前置阅读**：[00 · Example2 主机编译全景](./00-example2主机编译全景.md)
>
> **一句话本质**：保持模型权重语义和 `Context Length=2048` 不变，把 Example1 导出的 AR1073 固定图改造成 AR1 与 AR128 两套固定 Shape 图，分别为 Decode 和分块 Prefill 准备编译入口。

---

## 零、先建立概括性认知

在阅读代码前，先记住下面这张图：

```text
Example1 导出包
AR1073 + Past-KV975 + CL2048
        │
        ▼ change_hardcoding
        │
        ├── AR1 + Past-KV2047 + CL2048
        │      └── 通常用于逐 Token Decode
        │
        └── AR128 + Past-KV1920 + CL2048
               └── 通常用于分块 Prefill
```

本阶段不是重新训练，也不是重新寻找量化参数。可以把它理解成：

> **同一台计算机器，换成两种固定尺寸的进料模具。**

- AR1 模具一次放入 1 个新 Token；
- AR128 模具一次放入 128 个新 Token；
- 两者看到的“历史 KV + 当前 Token”总容量仍然是 2048；
- 模型层数、训练权重、词表以及量化 Encoding 的数值规则不在本阶段重新计算。

### 0.1 本阶段主要处理哪些对象

`change_hardcoding.execute()` 不是只修改一个 ONNX 输入 Shape，而是以 Example1 的整个导出包为输入，协调处理以下对象：

| 处理对象 | 处理方式 | 主要实现 |
|---|---|---|
| ONNX 计算图 | 修改输入、输出、`value_info` Shape，以及部分节点 Tensor 常量和全 1 辅助 Initializer | `apply_fix()` |
| Test Vector | 递归调整 `.pkl` 中 PyTorch/NumPy Tensor 的 Current Length 与 Past-KV Length | `fix_shapes()` |
| Encoding 与配置 | `.encodings`、JSON、YAML 原样复制到目标 AR 目录 | `shutil.copyfile()` |
| 输出目录结构 | 保持输入文件的相对目录层级，分别写入 AR1/AR128 目录 | `remap_path()` |

可以把处理边界概括为：

```text
真正修改
├── ONNX 计算图
└── Test Vector

只复制
├── Encoding
├── JSON
└── YAML

不会重新执行
├── 模型训练或微调
├── 普通权重数值优化
└── 量化 Calibration
```

因此，本阶段的核心不是改变模型“学会了什么”，而是让 ONNX 图和配套测试输入共同满足 AR1/AR128 的固定 Shape 契约。

### 0.2 本篇先记住十个结论

1. AR 可以先理解为“一张固定图一次并行处理的新 Token 数”。
2. `Current Length + Past-KV Length = Context Length`。
3. 原始图是 `1073 + 975 = 2048`。
4. AR1 图是 `1 + 2047 = 2048`。
5. AR128 图是 `128 + 1920 = 2048`。
6. 本阶段直接产物是两套 ONNX 导出目录，不是 DLC，也不是生成后的文本。
7. 模型执行时仍输出 `logits + 36 层的新 Key/Value`。
8. 实现上不只是改 ONNX 输入 Shape，还会修改输出/中间 Shape、Tensor Attribute、少量 Shape Initializer 和 Test Vector。
9. `.encodings`、JSON、YAML 在这里主要是复制，不会重新做 AIMET Calibration。
10. Shape 最终会一路固化到 Split ONNX、SHA ONNX、普通 DLC 和 Quantized DLC。

### 0.3 学完本篇应该会什么

学完后应能独立回答：

- 为什么一套权重需要 AR1 和 AR128 两张图？
- 为什么把 AR1073 改为 AR1 时，Past-KV 必须从 975 改为 2047？
- `gen_ar()` 给 `change_hardcoding.execute()` 传了什么？
- `change_hardcoding.py` 实际修改哪些对象、复制哪些文件？
- AR1/AR128 的输入和输出 Shape 分别是什么？
- Shape 最终影响 QNN 编译和端侧运行的哪些地方？
- 为什么不能仅凭日志中的 `done` 判断 AR 图生成成功？

---

## 一、为什么部署前需要 AR 图适配

### 1.1 PyTorch 运行与 QNN 固定图的差别

上层 PyTorch 推理比较容易接受变长序列：这次输入 37 个 Token，下次输入 128 个 Token，框架可以在运行时重新分配 Tensor。

端侧 QNN/HTP 编译更依赖固定的 Tensor Shape。编译器会根据 Shape 提前确定：

- 输入输出 Buffer 大小；
- 中间 Activation 的内存；
- Attention MatMul 的维度；
- 算子切块与调度；
- 图能够接受的运行时接口。

因此，Prefill 与 Decode 通常要准备不同的固定 Shape 图。

### 1.2 AR1 和 AR128 分别解决什么问题

| 图 | 一次处理的新 Token | 主要目标 | 特点 |
|---|---:|---|---|
| AR1 | 1 | Decode | 单步延迟低，生成一个 Token 后反复调用 |
| AR128 | 128 | 分块 Prefill | 一次并行处理一块 Prompt，吞吐量更高 |

AR128 是本项目的部署选择，不是所有模型都必须用 128。其他部署也可能选择 AR64、AR256 或多套 AR-N 图。

### 1.3 为什么不能只改 `inputs_embeds`

假如只把 `inputs_embeds` 从 1073 改成 1，却仍保留 Past-KV975：

```text
Past-KV975 + Current1 = 976
```

但 Attention Mask 和总 Context 仍按 2048 设计，K/V 拼接、Attention MatMul、Mask 广播和 Reshape 就会互相不匹配。

所以必须协调修改一整套 Shape 契约：

```text
Current Token Shape
Past Key / Past Value Shape
Attention Mask Query 维
RoPE cos/sin 的序列维
New Key / New Value 输出维
Logits 和中间 ValueInfo
图内 Reshape / Slice 等相关常量
Test Vector 中对应 Tensor Shape
```

---

## 二、前置准备

### 2.1 上游必须已有 Example1 导出包

主脚本用 `LLAMA_MODELS` 指向 Example1 的输出：

```python
LLAMA_MODELS = "/root/autodl-tmp/zgj/Qwen25/outputs/output"
```

至少应确认：

```text
output/
├── onnx/
│   ├── qwen25llm.onnx
│   ├── qwen25llm.encodings
│   └── ONNX 引用的外置权重数据
└── test_vectors/
    └── qt_0.pkl
```

| 输入 | 本阶段用途 |
|---|---|
| `qwen25llm.onnx` | 修改固定 Shape 和 Shape 相关常量 |
| ONNX 外置权重 | 继续为新 ONNX 提供训练后的 Weight 数据 |
| `qwen25llm.encodings` | 复制到 AR1/AR128 导出目录，供后续阶段使用 |
| `qt_0.pkl` | 把测试 Tensor 同步调整到 AR1/AR128 Shape |

注意：ONNX 文件可以只保存图和外置权重引用。不能因为 `.onnx` 文件本身存在，就断定它能独立加载全部权重。

### 2.2 主脚本的四个长度参数

[`qnn_compile_deploy.py`](../../../example2/host_linux/qnn_compile_deploy.py) 当前配置：

```python
CL = 2048
ARNs = [1, 128]
EXPORT_AR = 1073
EXPORT_CONTEXT_LENGTH = 2048
```

含义如下：

| 参数 | 含义 |
|---|---|
| `EXPORT_AR=1073` | Example1 原始 ONNX 的当前 Token 固定长度 |
| `EXPORT_CONTEXT_LENGTH=2048` | Example1 原始图的总 Context Length |
| `CL=2048` | Example2 目标图的总 Context Length |
| `ARNs=[1,128]` | 要生成的两种目标 Current Length |

### 2.3 先算出 Past-KV 长度

公式：

```text
Past-KV Length = Context Length - AR
```

所以：

| 图 | AR/Current | Past-KV | Context Length |
|---|---:|---:|---:|
| 原始导出图 | 1073 | 975 | 2048 |
| AR1 | 1 | 2047 | 2048 |
| AR128 | 128 | 1920 | 2048 |

这不是三个相互独立的数字，而是一条必须始终成立的 Shape 约束。

### 2.4 运行目录与环境

主脚本使用：

```python
workfolder = os.getcwd()
```

并根据当前工作目录拼接 `../G2G`、`assets/` 等路径。因此完整流程应从 Linux QAIRT 主机的 `example2/host_linux` 启动：

```bash
cd example2/host_linux
PYTHONUNBUFFERED=1 python qnn_compile_deploy.py 2>&1 | tee qnn_compile.log
```

完整主脚本在进入 AR 阶段前就会断言：

```text
QNN_SDK_ROOT 必须存在
LLAMA_MODELS 必须存在
```

但是从职责上说，单独执行 `change_hardcoding.py` 的核心逻辑只依赖 Python、ONNX、NumPy 和 PyTorch；QNN Converter/Quantizer 是后续阶段才会调用。

### 2.5 前置检查清单

正式运行前应确认：

- [ ] `LLAMA_MODELS` 指向本次 Example1 的真实输出；
- [ ] `qwen25llm.onnx` 能被 ONNX 加载；
- [ ] ONNX 的 External Data 路径可解析；
- [ ] `qwen25llm.encodings` 与这份 ONNX 对应；
- [ ] `test_vectors/qt_0.pkl` 存在且能反序列化；
- [ ] 原始图确实是 AR1073、CL2048、Past-KV975；
- [ ] 从 `example2/host_linux` 启动完整主脚本；
- [ ] 磁盘和内存足够容纳两套大模型图产物。

---

## 三、本阶段的输入和直接输出

### 3.1 必须区分两种“输出”

问题“AR 图适配后输出是什么”有两层含义。

第一层是**这个处理阶段生成的文件产物**：

```text
assets/models_ar_n/
├── ar1-cl2048/
│   ├── onnx/...
│   └── test_vectors/...
└── ar128-cl2048/
    ├── onnx/...
    └── test_vectors/...
```

第二层是**运行适配后的模型时，图会吐出什么 Tensor**：

```text
logits
+ 36层 New Key
+ 36层 New Value
= 73个模型输出端口
```

本阶段只生成第一层文件产物；第二层 Tensor 是未来真正执行 ONNX/DLC 时动态计算出来的值。

### 3.2 `change_hardcoding.execute()` 实际写出什么

按当前代码，它会：

1. 在输入目录的 `onnx/` 下递归查找 `.onnx`；
2. 修改后按原相对目录结构保存到目标 AR 目录；
3. 在 `test_vectors/` 下递归查找 `.pkl` 并调整 Tensor Shape；
4. 复制找到的 `.encodings`、`.json`、`.yaml`。

它不会在本阶段生成：

- Split ONNX；
- `.raw` 输入；
- `input_list`；
- SHA ONNX；
- DLC；
- Quantized DLC；
- Context Binary。

---

## 四、主入口 `gen_ar(arn)`

### 4.1 入口代码

主入口位于 [`qnn_compile_deploy.py`](../../../example2/host_linux/qnn_compile_deploy.py)：

```python
def gen_ar(arn):
    change_hardcoding.execute(
        LLAMA_MODELS,
        f"{workfolder}/assets/models_ar_n/ar{arn}-cl{CL}",
        [
            f" {EXPORT_AR},{arn}",
            f" -{EXPORT_AR},-1",
            f" {EXPORT_CONTEXT_LENGTH},{CL}",
            f" {EXPORT_CONTEXT_LENGTH-EXPORT_AR},{CL-arn}",
        ],
    )
```

三个参数依次是：

```text
input_path  = Example1 输出根目录
output_path = 当前 AR 的目标导出目录
fix_list    = “旧整数,新整数”替换列表
```

### 4.2 AR1 的替换表

当 `arn=1`：

```text
1073  → 1
-1073 → -1
2048  → 2048
975   → 2047
```

其中 `2048→2048` 的新旧值相同，`execute()` 会跳过，不会写入最终 `fix` 字典。

有效映射为：

```python
{
    1073: 1,
    -1073: -1,
    975: 2047,
}
```

### 4.3 AR128 的替换表

当 `arn=128`：

```text
1073  → 128
-1073 → -1
2048  → 2048
975   → 1920
```

有效映射为：

```python
{
    1073: 128,
    -1073: -1,
    975: 1920,
}
```

### 4.4 `-1073 → -1` 应该怎样理解

不能只看这个数字就断言它一定是“只保留最后一个 logits”。

原因是 `change_hardcoding.py` 不理解某个整数的业务含义，只要 Tensor Attribute 中出现相同数值就替换。`-1073` 可能属于 Slice、Reshape 或其他 Shape/索引常量，必须查看实际 ONNX 中该 Constant 的消费者才能判断。

当前最稳妥的结论是：

> 按 AR 图接口与后续 MHA2SHA 的 `seq_len` 构图逻辑，AR128 的序列维预期为 128；`-1073→-1` 的具体语义仍需在真实 AR128 ONNX 中检查，不能脱离节点上下文解释。

### 4.5 AR1 和 AR128 如何被调用

```python
ARNs = [1, 128]

with ProcessPoolExecutor(max_workers=1) as executor:
    results = executor.map(gen_ar, ARNs)
```

因为当前 `go_parallel=False`，所以 `max_workers=1`，两个任务按一个 Worker 的容量执行，以降低同时处理两份 3B 模型图的内存峰值。

---

## 五、`change_hardcoding.execute()` 的完整数据流

入口位于 [`change_hardcoding.py`](../../../example2/G2G/change_hardcoding.py)：

```text
fix_list
  │
  ▼
构造 fix 字典
  │
  ├── 找到 input_path/onnx/**/*.onnx
  ├── 找到 input_path/test_vectors/**/*.pkl
  └── 找到 encodings/json/yaml
  │
  ▼
加载 ONNX（不加载 External Data）
  │
  ▼
apply_fix()
  │
  ├── graph.input/output/value_info Shape
  ├── 部分 Initializer
  └── Tensor 类型 Node Attribute
  │
  ▼
保存新 ONNX
  │
  ▼
递归调整 PKL 中 Tensor Shape
  │
  ▼
复制 encodings/json/yaml
```

### 5.1 构造替换字典

```python
fix = {}
for opt in fix_list:
    old, new = [int(i) for i in opt.split(',')]
    if old != new:
        fix[old] = new
```

这段代码说明：

- 输入字符串前的空格不会影响 `int()`；
- 新旧值相同的映射直接忽略；
- 后续逻辑只知道整数映射，不知道 AR、KV、Slice 等业务语义。

### 5.2 发现输入文件

```python
onnxfiles = files(input_path + '/onnx', '.onnx')
picklefiles = files(input_path + '/test_vectors', '.pkl')
```

`files()` 使用 `os.walk()` 递归扫描，因此并不只处理 `qwen25llm.onnx` 和 `qt_0.pkl`；目录里其他同后缀文件也可能被处理。

### 5.3 保存时保持相对目录结构

`remap_path()` 把输入文件相对 `input_path` 的路径映射到 `output_path`：

```text
输入：output/onnx/qwen25llm.onnx
输出：assets/models_ar_n/ar1-cl2048/onnx/qwen25llm.onnx
```

AR128 同理。

---

## 六、ONNX 内部具体修改什么

### 6.1 第一类：输入、输出和 `value_info` Shape

```python
for vip in graph.input + graph.output + graph.value_info:
    for dim in vip.type.tensor_type.shape.dim:
        if dim.dim_value in fix:
            dim.dim_value = fix[dim.dim_value]
```

它会修改：

- 模型输入接口 Shape；
- 模型输出接口 Shape；
- ONNX 已记录的中间 Tensor Shape。

这部分改的是 ONNX Shape 元数据，但仅改这一层还不够，因为图内可能存在硬编码 Reshape/Slice 常量。

### 6.2 第二类：Tensor 类型的 Node Attribute

代码只处理：

```python
if attr.type == 4:
    fix_tensor_proto_in_attribute(attr.t)
```

ONNX 中 `4` 对应 Tensor 类型 Attribute。处理函数把 Attribute Tensor 中与 `fix` Key 相等的元素替换成新值。

精确边界是：

- 会处理 Tensor Attribute；
- 不代表所有标量 `INT` 或列表 `INTS` Attribute 都会被修改；
- 是否覆盖了目标图中的所有 Shape 常量，必须靠生成后检查确认。

### 6.3 第三类：部分 Initializer

代码不会普遍改写所有 Initializer，更不会重塑训练权重。它先检查 Initializer 的维度是否包含目标整数，然后只对“全部元素都等于 1”的 Tensor 重建 Shape：

```python
if (tensor == 1).all():
    arr = np.ones(new_shape)
```

这样做主要针对与固定 Shape 有关的全 1 辅助 Tensor，同时避免把 3B 模型的大权重全部加载进内存。

### 6.4 为什么这一步不改变模型知识

模型知识主要保存在 Linear/Conv/Embedding 等训练 Weight 中。本函数没有对一般训练 Weight 做重新量化、重新训练或数值优化。

所以应区分：

```text
改变计算图的固定尺寸和辅助 Shape 常量
≠
改变模型学习到的参数内容
```

---

## 七、Test Vector 如何同步修改

### 7.1 为什么 ONNX 改了，PKL 也必须改

原始 `qt_0.pkl` 中的 Tensor 是为 AR1073/Past-KV975 准备的。如果模型接口已经变成 AR1，但测试输入仍是 `[1,1073,2048]`，下一阶段无法把它作为匹配输入使用。

因此代码递归遍历：

- NumPy Array；
- PyTorch Tensor；
- List；
- Tuple；
- Dict。

只要某个 Tensor 的任一维度等于 `fix` 中的旧值，就计算新 Shape。

### 7.2 PyTorch Tensor 的处理方式

```python
nptensor = tensor.cpu().numpy().copy()
nptensor.resize(new_shape)
tensor = torch.tensor(nptensor)
```

这不是重新运行模型生成新的代表性输入，而是机械地缩小或扩展已有数组：

- 缩小时会截断数据；
- 扩大时 NumPy `resize` 会用 0 填充新增部分；
- 因此它的主要目标是满足新图的固定 Shape 契约。

不能把它理解成重新完成一次 Calibration。

### 7.3 当前 NumPy 分支的代码审阅点

当前代码：

```python
if isinstance(tensor, np.ndarray):
    tensor = tensor.resize(new_shape)
```

但 `np.ndarray.resize()` 是原地修改并返回 `None`。因此，如果 `qt_0.pkl` 中目标对象本身是 NumPy Array，这个分支可能把返回对象变成 `None`。

这属于需要实际产物验证的工程风险：

- 如果本次 PKL 主要保存 PyTorch Tensor，可能暂时没有触发；
- 如果包含直接的 NumPy Array，就应修正实现并重新生成测试向量。

---

## 八、AR1 / AR128 的预期输入 Shape

本项目原始 ONNX 使用：

```text
36 层
hidden_size = 2048
KV heads = 2
head_dim = 128
vocab_size = 151936
transposed_key_cache = true
return_new_key_value_only = true
```

### 8.1 三套图的输入对比

| 输入 | 原始 AR1073 | AR1 | AR128 |
|---|---|---|---|
| `inputs_embeds` | `[1,1073,2048]` | `[1,1,2048]` | `[1,128,2048]` |
| `attention_mask` | `[1,1,1073,2048]` | `[1,1,1,2048]` | `[1,1,128,2048]` |
| `position_ids_cos` | `[1,1,1073,64]` | `[1,1,1,64]` | `[1,1,128,64]` |
| `position_ids_sin` | `[1,1,1073,64]` | `[1,1,1,64]` | `[1,1,128,64]` |
| 每层 Past Key | `[1,2,128,975]` | `[1,2,128,2047]` | `[1,2,128,1920]` |
| 每层 Past Value | `[1,2,975,128]` | `[1,2,2047,128]` | `[1,2,1920,128]` |

这里两个 `2048` 容易混淆：

- `inputs_embeds` 最后一维的 2048 是模型 Hidden Size；
- `attention_mask` 最后一维的 2048 是 Context Length。

数字相同，但语义完全不同。

### 8.2 为什么 Past Key 和 Past Value Shape 不一样

配置开启：

```yaml
transposed_key_cache: true
```

因此：

```text
Past Value = [batch, kv_heads, past_length, head_dim]
Past Key   = [batch, kv_heads, head_dim, past_length]
```

Key 预先转置后，可以在 Attention 中更直接地参与 `Q × K`。

### 8.3 输入数量没有改变

三套图都是：

```text
1 个 inputs_embeds
+ 1 个 attention_mask
+ 2 个 position_ids_cos/sin
+ 36 × 2 个 Past Key/Value
= 76 个输入端口
```

改变的是这些端口的 Shape，不是输入端口的业务种类。

---

## 九、AR1 / AR128 的预期输出 Shape

配置 `return_new_key_value_only=true`，所以模型输出的是本轮新产生的 K/V，而不是把完整 Past-KV 全部再输出一遍。

### 9.1 输出种类

```text
1 个 logits
+ 36 × 2 个 New Key/Value
= 73 个输出端口
```

### 9.2 三套图的输出对比

| 输出 | 原始 AR1073 | AR1 预期 | AR128 预期 |
|---|---|---|---|
| `logits` | `[1,1073,151936]` | `[1,1,151936]` | `[1,128,151936]` |
| 每层 New Key | `[1,2,128,1073]` | `[1,2,128,1]` | `[1,2,128,128]` |
| 每层 New Value | `[1,2,1073,128]` | `[1,2,1,128]` | `[1,2,128,128]` |

AR128 的 New Key 和 New Value 恰好都显示为 `[1,2,128,128]`，但两个 128 的维度语义不同：一个是 `head_dim`，一个是本轮 `AR/current_length`。

### 9.3 关于 logits Shape 的验证边界

上表是根据原始接口、`1073→AR` 替换和 MHA2SHA 的 `seq_len` 构图意图得到的预期结果。

由于当前仓库没有生成后的 AR1/AR128 大模型 ONNX，且代码存在无语义的 `-1073→-1` 常量替换，所以最终必须检查真实 ONNX：

1. `graph.output` 中 `logits` 的 Shape；
2. 含 `-1` Constant 的消费者；
3. 是否存在通往 logits 的 `Slice(starts=-1)`；
4. MHA2SHA 后 `logits` Reshape 是否与 `seq_len` 一致。

因此不能只凭 `-1073→-1` 就断言 AR128 只输出最后一个位置的 logits。

---

## 十、Shape 对 Attention 计算的直接影响

输入 Query 的序列维就是 AR。简化表示：

```text
Q Shape            ≈ [1, 16, AR, 128]
完整 K Shape        ≈ [1, 16, 128, 2048]
Attention Score    ≈ [1, 16, AR, 2048]
```

所以：

```text
AR1 Attention Score   ≈ [1,16,1,2048]
AR128 Attention Score ≈ [1,16,128,2048]
```

这解释了为什么 Shape 虽然不改变模型知识，却会明显影响：

- 一次计算量；
- 中间 Activation 大小；
- 内存峰值；
- HTP Kernel 的切块与调度；
- Prefill 吞吐与 Decode 单步延迟。

---

## 十一、具体项目例子：Prompt 到逐 Token 生成

假设用户输入的 Prompt 经过 Tokenizer 后有 300 个有效 Token。

### 11.1 Prefill 阶段

运行时可以按 AR128 分块：

```text
第1块：128个有效Token
第2块：128个有效Token
第3块：44个有效Token + 84个Padding槽位
```

AR128 图的物理输入始终是固定 128 槽位。运行时通过 Attention Mask 区分有效 Token 和 Padding，并维护固定容量的 Past-KV Buffer。

每次 AR128 前向会生成这一块对应的新 K/V；运行时把有效部分写入会话 KV Cache。

### 11.2 Decode 阶段

Prompt 处理完后切换到 AR1：

```text
当前1个Token + Past-KV2047槽位
        │
        ▼ AR1 DLC
1份logits + 每层1个新Key/Value
        │
        ├── logits交给采样器选下一个Token
        └── 新KV写回会话Cache
```

然后把刚生成的 Token 再送入 AR1，循环执行。

注意：`example2` 当前负责生成 DLC，不负责实现这套端侧调度、采样与 KV Cache 管理；这些属于后续运行时阶段。

---

## 十二、Shape 最终影响到哪里

AR Shape 会沿完整流水线传播：

```text
AR1 / AR128 ONNX 导出包
          │
          ▼ Split ONNX
对应 Shape 的子图、RAW、input_list
          │
          ▼ MHA2SHA
对应 seq_len 的 SHA ONNX 和 Encoding
          │
          ▼ qairt-converter
固定接口的 AR1 / AR128 普通 DLC
          │
          ▼ qairt-quantizer
固定接口的 AR1 / AR128 Quantized DLC
```

当前最终文件预期为：

```text
assets/artifacts/ar1-cl2048/1_of_1/compiled_model/
└── ar1-cl2048_1_of_1_quantized.dlc

assets/artifacts/ar128-cl2048/1_of_1/compiled_model/
└── ar128-cl2048_1_of_1_quantized.dlc
```

### 12.1 对 Split 阶段的影响

Split 阶段必须分别为 AR1 和 AR128 生成匹配 Shape 的：

- Split ONNX；
- Input RAW；
- `input_list`；
- Golden Output。

AR128 的 RAW 不能直接喂给 AR1 图。

### 12.2 对 MHA2SHA 的影响

MHA2SHA 从第一个模型输入读取 `seq_len`。所以 AR1 和 AR128 会分别以 1 和 128 作为 Attention 图改写的 Current Length。

### 12.3 对 DLC 的影响

QNN Converter 会把输入输出 Shape、中间算子 Shape 和相关量化规则写入两份不同的 DLC 图。

因此，一份 AR1 DLC 不能在运行时突然接收 `[1,128,2048]` 的 `inputs_embeds`；静态接口不匹配。

### 12.4 对权重共享的边界

AR1 与 AR128 来源于同一套模型权重语义，但当前得到的是两份独立 DLC 文件。不能仅凭“权重数值相同”就断言两个 DLC 已经在物理存储上共享一份 Weight。

真正的 HTP Weight Sharing/Context Binary 仍属于后续阶段，而且当前主脚本中的相关代码尚未启用。

---

## 十三、哪些改变，哪些不改变

| 项目 | 是否改变 | 说明 |
|---|---|---|
| 模型层数 | 否 | 仍为同一套 36 层模型 |
| 训练权重语义 | 否 | 不重新训练、微调或优化 Weight |
| Context Length | 否 | 仍为 2048 |
| 当前 Token 长度 | 是 | 1073 改为 1 或 128 |
| Past-KV 长度 | 是 | 975 改为 2047 或 1920 |
| 输入输出 Shape | 是 | 所有相关 Tensor 必须协调变化 |
| 中间 Activation Shape | 是 | Attention、Reshape 等随 AR 变化 |
| Test Vector Shape | 是 | PKL 中对应 Tensor 被缩放 |
| Encoding 数值 | 本阶段不重算 | 文件主要直接复制给下游 |
| 最终 DLC 类型 | 间接改变 | 后续生成 AR1 与 AR128 两份 DLC |

一句话：

> **模型“会什么”没有变，模型“一次怎样装数据、算多少位置、吐多少新 KV”变了。**

---

## 十四、必须关注的工程风险

### 14.1 按整数全局匹配，不理解节点语义

`change_hardcoding.py` 看到 1073、975 等整数就按映射替换。它无法区分这个数字究竟表示：

- Sequence Length；
- Reshape 维度；
- Slice 起点；
- 某个无关的常量。

因此必须检查变更日志和最终 ONNX，确认没有误改同值但不同语义的常量。

### 14.2 外置权重没有被显式复制

ONNX 通过：

```python
onnx.load(onnxfile, load_external_data=False)
```

加载，复制列表只明确包含：

```text
.encodings
.json
.yaml
```

没有明确包含 `.data`、`.bin`、`.safetensors` 等外置权重文件。因此生成后必须验证新 ONNX 的 External Data 引用是否仍然可解析。

### 14.3 Test Vector 只是机械 Resize

PKL Tensor 并不是重新经过 Tokenizer、Mask、RoPE 和模型前向生成。缩放后的数值是否仍适合数值对拍，需要后续单独验证。

### 14.4 NumPy `resize()` 返回值问题

直接 NumPy Array 分支可能返回 `None`，必须检查生成后的 PKL 内是否出现异常对象。

### 14.5 日志 `done` 不等于成功

`gen_ar()` 捕获异常后调用 `exit(0)`；同时 `executor.map()` 的结果没有被显式遍历消费。异常传播和最终状态并不可靠。

所以必须以以下证据为准：

- 目标文件确实存在；
- 文件大小合理且非空；
- ONNX 可以加载并通过检查；
- External Data 可解析；
- 输入输出 Shape 与 AR 目标一致；
- Test Vector 类型和 Shape 正确。

---

## 十五、如何验收 AR1 / AR128 产物

### 15.1 第一层：目录和文件存在

在 `example2/host_linux` 下检查：

```bash
find assets/models_ar_n/ar1-cl2048 -maxdepth 3 -type f -ls
find assets/models_ar_n/ar128-cl2048 -maxdepth 3 -type f -ls
```

重点确认：

- AR1/AR128 ONNX 都存在且非空；
- Encoding 文件都存在；
- Test Vector 都存在；
- ONNX 所需 External Data 可以找到。

### 15.2 第二层：打印 ONNX 输入输出 Shape

可在 Linux QAIRT 环境中执行：

```python
from pathlib import Path
import onnx


def tensor_shape(value_info):
    dims = value_info.type.tensor_type.shape.dim
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in dims
    ]


root = Path("assets/models_ar_n")

for ar in (1, 128):
    model_path = root / f"ar{ar}-cl2048/onnx/qwen25llm.onnx"
    model = onnx.load(model_path, load_external_data=False)

    print(f"\n===== AR{ar} =====")
    for value_info in model.graph.input:
        if value_info.name in {
            "inputs_embeds",
            "attention_mask",
            "position_ids_cos",
            "position_ids_sin",
            "past_key_0_in",
            "past_value_0_in",
        }:
            print("INPUT ", value_info.name, tensor_shape(value_info))

    for value_info in model.graph.output:
        if value_info.name in {"logits", "past_key_0_out", "past_value_0_out"}:
            print("OUTPUT", value_info.name, tensor_shape(value_info))
```

### 15.3 第三层：检查 External Data

对于可能超过 2 GiB 的大模型，应优先把文件路径传给 Checker，避免把完整 ModelProto 作为内存对象再次序列化：

```python
from pathlib import Path
import onnx

model_path = Path(
    "assets/models_ar_n/ar1-cl2048/onnx/qwen25llm.onnx"
)

onnx.checker.check_model(str(model_path))
print("AR1 ONNX structure/external references: OK")
```

如需确认权重数据能够真正读入，并且主机内存充足，再执行：

```python
model = onnx.load(str(model_path), load_external_data=True)
print("AR1 external data loaded:", len(model.graph.initializer))
```

AR128 也要独立检查，不能只验证其中一张图。

### 15.4 第四层：检查 Test Vector

至少确认：

- 所有目标输入仍然是 Tensor/Array，不是 `None`；
- AR1 的 Current Length 为 1；
- AR128 的 Current Length 为 128；
- Past-KV 分别为 2047 和 1920；
- Tensor dtype 没有异常变化；
- 输入名称与 ONNX 输入端口一致。

### 15.5 第五层：数值验证

结构检查只能证明 Shape 基本正确，不能证明图的数值语义没有被盲替换破坏。

更严格的验证应比较：

- 相同有效 Token 和有效 Past-KV 条件下的 logits；
- 每层 New Key/New Value；
- 原图切片参考结果与 AR 图结果的最大误差、平均误差和余弦相似度。

---

## 十六、常见错误理解

### 16.1 “AR1 和 AR128 是两套重新训练的模型”

不对。它们来源于同一套权重语义，只是固定 Shape 和执行宽度不同。

### 16.2 “AR128 表示模型最多只能理解 128 个 Token”

不对。AR128 表示本次前向的 Current Token 槽位是 128；完整 Context Length 仍是 2048，历史内容由 Past-KV 提供。

### 16.3 “只改 `inputs_embeds` Shape 就够了”

不对。Past-KV、Mask、RoPE、输出新 KV 和图内 Shape 常量都必须协调一致。

### 16.4 “AR 适配会重新计算量化 scale/offset”

不对。本阶段复制已有 Encoding，真正的 QNN Converter/Quantizer 还在后面。

### 16.5 “输出目录里有 ONNX，就说明外置权重也完整”

不一定。必须使用 `load_external_data=True` 实际加载检查。

### 16.6 “AR128 一定输出 128 个可直接采样的 Token”

不准确。模型输出的是每个位置的 logits，不是 128 个最终 Token ID；采样由运行时完成。并且实际 logits Shape 仍需检查生成后的 ONNX。

### 16.7 “脚本打印 `Prepare AR128 AR1 export done` 就成功了”

不对。必须检查文件、Shape、External Data、PKL 类型和数值结果。

---

## 十七、用一句项目回答串起来

如果面试或复盘时需要用 30 秒说明，可以这样回答：

> Example1 导出的 Qwen2.5-VL-3B LLM 图固定为 AR1073、Past-KV975、Context Length2048。Example2 的第一阶段通过 `gen_ar()` 调用 `change_hardcoding.execute()`，把与 Current Length 和 Past-KV Length 有关的 ONNX 输入输出 Shape、ValueInfo、部分 Tensor Constant、辅助 Initializer 以及 Test Vector 同步改成 AR1/Past2047 和 AR128/Past1920。训练权重和量化 Encoding 不在这里重新计算。两套图随后分别进入 Split、MHA2SHA、QNN Converter 和 Quantizer，最终形成 Decode 用的 AR1 Quantized DLC 与 Prefill 用的 AR128 Quantized DLC。

---

## 十八、自测题

先不看答案，尝试回答：

1. AR 在本项目中可以先怎样理解？
2. 为什么 AR1 的 Past-KV Length 是 2047？
3. AR128 的 `attention_mask` 预期是什么 Shape？
4. AR1 每层 New Key 和 New Value 的 Shape 分别是什么？
5. `change_hardcoding.py` 会不会修改所有 Initializer？
6. 为什么 `.encodings` 在这一阶段不重新计算？
7. 为什么 `-1073→-1` 不能脱离真实 ONNX 节点解释？
8. Test Vector 的 Resize 是否等于重新 Calibration？
9. 为什么生成后的 ONNX 必须检查 External Data？
10. AR Shape 最终在哪一步固化成两份不同的 DLC？

### 18.1 参考答案

1. 一张固定图一次并行处理的新 Token 数。
2. 因为 `2048-1=2047`，Past-KV 加 Current 必须保持总 Context 2048。
3. `[1,1,128,2048]`。
4. New Key `[1,2,128,1]`，New Value `[1,2,1,128]`。
5. 不会，只重建 Shape 命中且元素全为 1 的 Initializer。
6. Example1 已确定 Encoding，本阶段只做 Shape 适配和文件复制。
7. 它是按整数匹配的无语义替换，必须确定该常量属于哪个节点及输入。
8. 不是，只是机械调整已有 Tensor Shape。
9. 当前代码没有明确复制所有外置权重后缀，ONNX 引用可能失效。
10. Shape 从 AR ONNX 一路传递，在 `qairt-converter` 生成的 DLC 图中形成固定接口，随后 Quantizer 生成对应 Quantized DLC。

---

## 十九、相关源码与笔记

- Example2 总入口：[`qnn_compile_deploy.py`](../../../example2/host_linux/qnn_compile_deploy.py)
- AR Shape 修改实现：[`change_hardcoding.py`](../../../example2/G2G/change_hardcoding.py)
- Example1 量化与模型覆盖配置：[`example1/config.yaml`](../../../example1/config.yaml)
- Example1 ONNX 输入输出命名：[`example1/llm_quant.py`](../../../example1/llm_quant.py)
- KV Shape 与输出整理：[`forward_pass_wrapper.py`](../../../example1/llm_utils/forward_pass_wrapper.py)
- MHA2SHA `seq_len` 处理：[`optimizer.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/optimizer.py)
- 上游 ONNX 与测试向量：[08 · ONNX 导出与测试向量](../08-ONNX导出与测试向量.md)
- Example2 总览：[00 · Example2 主机编译全景](./00-example2主机编译全景.md)
- 下一阶段：[02 · Split ONNX 与测试向量](./02-Split-ONNX与测试向量.md)

---

## 二十、本篇总结

> **AR 图适配的直接产物是 AR1/AR128 两套固定 Shape 导出包。`change_hardcoding.execute()` 主要改造两类对象：第一类是 ONNX 计算图，包括输入、输出、`value_info` Shape，以及部分节点 Tensor 常量和全 1 辅助 Initializer；第二类是 Test Vector，递归调整 `.pkl` 中 Tensor 的 Current Length 与 Past-KV Length。`.encodings`、JSON 和 YAML 只原样复制，不重新训练模型、不修改普通权重，也不重新执行量化 Calibration。在保持 Context Length2048 不变的前提下，Current/Past-KV 从 1073/975 分别变为 1/2047 与 128/1920，并在后续流程中固化进 Decode 使用的 AR1 Quantized DLC 和分块 Prefill 使用的 AR128 Quantized DLC。**
