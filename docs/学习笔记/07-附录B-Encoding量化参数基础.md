# 07-附录B · Encoding 量化参数基础

> **所属主篇**：[07 · 量化主流程](./07-量化主流程-QuantSim到Encoding.md)
>
> **一句话本质**：Encoding 是浮点数和低比特整数之间互相转换时使用的“量化尺子”。

---

## 一、Encoding 是什么

浮点 Tensor 本身是一组数值，Encoding 负责说明这些数值怎样映射成整数：

```text
浮点 Tensor
    │ 按 Encoding 量化
    ▼
低比特整数 Tensor
    │ 按同一 Encoding 反量化
    ▼
近似的浮点 Tensor
```

Encoding 不是权重，也不是激活数据，而是描述量化转换规则的元数据。

---

## 二、Encoding 主要包含什么

| 字段 | 含义 |
|---|---|
| `name` | 这套 Encoding 属于哪个 Tensor |
| `dtype` / `output_dtype` | 量化后的整数类型，例如 `int8`、`uint8` |
| `bitwidth` | 使用多少位表示，例如 4bit、8bit、16bit |
| `scale` / `delta` | 一个整数刻度代表多大的浮点范围 |
| `zero-point` / `offset` | 浮点 0 对应的整数位置；不同格式的字段约定可能不同 |
| `min` / `max` | 这把量化尺子覆盖的浮点范围 |
| `symmetric` | 是否使用以 0 为中心的对称量化范围 |

可以简化理解为：

```text
Encoding = Tensor 名字 + 整数类型/位宽 + 表示范围 + 刻度大小 + 零点位置
```

---

## 三、一个具体 Encoding 例子

这里使用项目 Prepare 产物中的真实 Tensor：

```text
past_value_0_in = 第 0 层的历史 Value Cache 输入 Tensor
```

这个 Tensor 名字来自实际模型；当前工作区还没有最终导出的 Encoding 文件，因此下面的 `scale/min/max` 使用便于计算的示例数值。位宽和对称方式则按照项目的 KV 8bit symmetric 规则设置。

```text
name         = "past_value_0_in"
dtype        = int8
bitwidth     = 8
scale        = 0.01
zero-point   = 0          # 新版 int8 表示
offset       = -128       # AIMET 旧格式常见表示
min          = -1.28
max          = 1.27
symmetric    = True
```

### 3.1 每个字段一一对应怎么理解

| 字段和值 | 具体解释 |
|---|---|
| `name="past_value_0_in"` | 这把尺子属于第 0 层历史 Value Cache 的输入 Tensor |
| `dtype=int8` | 量化后使用有符号整数，整数范围是 `-128～127` |
| `bitwidth=8` | 8bit 一共可以表示 `2⁸=256` 个整数刻度 |
| `scale=0.01` | 整数每变化 1，代表的浮点值变化 `0.01` |
| `zero-point=0` | 新版有符号 int8 中，整数 `0` 代表浮点数 `0` |
| `offset=-128` | AIMET 旧格式常用 `0～255` 编码位置描述同一把有符号尺子，位置 `128` 对应浮点 0 |
| `min=-1.28` | 最小 int8 值 `-128` 对应 `-128×0.01=-1.28` |
| `max=1.27` | 最大 int8 值 `127` 对应 `127×0.01=1.27` |
| `symmetric=True` | 正负范围围绕 0，符合项目对 KV 使用 symmetric 量化的规则 |

整把量化尺子可以画成：

```text
整数 q：   -128 ───────── 0 ───────── 53 ───────── 127
浮点 x：  -1.28 ───────── 0 ───────── 0.53 ───────── 1.27
```

### 3.2 用这套 Encoding 完成一次 QDQ

使用 `zero-point` 的通用公式：

```text
量化 Q：   q     = clip(round(x / scale) + zero-point, -128, 127)
反量化 DQ：x_hat = (q - zero-point) × scale
```

量化公式可以按照从里到外的顺序理解：

#### 第一步：`x / scale`——计算位于第几个格子

`scale` 是量化尺子的“每格长度”。当前例子中 `scale=0.01`，表示整数变化 1，对应浮点值变化 `0.01`。

```text
x / scale = 0.534 / 0.01 = 53.4
```

这表示浮点数 `0.534` 位于第 `53.4` 个量化格子附近。之所以使用除法，是因为这里计算的是：

```text
格子数量 = 浮点数值 ÷ 每格长度
```

#### 第二步：`round()`——落到最近的整数格

整数 Tensor 不能保存 `53.4`，所以需要选择距离它最近的整数刻度。

```text
round(53.4)  = 53
round(53.7)  = 54
round(-3.2)  = -3
round(-3.8)  = -4
```

`round` 是取最近整数，不是始终向上取整；始终向上取整对应的是 `ceil`。数值恰好位于 `.5` 时，具体舍入规则由实现决定，例如 `torch.round()` 使用“取最近的偶数”。

#### 第三步：加 `zero-point`——移动整数尺子的零刻度

`zero-point` 表示浮点数 `0` 对应整数尺子上的哪个位置：

```text
浮点 x = 0  →  整数 q = zero-point
```

当前对称 int8 例子使用 `zero-point=0`，所以浮点 `0` 对应整数 `0`。非对称量化中零点可以移动到其他整数位置，从而让有限的整数刻度更贴合偏向一侧的浮点分布。

例如，先看当前 `zero-point=0` 的情况：

```text
x = 0
q = round(0 / 0.01) + 0 = 0

所以：浮点 0  ↔  整数 0
```

再假设另一把量化尺子使用 `scale=0.01、zero-point=-128`：

```text
x = 0
q = round(0 / 0.01) - 128 = -128

所以：浮点 0  ↔  整数 -128
```

同一个浮点数 `x=0.534` 在这把新尺子上会变成：

```text
q = round(0.534 / 0.01) - 128
  = 53 - 128
  = -75

x_hat = (-75 - (-128)) × 0.01
      = 0.53
```

可以看到，`zero-point` 改变以后，保存下来的整数从 `53` 变成了 `-75`，但反量化后仍然约等于原来的浮点数。它改变的是**浮点零在整数尺子上的位置**，而不是每格的大小。

> 注意：这里的 `zero-point=-128` 是为了演示通用公式中的零点移动，不是前面 AIMET 旧格式的 `offset=-128`；两种字段的约定不能直接混用。

#### 第四步：`clip()`——限制在 int8 能表示的范围内

有符号 int8 只能保存 `-128～127`，所以最终结果必须截断到这个范围：

```text
clip(53,   -128, 127) = 53
clip(150,  -128, 127) = 127
clip(-200, -128, 127) = -128
```

超出量化范围的值会被压到最小值或最大值，这种现象称为**截断**或**饱和**。

#### 完整计算例子

现在把浮点数 `x=0.534` 量化：

```text
x / scale                = 0.534 / 0.01 = 53.4
round(x / scale)         = 53
round(x / scale) + 0     = 53
clip(53, -128, 127)      = 53

所以 q = 53
```

反量化：

```text
x_hat = (53 - 0) × 0.01
      = 0.53
```

因此量化误差为：

```text
0.53 - 0.534 = -0.004
```

这个例子同时说明：

- `scale` 决定尺子的刻度间隔；
- `zero-point` 决定浮点 0 在整数尺子上的位置；
- `min/max` 决定尺子能覆盖的范围；
- 超出 `[-1.28, 1.27]` 的数会被截断到边界；
- 范围内但落不到刻度上的数会发生取整误差。

可以用一句口诀记忆：

> **除以 scale 数格子，round 落到整数格，zero-point 移动零刻度，clip 防止超出尺子。**

### 3.3 AIMET 文件中字段名为什么可能不同

AIMET Encoding 文件有不同版本，但表达的是同一套量化规则：

新版 2.0.0 的核心写法：

```json
{
  "name": "past_value_0_in",
  "output_dtype": "int8",
  "y_scale": 0.01,
  "y_zero_point": 0
}
```

旧格式可能写成 `bitwidth、scale、offset、min、max、is_symmetric`。因此看文件时要先确认 Encoding 版本，不要把 `zero-point` 和旧格式 `offset` 的符号直接混用。

参考：[AIMET Encoding Format Specification](https://quic.github.io/aimet-pages/releases/latest/techniques/encoding_spec.html)。

---

## 四、对称量化和非对称量化

```text
对称量化：范围以 0 为中心，常用于权重和 KV
非对称量化：零点可以移动，常用于分布偏向一侧的激活
```

是否使用对称量化，也是 Encoding 规则的一部分。

---

## 五、Encoding 在当前流程中从哪里来

| 对象 | 当前项目中的主要确定方式 |
|---|---|
| 权重 Encoding | 根据权重数值建立，再由 SeqMSE 搜索更合适的量化范围 |
| 激活 Encoding | [`compute_encodings()`](./07-附录E-compute_encodings激活标定.md) 运行真实校准数据后统计得到 |

```text
权重 → SeqMSE → 优化 weight Encoding
真实样本 → compute_encodings → activation Encoding
```

最终导出时，Encoding 会和 ONNX 一起交给 QNN，告诉后端每个量化 Tensor 应该使用哪一把量化尺子。

---

## 六、Tensor、Quantizer 和 Encoding 的区别

| 名称 | 简单理解 |
|---|---|
| Tensor | 计算图中的多维数组数据，分权重、激活、KV Cache 三类（见 6.1） |
| Quantizer | 执行或模拟量化的组件 |
| Encoding | Quantizer 使用的量化参数 |

```text
Tensor 数据
   │
   ▼
Quantizer 使用 Encoding 完成 QDQ
```

### 6.1 三类 Tensor：权重、激活、KV Cache

Tensor 只是统一的数据容器，**决定它走哪条量化路径的是它在计算图中的位置**：

```text
节点(算子) ── 边(激活 Tensor) ──→ 节点(算子)
   │
   └── 挂在节点上的常量：权重 Tensor
```

| | 权重 (Weight) | 激活 (Activation) | KV Cache |
|---|---|---|---|
| 图中位置 | 挂在算子节点**上**的参数 | 算子节点**之间**的数据边 | 被提升为图 I/O 的激活 |
| 来源 | 训练得到，从 `.safetensors` 加载 | `forward` 运行时产生 | `forward` 运行时产生 |
| 生命周期 | 推理全程只读、不变 | 单次 forward 内，用完即释放 | 跨多次 forward 存活 |
| 与输入的关系 | 与输入无关，所有请求共享 | 与本次输入绑定 | 与本次会话绑定，不可跨请求共享 |
| 形状 | 固定，如 `[out, in]` | 随输入变化 | 随 seq 增长 |
| **落到哪个 encoding** | `param_encodings` | `activation_encodings` | `activation_encodings` |
| 量化时机 | 离线静态，常用 per-channel | 靠校准数据统计范围 | 靠校准数据统计范围 |

三点补充：

1. **权重不参与连线**。这是它和激活最本质的区别，也是 `_io_map.json` 把参数映射与激活 I/O 映射分成两部分的原因（见 [06-附录C](./06-附录C-QAIRT-model_preparer内部流程.md) 中「为什么叫激活 Tensor 连线映射」）。
2. **KV Cache 是激活的特例**，特殊在生命周期。普通激活是纯粹的内部中间边，而 KV 要跨 decode step 复用，因此在导出的定长图里必须显式变成图的输入和输出端口（`past_kv` 进、`present_kv` 出），否则外部无法传入传出。量化归类上它仍是激活。
3. **K、V 是权重算出来的，但不是权重**。`W_k`（权重，固定）× `hidden_state`（激活，变化）→ `K`（KV Cache 内容，变化）。前者属于模型，后者属于这次会话。

> 一句话：**权重挂在节点上，激活是节点之间的边，KV Cache 是被外置成图 I/O 的长生命周期激活；前者进 `param_encodings`，后两者进 `activation_encodings`。**

### 6.2 项目实例：同一层 Attention 中的 W4、A16 与 KV8

这里使用 Prepare 后模型的第 0 层 Attention。项目配置明确指定：

```text
default_param_bw  = 4     → 默认权重 W4
default_output_bw = 16    → 默认激活 A16
MatMul 第二输入   = 8bit symmetric → Attention 中的 K/V 使用 KV8
```

配置来源：[config.yaml](../../example1/config.yaml)；KV8 规则来源：[llm_quant.py](../../example1/llm_quant.py) 中的 `set_matmul_second_input_producer_to_8bit_symmetric(quantsim)`。

把第 0 层的真实结构串起来：

```text
普通输入激活 A16
       │
       ├─ q_proj_conv（权重 W4）→ Q16
       │
       ├─ k_proj_conv（权重 W4）→ 新 K8 ─┐
历史 past_key_0_in（K8）─────────────────┴─ Concat → K8
       │                                      ↓
       │                                  Q16 × K8
       │
       └─ v_proj_conv（权重 W4）→ 新 V8 ─┐
历史 past_value_0_in（V8）───────────────┴─ Concat → V8
                                              ↓
                                      Attention概率16 × V8
```

真实模块和 Tensor 可以在 Prepare 产物中找到：

- `model_layers_0_self_attn_q_proj_conv.weight`：第 0 层 Q 投影权重；
- `past_key_0_in`、`past_value_0_in`：外部传入的历史 KV Cache；
- `past_key_0_out`、`past_value_0_out`：本次前向新生成的 KV；
- `model_layers_0_self_attn_Concat_9/Concat_10`：分别拼接历史 K/V 和新 K/V；
- `model_layers_0_self_attn_MatMul/MatMul_1`：分别执行 `Q×K` 和 `Attention概率×V`。

对应代码见 [Prepare 重建模型](../../output/prepare/qwen25llm_kvcache_36_layer.py)，新旧权重与 Tensor 名称关系见 [io_map.json](../../output/prepare/qwen25llm_kvcache_36_layer_io_map.json)。

> 下面的位宽来自项目真实配置；由于当前工作区还没有最终 `.encodings` 文件，数值、`scale` 和 `zero-point` 仅用于演算，不是本项目最终导出的真实参数。

#### 6.2.1 权重量化：`q_proj_conv.weight` 使用 W4

假设权重 Tensor 中的一个浮点值为 `w=0.034`，并假设：

```text
scale_w = 0.01
zero-point = 0
signed int4 范围 = [-8, 7]
```

量化和反量化：

```text
q_w   = clip(round(0.034 / 0.01), -8, 7)
      = 3

w_hat = 3 × 0.01
      = 0.03
```

因此：

```text
浮点权重 0.034 → 4bit 整数 3 → 近似权重 0.03
```

权重是模型训练后固定下来的参数；本项目大部分权重默认 W4，权重 Encoding 还会经过 SeqMSE 优化。

#### 6.2.2 普通激活量化：投影输入/输出默认 A16

假设一次前向中产生的普通激活值为 `a=0.5344`，并假设：

```text
scale_a = 0.001
zero-point = 0
```

```text
q_a   = round(0.5344 / 0.001)
      = 534

a_hat = 534 × 0.001
      = 0.534
```

因此：

```text
浮点激活 0.5344 → 16bit 整数 534 → 近似激活 0.534
```

普通激活不是模型中固定保存的参数，而是随每次 `forward` 动态产生；它的 Encoding 主要由 `compute_encodings()` 使用真实校准数据统计得到。

#### 6.2.3 KV Cache 量化：K/V 使用 8bit symmetric

假设一个 Key 元素为 `K=0.534`，一个 Value 元素为 `V=-0.276`，并为了演算假设两者使用：

```text
scale_kv = 0.01
zero-point = 0
signed int8 范围 = [-128, 127]
```

Key：

```text
q_k   = round(0.534 / 0.01) = 53
K_hat = 53 × 0.01 = 0.53
```

Value：

```text
q_v   = round(-0.276 / 0.01) = -28
V_hat = -28 × 0.01 = -0.28
```

历史 KV 与本次新 KV 在 Concat 处需要使用相同 Encoding，拼接后再作为两个 MatMul 的第二输入：

```text
Q16 × K8
Attention概率16 × V8
```

KV Cache 不是第三种独立数据类型，它本质上仍是激活；只是它需要跨 token 保存并反复读取，所以项目把它从默认 A16 单独调整成 8bit，以减少缓存大小和数据搬运成本。

#### 6.2.4 KV 为什么有两次拼接

本项目的 KV 处理包含两次目的不同的拼接：

```text
模型内拼接：为本次 Attention 准备完整 KV
FPM 外拼接：维护下一次前向需要的真实历史 KV Cache
```

先纠正一个容易产生的误解：传给模型的固定 Shape `past KV` 不只是为了凑 Shape，它同时包含真实历史缓存：

```text
固定 Shape 的 past KV = Padding 部分 + 真实历史 KV
```

第一次没有历史数据时，它才全部是 Padding。项目当前：

```text
context_length      = 2048
ARN                 = 1073
past KV 固定长度    = 2048 - 1073 = 975
```

##### 第一次拼接：模型内部，用于当前 Attention

FPM 先把有效历史 KV 补齐或裁剪成固定的 975，再传入模型。模型计算本轮新 KV，然后在每一层内部执行：

```text
固定 past KV：[Padding + 真实历史 KV]
当前 new KV： [Padding + 本轮真实 new KV]
                         ↓
模型内部 Concat 成固定长度的完整 KV
                         ↓
用于 Q×K 和 Attention概率×V
```

这个完整 KV 只服务于当前 Attention。因为项目开启了 `return_new_key_value_only=true`，模型最终不会返回拼接后的完整 KV，而只返回本轮计算出的 `past_key_i_out/past_value_i_out`。

对应实现可见 [Prepare 重建模型](../../output/prepare/qwen25llm_kvcache_36_layer.py) 中第 0 层的 `Concat_9/Concat_10`，以及 [config.yaml](../../example1/config.yaml) 的 `return_new_key_value_only`。

##### 第二次拼接：FPM 外部，用于更新缓存状态

模型返回的新 KV 可能还包含本轮输入 Padding 对应的无效位置。`LLMForwardPassManager.prepare_outputs()` 会：

```text
① 从 new KV 中只保留本轮真实 input_length 对应的数据
② 取出 past KV 中的有效历史数据
③ 有效历史 KV + 本轮有效 new KV
④ 超过窗口时从左侧移除最早的数据
⑤ 得到下一次前向使用的 past_key_values
```

对应实现见 [forward_pass_wrapper.py](../../example1/llm_utils/forward_pass_wrapper.py) 中的 `prepare_outputs()` 和 `_update_kv_cache()`。

用一个不考虑 Shape 细节的小例子表示：

```text
模型输入的 past K：[Pad, Pad, K1, K2]
模型返回的 new K： [Pad, K3, K4]

模型内部临时使用： [Pad, Pad, K1, K2, Pad, K3, K4]
FPM 整理后保存：   [K1, K2, K3, K4]
```

下一次前向之前，FPM 再根据模型要求补成固定 Shape：

```text
[Pad, Pad, ..., K1, K2, K3, K4]
```

因此，两次拼接不能混为一件事：

> **模型内拼接解决“本次 Attention 看哪些 KV”；FPM 外拼接解决“下一次前向保存哪些有效 KV”。**

三者可以这样区分：

| 对象 | 第 0 层实例 | 项目位宽 | 特点 |
|---|---|---:|---|
| 权重 | `q_proj_conv.weight` | 默认 4bit | 固定参数，所有请求共享 |
| 普通激活 | Q 投影输入/输出等中间 Tensor | 默认 16bit | 单次前向动态产生 |
| KV Cache | `past_key_0_in/out`、`past_value_0_in/out` | 8bit symmetric | 长生命周期激活，跨 token 复用 |

---

## 七、Concat 共享 Encoding 怎么理解

同一个 Concat 的多个输入和输出需要使用同一把量化尺子：

```text
输入1 ─┐
输入2 ─┼─ Concat → 输出
输入3 ─┘

输入1、输入2、输入3、输出共享一套 Encoding
```

不同 Concat 仍然可以拥有各自不同的 Encoding。

Encoding 最终如何与 ONNX、External Weight 一起导出并交给 QNN，见 [08 · ONNX 导出与测试向量](./08-ONNX导出与测试向量.md)。

---

## 八、一句话总结

> **Encoding 就是量化尺子：它规定一个浮点 Tensor 使用多少 bit、覆盖多大范围，以及如何在浮点值与整数值之间转换。**
