# 06 · 附录A · Prepare Dummy Input 输入模具

> **关联主篇**：[06 · Prepare 模型准备阶段](./06-Prepare模型准备阶段.md)。本附录专门解析 `llm_quant.py` 中 `get_dummy_data()`。
>
> **一句话本质**：`get_dummy_data()` 手工制造一套“输入数量、顺序、shape、dtype、RoPE、mask、KV 布局都正确”的固定样例，供 QAIRT Prepare、QuantSim 和 ONNX 导出建立静态计算图；它不是实际文本/图像数据，也不是量化校准数据。
>
> **和 FPM 的关系**：它与 [05 · `FPM.prepare_inputs()`](./05-通用前向处理流程.md) 异曲同工——二者都要生成模型可接收的定长输入；FPM 处理真实数据，dummy input 直接伪造最终形态。

---

## 一、为什么静态图需要 dummy input

PyTorch 可以等真实输入来了再动态执行，但模型 Prepare、QuantSim 插桩和 ONNX 导出需要先看到一套具体输入，才能确认：

- forward 走哪条分支；
- 一共有多少输入输出；
- 每个 Tensor 是几维；
- 每一维长度是多少；
- dtype/device 是什么；
- KV Cache 是嵌套 tuple 还是扁平 Tensor；
- external mask 与 RoPE 是什么接口。

因此 dummy input 的目标不是“内容真实”，而是：

```text
接口真实 + 形状真实 + 数据类型真实 + 数值能够跑通
```

三条路径可以对照理解：

```text
真实 PPL：
dataloader → FPM.prepare_inputs → model → prepare_outputs → loss

Prepare/建图：
get_dummy_data → model_preparer.prepare_model

量化/导出：
get_dummy_data（扁平版）→ QuantSim / ONNX exporter
```

---

## 二、函数入口与参数

```python
def get_dummy_data(
    config,
    tokenizer,
    device,
    separate_tuple_input_output,
    num_tokens=None,
    dtype=torch.float32,
):
```

| 参数 | 作用 |
|---|---|
| `config` | 提供层数、hidden、头数、KV 布局、RoPE/mask/input 模式 |
| `tokenizer` | 提供词表大小与 `model_max_length` |
| `device` | dummy Tensor 创建在 CPU 还是 CUDA |
| `separate_tuple_input_output` | 保留原生嵌套 KV，还是展平成独立 Tensor |
| `num_tokens` | 当前固定图一次处理的 token 数，即 ARN |
| `dtype` | mask、RoPE、KV 等浮点 Tensor 的类型 |

当前调用明确传入：

```text
num_tokens = ARN = 1073
```

虽然形参默认是 `None`，函数内部会执行：

```python
max_tokens - num_tokens
```

所以它实际上是必填参数，稳妥写法应在开头增加：

```python
assert num_tokens is not None
```

---

## 三、读取模型结构参数

```python
num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
num_attention_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads
rope_theta = config.rope_theta
```

当前 Qwen2.5-VL-3B：

```text
num_layers          = 36
hidden_size         = 2048
num_attention_heads = 16   # Q 头数
num_kv_heads        = 2    # K/V 头数
head_dim            = 2048 / 16 = 128
```

Qwen2 使用 GQA：

```text
16 个 Q 头 / 2 个 KV 头 = 每组 KV 服务 8 个 Q 头
```

因此 dummy KV Cache 只保存 2 个 KV 头，而不是 16 个 Q 头：

```text
K/V Cache 头维 = num_kv_heads = 2
每头宽度       = head_dim = 128
```

`rope_theta` 在本函数中没有继续使用；RoPE 辅助类会从 `config` 重新读取，故这个局部变量可以删除。

---

## 四、建立长度关系

```python
max_tokens = tokenizer.model_max_length
```

当前：

```text
max_tokens = context_length = 2048
num_tokens = ARN            = 1073
past_size                    = 2048 - 1073 = 975
```

固定图的序列轴关系是：

```text
Past KV 975 + Current 1073 = Attention Source 2048
```

三个长度对应不同对象：

| 对象 | 长度 |
|---|---:|
| 当前 Q/输入槽位 | 1073 |
| 历史 K/V 槽位 | 975 |
| 当前 Q 可关注的总 K/V 范围 | 2048 |

---

## 五、创建基础 Padding Mask

```python
attention_mask = torch.ones(
    (1, max_tokens),
    dtype=torch.long,
    device=device,
)
```

得到：

```text
shape = [1, 2048]
value = [1, 1, 1, ..., 1]
```

Padding Mask 的通用语义是：

```text
1 = 真实/有效位置
0 = pad 或 dummy，无效位置
```

不过本函数全填 1，因此这份 dummy 输入宣称所有 2048 个 source 位置都有效。它只用于建图，所以不模拟真实的 KV padding 分布。

真实 FPM 会将补出来的 Past KV 和 Current input 位置标成 0。

---

## 六、生成 position IDs

```python
position_ids = torch.cumsum(attention_mask, dim=1) - 1
position_ids = position_ids.clip(0, max_tokens - 1)
position_ids = position_ids[..., :num_tokens]
```

由于基础 mask 全为 1：

```text
cumsum - 1 = [0, 1, 2, ..., 2047]
```

取前 `num_tokens=1073` 个：

```text
position_ids = [0, 1, 2, ..., 1072]
shape        = [1, 1073]
dtype        = int64
```

这里的数值只要能跑通 RoPE 查表即可。真实 FPM 会根据真实有效历史和 padding 计算位置。

---

## 七、合并 Padding Mask 与 Causal Mask

当前配置：

```yaml
use_combined_mask_input: true
```

于是执行：

```python
past_kv_length = max_tokens - num_tokens  # 975

attention_mask = prepare_combined_attention_mask(
    attention_mask,
    input_shape=(1, num_tokens),
    past_key_values_length=past_kv_length,
    device=device,
    mask_neg=-100,
    dtype=dtype,
)
```

### 7.1 Padding Mask 负责什么

Padding Mask 负责回答：

```text
这个 K/V 位置是真实数据，还是为了固定 shape 补出的 dummy？
```

原始形状：

```text
[B, source_len] = [1, 2048]
```

它会先扩维：

```text
[B, 2048]
  → [B, 1, 1, 2048]
  → [B, 1, 1073, 2048]
```

同时从布尔语义转为加法语义：

```text
原始 1（有效） → 0
原始 0（dummy）→ -100
```

### 7.2 Causal Mask 负责什么

Causal Mask 负责回答：

```text
这个 K/V 位置是否属于当前 Query 的未来？
```

先创建当前块的下三角 mask：

```text
[1073, 1073]
```

左侧再拼接 975 列 Past 对应的全 0 mask：

```text
[1073, 1073]
  → [1073, 975 + 1073]
  → [1073, 2048]
```

增加 batch/head 广播维：

```text
[B, 1, 1073, 2048]
```

左侧的 975 列不是 KV Tensor，而是与 Past KV Cache 位置一一对应的 mask 列。历史位置从因果关系上都允许当前 Query 关注，因此初始填 0；其中若存在 dummy，再由 Padding Mask 屏蔽。

### 7.3 为什么二者必须相加

两者最后形状相同：

```text
Padding additive mask：[B, 1, 1073, 2048]
Causal additive mask ：[B, 1, 1073, 2048]
```

逐元素相加：

| Padding | Causal | 合并结果 | 解释 |
|---:|---:|---:|---|
| `0` | `0` | `0` | 真实且非未来，可以看 |
| `-100` | `0` | `-100` | dummy，不能看 |
| `0` | `-100` | `-100` | 未来，不能看 |
| `-100` | `-100` | `-200` | dummy 且未来 |

随后：

```python
clamp_min(-100)
```

把 `-200` 截成 `-100`。逻辑上等价于：

```text
允许关注 = Padding 有效 AND Causal 允许
禁止关注 = 是 dummy OR 是未来
```

单独使用 Causal Mask 不够，因为它不知道历史/前序位置里哪些是 dummy；单独使用 Padding Mask 也不够，因为它不会阻止当前 token 偷看未来真实 token。

### 7.4 为什么 dummy 中 Padding Mask 没起实际作用

本函数的基础 mask 全是 1，扩展后 Padding additive mask 全是 0，所以：

```text
dummy Combined Mask = 0 + Causal Mask ≈ Causal Mask
```

这是允许的，因为 dummy 数值只用于跑图。真实 PPL/FPM 输入的 Padding Mask 才承担屏蔽 dummy 的数值职责。

最终 mask：

```text
shape = [1, 1, 1073, 2048]
dtype = 参数 dtype
value = 允许位置为 0，禁止位置为 -100
```

进入 Attention 后会广播到 16 个 Q 头：

```text
Attention Score：[1, 16, 1073, 2048]
Combined Mask ：[1,  1, 1073, 2048]
```

---

## 八、把整数位置转换成外部 RoPE

当前配置：

```yaml
use_position_embedding_input: true
```

执行：

```python
position_ids = get_position_embeddings_from_position_ids(
    position_ids,
    head_dim=hidden_size // num_attention_heads,
    max_length=max_tokens,
    device=device,
    dtype=dtype,
    config=config,
)
```

调用前：

```text
position_ids：[1, 1073]，整数 0～1072
```

调用后，变量虽然仍叫 `position_ids`，实际已经成为：

```python
(cos, sin)
```

Qwen 当前 `head_dim=128`，项目的复数半维实现得到：

```text
cos：[1, 1, 1073, 64]
sin：[1, 1, 1073, 64]
```

这些值将作为模型外部输入，QcAttention 不再在图内按动态 position ID 生成 RoPE。

当前函数只调用普通 RoPE helper，没有为 `use_mrope=true` 调用 `get_qwen_position_embeddings_from_position_ids()`；当前配置 `use_mrope=false`，所以现有路径可以正常使用。

---

## 九、创建当前 token ID 或 embedding

### 9.1 纯文本 `input_ids` 分支

```python
if not config.use_input_embeddings:
    input_ids = torch.randint(
        0,
        len(tokenizer),
        (1, num_tokens),
        device=device,
    )
```

得到：

```text
shape = [1, 1073]
dtype = int64
value = 词表范围内的随机 token ID
```

随机 token 没有语义，只要不越过词表索引即可。

### 9.2 当前项目的 `inputs_embeds` 分支

```python
inputs_embeds = torch.rand(
    (1, num_tokens, hidden_size),
    device=device,
)
```

得到：

```text
shape = [1, 1073, 2048]
value = 随机浮点数
```

它不是 tokenizer embedding，也没有经过视觉编码器；它只模拟语言模型实际接收的 embedding shape。

当前写法没有显式传 `dtype=dtype`，所以 `torch.rand()` 默认产生 FP32。当前 `enable_fp16=false` 没有问题；若将来用 FP16 dummy，更严谨的写法是：

```python
torch.rand(
    (1, num_tokens, hidden_size),
    device=device,
    dtype=dtype,
)
```

---

## 十、创建 36 层全零 Past KV Cache

```python
inputs["past_key_values"] = get_padded_kv_values(
    past_size=max_tokens - num_tokens,
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_attention_heads=num_attention_heads,
    num_kv_heads=num_kv_heads,
    transposed_key_cache=config.transposed_key_cache,
    device=device,
    dtype=dtype,
)
```

当前：

```text
past_size   = 975
num_layers  = 36
num_kv_heads= 2
head_dim    = 128
```

每层创建一对全零 Tensor：

```python
(K_i, V_i)
```

当前 `transposed_key_cache=true`：

```text
K_i：[1, 2, 128, 975]
V_i：[1, 2, 975, 128]
```

整体为嵌套结构：

```python
past_key_values = (
    (K0, V0),
    (K1, V1),
    ...
    (K35, V35),
)
```

共 72 个 KV Tensor。

辅助函数中这一行命名容易误解：

```python
head_dim = num_kv_heads
```

该局部变量实际表示 KV 头数；真正 head dimension 仍是：

```python
hidden_size // num_attention_heads = 128
```

虽然变量名不准确，但最终创建的 K/V shape 是正确的。

---

## 十一、嵌套字典版与扁平 Tuple 版

### 11.1 `separate_tuple_input_output=False`

返回原始模型熟悉的字典：

```python
{
    "inputs_embeds": Tensor,
    "attention_mask": Tensor,
    "position_ids": (cos, sin),
    "past_key_values": (
        (K0, V0),
        ...,
        (K35, V35),
    ),
}
```

第一次 Prepare 调用使用该格式：

```python
get_dummy_data(..., separate_tuple_input_output=False)
```

因为此时输入对象还是原始 `Qwen2ForCausalLM.forward()` 风格。

### 11.2 `separate_tuple_input_output=True`

先递归展开：

```python
flatten_tensors(past_key_values)
```

将：

```python
((K0, V0), (K1, V1), ...)
```

变成：

```python
(K0, V0, K1, V1, ..., K35, V35)
```

当前 embedding + external RoPE 的最终 tuple：

```python
(
    inputs_embeds,
    attention_mask,
    position_ids_cos,
    position_ids_sin,
    K0, V0,
    K1, V1,
    ...,
    K35, V35,
)
```

总输入数量：

```text
1 + 1 + 2 + 72 = 76
```

它和 `input_names` 严格按位置对应：

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

展平的原因是静态图更适合：

```text
一个输入名 ↔ 一个 Tensor
```

而不是多层 Python tuple。

---

## 十二、当前配置下的完整形状表

当前条件：

```text
use_input_embeddings        = true
use_combined_mask_input     = true
use_position_embedding_input = true
transposed_key_cache        = true
num_layers                  = 36
```

| 输入 | 数量 | shape | dtype 说明 |
|---|---:|---|---|
| `inputs_embeds` | 1 | `[1,1073,2048]` | 当前代码默认 FP32 |
| `attention_mask` | 1 | `[1,1,1073,2048]` | 参数 `dtype` |
| `position_ids_cos` | 1 | `[1,1,1073,64]` | 参数 `dtype` |
| `position_ids_sin` | 1 | `[1,1,1073,64]` | 参数 `dtype` |
| 每层 Past K | 36 | `[1,2,128,975]` | 参数 `dtype` |
| 每层 Past V | 36 | `[1,2,975,128]` | 参数 `dtype` |

扁平模式总计 76 个输入 Tensor。

模型输出签名预期为：

```text
logits                         1 个
每层 new/past key 输出        36 个
每层 new/past value 输出      36 个
总计                          73 个 Tensor
```

---

## 十三、dummy input 被使用三次

### 13.1 QAIRT Prepare

```python
model_preparer.prepare_model(
    model,
    dummy_input,
    ...
)
```

作用：跑通/追踪原始适配模型，建立并重建 prepared graph。

内部 `ONNX → QuIR → QNNIR → Emitter` 阶段见 [附录C · QAIRT model_preparer 内部流程](./06-附录C-QAIRT-model_preparer内部流程.md)。

### 13.2 QuantSim 创建

```python
QuantizationSimModel(
    model=sim_fpm.model,
    dummy_input=dummy_input,
    ...
)
```

作用：识别图结构并插入量化模拟节点。此时使用 `separate_tuple_input_output=True` 的扁平输入。

注意：dummy input 不是 compute encodings 的校准数据。真正统计激活范围时仍会使用 `train_dataloader` 的真实样本。

### 13.3 最终 ONNX 导出

```python
quantsim.export(
    onnx_dir,
    model_name,
    sample_inputs,
    ...
)
```

作用：确定最终 ONNX 输入签名与固定 shape。

最终 Test Vector、ONNX、External Weight、Encoding 的职责与交接关系，见 [08 · ONNX 导出与测试向量](./08-ONNX导出与测试向量.md)。

---

## 十四、与真实 `FPM.prepare_inputs()` 对照

| 对比项 | `get_dummy_data()` | `FPM.prepare_inputs()` |
|---|---|---|
| 数据来源 | 手工随机/全零伪造 | dataloader 的真实输入 |
| 使用目的 | Prepare、建图、QuantSim、导出 | PPL、校准、真实前向 |
| 当前输入长度 | 直接创建 1073 | 根据真实 L 左补到 1073 |
| Past KV | 36 层全零 975 | 真实历史不足时补到 975 |
| Padding Mask | 全 1，不模拟 dummy | 精确标记真实与 dummy |
| position | 固定 0～1072 | 根据真实有效历史计算 |
| 输出整理 | 不负责 | 配套 `prepare_outputs()` 裁 logits/KV |
| 数值是否有语义 | 没有 | 有，直接影响 PPL/校准 |

二者必须在这些接口属性上保持一致：

```text
输入数量、输入顺序、shape、dtype、KV 布局、RoPE 形式、mask 形式
```

否则会出现：

```text
dummy 建出的模型图 ≠ 真实 FPM 喂入的数据接口
```

---

## 十五、代码审阅时需要留意的点

1. `num_tokens=None` 实际不可用，建议增加断言。
2. `rope_theta` 局部变量未使用。
3. `inputs_embeds` 的 `torch.rand()` 未显式传 `dtype`。
4. `mask_neg=-100` 是硬编码，未使用 `config.mask_neg`。
5. dummy 基础 mask 全为 1，所以全零 Past KV 被标记为“可见”；适合建图，不适合数值评估。
6. position 固定取前 `ARN` 个位置，只保证建图输入合法，不模拟所有真实 cache offset。
7. 当前只构造普通 RoPE dummy，没有单独处理 mRoPE 路径。
8. `get_padded_kv_values()` 内部的局部变量 `head_dim = num_kv_heads` 命名不准确。

这些问题大多不影响当前 FP32、`use_mrope=false` 的 shape tracing，但切换 dtype、mRoPE 或拿 dummy 做数值验证时需要重新检查。

---

## 十五·补 · 常见误解速查（自测纠偏）

学完本篇后容易形成一句"顺口但不准"的总结：**"dummy 就是输入全 0、mask 全 1、KV 全 0"**。前后两半有偏差，这里集中纠偏。

### 1. 三个数值到底是什么

| 顺口的说法 | 实际代码 | 纠偏说明 |
|---|---|---|
| 输入是**全 0** | **随机值**：`torch.randint(...)` / `torch.rand(...)`（见第九节）| "值无所谓"的判断对，全 0 理论上也能 trace；但**事实是随机数**，别记成全 0 |
| padding mask **全 1** | ✅ 但**仅指最初那份 2D mask** `[1,2048]`（第五节）| 最终喂进模型的是 4D combined mask，**不是全 1**（见下）|
| KV Cache **全 0** | ✅ `get_padded_kv_values` 用 `torch.zeros`（第十节）| 正确 |

### 2. 最要紧的一条：别把两个 mask 混成一个

```text
① attention_mask = torch.ones((1, 2048))     ← 2D 原始 padding mask，全 1
        │   含义："假装 2048 个 source 位置全部有效"（dummy 不模拟 padding 分布）
        ▼   prepare_combined_attention_mask(...)
② combined mask [1, 1, 1073, 2048]           ← 真正进模型的，绝不是全 1
        内容是 0 / -100：历史 975 列放行(0) + 当前段 1073 的下三角(未来 = -100)
```

**结论**：dummy 省掉的是"**padding 标记**"（所以原料全 1、7.4 里 Padding additive mask 退化成全 0），但**没有省掉"因果掩码结构"**——因果结构属于图形状的一部分，必须造出真实的 `[1,1,1073,2048]` 且带下三角。

> 一句话：**全 1 是"原料"，不是"成品"。** 成品 = Padding(全 0) + Causal(下三角) ≈ Causal Mask。

### 3. 量化里的边界：建图用 dummy，标定必须用真实数据

13.2 已提过一句，这里明确成对照表——**别记成"量化全程用 dummy"**：

| 量化环节 | 喂什么 | 为什么 |
|---|---|---|
| `QuantizationSimModel(dummy_input=...)` | **dummy** | 只为识别图结构、插入量化模拟节点 |
| `apply_seq_mse(...)` | **真实数据**（`train_dataloader`）| 要按真实激活分布搜索最优量化范围 |
| `quantsim.compute_encodings(...)` | **真实数据**（`train_dataloader`）| 要统计真实激活的 min/max 定 scale/offset，喂假数据会算出错误范围 |

所以 `llm_quant.py` 里 dummy 只出现在 **QuantSim 构造**那一行；`SeqMSE` 与 `compute_encodings` 都走 dataloader。

### 4. 修正后的一句话总结

> dummy input 是给 **prepare / QuantSim 建图 / ONNX 导出**用的"**接口与 shape/dtype 全对、内容全假**"的样例——当前输入用**随机值**、历史 KV **全 0**、原始 2D mask **全 1**（但合成出的 combined mask 仍是真实的 0/-100 因果结构）；而量化的**标定环节必须换成真实数据**。

---

## 十六、记忆锚点

```text
get_dummy_data = FPM.prepare_inputs 的“无真实语义静态模板版”
```

- 先定长度：Current 1073、Past 975、Total 2048。
- 造当前输入：随机 token ID 或随机 embedding。
- 造 mask：Padding + Causal → `[1,1,1073,2048]`。
- 造位置：整数 position → external cos/sin。
- 造缓存：36 层全零 K/V，K 转置存储。
- 原始模型接嵌套 dict，prepared/ONNX 接扁平 tuple。
- dummy 只决定接口和图，不参与 PPL，也不代替 calibration dataset。
- **纠偏三连**（见十五·补）：输入是**随机值**不是全 0；**全 1 只是 2D 原料 mask**，成品 combined mask 仍是 0/-100 的因果结构；KV 确实全 0。
- **量化边界**：`QuantizationSimModel(dummy_input=...)` 用 dummy 建图，但 `SeqMSE` / `compute_encodings` **必须用真实数据**标定。

---

## 十七、源码入口

- `example1/llm_quant.py`
  - `get_dummy_data()`：约 L255～323
  - 第一次 Prepare dummy：约 L347
  - QuantSim dummy：约 L450～457
  - ONNX export dummy：约 L589～602
- `example1/llm_utils/forward_pass_wrapper.py`
  - `flatten_tensors()`：约 L23
  - `get_padded_kv_values()`：约 L30
  - `RopeEmbedding`：约 L103
  - `prepare_decoder_attention_mask()`：约 L151
  - `prepare_combined_attention_mask()`：约 L218
- 关联笔记：
  - [06 · Prepare 模型准备阶段](./06-Prepare模型准备阶段.md)
  - [06-附录C · QAIRT model_preparer 内部流程](./06-附录C-QAIRT-model_preparer内部流程.md)
  - [05 · 通用前向处理流程](./05-通用前向处理流程.md)
  - [02-附录E · 端侧定长与计算图导出](./02-附录E-端侧定长与计算图导出.md)
  - [02-附录G · RoPE](./02-附录G-RoPE位置编码.md)
  - [02-附录K · KV Cache](./02-附录K-KV%20Cache(键值缓存).md)
