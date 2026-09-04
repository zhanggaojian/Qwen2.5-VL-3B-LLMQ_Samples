# 07-附录C · 量化规则配置：MatMul、Concat 与混合精度

> **所属主篇**：[07 · 量化主流程](./07-量化主流程-QuantSim到Encoding.md)
>
> **流程位置**：QuantSim 创建完成以后、SeqMSE 和 `compute_encodings()` 以前。
>
> **一句话本质**：在默认 W4A16 的基础上，把 KV 改为 8bit、让每个 Concat 内部共享 Encoding，并对敏感算子设置例外精度。

---

## 一、三段代码总体在做什么

```text
QuantSim 默认 W4A16
        │
        ▼
① MatMul 第二输入改成 8bit symmetric
        │
        ▼
② 每个 Concat 的输入输出共享 Encoding
        │
        ▼
③ 对敏感算子应用手工混合精度例外
        │
        ▼
进入 SeqMSE 和 compute_encodings
```

这三段只是在修改 Quantizer 的配置，还没有通过真实数据计算最终 `scale` 和 `offset`。

---

## 二、第一段：让 MatMul 的 K/V 输入使用 8bit

```python
set_matmul_second_input_producer_to_8bit_symmetric(quantsim)
```

Attention 中有两个主要 MatMul：

```text
Q × Kᵀ
    ▲ 第二输入是 K

Attention概率 × V
                ▲ 第二输入是 V
```

所以修改 MatMul 的第二输入，在 Attention 中就对应 K 和 V：

```text
Q16 × K8
概率16 × V8
```

目的：让 KV Cache 以 8bit 保存和传输，降低缓存大小和数据 I/O。

这里的“producer”指产生 K/V Tensor 的上游算子；代码是根据计算图的位置找到 K/V，不是根据变量名称查找。

---

## 三、第二段：每个 Concat 内部共享 Encoding

```python
propagate_output_encodings(quantsim, aimet_ops.Concat)
```

规则是：

```text
对于每一个 Concat：
它的所有输入和输出使用相同的量化参数
```

例如：

```text
输入A1 ─┐
输入A2 ─┼─ Concat A → 输出A
        └─ 共享 Encoding A

输入B1 ─┐
输入B2 ─┼─ Concat B → 输出B
        └─ 共享 Encoding B
```

不是全模型所有 Concat 共用一套 Encoding，而是每个 Concat 各自形成一个共享组。

为什么需要共享？

```text
输入1：整数 10，scale=0.1，代表浮点 1.0
输入2：整数 10，scale=0.2，代表浮点 2.0
```

如果直接拼成 `[10, 10]`，输出 Tensor 无法用一套 scale 同时解释两个 `10`。因此要么重新量化，要么提前共享 Encoding；这里选择后者，从而减少 Requantize 的开销和误差。

共享的是 `bitwidth、scale、offset、对称方式`，不是模型权重。

---

## 四、第三段：手工混合精度例外

```python
quantsim_adjuster = ManualQuantsimMixedPrecisionConfig(
    mixed_precision_config_file=_mp_cfg_file
)
quantsim_adjuster.apply_exceptions(quantsim)
```

模型默认是 W4A16，但部分位置对量化更敏感，或者承担 KV 数据搬运，因此通过 `exceptions.json` 单独覆盖。可以把这个文件理解成一张“手工量化例外名单”：

```text
默认规则：大部分算子使用 W4A16
例外名单：部分算子改成 8bit、16bit 或不量化
```

配置来源：[exceptions.json](../../example1/config/mixed_precision_config/exceptions.json)。应用逻辑：[mixed_precision_overrides.py](../../example1/llm_utils/mixed_precision_overrides.py)。

### 4.1 它修改的是规则，不是最终 Encoding

`exceptions.json` 在 QuantSim 创建后、SeqMSE 和 `compute_encodings()` 以前应用：

```text
QuantSim 创建默认 W4A16 Quantizer
        ↓
exceptions.json 修改局部 Quantizer 的位宽、对称方式或启用状态
        ↓
SeqMSE / compute_encodings 根据新规则计算 min/max、scale、offset
```

因此该文件不保存最终量化数值，也不是导出的 `.encodings` 文件。

### 4.2 JSON 字段怎么读

文件中的 `module_list` 用于按模块类型匹配，当前为空；实际规则都放在 `name_list`，使用 `module_name` 匹配 `quantsim.model.named_modules()` 中的模块名。

| 字段 | 作用 |
|---|---|
| `module_name` | 模块名匹配表达式 |
| `param_exceptions` | 修改 `param_quantizers["weight"]` |
| `input_exceptions` | 修改输入 Activation Quantizer |
| `output_exceptions` | 修改输出 Activation Quantizer |
| `null` | 对这一类 Quantizer 不做修改 |
| `bitwidth` | 指定位宽 |
| `asymmetric: false` | 使用 signed symmetric 量化 |
| `asymmetric: true` | 使用 asymmetric 量化 |
| `enabled: false` | 将对应输入/输出 Quantizer 设为 `None`，即关闭该处假量化 |

例如：

```json
{
  "module_name": "\\w*lm_head_(MatMul|conv_Conv)",
  "exceptions": {
    "param_exceptions": {"bitwidth": 8},
    "input_exceptions": null,
    "output_exceptions": null
  }
}
```

意图是找到名称符合表达式的 `lm_head`，只把它的 Weight Quantizer 从默认 4bit 提高到 8bit，输入和输出 Quantizer 保持不变。

### 4.3 当前六条规则的设计意图

| 匹配位置 | 实际修改 | 目的 |
|---|---|---|
| `lm_head...` | Weight 改为 8bit symmetric | 保护最终 logits 精度 |
| `norm_Mul_1...` | 输入 0 改为 A16 asymmetric | 保护 RMSNorm 最后乘法的输入 |
| `norm_(Pow/ReduceMean/Add/Sqrt/Div/Mul)` | 关闭输出 0 Quantizer | 避免量化敏感的归一化中间结果 |
| `self_attn_Concat_1` | 输出 0 改为 A8 symmetric | 设计意图是让 KV Cache 相关拼接结果保持 8bit |
| `v_proj...` | 输出 0 改为 A8 symmetric | 让新生成的 Value 在进入 Cache 前保持 8bit，降低 KV I/O |
| `rms_norm_\d+` | Norm Weight 改为 16bit symmetric | 保护归一化缩放参数 |

可以把精度分配记成：

```text
普通大权重：W4，优先节省体积
lm_head：W8，保护最终预测
RMSNorm Weight：W16，保护归一化
Value/KV Cache：A8，减少缓存和数据搬运
Norm 内部敏感结果：不插入 QDQ
```

由于 SeqMSE 只优化满足其支持条件的低位宽权重层，提高到 W8/W16 的 Weight Quantizer 通常会跳过 SeqMSE，随后由 `compute_encodings()` 补齐其 Parameter Encoding。

---

## 五、三个规则的区别

| 规则 | 解决的问题 |
|---|---|
| MatMul 第二输入 A8 | 压缩 K/V Cache，减少数据搬运 |
| Concat 共享 Encoding | 让拼接的数据使用同一把量化尺子 |
| 手工混合精度 | 在性能、内存和精度之间做局部调整 |

---

## 六、工程检查点

### 6.1 配置被读取，不代表规则已经命中

匹配代码的核心路径使用 `re.fullmatch()`，要求表达式覆盖完整模块名。初始化时打印的：

```text
Applying ...
```

只表示 JSON 规则读取成功，不代表在 QuantSim 模型中找到了目标模块。

对当前生成文件 `output/prepare/qwen25llm_kvcache_36_layer.py` 中的顶层模块名做静态匹配，六条规则均为零命中。几个明显差异如下：

| 配置期望名称 | 当前生成模型名称示例 | 风险 |
|---|---|---|
| `...self_attn_Concat_1` | `model_layers_0_self_attn_Concat_10`（Value）和 `Concat_9`（Key） | `fullmatch` 下不会命中 |
| `...v_proj_conv_Conv` | `model_layers_0_self_attn_v_proj_conv` | 后缀不一致 |
| `...lm_head_conv_Conv` | 最终词表投影模块名为 `Conv` | 不包含 `lm_head` |
| `...rms_norm_0` | `rms_norm_model_layers_0_input_layernorm` | 命名结构不一致 |
| `norm_Pow/Mean/Sqrt/...` | 当前生成图将这部分表示为 `RmsNorm` 模块 | 可能不存在可匹配的分解模块 |

这说明当前 `exceptions.json` 很可能沿用了另一版 Prepare 图的名称。静态文件只能提供预警，最终应在 QuantSim 创建后对真实的 `named_modules()` 做运行时审计。

### 6.2 建议打印每条规则的真实命中结果

```python
import json
import re

with open(_mp_cfg_file) as f:
    rules = json.load(f)["name_list"]

module_names = [name for name, _ in quantsim.model.named_modules()]

for rule in rules:
    pattern = rule["module_name"]
    hits = [name for name in module_names if re.fullmatch(pattern, name)]
    print(pattern, len(hits), hits[:5])
```

每条重要规则都应至少看到预期数量的命中；否则 JSON 写得再合理也不会改变 Quantizer。

### 6.3 当前加载器的两个细节

- JSON 虽然写了 `input_index` 和 `output_index`，但当前实现实际按数组位置 `0、1……` 修改 Quantizer，并没有读取字段里的索引值。现有规则都只有索引 0，所以暂时结果一致；以后指定非零端口时需要修正加载器。
- 如果同一模块同时匹配多条 `name_list` 规则，当前实现不会合并多条 exception，而是保留遍历过程中最后命中的那一条。

---

## 七、一句话总结

> **第一段压缩 KV，第二段统一每个 Concat 内部的量化尺子，第三段通过例外名单保护敏感算子；但例外规则依赖模块名匹配，必须确认实际命中后，才能认为默认 W4A16 已被成功调整。**
