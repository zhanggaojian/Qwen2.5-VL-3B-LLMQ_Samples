# 07-附录D · SeqMSE 权重量化优化

> **所属主篇**：[07 · 量化主流程](./07-量化主流程-QuantSim到Encoding.md)
>
> **流程位置**：QuantSim 与量化规则配置完成以后，`compute_encodings()` 激活标定以前。
>
> **对应代码**：`example1/llm_quant.py` 约 L480～531。
>
> **一句话本质**：针对 SeqMSE 支持的低位宽权重层，用代表性样本构造配对输入，逐层从多组候选量化范围中选择局部输出误差最小的 Parameter Encoding。

---

## 一、这段代码属于同一个部分吗

属于。下面这些代码共同组成一个完整的 **SeqMSE 后训练量化优化阶段**：

```text
导入 SeqMSE 接口
        ↓
定义 input_ids / input_embeddings 两种前向回调
        ↓
根据模型配置选择回调
        ↓
创建 SeqMSE 搜索参数
        ↓
把浮点模型和量化模型放到 CUDA
        ↓
执行逐层 MSE 搜索
        ↓
删除浮点参考模型并恢复量化模型 dtype
```

其中，`print(train_dataloader)` 只是调试输出，被注释的 `.to(torch.half)` 代码也不会执行。

---

## 二、SeqMSE 是什么

SeqMSE 是 **Sequential Mean Squared Error**，即“顺序均方误差优化”。它是一种后训练量化（PTQ）方法，不需要重新训练整个语言模型。

对于一个支持优化的层，当前 `inp_symmetry=symqt` 配置可以抽象成：

```text
QuantSim 前序路径产生的层输入 x_q
          ├─→ 浮点权重 W ─────────→ y_ref = x_q · W
          └─→ 候选量化权重 Q(W) ──→ y_q   = x_q · Q(W)

                         计算 MSE(y_ref, y_q)
```

也就是说，参考分支仍使用浮点权重，但两边共享 QuantSim 前序路径采集到的层输入。SeqMSE 会为当前层尝试多组候选量化范围，选择使 `y_q` 最接近 `y_ref` 的那一组，然后继续处理下一层，因此叫“Sequential”。

SeqMSE 不会处理所有权重层。AIMET 的 PyTorch 实现会筛选支持的模块和参数位宽；当前官方实现列出的主要模块是 `torch.nn.Linear` 与 `torch.nn.Conv2d`，支持的参数位宽上限为 4 bit。本项目默认 W4，并将大量 Linear 转成 1×1 Conv，因此这些层正是主要优化对象。

它优化的主要是权重 Quantizer 的 Encoding，例如：

- 量化上下界；
- `scale`；
- `offset`；
- 与量化范围相关的参数。

它不是反向传播训练，也不会把模型权重重新训练成另一套浮点值。

### 2.1 SeqMSE 与普通 Min-Max 权重量化的核心区别

这里的“普通量化”特指当前项目中**不使用 SeqMSE，直接按 `post_training_tf` Min-Max 规则确定 Weight Encoding 的基础 PTQ**，不是泛指所有量化方法。

最核心的区别只有一句话：

```text
普通 Weight Min-Max：看权重 W 自身的 min/max
SeqMSE：看真实层输入下，量化权重造成的层输出误差
```

| 对比点 | 普通 Min-Max 权重量化 | SeqMSE |
|---|---|---|
| 根据什么 | Weight 自身的最小值和最大值 | 参考层输出与候选量化层输出的误差 |
| 怎么处理 | 直接计算一组 Encoding，通常配合 RTN 舍入 | 每层尝试多组候选 Encoding，选择损失最小者 |
| 是否需要校准数据／前向 | Weight Encoding 本身通常不需要 | 需要代表性数据和离线前向来采集层输入 |
| 优化目标 | 覆盖权重数值范围 | 最小化层输出重构误差 |
| 当前项目配置 | `post_training_tf` Min-Max | 20 个 Batch、每层 20 组候选、`symqt + mse` |
| 处理范围 | 按配置启用且允许计算的 Parameter Quantizer | 仅优化 AIMET 支持且满足条件的低位宽层 |
| 最终得到 | 普通 Weight Encoding | 搜索并冻结的 Weight Encoding |
| 计算成本 | 低 | 较高，需要逐层采样和局部候选计算 |

两者的共同点也很重要：

- 都属于静态 PTQ，最终得到的都是 Weight Encoding；
- 都不重新训练或改写原始浮点 Weight；
- SeqMSE 不会改变 INT4 量化公式、位宽或导出格式，它只是换了一种更贴近真实层输出的 Encoding 选择方法；
- SeqMSE 不是完整量化流程的替代品，之后仍要运行 `compute_encodings()` 标定 Activation，并处理其他允许覆盖的 Parameter Encoding。

因此，当前项目中二者是“**基础量化 + 可选精度优化**”的关系：

```text
QuantSim + Min-Max 规则建立普通量化基础
                   ↓
SeqMSE 优化并冻结满足条件的 W4 Weight Encoding
                   ↓
compute_encodings 标定 Activation，并处理其他 Parameter Encoding
```

更完整的异常值例子和其他量化方法对比，见 [07-附录F · 量化方法总览与选型](./07-附录F-量化方法总览与选型.md#八seqmse-与普通量化的区别)。

### 2.2 量化对象与静态／动态量化要分开理解

从**量化对象**来看，大模型量化通常可以分成三类：

1. **权重量化**：量化 Linear、Conv 等层中已经保存好的模型参数；
2. **激活量化**：量化模型前向过程中产生的中间张量；
3. **KV Cache 量化**：量化 Prefill／Decode 过程中生成并跨 Token 保存的 K、V 张量。KV Cache 本质上也是一种特殊的激活或模型状态，但由于它会长期占用显存和内存带宽，通常单独讨论。

“量化什么”和“Encoding 如何确定”是两个不同维度。运行时动态产生的张量，并不等于采用了动态量化：

| 量化对象／方法 | Encoding 如何获得 | 确定 Encoding 是否需要前向 |
|---|---|---|
| 普通 Min-Max 权重量化 | 直接扫描已有权重的范围，离线计算并固定 | 通常不需要 |
| SeqMSE 权重量化 | 前向比较多组候选量化权重造成的层输出误差，选出最佳 Encoding | **需要** |
| 静态激活量化 | 跑代表性校准数据，统计激活范围，校准后固定 | **需要** |
| 静态 KV Cache 量化 | 跑有代表性的 Prefill／Decode 数据，统计 K、V 范围，校准后固定 | **需要** |
| 动态激活／KV Cache 量化 | 推理时根据当前张量实时计算或更新 Encoding | 推理时动态计算 |

因此，“权重量化不需要前向”只适用于普通的 Min-Max 等直接统计方法。本项目使用的 SeqMSE 虽然最后得到的是固定的 Weight Encoding，但搜索过程仍要执行前向：

```text
浮点权重 W
    ↓ 生成多组候选 Encoding
Q1(W)、Q2(W)、...、Qn(W)
    ↓ 用代表性输入执行逐层计算
比较各候选的层输出 MSE
    ↓
固定误差最小的 Weight Encoding
```

激活和 KV Cache 的值由输入决定，模型文件中没有现成的数据分布，所以离线校准通常必须跑前向。但如果校准结束后把 `scale`、`offset` 等 Encoding 固定并导出，它们仍然属于**静态 PTQ**；只有在推理时根据当前激活或 KV Cache 重新计算 Encoding，才属于严格意义上的动态量化。

量化对象、PTQ/QAT、静态/动态、粒度以及其他常见算法的完整关系，见 [07-附录F · 量化方法总览与选型](./07-附录F-量化方法总览与选型.md)。

---

## 三、执行 SeqMSE 需要什么

核心调用是：

```python
apply_seq_mse(
    fp_prepared_fpm.model,
    quantsim,
    train_dataloader,
    params,
)
```

四个输入分别承担不同角色：

| 输入 | 作用 |
|---|---|
| `fp_prepared_fpm.model` | 提供对应的浮点权重与模型结构，参与构造逐层参考计算 |
| `quantsim` | 带 Quantizer 的量化模拟模型，尝试候选 Encoding |
| `train_dataloader` | 提供有代表性的真实样本 |
| `params` | 指定 batch 数、候选数、损失函数和前向回调 |

这里应使用真实、有代表性的校准数据。单个建图 Dummy input 只能帮助 QuantSim 确定接口、shape 和计算图，不能充分代表真实输入分布，不适合用来选择最佳量化范围。

---

## 四、为什么定义了两个前向函数

当前模型支持两种输入接口：

```text
模式一：input_ids
        Token ID → 模型内部 Embedding → Transformer

模式二：input_embeddings
        外部准备好的 Embedding → Transformer
```

因此代码提供了两种 SeqMSE 前向回调。

### 4.1 `input_ids` 回调

```python
def _forward_fn_inputs_id(model, inputs):
    if model == fp_prepared_fpm.model:
        fpm = fp_prepared_fpm
    else:
        fpm = sim_fpm

    input_length = inputs["input_ids"].shape[1]
    prepared_inputs, _ = fpm.prepare_inputs(
        input_ids=inputs["input_ids"][:, :min(input_length, fpm.num_tokens), ...]
    )
    fpm.model(**prepared_inputs)
```

它接收 dataloader 中的 `input_ids`，然后：

1. 找到当前模型对应的 `LLMForwardPassManager`；
2. 将序列裁剪到 `fpm.num_tokens` 以内；
3. 通过 `prepare_inputs()` 补齐模型需要的定长输入；
4. 执行浮点模型或量化模拟模型。

### 4.2 `input_embeddings` 回调

```python
def _forward_fn(model, inputs):
    if model == fp_prepared_fpm.model:
        fpm = fp_prepared_fpm
    else:
        fpm = sim_fpm

    input_length = inputs["input_embeddings"].shape[1]
    prepared_inputs, _ = fpm.prepare_inputs(
        input_embeddings=inputs["input_embeddings"][
            :, :min(input_length, fpm.num_tokens), ...
        ]
    )
    fpm.model(**prepared_inputs)
```

逻辑与上一种相同，只是输入由 Token ID 换成了外部 Embedding。这对于包含视觉 Token 的 Qwen2.5-VL 模型尤其重要，因为多模态预处理后可以直接向 LLM 部分提供融合后的 Embedding。

### 4.3 根据配置选择回调

```python
if not llm_config.use_input_embeddings:
    _forward_fn = _forward_fn_inputs_id
```

因此：

| `use_input_embeddings` | 最终使用的输入 |
|---|---|
| `True` | `input_embeddings` |
| `False` | `input_ids` |

### 4.4 `fpm.model(**prepared_inputs)` 何时执行

这行是普通的 PyTorch 前向调用，`**` 会把 `prepared_inputs` 字典展开成模型的关键字参数：

```python
fpm.model(**prepared_inputs)
```

它写在 `_forward_fn` 的函数体里，创建 `SeqMseParams` 时并不会提前运行。`SeqMseParams` 只是保存函数引用，真正的调用时机在 `apply_seq_mse()` 内部：

```text
SeqMseParams 保存 forward_fn
        ↓
apply_seq_mse 为目标层注册 Hook
        ↓
AIMET 调用 forward_fn(model, batch)
        ↓
prepare_inputs() 准备定长输入
        ↓
fpm.model(**prepared_inputs) 执行到目标层
        ↓
Hook 捕获该层输入激活
```

因此这里不需要保存最终 logits；执行模型的主要目的是让 AIMET 捕获 SeqMSE 所需的中间层输入。

---

## 五、为什么要先调用 `prepare_inputs()`

Dataloader 的 batch 可能同时包含 Token ID、Embedding、Label、Mask 等字段；当前回调只选择 `input_ids` 或 `input_embeddings` 作为主输入，再由 `prepare_inputs()` 构造 prepared 模型真正需要的完整接口，其中通常还包括：

- 定长后的当前 Token；
- Attention Mask；
- Position ID 或外置 RoPE 的 `cos/sin`；
- 每一层的 Past Key/Value；
- Prepare 后模型要求的其他扁平化输入。

因此不能直接写：

```python
model(**inputs)
```

而要经过：

```text
dataloader 原始 batch
        ↓
裁剪到 num_tokens
        ↓
LLMForwardPassManager.prepare_inputs()
        ↓
生成 prepared 模型需要的完整输入
        ↓
fpm.model(**prepared_inputs)
```

### 5.1 当前项目固定了哪些 Shape

当前配置是 `context_length=2048`、`num_tokens=1073`，因此 Past KV 长度固定为 `2048-1073=975`：

| 输入 | 固定 Shape | 数值来源 |
|---|---|---|
| Current Embedding | `[B, 1073, hidden_size]` | 真实 Embedding；不足 1073 时左侧补 0 |
| Past Key/Value | 每层固定保存 975 个历史位置 | 本次回调没有传入历史 KV，因此使用全 0 Padding KV |
| RoPE `cos/sin` | 序列维固定为 1073 | 根据有效 Token 的位置计算；Shape 固定不等于数值永远固定 |
| Combined Attention Mask | `[B, 1, 1073, 2048]` | 根据 Padding 和因果关系生成 |

切片表达式只负责让输入长度不超过 `fpm.num_tokens`；不足 1073 时的补齐、975 长度 KV 的构造以及 RoPE/Mask 的生成，都发生在 `prepare_inputs()` 内部。

### 5.2 Attention Mask 的两个组成部分

最终的 Combined Attention Mask 逻辑上由两部分组成：

```text
Padding Mask：遮挡补零输入和无效的 Padding KV
Causal Mask ：遮挡当前 Token 后面的未来 Token
```

它们最终不是彼此拼接，而是先扩展到相同的 `[B, 1, 1073, 2048]`，再逐元素相加：

```text
Combined Mask = Expanded Padding Mask + Causal Mask

0      + 0      = 0       允许关注
-100   + 0      = -100    遮挡 Padding
0      + -100   = -100    遮挡未来 Token
```

Causal Mask 自己内部则会沿最后一维拼接“Past KV 的全 0 区域”和“当前 Token 的下三角区域”。最终效果可以记成：一个位置只有同时满足“数据有效”并且“不是未来位置”时才能被 Attention 看到。

回调本身不需要返回 logits。AIMET 会用 Hook 捕获受支持目标层的输入，然后围绕该层分别计算浮点权重输出与候选量化权重输出，以得到局部重构误差。

---

## 六、`SeqMseParams` 参数含义

代码：

```python
params = SeqMseParams(
    num_batches=_seq_mse_cfg["num_batches"],
    inp_symmetry=_seq_mse_cfg["inp_symmetry"],
    num_candidates=_seq_mse_cfg["num_candidates"],
    loss_fn=_seq_mse_cfg["loss_fn"],
    forward_fn=_forward_fn,
)
```

当前配置来自 `example1/config.yaml`：

```yaml
seq_mse:
  num_batches: 20
  inp_symmetry: symqt
  num_candidates: 20
  loss_fn: mse
```

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `num_batches` | `20` | 使用多少个真实 batch 搜索 Encoding |
| `num_candidates` | `20` | 每层尝试多少组候选量化范围 |
| `loss_fn` | `mse` | 用均方误差评价候选结果 |
| `inp_symmetry` | `symqt` | 两侧共享 QuantSim 路径采集到的层输入，比较 `x_q·W` 与 `x_q·Q(W)` |
| `forward_fn` | `_forward_fn` | 告诉 AIMET 如何把一个 batch 喂给当前模型 |

### 6.1 先记住“根据什么、处理什么、得到什么”

`SeqMseParams` 自己不处理模型，它只是 `apply_seq_mse()` 的搜索规则：

| 问题 | 答案 |
|---|---|
| 根据什么 | 代表性校准数据产生的层输入，以及参考输出与候选量化输出之间的误差；当前 `symqt` 具体比较 `x_q·W` 和 `x_q·Q(W)` |
| 处理什么 | QuantSim 中受支持层的 Weight Encoding |
| 怎么处理 | 每层尝试 20 组候选范围，选择输出 MSE 最小的一组 |
| 得到什么 | 该层最佳的权重 `min/max、scale、offset` |

主线可以记成：

```text
20 个校准 batch
      ↓
每层尝试 20 组权重量化范围
      ↓
比较各候选的层输出 MSE
      ↓
保存并冻结最佳 Weight Encoding
```

### 6.2 20 组候选参数从哪里来

配置文件只写候选数量：

```yaml
num_candidates: 20
```

具体的 20 组数值不是预先写在配置里的，而是 AIMET 根据每一层自己的初始权重范围动态生成。例如初始范围为 `[-1, 1]` 时，概念上会依次尝试：

```text
候选1 ：[-0.05, 0.05]
候选2 ：[-0.10, 0.10]
...
候选20：[-1.00, 1.00]
```

每组候选 `min/max` 都会产生相应的 `scale/offset`。不同层的权重范围不同，所以它们生成的 20 组候选数值也不同。

### 6.3 MSE 比较的是什么

对于当前 `symqt` 配置，AIMET 用同一个量化路径层输入 `x_q` 计算：

```text
参考输出：y_ref = x_q · W
候选输出：y_q   = x_q · Q(W)
```

若输出中共有 `n` 个参与比较的数：

```text
MSE = [(y_ref1-y_q1)² + ... + (y_refn-y_qn)²] / n
```

因此 SeqMSE 不是直接比较 `W` 和 `Q(W)` 的权重数值误差，而是比较权重误差经过真实层输入以后造成的输出误差：

```text
输出误差 = x_q · [W - Q(W)]
```

### 6.4 `inp_symmetry` 决定两边使用哪一路输入

先定义：

```text
x_fp = 浮点模型到达当前层的输入
x_q  = QuantSim 模型到达当前层的输入
W    = 当前层浮点权重
Q(W) = 当前候选 Encoding 下的量化再反量化权重
```

`inp_symmetry` 决定参考分支和候选分支如何配对输入：

| 模式 | 参考输出 | 候选量化输出 |
|---|---|---|
| `asym` | `x_fp · W` | `x_q · Q(W)` |
| `symfp` | `x_fp · W` | `x_fp · Q(W)` |
| `symqt` | `x_q · W` | `x_q · Q(W)` |

本项目使用 `symqt`：

```text
同一个 QuantSim 路径输入 x_q
        ├─→ 浮点权重 W    → x_q·W
        └─→ 量化权重 Q(W) → x_q·Q(W)
                                ↓
                             计算 MSE
```

这样既能让两边使用同一个输入、公平隔离当前层的权重量化误差，又能让 `x_q` 带上前面已量化层对输入产生的影响。

名称可以记成：

```text
sym = 两边使用同一路输入
fp  = 这路输入来自浮点模型
qt  = 这路输入来自 QuantSim 模型
```

注意：这里的 `inp_symmetry` 是 SeqMSE 的**输入配对策略**，不是 Quantizer 的 `is_symmetric` 对称量化开关。

增大 `num_batches` 或 `num_candidates` 通常会提高搜索成本；实际耗时还取决于可优化层数、模型大小，以及 AIMET 内部的缓存和候选计算方式。

---

## 七、CUDA 上真正执行优化

```python
with fp_prepared_fpm.place_on_device("cuda"), \
     sim_fpm.place_on_device("cuda"):
    apply_seq_mse(fp_prepared_fpm.model, quantsim, train_dataloader, params)
```

`place_on_device("cuda")` 是上下文管理器。在 `with` 代码块内，它让：

- 浮点参考模型位于 GPU；
- QuantSim 模型位于 GPU；
- 两边能使用相同数据完成逐层输出比较。

这一行才是前面所有准备工作的执行入口：

```python
apply_seq_mse(...)
```

执行结束后，受支持层的最佳候选结果会保留并冻结在 `quantsim` 对应的参数 Quantizer 中，供后续 Encoding 补齐、评估和导出使用。

### 7.1 最佳 Weight Encoding 保存在哪里

SeqMSE 刚完成时，结果只保存在内存中的 QuantSim 权重 Quantizer，概念位置是：

```python
quantsim.model.<某一层>.param_quantizers["weight"]
```

当前 `QuantizationSimModel(..., in_place=True)`，因此 `quantsim.model` 与 `sim_fpm.model` 对应同一个量化模型。此时还没有生成独立磁盘文件；若程序直接退出，内存结果不会自动保留。

后续执行 `compute_encodings()` 并调用：

```python
quantsim.export(onnx_dir, model_name, sample_inputs, ...)
```

才会把 Encoding 写入磁盘。按当前配置，目标文件是：

```text
/root/autodl-tmp/zgj/Qwen25/outputs/output/onnx/qwen25llm.encodings
```

该文件是最终 Encoding 总表，既包含 SeqMSE 优化并冻结的 Weight Encoding，也包含 `compute_encodings()` 得到的 Activation Encoding，以及从参数张量计算或重算的其他可覆盖 Parameter Encoding。

---

## 八、执行后的清理

```python
del fp_prepared_fpm
del prepared_model
sim_fpm.model.to(torch.float32)
```

含义如下：

| 代码 | 目的 |
|---|---|
| `del fp_prepared_fpm` | 删除 SeqMSE 使用的浮点参考 FPM |
| `del prepared_model` | 删除不再需要的 prepared 浮点模型引用 |
| `sim_fpm.model.to(torch.float32)` | 显式确保保留下来的量化模拟模型以 FP32 承载假量化计算 |

这里的“确保 FP32”不代表取消量化；如果模型原本已经是 FP32，这一步可能不会产生实际 dtype 变化。QuantSim 的 Quantizer 仍然存在，模型只是用浮点运算模拟低位宽的量化、反量化误差。

---

## 九、SeqMSE 与 `compute_encodings()` 的关系

这两个步骤都会使用真实数据，但目的不同：

| 阶段 | 主要对象 | 核心动作 | 结果 |
|---|---|---|---|
| SeqMSE | 参数/权重 Quantizer | 逐层尝试候选范围并比较输出 MSE | 优化后的 Weight Encoding |
| `compute_encodings()` | 激活 Quantizer，以及允许覆盖的参数 Quantizer | 前向统计实际执行到的激活；直接从参数张量计算或重算参数 Encoding | Activation Encoding 与可覆盖的 Parameter Encoding |

流程顺序是：

```text
QuantSim 模型骨架
       ↓
SeqMSE：优化权重 Encoding
       ↓
compute_encodings：确定激活 Encoding，并计算／重算可覆盖参数 Encoding
       ↓
量化后 PPL 评估
       ↓
导出 ONNX + Encodings
```

因此，SeqMSE 不能完全代替 `compute_encodings()`：前者只优化满足支持条件的权重层，后者还要为校准前向实际执行到且已启用、允许更新的 Activation Quantizer 确定 Encoding，并从参数张量计算或重算其他已启用、允许覆盖的 Parameter Encoding。SeqMSE 设置为不可覆盖的 Weight Encoding 会被保留。

`compute_encodings()` 的回调、Quantizer 观察状态、切块 KV 路径和当前校准覆盖边界，详见 [07-附录E · `compute_encodings()` 激活标定](./07-附录E-compute_encodings激活标定.md)。

---

## 十、哪些东西发生了变化

### 发生变化的

- SeqMSE 支持且满足位宽条件的层，其参数 Quantizer Encoding；
- 对应的量化上下界、`scale` 和 `offset`；
- 后续假量化时产生的量化误差。

### 没有发生变化的

- 模型网络结构；
- 原始浮点权重数值本身；
- Tokenizer；
- Dataloader 中的样本；
- 最终 ONNX 文件——此时还没有执行导出；
- 目标端 HTP/NPU 模型——当前仍是在主机 GPU 上做量化模拟和优化。

---

## 十一、常见误解

### 11.1 “SeqMSE 是重新训练模型”

不是。它不进行完整的梯度反向传播训练，主要是在候选量化范围中搜索误差较小的 Encoding。

### 11.2 “传入 `train_dataloader` 就是在更新权重”

不是。这里的数据用于比较浮点输出与量化输出，起到校准和搜索依据的作用。

### 11.3 “SeqMSE 完成后已经得到真正的 INT4 模型”

不是。此时仍是 QuantSim 假量化模型。还需要完成激活 Encoding、导出 ONNX/Encoding，并经过 QNN 编译，才会进入端侧整数执行阶段。

### 11.4 “两个 `_forward_fn` 会同时执行”

不会。代码会根据 `llm_config.use_input_embeddings` 选择其中一个传给 `SeqMseParams`。

### 11.5 “SeqMSE 和 PPL 评估是一回事”

不是。SeqMSE 用层输出 MSE 选择 Encoding；PPL 从语言模型任务层面评估整体预测质量。局部层误差更小通常有助于最终精度，但仍需用 PPL 做整体验收。

---

## 十二、面试速答

### Q：这段代码主要做什么？

> 它执行 AIMET SeqMSE 后训练量化优化。代码通过代表性校准样本为受支持层采集输入，逐层尝试多组权重量化 Encoding；当前 `symqt` 策略比较 `x_q·W` 与 `x_q·Q(W)`，并用 MSE 选择误差最小的候选结果。

### Q：为什么需要两个模型？

> 两个模型用于建立浮点层和量化层的对应关系。当前 `symqt` 下，两侧使用相同的 QuantSim 路径输入：浮点权重产生参考输出，候选量化权重产生待比较输出。

### Q：为什么还需要自定义 `forward_fn`？

> AIMET 不知道当前 Qwen2.5-VL prepared 模型的定长输入、KV Cache、Mask 和 RoPE 如何构造，因此需要项目通过 `LLMForwardPassManager.prepare_inputs()` 把 dataloader batch 转成完整模型输入。

### Q：SeqMSE 完成后为什么还要 `compute_encodings()`？

> SeqMSE 只优化受支持层的 Weight Encoding；`compute_encodings()` 还要根据真实激活分布确定 Activation Encoding，并直接从参数张量计算或重算其他允许覆盖的 Parameter Encoding。SeqMSE 冻结的结果保持不变。

---

## 十三、参考资料

- [AIMET · Sequential MSE 功能指南](https://quic.github.io/aimet-pages/releases/2.0.0/featureguide/seq_mse.html)
- [AIMET · PyTorch SeqMSE 源码文档](https://quic.github.io/aimet-pages/releases/2.30.0/_modules/aimet_torch/_base/seq_mse.html)

---

## 十四、一句话总结

> **这段代码用代表性校准数据为受支持层采集输入，在相同 `x_q` 下逐层比较 `x_q·W` 和 `x_q·Q(W)`，从 AIMET 动态生成的候选范围中选择 MSE 最小的一组；最佳 Weight Encoding 先冻结在 QuantSim 权重 Quantizer 中，最后随 `quantsim.export()` 写入 `.encodings` 文件。**
