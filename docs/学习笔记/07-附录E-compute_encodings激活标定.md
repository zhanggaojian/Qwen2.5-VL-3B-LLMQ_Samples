# 07-附录E · `compute_encodings()` 激活标定

> **所属主篇**：[07 · 量化主流程](./07-量化主流程-QuantSim到Encoding.md)
>
> **流程位置**：SeqMSE 固定受支持层的 Weight Encoding 以后，量化后 PPL 评估以前。
>
> **对应代码**：`example1/llm_quant.py` 约 L534～568；切块和 KV Cache 前向见 `example1/llm_utils/forward_pass_wrapper.py` 约 L627～661。
>
> **一句话本质**：在 SeqMSE 已确定受支持 Weight Encoding 的 QuantSim 模型上运行代表性数据，让实际执行到且已启用的 Activation Quantizer 观察张量分布并确定 Encoding；同时从参数张量计算或重算允许覆盖的 Parameter Encoding，并保留 SeqMSE 已冻结的结果。

---

## 零、先抓住五个重点

1. `compute_encodings()` 不是训练：Activation Encoding 来自离线前向观察，Parameter Encoding 则可直接从参数张量计算。
2. 它应放在 SeqMSE 后面，因为这时受支持层的最佳 Weight Encoding 已经确定，才能观察该 Weight 路径下、尚未叠加 Activation QDQ 误差的统计。
3. 当前实际选择 `input_embeddings` 回调，对前 20 个 Batch 执行 QuantSim 校准前向。
4. 切块 helper 支持连续传递 KV，但当前每条 VL calibration 样本恰好为 ARN=1073，所以每 Batch 只前向一次，Past KV 输入是 975 个零占位。
5. 结果先保存在 QuantSim Quantizer 中；代码随后计算并打印 PPL，再继续执行 `quantsim.export()`。当前没有自动精度门禁，是否可接受需要人工判断。

---

## 一、这段代码整体在做什么

对应代码：

```python
# compute activation encodings using AIMET
def _forward_fn_inputs_id(model, kwargs):
    data_loader = kwargs['data_loader']
    fpm = kwargs['fpm']
    max_iterations = kwargs['num_batches']
    for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
        if batch_id < max_iterations:
            slice_inputs_and_run_successive_kvcache_inference(
                fpm,
                input_ids=batch['input_ids'],
            )
        else:
            break


def _forward_fn(model, kwargs):
    data_loader = kwargs['data_loader']
    fpm = kwargs['fpm']
    max_iterations = kwargs['num_batches']
    for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
        if batch_id < max_iterations:
            slice_inputs_and_run_successive_kvcache_inference(
                fpm,
                input_embeds=batch['input_embeddings'],
            )
        else:
            break


if not llm_config.use_input_embeddings:
    _forward_fn = _forward_fn_inputs_id

kwargs = {
    'data_loader': train_dataloader,
    'fpm': sim_fpm,
    'num_batches': _eval_cfg['compute_encodings_num_batches'],
}

with sim_fpm.place_on_device("cuda"):
    quantsim.compute_encodings(_forward_fn, kwargs)
```

完整流程可以简化成：

```text
SeqMSE 已经固定部分 Weight Encoding
                   ↓
选择 input_ids 或 input_embeddings 校准回调
                   ↓
对前 20 个 calibration batch 执行前向
                   ↓
FPM 构造固定 Shape、Mask、RoPE 和 Past KV 输入
                   ↓
运行 QuantSim 模型前向
                   ↓
已启用且实际经过的 Activation Quantizer 观察张量统计
                   +
允许覆盖的 Parameter Quantizer 从参数张量计算／重算 Encoding
                   ↓
Activation Encoding + 可覆盖 Parameter Encoding 保存在 Quantizer 中
                   ↓
后续 PPL 验证和 quantsim.export()
```

这一步不是训练，也不是 PPL 评估。执行回调前向的主要目的，是触发 Activation Quantizer 对中间张量进行观察和统计；Parameter Encoding 不依赖这次前向，而是直接从参数张量计算。

---

## 二、先记住“根据什么、处理什么、得到什么”

| 问题 | 当前项目中的答案 |
|---|---|
| 根据什么 | Activation：前 20 个校准 Batch 产生的模型中间张量；Parameter：当前 Weight 等参数张量 |
| 处理什么 | 校准路径实际执行到且已启用的 Activation Quantizer；以及已启用、允许覆盖的 Parameter Quantizer |
| 怎样处理 | Activation 通过前向观察分布；Parameter 直接根据参数张量计算或重算；两者按 `post_training_tf` 等已有规则确定范围 |
| 得到什么 | `min/max`、`scale`、zero-point／AIMET Encoding `offset` 等量化参数 |
| 保存在哪里 | 首先写入 QuantSim 模型中对应的 Quantizer；导出时再写入 `.encodings` 文件 |

可以进一步简记成：

```text
代表性样本
    ↓
SeqMSE 已确定的 Weight Encoding 路径下执行 QuantSim 前向
    ↓
观察 Activation/KV 相关张量范围
    ↓
确定 Activation Encoding，供后续 QDQ 推理使用
```

注意：`compute_encodings()` 不负责决定哪些节点要量化成 A16、A8，也不负责决定对称性或 Quantizer 是否启用。这些结构和规则已经在 QuantSim 创建、HTP 配置、MatMul/Concat 规则和混合精度配置阶段确定，详见 [07-附录C · 量化规则配置](./07-附录C-量化规则配置-MatMul-Concat与混合精度.md)。

---

## 三、为什么定义两个前向回调

### 3.1 `input_ids` 路径

```python
slice_inputs_and_run_successive_kvcache_inference(
    fpm,
    input_ids=batch['input_ids'],
)
```

适用于 dataloader 提供 Token ID 的情况：

```text
input_ids
   ↓
模型内部 Embedding
   ↓
Transformer
```

### 3.2 `input_embeddings` 路径

```python
slice_inputs_and_run_successive_kvcache_inference(
    fpm,
    input_embeds=batch['input_embeddings'],
)
```

适用于视觉模型已经将文本 Token Embedding 与图像特征融合为输入 Embedding 的情况：

```text
文本 Embedding + 图像 Embedding
              ↓
       input_embeddings
              ↓
        LLM Transformer
```

### 3.3 当前实际选择哪一个

`example1/config.yaml`：

```yaml
model_overrides:
  use_input_embeddings: true
```

代码只有在该配置为 `false` 时才重新赋值：

```python
if not llm_config.use_input_embeddings:
    _forward_fn = _forward_fn_inputs_id
```

所以当前默认实际执行的是：

```text
_forward_fn
  → batch['input_embeddings']
  → slice_inputs_and_run_successive_kvcache_inference(...)
```

两个函数不会同时执行。

---

## 四、`kwargs`、`model` 和循环分别是什么

### 4.1 `kwargs` 不是模型输入字典

```python
kwargs = {
    'data_loader': train_dataloader,
    'fpm': sim_fpm,
    'num_batches': 20,
}
```

这里的 `kwargs` 是传给 AIMET 回调的参数包：

| 字段 | 作用 |
|---|---|
| `data_loader` | 提供代表性 calibration 数据 |
| `fpm` | 把 batch 转换成 prepared 定长模型需要的完整输入，并管理 KV Cache |
| `num_batches` | 限制本次标定最多处理多少个 Batch |

它不会直接执行：

```python
model(**kwargs)
```

### 4.2 `model` 参数为什么没有直接使用

AIMET 的回调接口会执行：

```text
forward_pass_callback(quantsim.model, callback_args)
```

所以函数签名必须接收：

```python
def _forward_fn(model, kwargs):
```

但当前 prepared LLM 不能直接用一个 Embedding 调用，还要由 FPM 准备固定 Shape、Mask、RoPE 和 KV Cache。因此项目改用：

```python
fpm = kwargs['fpm']
fpm(...)
```

QuantSim 创建时使用：

```python
QuantizationSimModel(
    model=sim_fpm.model,
    in_place=True,
)
```

所以 `sim_fpm.model` 已经被原地改造成 `quantsim.model` 对应的 QuantSim 模型。虽然回调没有直接使用 `model` 变量，实际前向仍经过同一套 Quantizer。

### 4.3 `tqdm(total=...)` 只负责进度显示

```python
for batch_id, batch in enumerate(
    tqdm(data_loader, total=max_iterations)
):
```

真正限制前向次数的是：

```python
if batch_id < max_iterations:
    ...
else:
    break
```

当前 `max_iterations=20`，实际执行 QuantSim 前向的 Batch 是：

```text
batch_id = 0, 1, ..., 19
```

如果 dataloader 少于 20 个 Batch，只会处理实际存在的 Batch。

---

## 五、`compute_encodings()` 内部发生了什么

核心调用：

```python
quantsim.compute_encodings(_forward_fn, kwargs)
```

根据 AIMET v2 的实现，可以概括为：

```text
接收 quantsim.model 和 callback_args
             ↓
临时把模型设为 eval 模式
             ↓
进入 torch.no_grad()
             ↓
进入各量化模块的 Encoding 计算上下文
             ↓
允许覆盖的 Parameter Quantizer
直接根据 Parameter Tensor 计算／重算 Encoding
             ↓
允许更新的 Activation Quantizer 进入观察／直通状态
             ↓
调用 _forward_fn(quantsim.model, kwargs)
             ↓
实际执行到的 Activation Quantizer 累积张量统计
             ↓
退出上下文并完成 Activation Encoding 初始化
```

这里必须把两类 Quantizer 分开：

- **Activation Quantizer** 依赖校准前向；没有执行到的分支就没有对应的真实激活统计。
- **Parameter Quantizer** 直接读取 Weight 等 Parameter Tensor，不依赖校准样本是否执行到该层；只要已启用且允许覆盖，即使原来已有 Encoding，也可能被重新计算。
- SeqMSE 对选中的 Weight Encoding 设置了不可覆盖状态，因此这些结果会被保留，不会在此处被普通 Min-Max Encoding 覆盖。

因此，项目回调外面虽然没有显式写：

```python
with torch.no_grad():
```

AIMET 的 `compute_encodings()` 本身会在 Eval 和 No-Grad 上下文中调用回调。

### 5.1 当前 `quant_scheme` 怎样使用统计结果

配置为：

```yaml
quant_scheme: post_training_tf
```

它属于 AIMET legacy TF-style Min-Max 方案：使用观察到的 Tensor 最小值和最大值计算 Encoding。这里的 `tf` 是历史方案名，不代表执行 TensorFlow。

概念上：

```text
Batch 0 观察范围：[-2.1, 2.5]
Batch 1 观察范围：[-3.0, 2.2]
Batch 2 观察范围：[-1.8, 3.4]
               ...
汇总观察范围并结合 bitwidth、对称性等已有配置
               ↓
计算最终 min/max、scale、zero-point／offset
```

具体公式、AIMET `offset` 与通用 zero-point 的符号区别，见 [07-附录B · Encoding 量化参数基础](./07-附录B-Encoding量化参数基础.md)。

### 5.2 为什么回调不需要返回 logits

`slice_inputs_and_run_successive_kvcache_inference()` 会返回 logits 和 Past KV，但 `_forward_fn` 没有接收返回值。

这是因为校准依赖的是前向副作用：

```text
中间 Tensor 经过 Quantizer
            ↓
Quantizer 更新内部统计信息
```

它不需要最终预测结果。因此这一阶段：

- 不读取 labels；
- 不计算 Cross Entropy；
- 不计算 MSE；
- 不计算 PPL；
- 不反向传播；
- 不更新原始模型权重。

### 5.3 校准过程中是否已经对 Activation 做 QDQ

按当前官方 AIMET v2 实现，`compute_encodings()` 的校准上下文会让已启用、允许更新的 Activation Quantizer 观察输入统计，并在这一阶段主要以直通方式传递 Activation；Parameter Quantizer 则先直接根据参数张量计算或重算允许覆盖的 Encoding。已经具有有效 Encoding 的 Weight Quantizer 可以用相应的模拟量化权重参与回调前向。退出上下文后，Activation Quantizer 才根据累计统计完成 Encoding 初始化。

因此可以概括为：

```text
SeqMSE 已选定的模拟量化 Weight
                ↓
产生该 Weight 路径下、尚未叠加上游 Activation QDQ 误差的张量
                ↓
Activation Observer 观察范围，校准时主要直通
                ↓
退出 compute_encodings 上下文
                ↓
确定 Activation Encoding，后续前向执行已启用 Quantizer 的 QDQ 模拟
```

这里的“确定”是指将 Encoding 保存到 Quantizer，供随后的 PPL 和导出使用；它不等于 SeqMSE 的 `allow_overwrite(False)` 冻结。若以后再次允许覆盖并调用 `compute_encodings()`，Activation Encoding 仍可能被重新标定。

本仓库没有固定记录 AIMET 的精确安装版本；上述描述对应当前官方 v2 行为，实际运行容器升级或降级 AIMET 后，应再次核对对应版本源码。

---

## 六、切块和连续 KV Cache 前向的设计

辅助函数：

```python
slice_inputs_and_run_successive_kvcache_inference(...)
```

设计目的，是在输入长度大于 `fpm.num_tokens` 时，把长序列按时间顺序拆成多个块，并把上一块产生的 Past KV 传给下一块。

核心逻辑：

```python
for idx in range(0, input_length, fpm.num_tokens)[::-1]:
    idx = input_length - idx
    cur_outputs = fpm(current_slice, past_key_values=...)
    kwargs['past_key_values'] = cur_outputs['past_key_values']
```

### 6.1 为什么从尾部计算切片，再按正序运行

当前配置：

```text
context_length = 2048
num_tokens / ARN = 1073
固定 Past KV 槽位 = 2048 - 1073 = 975
```

如果输入长度是完整的 2048，切块结果是：

```text
第 1 块：Token [0:975]       → 975 个有效 Token
第 2 块：Token [975:2048]    → 1073 个有效 Token
```

第一块产生长度 975 的真实 KV，正好可以作为第二块的 Past KV：

```text
第 1 块 975 Token
       ↓ 产生 975 长度 KV
第 2 块 1073 Token + 975 Past KV
       ↓
覆盖完整 2048 上下文
```

如果最前面的块短于 1073，FPM 会在左侧补 0，使 prepared 模型始终接收固定 Shape。

### 6.2 每次 `fpm(...)` 做什么

`LLMForwardPassManager.__call__()`：

```text
prepare_inputs()
    ↓
构造固定 Current Input、Attention Mask、RoPE、Past KV
    ↓
fpm.model(**prepared_inputs)
    ↓
prepare_outputs()
    ↓
裁掉 Dummy logits，提取并更新有效 Past KV
```

当前固定 Shape 的关键数值是：

| 输入 | 固定 Shape／长度含义 |
|---|---|
| Current Embedding | 当前为 `[1, 1073, 2048]`；不足 1073 时左补 0 |
| 每层 Past Key | `[1, 2, 128, 975]`，Key Cache 已转置 |
| 每层 Past Value | `[1, 2, 975, 128]` |
| Past KV 输入数量 | 36 层 × Key/Value = 72 个扁平输入张量 |
| 总 Attention 长度 | 2048 |
| Combined Mask | `[1, 1, 1073, 2048]` |

FPM 的详细 Padding、Mask、RoPE 和 Cache 更新逻辑，见 [05 · 通用前向处理流程](./05-通用前向处理流程.md)。

---

## 七、设计能力不等于当前一定执行了多块 KV 前向

这是当前代码最容易忽略的边界。

### 7.1 当前 VL 校准数据被固定成 ARN 长度

当前配置：

```yaml
use_input_embeddings: true
arn: 1073
```

`QwenDataset` 在 calibration 模式下，会把每个样本重复或截断到：

```text
emb_length = ARN = 1073
```

也就是说，当前每个 `batch['input_embeddings']` 的序列长度就是 1073。

### 7.2 当前每个 Batch 实际只有一次 FPM 前向

当：

```text
input_length = 1073
fpm.num_tokens = 1073
```

切块循环只产生一个块：

```text
当前 1073 Token
      +
FPM 构造的 975 个零 Padding Past KV
      ↓
一次固定 Shape 前向
```

因此，辅助函数虽然支持 successive KV Cache inference，但在当前 VL calibration 数据上：

- 每个 Batch 只调用模型一次；
- 不会在同一个 Batch 内进入第二个带真实历史 KV 的块；
- 每个新 Batch 都重新开始，不会继承上一个无关样本的 Cache；
- Past KV 输入由 FPM 构造的 975 长度零 Padding 组成。

它仍会观察当前 1073 Token 产生的新 K/V 以及实际执行到的 KV 相关算子，但不能声称当前校准已经覆盖了“非零真实 Past KV 输入”的分布。

### 7.3 什么情况下才会真正连续两次

例如传入长度 2048 的 `input_ids` 或 `input_embeddings`：

```text
975 Token → 产生真实 Past KV
                      ↓
1073 Token + 975 真实 Past KV → 第二次前向
```

因此，需要分别区分：

```text
函数能力：支持长序列切块并连续传递 KV
当前配置：VL calibration embedding 长度恰好为 1073，只跑一个块
```

两条分支可以直接对比：

| 分支 | 典型输入长度 | 每 Batch FPM 前向次数 | 是否把非零历史 KV 回喂给下一块 |
|---|---:|---:|---:|
| 当前 VL `input_embeddings` | 1073 | 1 | 否 |
| 备用 WikiText `input_ids` | 2048 | 2（975 + 1073） | 是 |

如果部署精度高度依赖真实非零 Past KV 分布，应使用能触发第二块的代表性校准输入，或增加专门的 Prefill/Decode/KV Cache 校准流程，并重新检查 PPL 和端侧结果。

---

## 八、最终哪些 Quantizer 会得到 Encoding

不能简单说“模型里的每一个 Quantizer 都一定得到 Encoding”，因为 Activation 与 Parameter 的计算条件不同。

### 8.1 Activation Quantizer

Activation Quantizer 要获得或更新 Encoding，通常需要同时满足：

```text
Quantizer 已启用
      +
允许初始化／覆盖
      +
校准前向实际执行到该路径
```

可能包括：

- 普通算子输入／输出 Activation Quantizer；
- MatMul、Concat 等规则保留下来的 Activation Quantizer；
- K/V Projection、Cache 拼接和 Attention 路径上实际启用的 Quantizer；
- 模型输入或输出 Quantizer，但前提是配置启用。

KV Cache 在 Encoding 文件中通常仍属于 Activation Encoding，而不是单独生成第三个 `kv_cache_encodings` 文件。

### 8.2 Parameter Quantizer

Parameter Quantizer 的规则不同：它可以直接根据 Weight 等 Parameter Tensor 计算 Encoding，不要求 calibration batch 在前向中执行到该模块。对于已启用且允许覆盖的 Parameter Quantizer，`compute_encodings()` 不仅会补齐未初始化项，也可能重算已有 Encoding。

当前流程中可分成两类：

- SeqMSE 已选中并设置 `allow_overwrite(False)` 的 Weight Encoding：保持不变；
- 其他已启用且允许覆盖的 Parameter Quantizer：根据 Parameter Tensor 计算或重算 Encoding。

所以源码注释：

```python
# compute activation encodings using AIMET
```

是对主要目的的简写，不代表 API 绝对只处理 Activation。

### 8.3 已禁用、共享和未经过的 Quantizer

- 被配置为 `None` 或关闭的 Quantizer 不会独立标定；
- 通过 Encoding 传播或共享规则绑定的 Quantizer，不一定各自产生独立 Encoding；
- 校准数据没有走到的条件分支，无法获得该分支 Activation 的真实统计；这条路径条件不适用于直接读取参数张量的 Parameter Encoding；
- 手工混合精度规则是否真正命中，需要审计实际 prepared 模块名称。

具体规则和当前 `exceptions.json` 的命中风险，见 [07-附录C · 量化规则配置](./07-附录C-量化规则配置-MatMul-Concat与混合精度.md)。

---

## 九、为什么不能在 SeqMSE 前向中顺便完成

技术上可以让一个更高级的流程在 SeqMSE 结束后立即启动 Calibration，但不能把 SeqMSE 搜索过程中的中间统计直接当成最终 Activation Encoding。

### 9.1 SeqMSE 会临时关闭 Activation Quantizer

AIMET SeqMSE 的算法流程会临时关闭 Input/Output Activation Quantizer，也会临时移除不支持或被排除模块的 Parameter Quantizer；受支持的 Weight Quantizer 随后还会经过适用性筛选。搜索结束后再恢复原配置。

所以 SeqMSE 的回调不是 Activation Quantizer 的正常观察上下文。

### 9.2 Weight Encoding 候选正在变化

SeqMSE 不是为每个候选完整跑一遍全模型。它先捕获当前目标层的浮点输入 `x_fp` 和量化路径输入 `x_q`；当前 `inp_symmetry="symqt"` 使用 `x_q` 作为参考和候选两边的共同输入，然后在该层局部尝试 Weight Encoding。假设某层尝试 20 组候选：

```text
y_ref = layer(x_q, W)

y1  = layer(x_q, Q1(W))
y2  = layer(x_q, Q2(W))
y3  = layer(x_q, Q3(W))
...
y20 = layer(x_q, Q20(W))
```

这些 `y1 ... y20` 用来和浮点参考层输出比较局部重构误差，并不等于 20 组全模型 Activation。概念上，如果在候选不断切换时强行让全局 Activation Observer 同时累计，统计就会混入多组临时 Weight 状态：

```text
Activation 统计 = 候选 Q1 路径 + Q2 路径 + ... + Q20 路径
```

但最终可能只选择 `Q7(W)`。正确做法是：

```text
先选定并冻结 Q7(W) 的 Encoding
      ↓
重新前向
      ↓
观察选定 Weight 路径下的 Activation
```

Activation 和 KV Cache 都由最终有效权重参与计算，因此应在 Weight Encoding 确定后标定。

### 9.3 两次回调的执行目标不同

SeqMSE 回调：

```python
input_embeddings=inputs['input_embeddings'][
    :, :min(input_length, fpm.num_tokens), ...
]
fpm.model(**prepared_inputs)
```

它最多取一个 `num_tokens` 块，主要用于 AIMET Hook 捕获目标层输入；逐层采样时还可能在捕获目标层后提前终止当前前向。

`compute_encodings()` 回调则调用通用切块 helper，目标是让 SeqMSE 完成后的 QuantSim 模型沿校准路径运行，使实际经过且已启用、允许更新的 Activation Quantizer 观察输入；Parameter Encoding 按第八节所述直接从参数张量计算。

即使当前 VL 数据长度导致 helper 也只运行一个块，两者的 Quantizer 状态和 Weight 状态仍然不同，不能直接合并统计。

### 9.4 正确顺序

```text
SeqMSE
  → 确定并冻结支持层 Weight Encoding

compute_encodings
  → 在选定的 Weight Encoding 路径下标定 Activation
  → 计算／重算允许覆盖的 Parameter Encoding

PPL
  → 验证整体量化误差
```

SeqMSE 的候选搜索、MSE 和 `inp_symmetry` 细节见 [07-附录D · SeqMSE 权重量化优化](./07-附录D-SeqMSE权重量化优化.md)。

---

## 十、SeqMSE 与 `compute_encodings()` 对比

| 对比项 | SeqMSE | `compute_encodings()` |
|---|---|---|
| 主要目标 | 优化受支持层的 Weight Encoding | 标定 Activation，并计算／重算允许覆盖的 Parameter Encoding |
| 主要模型 | 浮点参考路径 + QuantSim 路径 | SeqMSE 完成后的 QuantSim 模型 |
| Activation Quantizer | 搜索期间临时关闭 | 已启用、允许更新且实际执行到者进入观察状态 |
| Parameter Quantizer | 对受支持目标层搜索候选 Encoding | 直接根据参数张量计算／重算；SeqMSE 冻结者保持不变 |
| Weight 状态 | 候选 Encoding 正在变化 | SeqMSE 最佳 Encoding 已经选定并冻结 |
| 判断依据 | 层输出 MSE/L1/SQNR 等配置损失 | 当前 `post_training_tf` 下的观察 min/max |
| 前向目的 | 捕获目标层输入并测试候选 | 让 Quantizer 观察代表性张量分布 |
| 是否使用 labels | 否 | 否 |
| 是否反向传播 | 否 | 否 |
| 最终结果 | 最佳 Weight Encoding | Activation Encoding + 可覆盖的 Parameter Encoding |

两者都使用真实 calibration 数据，但并不是重复工作。

---

## 十一、这是静态量化还是动态量化

当前流程属于静态 PTQ：

```text
离线运行 calibration 数据
          ↓
计算 Activation/KV 相关 Encoding
          ↓
推理前确定并导出 Encoding
```

Activation 和 KV Cache 的数值虽然在运行时动态产生，但如果推理期间一直使用校准后固定的 Encoding，就不是严格意义上的动态量化。

此处“固定”描述的是部署推理时重复使用离线 Encoding，而不是说 `compute_encodings()` 自动把 Quantizer 设置成永久不可覆盖。

只有推理时根据当前 Batch、Token 或 KV Cache 重新计算量化参数，才属于动态 Activation/KV 量化。

完整的量化方法分类见 [07-附录F · 量化方法总览与选型](./07-附录F-量化方法总览与选型.md)。

---

## 十二、标定完成后结果保存在哪里

### 12.1 `compute_encodings()` 刚结束

Encoding 首先保存在内存中的 QuantSim Quantizer：

```text
quantsim.model
  └─ Quantized Module
       ├─ input_quantizers
       ├─ output_quantizers
       └─ param_quantizers
```

这一步还没有写出最终 ONNX，也没有生成真实端侧整数模型。

### 12.2 后续 PPL 验证

代码紧接着运行量化模拟模型的 PPL：

```python
sim_ppl = ppl_eval_embedding(
    test_dataloader,
    sim_fpm,
    num_batches=_eval_cfg['ppl_num_batches'],
)
```

此时 QuantSim 使用已经确定的 Weight 和 Activation Encoding 执行 QDQ 模拟，以检查整体精度损失。

### 12.3 导出

后续：

```python
quantsim.export(
    onnx_dir,
    model_name,
    sample_inputs,
    onnx_export_args=onnx_api_args,
)
```

才会把结果写入：

```text
ONNX 模型
+ .encodings 文件
```

最终导出的有效 Encoding 会包含：

- SeqMSE 已冻结的 Weight Encoding；
- `compute_encodings()` 得到且成功初始化、可导出的 Activation Encoding；
- 允许覆盖并成功计算、可导出的其他 Parameter Encoding；
- bitwidth、dtype、对称性、粒度等导出描述。

QuantSim、QDQ 和真实端侧整数执行的区别见 [07-附录A · QuantSim 模型骨架与 QDQ](./07-附录A-QuantSim模型骨架与QDQ.md)。

`quantsim.export()` 的完整输入、输出文件和 Test Vector 对拍方法，见 [08 · ONNX 导出与测试向量](./08-ONNX导出与测试向量.md)。

---

## 十三、当前配置汇总

| 配置 | 当前值 | 对标定的影响 |
|---|---:|---|
| `quant_scheme` | `post_training_tf` | 使用观察到的 Tensor min/max 确定基础 Encoding |
| `default_output_bw` | 16 | 为配置中启用的 Activation Quantizer 设置默认 16 bit |
| `default_param_bw` | 4 | 默认 Weight 4 bit；受规则和例外覆盖 |
| `compute_encodings_num_batches` | 20 | 最多执行 20 个校准 Batch |
| `use_input_embeddings` | `true` | 当前选择 Embedding 回调 |
| `context_length` | 2048 | 固定总 Attention 上下文长度 |
| `ARN / num_tokens` | 1073 | 单次 Current Input 固定长度 |
| 固定 Past KV 槽位 | 975 | `2048 - 1073` |
| VL calibration 样本长度 | 1073 | 当前每个 Batch 只触发一次 FPM 前向 |

默认 W4/A16 只是基础规格。HTP 配置、Quantizer 禁用／共享、MatMul/Concat 规则和手工例外都可能覆盖局部行为，不能理解为全图每个 Weight 都是 W4、每个 Activation 都有独立 A16 Quantizer。

---

## 十四、代码审查时值得注意的细节

### 14.1 当前只对前 20 个执行前向，而且不打乱

`train_dataloader` 使用 `shuffle=False`，因此标定使用数据文件顺序中的前 20 个 Batch。Encoding 质量取决于这些样本能否代表真实部署分布。

增加 Batch 数量不一定总能提高精度；关键是覆盖文本长度、图像内容、异常激活和真实使用场景。

### 14.2 `if batch_id < max_iterations` 可能多取一个 Batch

Python 会先从 dataloader 取出 `batch_id=20` 的 Batch，再进入：

```python
else:
    break
```

它不会执行第 21 次 QuantSim 前向，但可能已经产生一次额外的数据加载／图像预处理开销。可以用 `itertools.islice(data_loader, max_iterations)` 避免。

### 14.3 dataloader 的 Mask 和 Position 没有传入 helper

当前回调只传：

```python
input_embeds=batch['input_embeddings']
```

没有显式传入 `batch['attention_mask']` 或 `position_ids`。当前 VL calibration 数据构造出的 `attention_mask` 全为 1，且 `use_mrope=false`，因此现有路径会由 FPM 自行构造 Mask 和普通 RoPE。

如果以后启用 MRoPE，当前回调没有传入 `position`，而 `prepare_inputs()` 会访问 `kwargs['position']`，可能直接触发 `KeyError`；复杂 Padding Mask 或特殊位置输入也需要把相应字段同步传给 helper。

### 14.4 helper 会构造并拼接 logits，但校准不使用

`slice_inputs_and_run_successive_kvcache_inference()` 是通用推理辅助函数，会把各块 logits 拼接起来。Calibration 只需要 Quantizer 统计和下一块 Past KV，最终 logits 被丢弃。

当前 `vocab_size=151936` 时，一个 FP32 logits 张量 `[1, 1073, 151936]` 约为 621.9 MiB；即使只有一个块，`torch.cat()` 也可能再分配并复制整块数据，显存影响已经很显著。若未来使用多块长序列，累计拼接还会继续放大开销。专门的 calibration helper 可以丢弃无用 logits，只保留下一块需要的 KV 状态。

### 14.5 `model` 参数未使用形成隐式耦合

当前正确性依赖：

```text
kwargs['fpm'].model 与 AIMET 传入的 quantsim.model 是同一个 QuantSim 模型
```

当前代码依赖两者确实指向同一个量化模型；`in_place=True` 是这一关系成立的重要前提，但最好增加身份断言做运行时确认，或让回调显式使用 AIMET 传入的模型并交给 FPM。

### 14.6 `_forward_fn` 名称被重复定义

SeqMSE 阶段和 `compute_encodings()` 阶段各自重新定义了一组同名 `_forward_fn`。由于 Python 按顺序执行，后面的定义发生在 SeqMSE 已经完成之后，当前流程不会因此调用错函数；但阅读和调试时容易混淆。

更清晰的命名可以是：

```text
_seqmse_forward_fn
_calibration_forward_fn
```

### 14.7 helper 不是任意长度的无限流式推理器

FPM 围绕 `context_length=2048`、Current 1073 和 Past KV 975 的固定图设计。它会裁剪 Cache，并对输入与 KV 长度做校验；超过 2048 Token 后，第三块前传入的 Past KV 可能大于允许的 975，从而在 `validate_inputs()` 中失败。因此它不是完善的无限滑动窗口实现，不能因为函数名包含 `successive` 就假设它能无条件处理任意超长上下文。

---

## 十五、哪些东西发生了变化

### 发生变化的

- 实际执行到且已启用、允许更新的 Activation Quantizer 获得或更新 Encoding；
- 已启用且允许覆盖的 Parameter Quantizer 根据参数张量获得或更新 Encoding；
- SeqMSE 已禁止覆盖的 Weight Encoding 保持不变；
- QuantSim 模型可以使用这些 Encoding 执行稳定的 QDQ 模拟；
- 后续导出文件具备 Activation 和 Parameter 量化描述。

### 没有发生变化的

- 原始浮点 Weight 数值本身；
- Transformer 网络结构；
- Tokenizer；
- dataloader 样本内容；
- 模型不会因为校准执行反向传播；
- 尚未直接获得打包后的 INT4/INT8 端侧模型；
- 尚未证明 PPL 和端侧性能满足要求。

---

## 十六、常见误解

### 16.1 “`compute_encodings()` 只处理 Activation”

不完全正确。Activation 是回调前向的主要标定对象；此外，API 还会直接根据参数张量计算或重算已启用且允许覆盖的 Parameter Encoding。SeqMSE 已禁止覆盖的 Weight Encoding 会被保留。

### 16.2 “模型里的每个 Quantizer 都一定获得 Encoding”

不一定。Activation Quantizer 还必须允许更新并在校准前向中实际执行到；Parameter Quantizer 则不依赖前向路径，而取决于是否启用、是否允许覆盖。禁用或共享 Quantizer 的情况也不同。

### 16.3 “20 个 Batch 就等于 20 次模型 forward”

不一定。一个 Batch 如果长于 `num_tokens`，helper 可能切成多个块并执行多次 FPM 前向；不过当前 VL calibration 样本长度正好为 1073，所以当前是每 Batch 一次。

### 16.4 “函数名有 successive KV，就一定标定了真实 Past KV”

不是。当前 Embedding 样本长度等于单块 ARN，只执行一次，Past KV 主要是零 Padding。只有输入长于 1073 并进入第二块，才会把上一块生成的真实 KV 作为下一块输入。

### 16.5 “不使用 logits，就没有真正跑模型”

不是。模型完整前向已经执行，Quantizer 也在中间张量流过时完成统计；只是最终 logits 对校准目标没有用途。

### 16.6 “跑前向标定就是动态量化”

不是。这里是离线前向后固定 Encoding，仍属于静态 PTQ。

### 16.7 “`compute_encodings()` 会重新训练或修改 Weight”

不会。它不计算训练损失和梯度，也不改写原始 Weight Tensor；它更新的是 Quantizer Encoding，SeqMSE 已设置为不可覆盖的 Weight Encoding 会保持不变。

### 16.8 “校准完成就一定有精度和速度收益”

不一定。还要用 PPL／任务指标检查精度，并经过 ONNX 导出、QNN 编译和目标 HTP/NPU 实测延迟、内存及算子回退。

---

## 十七、面试速答

### Q：这段代码主要做什么？

> 它调用 AIMET `compute_encodings()`，在 Eval 和 No-Grad 模式下用代表性数据运行 SeqMSE 完成后的 QuantSim 模型，让实际执行到且已启用、允许更新的 Activation Quantizer 观察张量分布；同时直接从参数张量计算或重算允许覆盖的 Parameter Encoding。

### Q：为什么需要自定义 `_forward_fn`？

> AIMET 不知道当前 prepared Qwen 模型怎样构造定长 Embedding、Attention Mask、RoPE 和 Past KV，也不知道怎样切分长序列，所以项目通过 FPM 和切块 helper 完成模型专用前向。

### Q：为什么不用最终 logits？

> `compute_encodings()` 不使用标签或最终预测，而是依赖前向观察副作用；当中间 Tensor 流过已启用、允许更新且正处于观察上下文的 Activation Quantizer 时，统计才会发生。

### Q：为什么要放在 SeqMSE 后面？

> SeqMSE 期间 Activation Quantizer 被临时关闭，而且 Weight Encoding 候选不断变化。只有先选定最终 Weight Encoding，再重新前向，才能观察该 Weight 路径下、尚未叠加上游 Activation QDQ 误差的稳定统计。

### Q：当前是否真的做了多轮连续 KV Cache 标定？

> helper 具备长序列切块和 KV 传递能力，但当前 `use_input_embeddings=true`，VL calibration 样本又被固定为 `ARN=1073`，等于单次 `num_tokens`，所以每个 Batch 只前向一次，未进入带真实非零 Past KV 的第二块。

### Q：标定结果保存在哪里？

> 先保存在 QuantSim 的 Activation／Parameter Quantizer 中；后续 `quantsim.export()` 才把 Weight 和 Activation Encoding 一起写入 `.encodings` 文件。

---

## 十八、参考资料

- [AIMET · QuantizationSimModel.compute_encodings API](https://quic.github.io/aimet-pages/releases/latest/apiref/torch/quantsim.html)
- [AIMET · PyTorch QuantSim `compute_encodings()` 源码](https://quic.github.io/aimet-pages/releases/latest/_modules/aimet_torch/quantsim/quantsim.html)
- [AIMET · Post Training Quantization / Calibration](https://quic.github.io/aimet-pages/releases/latest/techniques/ptq.html)
- [AIMET · Sequential MSE](https://quic.github.io/aimet-pages/releases/2.13.0/ptq_techniques/seq_mse.html)
- [AIMET · Encoding Format Specification](https://quic.github.io/aimet-pages/releases/latest/techniques/encoding_spec.html)

---

## 十九、一句话总结

> **`compute_encodings()` 在 SeqMSE 选定并冻结目标 Weight Encoding 后，用代表性数据运行 QuantSim 路径，为实际执行到且已启用、允许更新的 Activation Quantizer 统计并确定 Encoding；同时直接从参数张量计算或重算允许覆盖的 Parameter Encoding。当前 VL 样本长度等于 ARN=1073，因此每 Batch 只执行一个带 975 个零 Past-KV 槽位的固定 Shape 块。**
