# 04 · 附录A · PPL 面试速答

> **定位**：本篇只保留 PPL 与固定 shape 前向的面试结论。完整原理见 [04 · PPL 困惑度评估](./04-PPL困惑度评估.md)，FPM 内部细节见 [05 · 通用前向处理流程](./05-通用前向处理流程.md)。

## 一、30 秒总答

PPL 用来衡量模型预测真实下一个 token 时有多不确定，计算公式是 `PPL = exp(平均交叉熵)`，越低越好。本项目为了适配端侧固定 shape 模型，先由外层函数把真实序列按 ARN 分块，再由 `LLMForwardPassManager` 将每块输入、Mask、RoPE 和历史 KV 整形成固定 shape，逐块前向并拼回有效 logits；最后 PPL 函数将第 `t` 个位置的 logits 与第 `t+1` 个真实 token 对齐，计算交叉熵并取指数。

```text
真实变长序列
    │
    ▼
切分函数：按 ARN 分块、循环、传 KV、拼 logits
    │
    ▼
FPM：单块补齐固定 shape → 模型前向 → 去除 dummy
    │
    ▼
ppl_eval：shift 对齐 → CrossEntropyLoss → exp → PPL
```

---

## 二、高频面试速答

### 1. PPL 是什么？

PPL（困惑度）衡量模型预测真实下一个 token 时有多不确定：

```text
PPL = exp(所有有效 token 的平均交叉熵)
```

PPL 越低越好；量化后 PPL 相比浮点基线涨得越少，说明精度保持得越好。

### 2. 为什么 PPL 使用交叉熵？

因为 PPL 的定义就是平均负对数似然的指数，而语言模型上的平均负对数似然就是交叉熵。它不是随意挑选的一种评估函数。

### 3. `LLMForwardPassManager` 是做什么的？

FPM 负责**一次固定 shape 前向**：

- 保存真实 `input_length`；
- 当前输入不足 ARN 时左补 dummy；
- 从传入的 `past_key_values` 读取真实历史长度；
- 将进入本次模型的历史 KV 补齐到固定 KV 槽；
- 准备 Mask、RoPE 和模型输入；
- 调用模型后，裁掉 dummy logits/KV，返回有效结果。

FPM 是类，不是负责整条长序列循环的函数。

### 4. 谁负责切分和循环前向？

独立函数 `slice_inputs_and_run_successive_kvcache_inference()` 负责：

```text
长序列 → 按 ARN 分块 → 逐块调用 FPM
       → 上一块 KV 传给下一块 → 拼接有效 logits
```

因此：**切分函数管循环，FPM 管单块。** 这种分块是连续 prefill，不是一次生成一个 token 的 decode。

### 5. 当前固定 shape 和 KV 长度是什么关系？

当前配置：

```text
最大上下文 Context = 2048
当前输入槽 ARN      = 1073
固定历史 KV 槽      = 2048 - 1073 = 975
```

传入单次 FPM 的真实历史 KV 来自前一块的有效缓存，并且必须满足本次接口的长度上限；FPM 读取其实际长度，再用全零 KV 补齐到固定 975 槽。模型执行后，FPM 去掉 dummy，并把有效旧 KV 与本轮新 KV 整理成输出缓存。历史 KV 不是仅根据输入长度凭空计算出来的。

例如真实序列长度正好为 2048 时，当前切分逻辑先处理余数 975 个 token，再处理 1073 个 token；第二块携带第一块产生的 975 个真实 KV。

### 6. 模型返回的是 score、概率还是 logits？

模型返回的是**未归一化的全词表 logits**。FPM 不显式计算 Softmax，只负责裁出真实输入位置对应的 logits。

但不能说整个 PPL 过程“不计算 Softmax”：`CrossEntropyLoss` 内部等价于：

```text
log_softmax + NLLLoss
```

因此每个位置的整个词表 logits 都会影响 loss，不是只读取真实 token 的一个 score。

### 7. logits 和 labels 为什么要错开一位？

语言模型用当前位置预测下一个 token：

```text
shift_logits = logits[..., :-1, :]
shift_labels = input_ids[..., 1:]
```

也就是第 `t` 个位置的全词表 logits，对应第 `t+1` 个真实 token ID。

### 8. PPL 评估是生成过程吗？

不是。PPL 使用真实 token 作为上下文和答案，这叫 teacher forcing。它不会采样模型输出，也不会把预测 token 再喂回模型，所以是“测量真实文本概率”，不是自回归生成。

### 9. `ppl_eval` 和 `ppl_eval_embedding` 有什么区别？

| 函数 | 模型输入 | 交叉熵标签 |
|---|---|---|
| `ppl_eval` | token ID | 真实 `input_ids` |
| `ppl_eval_embedding` | 多模态 input embedding | 对应的真实 token ID / labels |

两者的分块、shift、交叉熵和 `exp` 逻辑相同。

### 10. 本项目为什么在三个阶段都测 PPL？

```text
端侧适配后的浮点模型 → 建立基线
Prepare 后浮点模型    → 检查图重写是否等价
QuantSim 模型         → 检查量化精度损失
```

三次评估必须尽量使用相同数据和前处理，差值才有意义。

---

## 三、当前实现的统计注意点

当前代码先计算每个 batch 的平均交叉熵，再对 batch loss 求平均，最后取 `exp`：

```text
PPL = exp(mean(batch_mean_loss))
```

当各 batch 有效 token 数相同时，这与全 token 平均一致；若长度差异较大，严格做法应按每个 batch 的有效 token 数加权。

---

## 四、面试易错句纠正

| 容易说错 | 更准确的说法 |
|---|---|
| FPM 负责切完整序列并循环 | 外层切分函数负责循环，FPM 负责一次固定 shape 前向 |
| 历史 KV 长度由公式直接算出 | 真实长度来自传入 KV；公式只确定固定输入槽，FPM 再补齐 dummy |
| 模型输出的是概率 | 模型输出原始 logits |
| PPL 完全不计算 Softmax | 没有显式调用；CrossEntropyLoss 内部包含 log-softmax |
| 交叉熵只取真实 token 的 score | 真实 token 是 target，但整个词表 logits 都参与归一化 |
| 分块使用 KV 就是在 decode | PPL 使用真实序列 teacher forcing，分块是连续 prefill |
| loss 就是 PPL | `PPL = exp(平均 loss)` |

---

## 五、推荐面试表述

> 外层函数把真实序列按 ARN 分块；`LLMForwardPassManager` 将每块真实输入、Mask、RoPE 和历史 KV 整形成模型要求的固定 shape，逐块前向并更新 KV，最后拼接每个真实位置的原始 logits；PPL 函数再用第 `t` 个位置的全词表 logits 预测第 `t+1` 个真实 token，计算交叉熵并对平均 loss 取指数，得到 PPL。

## 六、一句话总结

> **切分函数负责“分块循环”，FPM 负责“固定 shape 前向”，PPL 函数负责“真实下一个 token 的交叉熵取指数”。**
