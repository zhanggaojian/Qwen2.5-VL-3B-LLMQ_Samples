# 附录 · Attention 注意力机制

> **关联**：笔记 [02-模型适配(Monkey-Patch).md](./02-模型适配(Monkey-Patch).md) 里第 1 个被替换的零件 `QcAttention` 的原理出处。
> **前置地基**：先搞懂张量三维含义 → [02-附录C-张量维度(B,seq,hidden).md](./02-附录C-张量维度(B,seq,hidden).md)。
> **配套概念**：本篇的两个定语"**自**注意力"和"**自回归**"单独成篇 → [02-附录D-自回归与自注意力.md](./02-附录D-自回归与自注意力.md)。
> **一句话本质**：Attention 让序列里**每个词去"看"其他词、按相关性加权汇总信息**，从而理解上下文。它是 Transformer 的核心。

---

## 一、Attention 是做什么的？（直觉）

一句话：**让每个词根据"和其他词的相关程度"，有重点地从别的词那里收集信息。**

例子：句子 "猫累了，因为**它**睡了一天"。
- 模型读到"它"时，需要知道"它"指代谁。
- Attention 会让"它"对"猫"的关注度很高、对"睡"较低 → 于是"它"的表示里融入了"猫"的信息。

这种"按相关性加权汇总"的能力，就是 Transformer 能理解长距离上下文的关键。

---

## 二、核心三件套：Q、K、V

每个词的向量会被**三个不同的线性层**投影成三个角色：

| 名称 | 全称 | 角色比喻 |
|------|------|----------|
| Q (Query) | 查询 | 我**想找**什么样的信息 |
| K (Key) | 键 | 我**能提供**什么样的信息（用于被匹配） |
| V (Value) | 值 | 我**实际携带**的信息内容 |

计算分三步：

1. **算相关性**：用 Q 和每个 K 做点积 → 得到"这个词该关注其他词多少"的分数。
2. **归一化**：除以 $\sqrt{d}$ 再做 softmax → 分数变成和为 1 的权重。
3. **加权汇总**：用这些权重对 V 加权求和 → 得到该词融合上下文后的新表示。

公式（"Attention is All You Need"的经典式子）：

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V$$

- $QK^T$：相关性矩阵（`seq × seq`），第 i 行=第 i 个词对所有词的关注分。
- $\sqrt{d}$：缩放，防止点积太大导致 softmax 梯度消失（`d`=head_dim）。
- softmax：把每行变成概率（和为 1）。
- 乘 V：按权重把信息汇总。

> **因果掩码**就加在 softmax 之前的分数上，把"未来位置"置 -∞，详见笔记 02 第 2 节。

---

## 二·补A · `q/k/v/o_proj` 是什么 + 三种张量别混淆

### proj = projection（投影）

`proj` = 投影，就是用一个矩阵做一次线性变换（矩阵乘法）。四个投影层：

| 名称 | 全称 | 含义 |
|------|------|------|
| `q_proj` | query projection | 把输入**投影成 Query** 的 Linear 层 |
| `k_proj` | key projection | 把输入**投影成 Key** 的 Linear 层 |
| `v_proj` | value projection | 把输入**投影成 Value** 的 Linear 层 |
| `o_proj` | **output** projection | 注意力算完后，把结果**投影回 hidden** 的 Linear 层（`o`=output） |

- 前三个在注意力**开始前**，把输入拆成 Q/K/V 三个角色。
- `o_proj` 在注意力**结束后**，融合多头结果并变回 hidden 维（好和残差 `x + Attn(x)` 相加）。

⚠️ 关键：**`q_proj` 是"层/权重"本身，不是结果。** `q_proj(hidden_states)` 的调用结果才是 `query_states`。
```
hidden_states ──q_proj(权重)──▶ query_states (Q)
   输入            投影层           结果
# 即 query_states = hidden_states × W_q^T + b
```

### 三种张量别混淆（很重要）

| 名称 | 是什么 | 形状 | 和 B/seq 有关吗 |
|------|--------|------|-----------------|
| ① 投影权重 `W_q/W_k/W_v/W_o` | **可学习参数** | `[out, in]`（与 batch/seq 无关） | ❌ |
| ② Q/K/V 张量 | 输入投影后的**激活值** | `[B, seq, hidden]` | ✅ |
| ③ 注意力分数 `attn_weights` | `softmax(QK^T)` 的结果 | `[B, heads, seq, seq]` | ✅ |

口语说的"QKV 权重和输入一样形状"，其实指的是 ② 激活，不是 ① 可学习权重。

### 为什么投影权重常说是 `hidden × hidden`？

Linear 权重 = `[out, in]`，`in = hidden`（输入是 hidden 维向量）；`out` 是设计出来的 = `num_heads × head_dim`。
**设计上让 `num_heads × head_dim = hidden`**（如 16 头 × 128 = 2048 = hidden），所以 `q_proj`/`o_proj` 是 `[hidden, hidden]`。两个原因：
1. **切分而非增维**：多头是把 hidden 切成 num_heads 段各算各的，拼回来还是 hidden（否则计算量翻 num_heads 倍）。
2. **残差对齐**：`o_proj` 输出必须是 hidden，才能和输入做 `x + Attn(x)`。

> **例外**：Qwen2 用 GQA，`k_proj/v_proj` 的 out = `num_kv_heads × head_dim` < hidden，所以 **K/V 权重比 hidden×hidden 小**（见下一节 GQA）。

---

## 二·补B · 为什么除以 $\sqrt{d}$（缩放点积）

代码：`attn_weights = matmul(Q, K^T) / math.sqrt(self.head_dim)`，**`d` 就是 `head_dim`，直接开根号**。

- **为什么是 head_dim 不是 hidden**：点积 $QK^T$ 是在每个头内部、沿 head_dim 维求和的：$\text{score} = \sum_{k=1}^{head\_dim} q_k k_k$，所以缩放只跟 head_dim 有关。（如 head_dim=128 → 分母 $\sqrt{128}≈11.3$）
- **为什么要除**：若 q、k 分量均值0方差1，head_dim 个乘积相加 → **方差≈head_dim，标准差≈$\sqrt{head\_dim}$**。head_dim 越大点积越大，送进 softmax 会**过度尖锐→饱和→梯度消失**。除以 $\sqrt{head\_dim}$ 正好把方差拉回约 1，softmax 不饱和、梯度健康。
- 这就是原论文叫 **Scaled Dot-Product Attention（缩放点积注意力）** 的由来。

> 项目 `QcAttention` 有个 `advance_attention_div`：把 $/\sqrt{d}$ 提前作用到 K 上，**数学等价**，仅为端侧算子排布；默认行为与官方一致。

---

## 三、多头注意力（Multi-Head）与 GQA

### 多头
不是只做一次 Attention，而是把特征切成 `num_heads` 份，**每个头独立做一次**，最后拼起来。好处：不同的头可以关注不同种类的关系（语法、指代、位置……）。

### 多头的形状变换（输入要 reshape + transpose）

多头**需要对 Q/K/V 张量做形状变换**，但只是"切开 hidden 再换轴"，不增减数据。前提：`hidden = num_heads × head_dim`。

```python
# 第1步 view：把 hidden 拆成 (heads, head_dim)
[B, seq, hidden] → [B, seq, num_heads, head_dim]
# 第2步 transpose(1,2)：把 head 维提到前面
[B, seq, num_heads, head_dim] → [B, num_heads, seq, head_dim]
```
对应源码第 261-263 行的 `.view(...).transpose(1, 2)`。

**为什么 transpose 成 `[B, heads, seq, head_dim]`**：这样能把每个头当成独立的 `[seq, head_dim]` 矩阵并行算 $QK^T$，得到分数 `[B, heads, seq, seq]`。

### GQA（分组查询注意力）
Qwen2 用了 GQA：**Q 的头数多，K/V 的头数少**，多个 Q 头共享同一组 K/V，省显存和计算。代码里：
- `num_heads`：Q 的头数
- `num_key_value_heads`：K/V 的头数（更少）
- `num_key_value_groups = num_heads // num_key_value_heads`：每组共享数
- `repeat_kv(...)`：把少量 K/V 头"复制"到和 Q 头一样多，再做 matmul。

> 正因 K/V 头少，`k_proj/v_proj` 输出维度 = `num_kv_heads × head_dim` < hidden，权重比 Q 小，KV Cache 也更小。

---

## 四、RoPE：位置信息怎么进来的

Attention 本身不区分词的顺序（打乱也一样）。**RoPE（旋转位置编码）** 通过对 Q、K 做"旋转"，把位置信息注入进去。代码里：
- `self.rotary_emb(...)` 算出 `cos, sin`
- `apply_rotary_pos_emb(query, key, cos, sin, ...)` 把旋转作用到 Q、K 上

> 本项目端侧版支持把 cos/sin **从外部输入**（`use_position_embedding_input`），见 QcAttention 里的 `_apply_rope_single`。

---

## 五、KV Cache：为什么推理要缓存

自回归生成时一个词一个词蹦，每生成新词都要和**前面所有词**算 Attention。若每次重算所有历史 K/V 太浪费，于是把历史 K/V **缓存**起来，新词只算自己的、拼接上历史即可。
- 代码里 `past_key_value.update(...)` 就是在更新这个缓存。
- 本项目对它做了端侧改写（定长/只存新值/转置），见笔记 02 第 4 节。

---

## 六、官方源码逐段对照（`modeling_qwen2.py` 第 245-316 行）

```python
def forward(self, hidden_states, attention_mask, position_ids, past_key_value, ...):
    bsz, q_len, _ = hidden_states.size()

    # 1. 投影出 Q/K/V（三个 Linear）
    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # 2. 拆成多头：[B, seq, hidden] → [B, heads, seq, head_dim]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 3. 加位置编码 RoPE
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # 4. 更新 KV Cache（推理时拼接历史）
    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 5. GQA：把少的 K/V 头复制到和 Q 头一样多
    key_states   = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 6. 算相关性分数 QK^T / sqrt(d)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # 7. 加因果掩码（屏蔽未来）
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]

    # 8. softmax 归一化（升到 fp32 算更稳）
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # 9. 加权汇总 V
    attn_output = torch.matmul(attn_weights, value_states)

    # 10. 多头拼回 + 输出投影 o_proj
    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value
```

这 10 步就是标准 Attention 的完整流程，**任何 Transformer 模型都大同小异**，记住这条线即可。

---

## 七、本项目的 QcAttention 改了什么（对照笔记 02）

`QcAttention` 继承官方 `Qwen2Attention`，主线逻辑不变，主要差异：

| 改动 | 官方 | QcAttention | 为什么 |
|------|------|-------------|--------|
| 投影层 | `q/k/v/o_proj` 是 Linear | 换成 1×1 Conv2d | HTP 硬件对 Conv 更友好（见 [02-附录B-Linear与Conv算子转换.md](./02-附录B-Linear与Conv算子转换.md)） |
| 加掩码 | `attn_weights + mask` | 用 `aimet_torch` 的 `Add()`，且部分层 `mask*2 / *10` | 让量化工具能识别该算子 + 经验性防溢出 |
| RoPE | 内部算 cos/sin | 支持外部传入（`_apply_rope_single`） | 适配端侧定长导出 |
| KV Cache | 标准 DynamicCache | 定长/只存新值/转置 | 适配端侧 KVCache 模式 |
| 掩码生成 | 动态构造因果掩码 | 外部输入（bypass） | 利于导出固定计算图 |

> 结论：**算法没变，是同一个 Attention**；变的全是"用什么算子、数据从哪来"这些**端侧工程适配**。

---

## 八、记忆锚点

- Attention = **Q 找、K 配、V 取**，softmax 加权汇总。
- 公式：$\text{softmax}(QK^T/\sqrt{d})\,V$。
- 多头=多个视角；GQA=K/V 头更少省资源；RoPE=注入位置；KV Cache=推理提速。
- 项目里的 QcAttention 只改"算子和数据通路"，**核心算法和官方一致**。

---

## 九、待深入（自己往下填）

- [ ] `repeat_kv` 具体怎么把 K/V 头复制对齐的？看一下实现。
- [ ] RoPE 的"旋转"在数学上是怎么编码位置的？（复数/旋转矩阵视角）
- [ ] 为什么 QcAttention 第 0 层 `mask*2`、第 27 层 `mask*10`？这些倍数怎么定的？
- [ ] softmax 为什么要 upcast 到 fp32 再转回来？
