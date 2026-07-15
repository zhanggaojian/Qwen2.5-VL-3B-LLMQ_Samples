# 附录 · Attention 注意力机制

> **关联**：笔记 [02-模型适配(Monkey-Patch).md](./02-模型适配(Monkey-Patch).md) 里第 1 个被替换的零件 `QcAttention` 的原理出处。
> **前置地基**：先搞懂张量三维含义 → [02-附录C-张量维度(B,seq,hidden).md](./02-附录C-张量维度(B,seq,hidden).md)。
> **配套概念**：本篇的两个定语"**自**注意力"和"**自回归**"单独成篇 → [02-附录D-自回归与自注意力.md](./02-附录D-自回归与自注意力.md)。
> **横向分类**：Attention 有哪些种类（自/交叉、因果/双向、MHA/MQA/GQA/MLA、FlashAttention…）面试向整理 → [02-附录F-Attention分类大全(面试向).md](./02-附录F-Attention分类大全(面试向).md)。
> **一句话本质**：Attention 让序列里**每个词去"看"其他词、按相关性加权汇总信息**，从而理解上下文。它是 Transformer 的核心。

> **本篇按四段式组织**（全笔记统一风格）：**① 介绍/为什么 → ② 原理 → ③ 官方 Qwen2 做法 → ④ 本项目改造后做法**。

---

## 一、介绍：Attention 是什么 & 为什么要用

### 1.1 Attention 是做什么的？（直觉）

一句话：**让每个词根据"和其他词的相关程度"，有重点地从别的词那里收集信息。**

例子：句子 "猫累了，因为**它**睡了一天"。
- 模型读到"它"时，需要知道"它"指代谁。
- Attention 会让"它"对"猫"的关注度很高、对"睡"较低 → 于是"它"的表示里融入了"猫"的信息。

这种"按相关性加权汇总"的能力，就是 Transformer 能理解长距离上下文的关键。

---

## 二、原理

> 本部分把 Attention 的机制拆透：三件套 Q/K/V、投影层、缩放、多头/GQA，以及位置编码(RoPE)与 KV Cache 的简述（后两者各有专篇）。

### 2.1 核心三件套：Q、K、V

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

### 2.2 `q/k/v/o_proj` 是什么 + 三种张量别混淆

#### proj = projection（投影）

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

#### 三种张量别混淆（很重要）

| 名称 | 是什么 | 形状 | 和 B/seq 有关吗 |
|------|--------|------|-----------------|
| ① 投影权重 `W_q/W_k/W_v/W_o` | **可学习参数** | `[out, in]`（与 batch/seq 无关） | ❌ |
| ② Q/K/V 张量 | 输入投影后的**激活值** | `[B, seq, hidden]` | ✅ |
| ③ 注意力分数 `attn_weights` | `softmax(QK^T)` 的结果 | `[B, heads, seq, seq]` | ✅ |

口语说的"QKV 权重和输入一样形状"，其实指的是 ② 激活，不是 ① 可学习权重。

#### 为什么投影权重常说是 `hidden × hidden`？

Linear 权重 = `[out, in]`，`in = hidden`（输入是 hidden 维向量）；`out` 是设计出来的 = `num_heads × head_dim`。
**设计上让 `num_heads × head_dim = hidden`**（如 16 头 × 128 = 2048 = hidden），所以 `q_proj`/`o_proj` 是 `[hidden, hidden]`。两个原因：
1. **切分而非增维**：多头是把 hidden 切成 num_heads 段各算各的，拼回来还是 hidden（否则计算量翻 num_heads 倍）。
2. **残差对齐**：`o_proj` 输出必须是 hidden，才能和输入做 `x + Attn(x)`。

> **例外**：Qwen2 用 GQA，`k_proj/v_proj` 的 out = `num_kv_heads × head_dim` < hidden，所以 **K/V 权重比 hidden×hidden 小**（见下一节 GQA）。

### 2.3 为什么除以 $\sqrt{d}$（缩放点积）

代码：`attn_weights = matmul(Q, K^T) / math.sqrt(self.head_dim)`，**`d` 就是 `head_dim`，直接开根号**。

- **为什么是 head_dim 不是 hidden**：点积 $QK^T$ 是在每个头内部、沿 head_dim 维求和的：$\text{score} = \sum_{k=1}^{head\_dim} q_k k_k$，所以缩放只跟 head_dim 有关。（如 head_dim=128 → 分母 $\sqrt{128}≈11.3$）
- **为什么要除**：若 q、k 分量均值0方差1，head_dim 个乘积相加 → **方差≈head_dim，标准差≈$\sqrt{head\_dim}$**。head_dim 越大点积越大，送进 softmax 会**过度尖锐→饱和→梯度消失**。除以 $\sqrt{head\_dim}$ 正好把方差拉回约 1，softmax 不饱和、梯度健康。
- 这就是原论文叫 **Scaled Dot-Product Attention（缩放点积注意力）** 的由来。

> 项目 `QcAttention` 有个 `advance_attention_div`：把 $/\sqrt{d}$ 提前作用到 K 上，**数学等价**，仅为端侧算子排布；默认行为与官方一致（详见第四节）。

### 2.4 多头注意力（Multi-Head）与 GQA

#### 多头
不是只做一次 Attention，而是把特征切成 `num_heads` 份，**每个头独立做一次**，最后拼起来。好处：不同的头可以关注不同种类的关系（语法、指代、位置……）。

#### ⚠️ 两个常见误解（重要）

**误解1：`num_heads` 是"多个独立的注意力模型"？**
→ 不是。头**不是**各自完整、互不相关的模型，而是**同一层注意力被切成的多个并行子空间**。它们共享同一份输入 `hidden_states`，只是各自在一个更小的 `head_dim` 维子空间里算注意力，最后拼回来。

```
hidden = num_heads × head_dim      例：2048 = 16头 × 128
"16 个头" ≠ "16 个模型"，而是"把 2048 维注意力切成 16 份各算 128 维"
```

**误解2：每个头是不是各有一套独立的 Q/K/V 投影层？**
→ 每个头**确实用自己那一段 Q/K/V**，但不是每头一个独立投影层。真实流程是：

```
① 整层用一个大投影 q_proj/k_proj/v_proj，把 hidden_states 投影成完整 Q/K/V  [B, seq, hidden]
② 把 hidden 这一维【切成 num_heads 段】→ 每个头分到互不重叠的一段(各 head_dim 维)
③ 每个头用自己那段 Q/K/V 独立算一次注意力
④ 各头结果拼回 [B, seq, hidden]，再过 o_proj
```

> 数学上，"一个大投影后切成 N 段" 与 "N 个小投影各算各的" **完全等价**（大矩阵可看成 N 个头的小矩阵拼起来）。所以"每个头有自己的 Q/K/V"概念上没错，只是代码里合成一个大矩阵一次算完更高效。
>
> 多头也**不增加计算量**：`num_heads × head_dim = hidden`，总量和单头一样，只是切开让不同头关注不同类型的关联。

#### 多头的形状变换（输入要 reshape + transpose）

多头**需要对 Q/K/V 张量做形状变换**，但只是"切开 hidden 再换轴"，不增减数据。前提：`hidden = num_heads × head_dim`。

```python
# 第1步 view：把 hidden 拆成 (heads, head_dim)
[B, seq, hidden] → [B, seq, num_heads, head_dim]
# 第2步 transpose(1,2)：把 head 维提到前面
[B, seq, num_heads, head_dim] → [B, num_heads, seq, head_dim]
```
对应源码第 261-263 行的 `.view(...).transpose(1, 2)`。

**为什么 transpose 成 `[B, heads, seq, head_dim]`**：这样能把每个头当成独立的 `[seq, head_dim]` 矩阵并行算 $QK^T$，得到分数 `[B, heads, seq, seq]`。

#### GQA（分组查询注意力）
Qwen2 用了 GQA：**Q 的头数多，K/V 的头数少**，多个 Q 头共享同一组 K/V，省显存和计算。代码里：
- `num_heads`：Q 的头数
- `num_key_value_heads`：K/V 的头数（更少）
- `num_key_value_groups = num_heads // num_key_value_heads`：每组共享数
- `repeat_kv(...)`：把少量 K/V 头"复制"到和 Q 头一样多，再做 matmul。

> 正因 K/V 头少，`k_proj/v_proj` 输出维度 = `num_kv_heads × head_dim` < hidden，权重比 Q 小，KV Cache 也更小。

### 2.5 RoPE：位置信息怎么进来的（简述）

> **想从零搞懂（为什么要位置编码 → 绝对编码 → RoPE → 官方/改造做法）** → [02-附录G-RoPE位置编码.md](./02-附录G-RoPE位置编码.md)。

Attention 本身不区分词的顺序（打乱也一样）。**RoPE（旋转位置编码）** 通过对 Q、K 做"旋转"，把位置信息注入进去。代码里：
- `self.rotary_emb(...)` 算出 `cos, sin`
- `apply_rotary_pos_emb(query, key, cos, sin, ...)` 把旋转作用到 Q、K 上

> 本项目端侧版支持把 cos/sin **从外部输入**（`use_position_embedding_input`），见 QcAttention 里的 `_apply_rope_single`（详见第四节 / [附录G](./02-附录G-RoPE位置编码.md)）。

### 2.6 KV Cache：为什么推理要缓存（简述）

自回归生成时一个词一个词蹦，每生成新词都要和**前面所有词**算 Attention。若每次重算所有历史 K/V 太浪费，于是把历史 K/V **缓存**起来，新词只算自己的、拼接上历史即可。
- 代码里 `past_key_value.update(...)` 就是在更新这个缓存。
- 本项目对它做了端侧改写（定长/只存新值/转置），见笔记 02 第 4 节。

---

## 三、官方 Qwen2 的做法（源码逐段对照，`modeling_qwen2.py` 第 245-316 行）

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

## 四、本项目改造后的做法（官方 vs QcAttention 逐段对照）

`QcAttention` 继承官方 `Qwen2Attention`，但把 `forward` 换成了 `forward_conv`（并备份原版为 `forward_no_conv`）。**注意力的数学骨架没变**（还是第三节那 10 步），变的全是"用什么算子、数据从哪来、数值怎么稳"这些**端侧 + 量化工程适配**。

### 4.0 先看概览

| # | 阶段 | 官方 `Qwen2Attention` | QcAttention | 性质 |
|---|------|----------------------|-------------|------|
| 0 | 权重定义 | `q/k/v/o_proj` = `nn.Linear` | `*_conv` = `nn.Conv2d(...,1)`（`prepare_conv` 搬权重后删 Linear）| 🔧改 |
| 1 | 投影 Q/K/V | Linear 直接算 | 先摆成 NCHW 再 1×1 Conv | 🔧改 |
| 2 | 拆多头 | `view+transpose(1,2)` | `reshape+transpose(2,3)`（因 Conv 输出布局不同）| 🔧改 |
| 3 | RoPE | 内部 `rotary_emb` 现算 | **支持外部传入 cos/sin** | 🔧半改 |
| 4 | KV Cache | 标准 `update` | 先转置 K + 传 `return_new_key_value_only`/`transposed` 开关 | 🔧改 |
| 5 | GQA `repeat_kv` | ✅ | ✅ 一样 | ⬜不变 |
| 6 | 打分 `QKᵀ/√d` | matmul 后除 √d | 可**提前把 √d 除到 K**、转置存时**免运行时转置** | 🔧改 |
| 7 | 加掩码 | `+ mask`（切片）| **量化感知 `Add` + 逐层掩码缩放**（layer0×2 / layer27×10）| 🔧改 |
| 8 | softmax+加权V | softmax→**dropout**→@V | softmax→@V（**去掉 dropout**）| 🔧微改 |
| 9 | 输出投影 | reshape→Linear | transpose→Conv→transpose | 🔧改 |
| 10 | return | ✅ | ✅ 一样 | ⬜不变 |

### 4.1 逐段代码对照

**⓪ 继承官方类：先完整复用，再做工程替换**

`QcAttention` 不是从零重写参数定义，而是继承 `Qwen2Attention`：

```62:66:example1/llm_utils/qcqwen2_adaptation.py
class QcAttention(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_add = Add()
```

`super().__init__()` 会先创建官方的 `q/k/v/o_proj`、`rotary_emb`，并初始化 `num_heads`、`num_key_value_heads`、`head_dim`、`layer_idx` 等属性。这样可以直接加载官方权重，再由 `prepare_conv()` 做等价转换，不需要重新训练模型。

`self.attn_add = Add()` 数学上仍是 `attn_weights + attention_mask`，但它把普通的 `+` 表达成 AIMET 能识别和插入量化节点的显式模块，便于统计量化范围及生成量化图。

**① 权重定义：Linear → 1×1 Conv2d（`prepare_conv` 里搬权重）**

官方在 `__init__` 里就是四个 Linear：

```234:237:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
```

QcAttention 用 `prepare_conv` 新建同规格 Conv2d、把权重 `[out,in]→[out,in,1,1]` 搬过去、再删掉原 Linear：

```71:91:example1/llm_utils/qcqwen2_adaptation.py
            self.q_proj_conv = nn.Conv2d(self.hidden_size, self.num_heads * self.head_dim, 1, bias=True)
            ...
            self.forward_no_conv = self.forward
            self.forward = self.forward_conv
            self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
            ...
            del self.q_proj
```

这里还有两个容易忽略的机制：

1. `if not hasattr(self, "forward_no_conv")` 是**幂等保护**：第一次调用才转换；转换后该属性已存在，再调用不会重复创建 Conv，也不会再次删除 Linear。
2. `self.forward_no_conv = self.forward` 先备份官方 `forward`，然后 `self.forward = self.forward_conv` 动态切换入口。上层 Transformer 无需修改调用代码，之后执行 `attention(...)` 时会自动进入 `forward_conv()`。

权重搬运的形状变化为：

```text
Linear.weight: [out_channels, in_channels]
                         ↓ 增加两个长度为 1 的维度
Conv2d.weight: [out_channels, in_channels, 1, 1]
```

> **区别**：算子从 Linear 变成 1×1 Conv（数学等价），权重从原 Linear 原样扩维拷贝，不需要重新训练；最后删除原 Linear，避免模型同时保留两套权重。为什么转 Conv → 见 [附录B](./02-附录B-Linear与Conv算子转换.md)。

**② + ③ 投影 & 拆多头：多了 NCHW 摆形状**

官方：Linear 投影 → `view + transpose(1,2)` 拆头：

```257:263:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        query_states = self.q_proj(hidden_states)
        ...
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
```

QcAttention：先把 `[b,seq,hidden]` 拧成 Conv 要的 `[b,hidden,1,seq]`，Conv 完再 `reshape + transpose(2,3)` 拆头：

```109:122:example1/llm_utils/qcqwen2_adaptation.py
        hidden_states = torch.reshape(hidden_states, (bsz, q_len, 1, self.hidden_size)).transpose(1, 3)
        query_states = self.q_proj_conv(hidden_states)
        ...
        query_states = query_states.reshape(bsz, self.num_heads, self.head_dim, q_len).transpose(2, 3)
```

> **区别**：Conv 要求 NCHW 布局，所以进出各多一组 reshape/transpose；拆头的轴也因此不同（官方 `transpose(1,2)`，Qc 是 `reshape(...,head_dim,q_len).transpose(2,3)`），**最终形状同为 `[b,head,seq,head_dim]`**。

完整形状链如下：

```text
hidden_states [B, S, hidden]
  → reshape     [B, S, 1, hidden]
  → transpose   [B, hidden, 1, S]       # Conv2d 的 NCHW
  → 1×1 Conv    [B, heads×D, 1, S]
  → reshape     [B, heads, D, S]
  → transpose   [B, heads, S, D]        # 回到标准 Attention 布局
```

这里的卷积核只有 `1×1`，所以每个序列位置只读取该位置的全部 hidden 通道，**不会跨 token 混合信息**；它仍等价于对每个 token 独立执行同一个 Linear。真正让不同 token 交换信息的是后面的 `QKᵀ` 和对 V 的加权汇总。

**③ RoPE：支持外部传入 cos/sin**

官方只在内部现算：

```274:275:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
```

QcAttention：`position_ids` 若是 `(cos,sin)` 元组就**直接用外部预计算的**，否则回退官方：

```124:131:example1/llm_utils/qcqwen2_adaptation.py
        if isinstance(position_ids, (tuple, list)): # QC
            cos, sin = position_ids
            query_states = _apply_rope_single(query_states, position_ids)
            key_states = _apply_rope_single(key_states, position_ids)
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(...)
```

> **区别**：把 cos/sin 挪到外部预计算并供各层复用，主要为了避开端侧不友好的三角函数/动态频率逻辑、让量化图更干净并减少重复计算，**不是实现定长的必要条件**；固定输入长度后，官方内部计算 RoPE 也可以定长导出。详见 [附录G](./02-附录G-RoPE位置编码.md)。

**④ KV Cache：存 K 前先转置 + 传开关**

官方直接 update：

```277:279:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

QcAttention：按开关先 `transpose(2,3)` 把 K 转成 `[..,head_dim,seq]`，并把 `return_new_key_value_only`/`transposed_key_cache` 通过 `cache_kwargs` 传给**改写过的** `update`：

```133:143:example1/llm_utils/qcqwen2_adaptation.py
        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)
        if past_key_value is not None:
            cache_kwargs = {..., "return_new_key_value_only": ..., "transposed_key_cache": ...}
            key_states, value_states = past_key_value.update(...)
```

> **区别**：配合定长/只存新/转置存的 KV 缓存。详见 [附录K](./02-附录K-KV%20Cache(键值缓存).md)。

**⑥ 打分：√d 缩放可提前 + 转置存免转置**

官方一行搞定（matmul 后除 √d，且要临时转置 K）：

```285:285:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
```

QcAttention 拆成四种组合：`advance_attention_div` 决定 √d 除在 K 上（提前，见 118-121 行）还是这里；`transposed_key_cache` 决定要不要 `key.transpose`：

```148:157:example1/llm_utils/qcqwen2_adaptation.py
        if advance_attention_div:
            if transposed_key_cache:
                attn_weights = torch.matmul(query_states, key_states)          # 已提前除、且免转置
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        else:
            ... / math.sqrt(self.head_dim)
```

> **区别**：(a) 把 `1/√d` 提前折进 K，稳住 attn_weights 数值范围利于量化；(b) 转置存的 K 直接 matmul，省一次运行时转置。

**⑦ 加掩码：量化感知 Add + 逐层缩放**

官方：切片对齐后普通相加：

```293:295:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
```

QcAttention：大部分层用 `aimet_torch` 的量化感知 `self.attn_add`，但对**数值范围异常的特定层单独放大掩码**（layer0×2、layer27×10）：

```162:173:example1/llm_utils/qcqwen2_adaptation.py
            if self.layer_idx == 0:
                attn_weights = attn_weights + (attention_mask * 2)
            elif self.layer_idx == 27:
                attn_weights = attn_weights + (attention_mask * 10)
            else:
                attn_weights = self.attn_add(attn_weights, attention_mask)
```

> **区别**：(a) 用可被量化工具识别的 `Add` 算子；(b) 特定层（layer0、layer27）的 attn_weights min/max 过大（注释 `too huge minmax`），放大掩码保证被屏蔽位置仍被有效压到极小——经验性量化调优。⚠️ 本模型共 **36 层**（0~35），layer0 是首层、**layer27 是中间偏后的某一层（并非最后一层，最后是 layer35）**，只是它实测数值异常才被特殊处理。另外官方会 `[:, :, :, :key.shape[-2]]` 切片，Qc 因定长外部掩码不切。

**⑧ softmax + 加权 V：去掉 dropout**

官方 softmax 后还有一步 dropout：

```297:300:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
```

QcAttention 去掉了 dropout（推理/量化不需要）：

```192:193:example1/llm_utils/qcqwen2_adaptation.py
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
```

> **区别**：省掉推理期无用的 dropout；softmax 仍在 fp32 上算再转回（这点一致）。

**⑨ 输出投影：Linear → Conv（+摆形状）**

官方：拼头 → Linear：

```308:311:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
```

QcAttention：拼头 → 摆 NCHW → Conv → 摆回：

```201:207:example1/llm_utils/qcqwen2_adaptation.py
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, 1, self.hidden_size)
        attn_output = attn_output.transpose(1, 3)
        attn_output = self.o_proj_conv(attn_output)
        attn_output = attn_output.transpose(1, 3)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
```

> **区别**：o_proj 同样 Linear→Conv，进出各多一组 transpose 摆 NCHW。

### 4.2 一句话结论

**算法骨架不变**——第三节那 10 步一步不少、GQA/softmax/加权汇总全一致；改的是围绕它的 4 类工程适配：**① 全部投影 Linear→1×1 Conv（+ NCHW 摆形状）② RoPE 预计算、掩码输入化、KV 历史外部管理（其中定长主要由固定输入及掩码/KV 机制保证）③ 量化数值适配（√d 可提前、逐层掩码缩放、量化感知 Add）④ 去掉推理无用的 dropout**。

---

## 五、记忆锚点

- Attention = **Q 找、K 配、V 取**，softmax 加权汇总。
- 公式：$\text{softmax}(QK^T/\sqrt{d})\,V$。
- 多头=多个视角；GQA=K/V 头更少省资源；RoPE=注入位置；KV Cache=推理提速。
- 项目里的 QcAttention 只改"算子和数据通路"，**核心算法骨架和官方一致**（逐段对照见第四节）：投影 Linear→1×1 Conv、RoPE 预计算、掩码/KV 外部化、√d 可提前 + 逐层掩码缩放 + 量化感知 Add、去 dropout。

---

## 六、待深入（自己往下填）

- [ ] `repeat_kv` 具体怎么把 K/V 头复制对齐的？看一下实现。
- [x] RoPE 的"旋转"在数学上是怎么编码位置的？→ 见 [附录G](./02-附录G-RoPE位置编码.md)。
- [ ] 为什么 QcAttention 第 0 层 `mask*2`、第 27 层 `mask*10`？这些倍数怎么定的？
- [ ] softmax 为什么要 upcast 到 fp32 再转回来？
