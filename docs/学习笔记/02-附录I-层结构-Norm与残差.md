# 附录 · 层结构：Norm 与残差（Transformer 骨架）

> **关联**：把 [Attention（附录A）](./02-附录A-Attention注意力机制.md) 和 [MLP（附录H）](./02-附录H-MLP前馈网络.md) 这两个核心零件"连起来"的骨架，就是这篇讲的 Norm + 残差。结构全貌见 [总结篇](./02-模型适配总结篇-结构与替换全景.md)。
> **前置地基**：懂张量三维 → [附录C](./02-附录C-张量维度(B,seq,hidden).md)。
> **一句话本质**：一层 = `x → norm → 子层(Attn/MLP) → 加回残差`（Pre-Norm）；**norm 稳住数值分布，残差给梯度留一条高速公路**，两者是把网络堆到几十层还能训得动的关键。

> **本篇按四段式组织**（全笔记统一风格）：**① 介绍/为什么 → ② 原理 → ③ 官方 Qwen2 做法 → ④ 本项目改造后做法**。（Norm/残差是通用骨架，本项目未做结构性改造，故 ④ 说明"不涉及"。）

---

## 一、介绍：为什么需要 Norm 与残差、一层骨架长啥样

每层 `Qwen2DecoderLayer` 就是两个"归一化 + 子层 + 残差"的三明治叠在一起：

```
x ─┬────────────────────────────────┐
   └─ input_layernorm → Attention ──(+)──┬───────────────────────────────┐
                                          └─ post_attention_layernorm → MLP ──(+)── 输出
   每个虚线框都是：先 norm、再算子、再把进来的原值加回去（残差）
```

对应官方代码，一层就干这 4 步 × 2：

```758:778:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            ...
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
```

**为什么需要这两样**：深层网络里数值会飘、梯度会消失。**norm 管"数值稳不稳"，残差管"梯度传不传得回去"**——两者是深层 Transformer 能训练的地基。下面 ② 分别讲透它们的原理，③ 看官方如何把它们拼成完整的一层/整模型。

---

## 二、原理

### 2.1 RMSNorm

#### 归一化在干嘛

深层网络里，每层的输出数值分布会飘（有的通道特别大、有的特别小），越叠越不稳，训练容易发散。**归一化(Normalization)** 就是每层把向量"拉回一个稳定的尺度"，让后续计算和梯度都更稳。

#### RMSNorm 的公式与代码

RMSNorm（Root Mean Square Norm，均方根归一化）的做法：**用向量自身的均方根(RMS)把它缩放到单位尺度，再乘一个可学习的权重**。

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}\;\odot\; w
$$

对照官方实现，一行行都能对上：

```92:97:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

- `pow(2).mean(-1)`：对**最后一维（hidden 那一维）**求平方的均值 = 均方(mean square)。
- `rsqrt(variance + eps)`：开根号取倒数 = 除以 RMS；`eps` 防止除零。
- `self.weight * ...`：乘可学习的缩放向量 `w`（初始化为全 1，见 `__init__` 里 `torch.ones`）。

> 注意它是**在每个 token 的 hidden 维内部做归一化**（`mean(-1)`），token 之间互不影响——和 MLP"逐 token 独立"是一致的（见 [附录H 2.4](./02-附录H-MLP前馈网络.md)）。也注意它内部**临时转 float32** 算，算完再转回原 dtype，保证数值稳定。

#### RMSNorm vs LayerNorm：为什么大模型用 RMSNorm

老牌的 **LayerNorm** 要做两件事：先**减均值**（中心化）、再**除标准差**，还带一个偏置 `b`：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\cdot w + b
$$

**RMSNorm 砍掉了"减均值"和偏置 `b`**，只保留"除以 RMS + 乘 w"。好处：

| | LayerNorm | RMSNorm |
|---|-----------|---------|
| 减均值(中心化) | 有 | **无** |
| 除以 | 标准差 | **均方根(RMS)** |
| 偏置 b | 有 | **无** |
| 计算量 | 稍大 | **更少（少一次求均值和减法）** |
| 效果 | 好 | **相当，甚至更稳** |

实践发现"减均值"这步对大模型效果影响很小，去掉后**更快、更省**，效果几乎不变，所以 LLaMA、Qwen 等都改用 RMSNorm。

#### 均方根(RMS) vs 方差/标准差，到底差在哪

上表的关键分水岭是"除以 RMS 还是除以标准差"，而这俩的本质区别只有一句：**减不减均值**。设向量 $d$ 个数、均值 $\mu$：

$$
\text{RMS} = \sqrt{\tfrac{1}{d}\sum x_i^2}\qquad
\text{方差} = \tfrac{1}{d}\sum (x_i-\mu)^2 \qquad
\text{标准差} = \sqrt{\text{方差}}
$$

- **RMS**：直接对**原始值**平方求平均再开根 → 量"这些数**离 0** 有多远"，即**整体幅度**。
- **方差/标准差**：先**减掉均值 μ**，量"偏离均值的量" → 量"这些数**离自己平均值**有多散"，即**离散程度**。

一句话：**RMS 量"离原点 0 的距离"，标准差量"离均值 μ 的距离"。**

#### （承上）精确关系

$$
\text{RMS}^2 = \text{方差} + \mu^2
$$

- **μ = 0 时**：RMS = 标准差，完全相等；
- **μ ≠ 0 时**：RMS 比标准差大（多了个 $\mu^2$），因为它把"整体偏移"也算进幅度。

#### （承上）数字例子

取 `x = [3, 5]`，μ = 4：

```
RMS   = √((3²+5²)/2) = √17 ≈ 4.12     ← 不减均值，离 0 挺远
方差  = ((3-4)²+(5-4)²)/2 = 1
标准差 = √1 = 1                        ← 减了均值，其实彼此很集中
验证：RMS²=17 = 方差1 + μ²16 ✅
```

同一组数，RMS≈4.12、标准差=1，差别全来自 μ=4 那个整体偏移。

#### （承上）几何直觉

- **标准差平移不变**：所有数 +100，散布没变 → 标准差不变。
- **RMS 平移敏感**：所有数 +100 → 离 0 更远 → RMS 暴涨。

所以 RMSNorm 赌的是"**减均值这步对大模型没啥用**"，于是砍掉它只除 RMS——省一次求均值和减法（更快），代价是不再中心化、只把**整体幅度**拉到统一尺度。

### 2.2 Pre-Norm：norm 放在子层"前面"

#### 位置：norm 不在 Attention/MLP 里面，在它们前面

每层有**两个** RMSNorm，都放在子模块**之前**：

- `input_layernorm`：Attention **之前**；
- `post_attention_layernorm`：MLP **之前**（名字带 "post_attention" 是相对 Attention 而言，作用是给 MLP 做前置归一化）。

`self_attn(...)` 和 `self.mlp(...)` **内部本身不含 norm**，它们拿到的都是已经 norm 过的输入。

#### Pre-Norm vs Post-Norm

| | Post-Norm（原始 Transformer）| Pre-Norm（Qwen/LLaMA）|
|---|---|---|
| 公式 | `x = norm(x + 子层(x))` | `x = x + 子层(norm(x))` |
| norm 位置 | 残差相加**之后** | 子层**之前**、残差路径之外 |
| 训练稳定性 | 深层易梯度爆炸/消失，需要 warmup 等技巧 | **更稳**，容易堆到几十上百层 |
| 现代大模型 | 少用 | **主流** |

关键区别：**Pre-Norm 里那条残差主干上"没有 norm 挡着"**（`x + 子层(norm(x))`，加号左边的 `x` 是干净原值），梯度能一路畅通地回传——这正是下一节残差要讲的"高速公路"。

### 2.3 残差：输出 = 输入 + 子层(输入)

#### 是什么

代码里就是这三行的模式（Attention 和 MLP 各一组）：

```python
residual = hidden_states                     # ① 存下进来时的原值
hidden_states = 子层(norm(hidden_states))     # ② 走 Attention / MLP
hidden_states = residual + hidden_states      # ③ 把原值加回来
```

即 **输出 = 输入 + 子层(输入)**，子层学的是一个"**增量 Δ**"，而不是从零重造整个向量。

#### 为什么要有残差（三个核心理由）

1. **给梯度留一条高速公路（最关键）。** 反向传播时，加法 `+` 的梯度是 1，梯度能**原封不动地沿残差主干直达前面的层**。没有残差，几十层梯度连乘会指数级衰减（梯度消失），深层根本训不动。残差是能把网络堆到几十上百层的**前提**。

2. **每层只需学"微调"，更好优化。** 子层任务从"输出完整新表示"降级成"在原表示上加点修正"。哪怕某层没用，只要它输出 ≈ 0，`residual + 0` 也能把信息**无损透传**，不会帮倒忙。

3. **信息不丢失。** Attention/MLP 是有损加工，残差保证原始信息始终有一条旁路保留，加工结果只是叠加上去的补充。

> 直觉类比：残差像"**改稿**"——不是每层都从白纸重写，而是在上一版基础上批注修改；改错了大不了改动为 0，原稿还在。

#### 深挖第 1 点：为什么"加法的梯度是 1"、梯度能不衰减地直达前层

第 1 点最难，这里把它彻底推开。先澄清两个前提：

- **前向传"值"，反向传"梯度"** 是两码事：前向从头到尾算结果（传的是 hidden 向量）；反向从尾到头传"该怎么改参数"的信号（梯度）。残差的加号在两个方向各有用，别混。
- **"系数/梯度"通俗说就是斜率**：输入变一点点，输出变多少倍。梯度回传时每过一层就乘这层的系数；系数 < 1 就会越传越小（梯度消失）。

#### （承上）为什么残差层的系数是 `1 + f'(x)`

一层有残差时是 `输出 = x + f(x)`（f 是 Attention/MLP）。让 x 增加一个极小量 Δ，输出变化分两条路：

```
① "x" 这条路：  x 变 Δ  →  直接让输出变 Δ        （斜率 = 1）
② "f(x)" 这条路：x 变 Δ  →  f(x) 变 f'(x)·Δ      （斜率 = f'(x)）
────────────────────────────────────────────────
合计：输出变化 = (1 + f'(x))·Δ    →  系数 = 1 + f'(x)
```

那个 **`1` 的来源**：`输出 = x + f(x)` 对 x 求导，`x` 这一项是 **`d(x)/dx = 1`**——因为 **x 是变量（不是常数）**，它对自己求导恒等于 1（若是常数才会是 0）。这个 1 谁也拿不走。

#### （承上）为什么 `+1` 就等于"梯度原封不动直达前层"

把系数乘开，回传梯度拆成**两条相加的路**：

```
传回的梯度 = 上游梯度 × (1 + f'(x))
          = 上游梯度 × 1   +   上游梯度 × f'(x)
            └─ 直连路：原样通过 ─┘  └─ 加工路：可能变小 ─┘
```

- `× 1` 那项 = 上游梯度本身、一点没变（"原封不动"）；哪怕 `f'(x) ≈ 0`，这项也照样送满。
- 多层串起来时，连乘 `(1+f'₃)(1+f'₂)(1+f'₁)` 展开，必有一项是 `1×1×1 = 1`——这是一条**只走加号直连、绕开所有 f 加工**的路径，沿途全乘 1，于是梯度从尾部**一路 1 倍直达最前层**，不衰减。

#### （承上）数字对比（设 f'(x)=0.1，上游梯度=2.0）

```
没残差：系数 = f'(x)=0.1     → 2.0×0.1=0.2，再一层 0.02… 很快归零     ❌
有残差：系数 = 1+f'(x)=1.1   → 2.0×1.1=2.2，再一层 2.42… 保得住       ✅
```

> 本质：是**加法把两条路"并联相加"**（`1 + f'(x)`），才拆得出一条独立的 ×1 旁路；若是纯串联 `f(x)`（没有 +x），梯度必须穿过 f，f 一弱就没了。这条恒为 1 的旁路，就是"梯度高速公路"。

---

## 三、官方 Qwen2 的做法：把三者拼成完整层与模型

### 3.1 一层的完整数据流（把 Norm + 子层 + 残差 拼齐）

```
       ┌──────────────── 残差主干（干净原值，梯度高速公路）────────────────┐
x ─────┤                                                                    │
       │   input_layernorm(x)  →  Attention  ─────────────────────────────(+)── h1
       │                                                                        │
       │   ┌──────────────── 残差主干 ─────────────────────────────────────┐  │
h1 ────┤   │                                                                │  │
       │   post_attention_layernorm(h1) → MLP ─────────────────────────────(+)── h2 → 下一层
```

- 两条残差主干上都**没有 norm/子层挡着**，是梯度直达通道；
- norm 只作用在"喂给子层"的那条支路上；
- 子层（Attn/MLP）算出的是"增量"，加回主干。

### 3.2 收尾的 final norm（模型级组装）

因为 Pre-Norm 里每个 norm 都在子层**前面**，**最后一层的输出后面还没被 norm 过一次**，所以 `Qwen2Model` 在所有 N 层之后、进 `lm_head` 之前，补一个**收尾的 final norm**：

```1041:1041:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        hidden_states = self.norm(hidden_states)
```

这是 Pre-Norm 架构的标配收尾：把最终隐藏态归一化好，再交给 `lm_head` 算 logits。

完整位置串起来：

```
embed_tokens
  → DecoderLayer × N   （每层：input_norm→Attn→残差，post_norm→MLP→残差）
  → self.norm          ← 收尾的 final RMSNorm（本节）
  → lm_head
  → logits
```

---

## 四、本项目改造后的做法

**本篇涉及的 RMSNorm / Pre-Norm / 残差结构，本项目保留官方实现、未做结构性改造。** 一层的骨架（两组 `norm → 子层 → +残差`）、Pre-Norm 位置、final norm 收尾，端侧版本和官方完全一致。

> 端侧真正改动的是骨架里挂着的**子层**和**数据来源**（Attention 换 `QcAttention`、MLP 的 Linear→1×1 Conv、掩码/KV 外部化），这些见各自专篇（[附录A](./02-附录A-Attention注意力机制.md) / [附录H](./02-附录H-MLP前馈网络.md) / [附录E](./02-附录E-端侧定长与计算图导出.md) / [附录K](./02-附录K-KV%20Cache(键值缓存).md)）。至于量化时 RMSNorm 的 `rsqrt`、逐通道 scale 在端侧算子里怎么表达，属于量化实现细节，见文末待深入。

---

## 五、记忆锚点

- 一层骨架：**`x → norm → 子层 → +残差`**，Attention 和 MLP 各一组（Pre-Norm）。
- **RMSNorm**：只"除以均方根 + 乘可学习 w"，比 LayerNorm 砍掉了"减均值"和偏置，**更快更省、效果相当**；在每个 token 的 hidden 维内部做。
- **Pre-Norm**：norm 放子层**前面**，残差主干干净 → 训练稳、能堆深层；对比 Post-Norm（norm 在相加之后）。
- **残差 = 输入 + 子层(输入)**：主要为**梯度走捷径直达深层**，让每层只学增量、信息可无损透传。
- **Final norm**：所有层跑完后补一个 `Qwen2Model.norm`，在 `lm_head` 前给最终隐藏态归一化（Pre-Norm 收尾标配）。
- **本项目改造**：Norm/残差结构**保持官方原样**；改的是骨架里挂的子层和数据来源（见附录 A/H/E/K）。
- 一句话：**norm 管"数值稳不稳"，残差管"梯度传不传得回去"，两者是深层 Transformer 能训练的地基。**

---

## 六、待深入（自己往下填）

- [ ] RMSNorm 里为什么要临时 `.to(torch.float32)` 再转回？（提示：低精度下平方求和易溢出/损失精度）
- [ ] 量化时 RMSNorm 怎么处理？`rsqrt`、逐通道 scale 在端侧算子里怎么表达？
- [ ] Post-Norm 到底为什么难训？（从梯度公式看 norm 在残差路径上的影响）
- [ ] 为什么两个 norm 的名字是 `input_layernorm` / `post_attention_layernorm` 而不是统一命名？
