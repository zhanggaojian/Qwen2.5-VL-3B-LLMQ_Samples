# 附录 · RoPE 旋转位置编码（从零到懂）

> **关联**：[02-附录A-Attention注意力机制.md](./02-附录A-Attention注意力机制.md) 第四节只说了"RoPE 把位置注入 Q、K"，这篇专门把它讲透，并落到本项目实际代码。
> **前置地基**：先懂 Q/K/V 和 $QK^T$ → [02-附录A-Attention注意力机制.md](./02-附录A-Attention注意力机制.md)；懂张量三维 → [02-附录C-张量维度(B,seq,hidden).md](./02-附录C-张量维度(B,seq,hidden).md)。
> **一句话本质**：Attention 天生"不认顺序"，**RoPE 靠把 Q、K 向量按位置转一个角度**，让模型算相关性时自动感知"两个词隔多远"；本项目为端侧定长导出，把 cos/sin 改成**外部预算好喂进来**。

> **本篇按四段式组织**（后续笔记也照此风格）：**① 是什么/为什么用 → ② 原理 → ③ 官方 Qwen2 怎么做 → ④ 本项目改造后怎么做**。

---

## 一、位置信息是什么、为什么要用（介绍）

### 1.1 一个扎心的事实：Attention 是"无序"的

Attention 算相关性靠的是 $QK^T$ —— 每个词和每个词做点积。**点积不关心谁先谁后**。

举例，两句话：

- "**狗** 咬 **人**"
- "**人** 咬 **狗**"

如果只看 Attention 的数学，把词打乱顺序，每个词的 Q、K、V 都没变，算出来的相关性矩阵里"狗↔人"的分数**完全一样**。模型根本分不清是狗咬人还是人咬狗。

> 一句话：**Attention 把句子当成"一袋词"（词的集合），而不是"一串词"（有顺序的序列）。**

### 1.2 所以必须额外告诉模型"顺序"

语言的意思严重依赖顺序，所以必须**在词的表示里掺进"我排在第几个"这个信息**。这个"想办法"就叫**位置编码（Positional Encoding）**。目标：

- 让第 1 个词和第 5 个词"带着不一样的位置标签"；
- 最好还能让模型感觉到"这俩词隔了 4 个位置"这种**相对距离**。

---

## 二、位置信息的原理（从绝对编码到 RoPE）

### 2.1 最直接的做法：绝对位置编码（给每个位置发工牌）

**第几个位置，就配一个固定向量，直接加到词向量上。**

```
第0个位置 → 位置向量 p₀
第1个位置 → 位置向量 p₁
...
实际输入 = 词向量 + 对应位置向量
```

因为由"第几个"这个**绝对编号**决定，所以叫**绝对位置编码**。位置向量两种来源：

| 做法 | 代表 | 怎么来的 |
|------|------|----------|
| **可学习** | BERT、GPT-2 | 建一张"位置向量表"当参数一起训练 |
| **公式算（正弦）** | 原始 Transformer | 用 sin/cos 按固定公式算，不训练 |

正弦公式（了解即可）：

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\quad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

直觉：**每个维度是一个不同频率的正弦波**，不同位置就有了不同的"波形指纹"。

### 2.2 绝对位置编码的三个问题

- **问题1：只懂"绝对第几个"，不直接懂"相隔多远"**——语言里真正重要的常是相对距离（"它"指代前面不远的名词），绝对编码要模型自己费劲学。
- **问题2：外推能力差（最致命）**——只训了 `max_len`（如 512）个位置向量，句子超长就没向量可用，得重训。
- **问题3："相加"注入有点糙**——位置和语义硬加在一个向量里纠缠。

### 2.3 RoPE 的核心：把"加位置"改成"转角度"

RoPE 换思路：**不往向量上加东西，而是把 Q、K 按位置转一个角度。**

- 位置 0：不转；位置 1：转 θ；位置 m：转 m·θ。
- 把 Q、K 想象成钟表指针，**位置越靠后转得越多**。

**为什么妙？相对位置自动出现**：点积大小取决于两向量夹角。位置 m 的 Q 转了 m·θ、位置 n 的 K 转了 n·θ，它俩**夹角差正好是 (m−n)·θ**——只和"相隔多远"有关，与绝对位置无关！

```
Q 转了 mθ ┐
          ├─ 点积只在乎夹角差 = (m−n)θ  →  只依赖【相对距离 m−n】
K 转了 nθ ┘
```

于是三个问题一起缓解：相对距离直接进数学（问题1）、角度无上限可外推（问题2）、乘法旋转而非相加（问题3）。

**具体怎么转（二维一组）**：把向量维度两两一组 $(x_0,x_1)$ 当平面点，转 m·θ：

$$
\begin{pmatrix} x_0' \\ x_1' \end{pmatrix}=
\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
$$

- 不同组用不同 θ（快慢不一），同时编码近/远距离；$\theta_i = 10000^{-2i/d}$。

> ⚠️ **别被 2.3 这里的单个 θ 误导**：上面为讲清"位置→角度"只写了一个 θ，实际是**每对维度各有自己的 $\theta_i$**（`i` 越大转得越慢），所以**同一个 token 内、不同维度对的角度并不相同**。角度由两个下标共同决定：$\text{角度}=m\cdot\theta_i$（`m`=第几个 token 决定"转几倍"，`i`=第几对维度决定"基础步长"）。类比钟表：同一时刻秒针/分针/时针指向本就不同。

#### 2.3.1 为什么必须"两两一组"

因为**"旋转"这个动作天生需要 2 个数**：

- 1 个数 = 数轴上一个点，没法"转"（只能左右挪）。
- 2 个数 = 平面上一个点 $(x,y)$，即从原点出发的一支箭 → 才能绕原点转角度。

所以 `head_dim` 个数要两两拼成"箭"才有得转：`head_dim=128` → 分成 **64 对** → 64 支小箭，每支在自己的小平面里按 $m\cdot\theta_i$ 旋转。

#### 2.3.2 旋转公式是怎么来的（正向推导）

目标：点 $(x,y)$ 绕原点逆时针转 θ，求新坐标 $(x',y')$。

1. **写成极坐标**：任何点可写成"距离 + 角度"，$x=r\cos\varphi,\;y=r\sin\varphi$（$r=\sqrt{x^2+y^2}$，$\varphi$ 是当前角度）。
2. **旋转 = 角度加 θ、长度不变**：$x'=r\cos(\varphi+\theta),\;y'=r\sin(\varphi+\theta)$。
3. **代入和角公式**并把 $r\cos\varphi=x,\;r\sin\varphi=y$ 换回：

$$
x' = x\cos\theta - y\sin\theta,\qquad y' = x\sin\theta + y\cos\theta
$$

> 减号来自和角公式里的 $-\sin\varphi\sin\theta$；几何上对应"y 轴转 θ 后横分量变负（往左偏）"。

#### 2.3.3 "相对位置只看 m−n"的证明（三步 + 数字）

用点积的几何意义 $a\cdot b=|a||b|\cos(\text{夹角})$，且**旋转不改长度、只改方向**：

```
① 未加位置：  Q·K = |Q||K|·cos(φ_q − φ_k)             （φ 是各自原方向）
② 加位置转： Q 在位置 m 转 mθ → 新方向 φ_q + mθ
             K 在位置 n 转 nθ → 新方向 φ_k + nθ
③ 再点积：   新夹角 = (φ_q+mθ) − (φ_k+nθ) = (φ_q−φ_k) + (m−n)θ
             Q'·K' = |Q||K|·cos( (φ_q−φ_k) + (m−n)·θ )
                                  └内容决定┘   └位置只以 m−n 出现┘
```

绝对位置 `m`、`n` 不单独出现，只以 **`m−n`（相隔多远）** 出现 → 点积只依赖相对距离。数字验证（长度 1，$φ_q=10°,φ_k=40°,θ=10°$）：

| 情况 | Q 位置 | K 位置 | 相隔 | Q 转后 | K 转后 | 夹角 | 点积 |
|------|--------|--------|------|--------|--------|------|------|
| A | 2 | 5 | **3** | 30° | 90° | 60° | **0.5** |
| B | 5 | 8 | **3** | 60° | 120° | 60° | **0.5** |

两词整体后移、间距不变 → 注意力分数不变（都 0.5），**位置感只认相对距离**。

### 2.4 通用工程写法（rotate_half）

工程上不真用旋转矩阵，用等价小技巧 `rotate_half`：

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated
```

`q*cos + rotate_half(q)*sin` 展开正好等于旋转矩阵效果，用逐元素乘加实现、快。

> HF 的配对是"**前半配后半**"（`a` 配 `c`、`b` 配 `d`），所以 `cos`/`sin` 是前后半重复的全长向量：`cos=[cosθ₀,cosθ₁,cosθ₀,cosθ₁]`。

#### 2.4.1 rotate_half 代码逐行（以 `x=[a,b,c,d]`）

```
x1, x2 = x.chunk(2, dim=-1)     # 沿最后一维平均切 2 段
   x1 = [a, b]  (前半)   x2 = [c, d]  (后半)

torch.cat((-x2, x1), -1)        # 后半取负挪到前面，前半挪到后面
   -x2 = [-c, -d]  → cat(-x2, x1) = [-c, -d, a, b]
```

一句话：**切成前后两半 → 后半取负放前、前半放后**。

#### 2.4.2 展开为什么正好等于旋转

三块逐元素算（角度 θ₀ 给第 0 对、θ₁ 给第 1 对）：

```
q            = [ a,        b,        c,        d       ]
q*cos        = [ a·cosθ₀,  b·cosθ₁,  c·cosθ₀,  d·cosθ₁ ]
rotate_half(q)      = [ -c, -d,  a,  b ]
rotate_half(q)*sin  = [ -c·sinθ₀, -d·sinθ₁, a·sinθ₀, b·sinθ₁ ]

相加：
位置0: a·cosθ₀ − c·sinθ₀   = (a,c) 这支箭的 x'   ✓ 对上 x'=x·cosθ−y·sinθ
位置2: c·cosθ₀ + a·sinθ₀   = (a,c) 这支箭的 y'   ✓
位置1: b·cosθ₁ − d·sinθ₁   = (b,d) 的 x'         ✓
位置3: d·cosθ₁ + b·sinθ₁   = (b,d) 的 y'         ✓
```

`rotate_half` 的**取负 + 换位**，正好补上旋转公式里的**减号**和**另一分量**——所以逐元素乘加的结果和旋转矩阵完全一致。

#### 2.4.3 为什么不直接用旋转矩阵（大部分是 0）

旋转是"对内各转各、对与对互不干涉"，写成矩阵就是**分块对角**：只有对角线上一串 2×2 小块非零，其余全 0。

```
        输入0   输入1   输入2   输入3
输出0 [ cosθ₀ -sinθ₀   0      0   ]
输出1 [ sinθ₀  cosθ₀   0      0   ]
输出2 [   0      0    cosθ₁ -sinθ₁]
输出3 [   0      0    sinθ₁  cosθ₁]
```

非零比例 = $2d/d^2 = 2/d$：`head_dim=128` 时只有约 **1.6% 非零**，98% 是 0。直接矩阵乘等于狂算"乘 0"，浪费算力+显存；`rotate_half` 只算那 2/d 的有效部分，结果相同、快得多。

### 2.5 RoPE 作用在什么阶段（重点）

对照 [附录A 第三节](./02-附录A-Attention注意力机制.md) 官方源码那 10 步，RoPE 卡在**第 2 步之后、第 6 步之前**：

```
1. q/k/v_proj 投影
2. 拆多头 [B, heads, seq, head_dim]
3. 👉 RoPE：把 Q、K 各转一个角度   ← 就在这一步
4. 更新 KV Cache
5. GQA repeat_kv
6. QKᵀ/√d 算相关性   ← 转过的 Q、K 在这里点积，相对位置生效
7~10. 掩码 → softmax → 加权 V → o_proj
```

三个关键点：**① 只作用在 Q、K，不动 V；② 每层都做一次；③ 发生在 $QK^T$ 之前。**

### 2.6 三种做法一图对比

| | 绝对·可学习 | 绝对·正弦 | **RoPE** |
|---|---|---|---|
| 怎么注入 | 位置向量**加**到词向量 | 位置向量**加**到词向量 | 把 Q、K **旋转**一个角度 |
| 作用位置 | 输入层，加一次 | 输入层，加一次 | **每层 Attention，作用在 Q、K** |
| 懂相对距离 | 靠模型硬学 | 一定程度 | **天生就懂** ✅ |
| 长度外推 | 差 | 一般 | **好**（角度可无限转）✅ |
| 额外参数 | 有（一张表） | 无 | **无** ✅ |
| 代表模型 | BERT、GPT-2 | 原始 Transformer | LLaMA、Qwen、Qwen2.5-VL |

---

## 三、官方 Qwen2 用的什么方法

Qwen 系列用的正是 **RoPE**。但要分两种模型：

### 3.1 Qwen2（纯文本）：标准 1D RoPE

**第一步：预算 cos/sin 缓存表**（`Qwen2RotaryEmbedding`）——按 θ 和位置把整表算好缓存，避免每次现算：

```116:134:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
```

- `inv_freq = 1/base^(2i/d)`：每维一个频率（就是 θ）。
- `freqs = outer(t, inv_freq)`：位置 t × 频率 → 每个(位置,维度)的角度。
- `emb = cat(freqs, freqs)`：拼成全长 `head_dim`，再取 cos/sin。

**第二步：按 position_ids 索引并旋转**（`apply_rotary_pos_emb`，用 `rotate_half`）：

```167:171:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**第三步：在 Attention.forward 里调用**（cos/sin 在**模型内部**现算）：

```274:275:example1/huggingface/baseline_models/qwen2/modeling_qwen2.py
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
```

### 3.2 Qwen2.5-VL（多模态）：mRoPE（多模态旋转位置编码）

图像/视频不是一维序列，而有**时间(t)、高(h)、宽(w)三个空间维度**。所以 Qwen2-VL 用 **mRoPE**：`position_ids` 是**3D**的（t/h/w 三套位置），把 head_dim 的角度按段分给三个维度：

- 用 `Qwen2VLRotaryEmbedding`（`transformers.models.qwen2_vl`）预算。
- **`mrope_section = [16, 24, 24]`**：把 `head_dim/2 = 64` 个频率分成 3 段——前 16 段用**时间**位置、中 24 段用**高**、后 24 段用**宽**。
- 纯文本 token 的 t/h/w 三套位置相同，mRoPE 自动退化成普通 1D RoPE。

> 一句话：**Qwen2 = 标准 1D RoPE；Qwen2.5-VL = mRoPE，把角度按 `[16,24,24]` 分给 时间/高/宽 三个维度**，让模型理解图像里 token 的空间位置。

---

## 四、本项目改造后怎么计算位置信息

### 4.1 核心改动：cos/sin 从"模型内部现算"→"外部预算好当输入喂入"

官方在 `Attention.forward` 里现算 cos/sin（3.1 第三步）。本项目把这步挪到模型外部：由 config 开关 `use_position_embedding_input` 驱动，**外部预算好 cos/sin，作为 `position_ids` 参数（一个 `(cos, sin)` 元组）喂进模型**。

> ⚠️ **常见误解纠正：这个改动的目的"不是"为了定长。** 定长来自"固定输入序列长度"（+ 掩码外部化、KV Cache 改写，见 [附录E](./02-附录E-端侧定长与计算图导出.md)/[附录K](./02-附录K-KV%20Cache(键值缓存).md)）——**官方 RoPE 在内部现算 cos/sin，只要固定长度照样能定长导出**。把 cos/sin 外部化的真正动机是 **NPU/量化友好 + 省重复计算**：
> - **NPU 友好**：`sin/cos` 三角算子在 HTP 上支持差/慢，官方 `RotaryEmbedding` 还可能带动态频率表 / 按位置 gather 等杂算子；移到图外后，图里不再有三角函数和这些动态逻辑，更干净、更好在 NPU 跑。
> - **量化干净**：三角函数与频率表不好量化，移出去让量化图更清爽。
> - **省重复**：cos/sin 只依赖位置、与权重无关，28 层每步都一样 → 外部预算一次、全层复用，不必在 NPU 上层层重算。
>
> 一句话：**定长是别处（固定长度 + 掩码/KV）实现的；RoPE 外部化是为 NPU/量化/省算力，属优化不是定长的必要条件。**

### 4.2 计算代码链（都在 `forward_pass_wrapper.py`）

按模型分两个类：`MRopeEmbedding`（mRoPE，Qwen2.5-VL 走这个）、`RopeEmbedding`（纯文本）。

**① 预算全表（`precompute`）**：复用官方 rotary embedding 把整表 cos/sin 算好，**并只取前半 `head_dim/2`**（因为消费端 `_apply_rope_single` 用半长，见 4.3）：

```64:78:example1/llm_utils/forward_pass_wrapper.py
        rope = Qwen2VLRotaryEmbedding(dim=head_dim, **kwargs)
        dummy_x = torch.Tensor([1.0]).to(device)
        position_ids = example_position_ids
        position_ids = position_ids.to(device)
        if hasattr(rope, '_original_forward'):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[..., :emb_size] for emb in embeddings]
        embeddings = [emb.unsqueeze(0) for emb in embeddings]
        return embeddings
```

**② 取当前窗口 + mRoPE 分段重组（`get_embedding`）**：按 `at_mask`/`num_tokens` 切出本次要的那段，再按 `mrope_section=[16,24,24]` 把三段分别取对应维度拼回：

```85:101:example1/llm_utils/forward_pass_wrapper.py
        at_mask ,num_tokens = position_ids[0],position_ids[1]
        cos = self.cos[0,:,:,:]  # [seq_len, dim]
        sin = self.sin[0,:,:,:]  # [seq_len, dim]

        cos_position_ids = cos[:,:,:at_mask,:]
        sin_position_ids = sin[:,:,:at_mask,:]
        cos = cos_position_ids[:,:,-num_tokens:,:].to(dtype=dtype)
        sin = sin_position_ids[:,:,-num_tokens:,:].to(dtype=dtype)

        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))], dim=-1).unsqueeze(
            1
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))], dim=-1).unsqueeze(
            1
        )
        return cos, sin
```

> 纯文本版 `RopeEmbedding.get_embedding`（140-149 行）更简单：直接 `cos[position_ids]` 索引，无 mrope_section 分段。

**③ 对外入口**（`get_[qwen_]position_embeddings_from_position_ids`）：

```212:216:example1/llm_utils/forward_pass_wrapper.py
def get_qwen_position_embeddings_from_position_ids(position_ids, head_dim, max_length, device, dtype, config, example_position_ids):
    return MRopeEmbedding(device=device, head_dim=head_dim, max_length=max_length, config=config,example_position_ids=example_position_ids).get_embedding(position_ids, dtype=dtype)

def get_position_embeddings_from_position_ids(position_ids, head_dim, max_length, device, dtype, config):
    return RopeEmbedding(device=device, head_dim=head_dim, max_length=max_length, config=config).get_embedding(position_ids, dtype=dtype)
```

**④ 造 dummy 输入时注入**（`llm_quant.py`）：`use_position_embedding_input` 时把 `position_ids` 换成算好的 `(cos, sin)`，随模型输入一起喂：

```275:280:example1/llm_quant.py
    if config.use_position_embedding_input:
        position_ids = get_position_embeddings_from_position_ids(position_ids,
                                                                 head_dim=hidden_size // num_attention_heads,
                                                                 max_length=max_tokens,
                                                                 device=device, dtype=dtype,
                                                                 config=config)
```

### 4.3 消费端：`_apply_rope_single`（vs 官方 `apply_rotary_pos_emb` 的区别）

`QcAttention.forward_conv` 里，若 `position_ids` 是 `(cos, sin)` 元组，就走自定义 `_apply_rope_single`：

```124:131:example1/llm_utils/qcqwen2_adaptation.py
        if isinstance(position_ids, (tuple, list)): # QC
            rope_embedding = position_ids
            cos, sin = rope_embedding
            query_states = _apply_rope_single(query_states, rope_embedding)
            key_states = _apply_rope_single(key_states, rope_embedding)
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
```

`_apply_rope_single` 用**复数（实部/虚部）**写法，且 cos/sin 是**半长 `head_dim/2`**：

```42:58:example1/llm_utils/qcqwen2_adaptation.py
def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

    x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
    x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements

    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real

    x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    return x
```

**和官方 `apply_rotary_pos_emb` 的区别**：

| | 官方 `apply_rotary_pos_emb` | 本项目 `_apply_rope_single` |
|---|---|---|
| cos/sin 长度 | 全长 `head_dim`（`cat(freqs,freqs)`）| **半长 `head_dim/2`** |
| 旋转写法 | `rotate_half`：后半取负拼前，`q*cos+rotate_half(q)*sin` | **复数式**：拆前半=实部、后半=虚部，按复数乘法转 |
| cos/sin 来源 | 模型内部 `self.rotary_emb` 现算 | **外部预算好当输入喂入** |
| 目的 | 通用 | **NPU/量化友好 + 省重复计算/内存**（不是为了定长，见 4.1 ⚠️）|

> 两者数学等价（都是"按位置旋转 Q/K"），只是实现风格 + 数据来源不同。
>
> **为什么复数式 = 旋转**：把一对 `(x_real, x_im)` 看成复数 `x_real + i·x_im`，乘以 `cos + i·sin`（即 e^{iθ}）就是转 θ：
> ```
> (x_real + i·x_im)(cos + i·sin)
>  = (x_real·cos − x_im·sin) + i·(x_real·sin + x_im·cos)
>       └── 实部 = x' ──┘         └── 虚部 = y' ──┘
> ```
> 对照 2.3.2 的旋转公式 `x'=x·cosθ−y·sinθ, y'=x·sinθ+y·cosθ` —— 完全一致。代码里 `x_prod_real/x_prod_im` 就是这实部/虚部。**复数式只需半长**：每对维度一个角度，复数乘法天然用"一个 cos + 一个 sin"同时处理实虚部，不像 `rotate_half` 要把 cos 复制成前后两份对齐全长。

### 4.4 完整闭环

```
外部（预算，不进定长图）                          模型内部（QcAttention）
precompute 全表 cos/sin (取前半 head_dim/2)
   │ get_embedding 切窗口 + mrope_section[16,24,24]分段
   ▼
(cos, sin) ──作为 position_ids 输入喂进模型──▶ forward_conv 判 tuple
                                             → _apply_rope_single 旋转 Q、K
                                             → 继续算 QKᵀ
```

### 4.5 RoPE 的定长：每步只转"新 token"，与 KV Cache 怎么咬合

RoPE 到了这里其实**早已是定长**（长度在输入端就固定，RoPE 之前就成立；RoPE 只是形状不变的旋转）。它每步的形状恒定，靠的是**只处理"新 token"那一段**：

- 以本项目 `max_tokens=2048 = 历史 past(975) + 新 token arn(1073)` 为例（数字见 [附录E](./02-附录E-端侧定长与计算图导出.md)、[附录K](./02-附录K-KV%20Cache(键值缓存).md)）：
- **RoPE 只对"新 token(1073)"的 Q、K 旋转**，形状 `q:[1,heads,1073,128]`、`cos/sin:[1,1,1073,64]` —— `num_tokens=1073` 是写死常数 → 定长。这正是 `get_embedding` 用 `[-num_tokens:]` 只取最新一段的原因。
- **历史(975)不重转**：它们当年"当新 token 时"就旋转过了，转好的 K 存进 KV 缓存；读出来直接用（代码顺序：先 `_apply_rope_single` 转、再 `past_key_value.update` 存）。

**每个 token 一辈子只转一次 RoPE**，这就是 RoPE 定长和 KV Cache 定长的咬合点：

```
新 token(1073) ─ 转 RoPE(用绝对位置 m) ─ 算注意力 ─ 转好的新 K 存进缓存 ─▶ 下一轮成为"历史"
历史(975)      ─ 从外部 KV 缓存读(已转好, 绝对位置 n) ─ 只读不重转
```

- **为什么"只转新的"不出错**：新 Q 按绝对位置 m 转、历史 K 早按绝对位置 n 转，点积夹角差 = `(m−n)` → 横跨整个 2048 窗口的相对位置依然正确（见 2.3.3 证明）。
- 结论：**RoPE 定长（只转固定数量的新 token）** 与 **KV Cache 定长（历史+新=2048 固定缓冲）** 靠"新 K 转完即存、历史只读不重转"衔接。

### 4.6 易踩坑：`position_ids` 在本项目有两种含义

同名变量在两处指不同东西，别混：

| 出现位置 | `position_ids` 是什么 | 用途 |
|----------|----------------------|------|
| `precompute`（喂给 `Qwen2VLRotaryEmbedding`） | **真实每 token 位置**（纯文本 1D；Qwen2.5-VL 是 t/h/w 3D） | 算整张 cos/sin 表 |
| `get_embedding`（`at_mask,num_tokens=position_ids[0],[1]`） | **`[at_mask, num_tokens]` 两个数** | 从表里切窗口（`at_mask` 定位真实末尾、`num_tokens` 取最新几个）|
| `QcAttention.forward_conv`（判 `isinstance(..., tuple)`） | **外部喂入的 `(cos, sin)` 元组** | 直接当算好的位置编码用，走 `_apply_rope_single` |

> `get_embedding` 里 `at_mask` 只改"切哪一段（内容）"，`[-num_tokens:]` 保证"取出长度恒为 num_tokens（形状）"——**内容随历史移动、形状始终固定**，所以不破坏定长。

---

## 五、记忆锚点

- **为什么用**：Attention 无序（$QK^T$ 不认先后），得额外喂"顺序"。
- **原理演进**：绝对编码（发工牌、相加）→ 缺点(不懂相对/不能外推) → **RoPE：按位置转角度**，点积夹角差=相对距离，天生懂相对、可外推；只转 Q/K、每层做、在 $QK^T$ 之前。
- **官方 Qwen2**：标准 1D RoPE（`Qwen2RotaryEmbedding` 预算 cache + `apply_rotary_pos_emb` 用 `rotate_half`，模型内部现算）。
- **官方 Qwen2.5-VL**：**mRoPE**，`position_ids` 3D，`mrope_section=[16,24,24]` 把角度分给 时间/高/宽。
- **本项目改造**：cos/sin **外部预算好当输入喂入**（`use_position_embedding_input`）；`MRopeEmbedding`/`RopeEmbedding` 的 `precompute`(全表,取半长) + `get_embedding`(切窗口+分段) → 入口函数 → `llm_quant` 造 dummy 注入 → `QcAttention._apply_rope_single`(复数式,半长) 消费。
- **一句话**：算法还是 RoPE，改的是"cos/sin 从模型内部现算 → 外部预算好喂进来"，为的是 **NPU/量化友好 + 省重复计算（不是为了定长）**。
- **定长关系（4.5）**：定长在输入端就定了（固定长度 + 掩码/KV 外部化），RoPE 只是形状不变的旋转；每步只转"新 token(如 1073)"、历史从 KV 缓存读已转好的，**一个 token 一辈子只转一次**。

---

## 六、待深入（自己往下填）

- [x] `rotate_half` 那一行为什么等价于旋转矩阵？→ 见 2.4.2（`[a,b,c,d]` 展开逐位对上 x'/y'）、2.4.3（矩阵 98% 是 0）。
- [x] `_apply_rope_single` 的复数式与官方 `rotate_half` 式，怎么证明等价？→ 见 4.3（复数乘 e^{iθ} 展开 = 旋转公式，复数式用半长的原因）。
- [ ] mRoPE 的 3D `position_ids` 具体怎么由 `grid_thw` 生成？（看 `Qwen_MROPE_Index.get_rope_idx`）
- [ ] `mrope_section=[16,24,24]` 为什么是这个划分？和图像 patch 的 t/h/w 网格怎么对应？
- [ ] 长文本外推的 NTK-aware、YaRN、位置插值，具体怎么"改 θ / 改角度"？
