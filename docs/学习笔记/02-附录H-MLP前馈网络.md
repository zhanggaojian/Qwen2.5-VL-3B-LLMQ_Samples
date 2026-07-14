# 附录 · MLP 前馈网络（从零到懂）

> **关联**：MLP 是 [README](./README.md) 里点名的「Transformer 四大块」之一（Attention / **MLP** / Causal Mask / KV Cache），却一直没有专篇，这篇补上。
> **前置地基**：懂张量三维 → [02-附录C-张量维度(B,seq,hidden).md](./02-附录C-张量维度(B,seq,hidden).md)；懂 Attention 在干嘛 → [02-附录A-Attention注意力机制.md](./02-附录A-Attention注意力机制.md)；MLP 里的 Linear 端侧怎么转 Conv → [02-附录B-Linear与Conv算子转换.md](./02-附录B-Linear与Conv算子转换.md)。
> **一句话本质**：Attention 管「token 之间互相看、交换信息」，**MLP 管「每个 token 自己关起门来做非线性深加工」**，是模型表达能力和知识存储的主力。

> **本篇按四段式组织**（全笔记统一风格）：**① 介绍/为什么 → ② 原理 → ③ 官方 Qwen2 做法 → ④ 本项目改造后做法**。

---

## 一、介绍：MLP 是什么、和 Attention 的分工、为什么需要它

### 1.1 读之前先记住一张图

```
一层 Transformer Block：

输入 x ──▶ Attention（跨 token 交流）──▶ +残差 ──▶ MLP（逐 token 深加工）──▶ +残差 ──▶ 输出
                  ↑                                      ↑
            token 之间互相看                    每个 token 各算各的，彼此不看
```

记住这个分工：**Attention 是"开会讨论"，MLP 是"会后各自消化整理"。** 后面都围绕这句话展开。

### 1.2 MLP 的作用（为什么非要它）

**1. 引入非线性、提升表达能力。**
Attention 主要做"加权求和"（偏线性的信息聚合），没有 MLP 的话表达能力很受限。MLP 的激活函数提供关键的非线性拟合能力。

**2. 逐 token 独立地做特征变换 / 提炼。**
Attention 负责"token 之间互相看、交换信息"（跨位置）；MLP 是**对每个 token 单独处理**（同一套权重作用在每个位置），把 Attention 汇总来的信息**深加工、抽取更高级特征**。

**3. 承载模型大部分参数与知识。**
大模型里 MLP 的参数量通常**比 Attention 还大**（中间维度撑宽 4~5 倍）。很多研究认为大量"事实知识"就存在 MLP 权重里。

---

## 二、原理

### 2.1 最原始的 MLP 是什么

抛开 Transformer，MLP（Multi-Layer Perceptron，多层感知机）是最基础的神经网络：**若干个全连接层（Linear）中间夹激活函数**。

```
输入 → Linear → 激活函数 → Linear → 激活函数 → ... → 输出
```

- **Linear（全连接层）**：`y = Wx + b`，本质是矩阵乘法，负责"线性变换、混合特征"。
- **激活函数（ReLU / SiLU / GELU 等）**：负责引入**非线性**。

> ⚠️ 关键点：**如果只堆 Linear 不加激活函数，几层叠起来数学上等价于一层 Linear**（线性套线性还是线性）。所以**激活函数是 MLP 的灵魂**——正是它让网络能拟合任意复杂的非线性函数（这就是"通用近似定理"的直觉）。

### 2.2 顺带把 SiLU 激活讲透

Qwen2.5 的 `act_fn` 是 **SiLU**（也叫 Swish）：

$$
\text{SiLU}(x) = x \cdot \sigma(x),\qquad \sigma(x) = \frac{1}{1+e^{-x}}
$$

#### 别把 SiLU 和 σ 搞混

| 函数 | 公式 | 值域 | 角色 |
|------|------|------|------|
| **σ(x)**（sigmoid） | $1/(1+e^{-x})$ | **(0, 1)** | "开关 / 权重"，压到 0~1 |
| **SiLU(x)** | $x\cdot\sigma(x)$ | **约 [-0.278, +∞)** | "激活函数"，平滑版 ReLU |

**压到 (0,1) 的是 σ，不是 SiLU。** SiLU 是拿 x 再乘上 σ(x)，输出跟着 x 走。

#### SiLU 的两个"区间"，别绕晕

1. **对固定的某个 x**：因为 $\sigma(x)\in(0,1)$，拿 x 乘一个 (0,1) 的数必然"往 0 缩"，所以 **SiLU(x) 一定夹在 0 和 x 之间**（更精确：正数在 `(x/2, x)`，负数在 `(x/2, 0)`）。
2. **让 x 扫过所有实数（整个函数值域）**：最小值 **-0.278**（在 x≈-1.28 处），最大到 +∞。

这俩不矛盾：x=-1.28 时 SiLU=-0.278，它既在 (x, 0) 里，又恰好是所有 x 能压出的最负值。

#### 图形直觉 & 对比 ReLU

```
        │           ╱
        │         ╱      ← x 大时 SiLU ≈ x（σ→1，几乎原样通过）
        │       ╱
────────┼─────╱──────────  x
     ╲__│  ╱
        ╲╱  ← 负半轴小凹陷(最低≈-0.278)，不硬切成 0
```

- **ReLU**：负数一律砍成 0，0 点有硬折角。
- **SiLU**：负数轻轻压小、还允许一点点负值，整条曲线平滑无折角 → 梯度更平滑，训练更稳。

### 2.3 如何理解 MLP 在"想"什么

上面说的作用有点抽象，这一节把"MLP 到底在干嘛"讲透。核心分工先钉死：

```
token 向量 ──▶ Attention（收集情报）──▶ MLP（独立思考）──▶ 下一层
```

- **Attention = 收集情报**：这个 token 环顾整句话，把相关 token 的信息**搬到自己身上**，得到一个"融合了上下文的向量"。
- **MLP = 独立思考/加工**：拿到融合向量后**关起门来自己琢磨**——提取特征、做判断、调取知识，产出更"成熟"的向量。

> 一句话：**Attention 负责"看到什么"，MLP 负责"想明白什么"。**

#### 把三步拆成"特征识别 + 知识调取"

- **① 升维（2048 → 11008）＝ 摊开一大排"检测器"**：这 11008 个维度可理解成 11008 个"这是不是某种模式？"的探测器同时开火，比如"是不是在描述颜色？""前文是不是有否定词？""是不是一个法国地名？"。
- **② 门控 + 激活 ＝ 只留下真正被触发的**：`SiLU(gate) * up` 决定哪些探测器被点亮、点亮多少，没触发的通道压到接近 0（就是门控，详见第三节 3.4）。
- **③ 降维（11008 → 2048）＝ 把结论写回向量**：把"哪些特征被激活"重新压回 2048 维，更新这个 token 的表示，让它携带比进来时更抽象、更丰富的信息。

#### 一个具体例子

生成 "The capital of France is ___"，要预测下一个词：

- **Attention 这一步**：`is` 这个 token 把 `capital`、`France` 的信息搬过来，向量里现在混进了 "France + capital" 的语义。
- **MLP 这一步**：拿到这个融合向量后，某些通道被强烈激活（命中"首都查询"模式），从权重里**调出"France 的首都 = Paris"这条知识**写进输出，让预测 `Paris` 的概率飙升。

**注意：真正"记得 France 首都是 Paris"这条事实，存在 MLP 的权重里，不在 Attention 里。** Attention 只是把 France 和 capital 凑到一起，MLP 才完成"查表得出 Paris"。

#### 更硬核的视角：MLP 是一块"记忆库"（Key-Value Memory）

有一派很有影响力的研究（Geva et al. 2021）把 MLP 直接看成一个**联想记忆表**：

```
gate_proj / up_proj 的每一行 → 一把"钥匙(key)"：识别某种输入模式
down_proj 的每一列          → 对应的"值(value)"：命中后要写回的知识
```

运作就像查字典：输入向量和所有 key 比对 → 命中某几个 key → 取出对应 value → 加权写回输出。**升维**在做"输入和哪些 key 匹配"，**降维**在做"把命中的 value 取出来累加"。

所以常把 **MLP 形容为"知识仓库 / 长期记忆"，Attention 是"工作记忆 / 信息路由"**。大模型能记住海量事实，主力就靠 MLP 权重（也因此它参数量最大）。

#### 为什么两者缺一不可

| | 只有 Attention | 只有 MLP |
|---|---------------|----------|
| 能力 | 只会搬运、加权组合已有信息（偏线性）| 只会对单个 token 加工，但**看不到上下文** |
| 缺陷 | 没有非线性深加工，也存不下知识 | 每个 token 是孤岛，无法理解句子结构 |

**Attention 提供"跨 token 的上下文"，MLP 提供"非线性加工 + 知识存取"**，交替堆叠 N 层，模型才既懂上下文、又有知识、还能层层抽象。

### 2.4 MLP 的特点

| 特点 | 说明 |
|------|------|
| **逐位置独立（Position-wise）** | 对每个 token 用**同一套权重**、各算各的，token 之间不交互 |
| **参数量大** | 中间维度撑宽 4~5 倍，是模型参数主要来源 |
| **计算量大** | 推理时 MLP 的矩阵乘法往往是耗时大头，是量化/加速重点对象 |
| **结构简单** | 就是 Linear + 激活，没有复杂位置关系，容易并行、容易量化 |
| **非线性来源** | 激活函数（SiLU/GELU）是它区别于纯线性变换的关键 |

#### "逐 token 独立"到底什么意思（别误解）

⚠️ 常见误解："逐 token"是指每个 token 用不同的维度或不同的权重。**恰恰相反——所有 token 共享同一套 MLP 权重**。准确含义是：**token 与 token 之间不混合，各算各的。**

数据是 `[B, seq, hidden]`（详见 [附录C](./02-附录C-张量维度(B,seq,hidden).md)），MLP 作用在最后一维 `hidden` 上，可以想成一叠向量各跑一遍同一个 MLP：

```
token0("我") : [2048维]  ──▶ MLP ──▶ [2048维]
token1("爱") : [2048维]  ──▶ MLP ──▶ [2048维]   ← 同一个 MLP
token2("北") : [2048维]  ──▶ MLP ──▶ [2048维]   ← 权重完全一样
token3("京") : [2048维]  ──▶ MLP ──▶ [2048维]
```

关键要分清**两个轴**上到底混不混：

| 轴 | MLP 会不会混合？ | 说明 |
|----|----------------|------|
| **hidden 维（一个 token 内部的 2048 个数）** | ✅ **会充分混合** | 矩阵乘法让输出每一维都是输入 2048 维的加权和 |
| **seq 维（token 与 token 之间）** | ❌ **完全不混合** | 每个 token 单独过 MLP，彼此看不到 |

#### 和 Attention 的分工对比（横向 vs 纵向）

```
Attention：横向混合（沿 seq 轴，token 之间互相看）
   token0 ←→ token1 ←→ token2 ←→ token3

MLP：纵向加工（沿 hidden 轴，token 内部维度混合，token 间隔离）
   token0: [2048维内部全交互] ─┐
   token1: [2048维内部全交互]  ├ 各自独立，同一套权重
   token2: [2048维内部全交互]  │
   token3: [2048维内部全交互] ─┘
```

| | Attention | MLP |
|---|-----------|-----|
| 处理方向 | **跨 token**（沿 seq 轴混合）| **单 token**（沿 hidden 轴混合，seq 轴隔离）|
| 主要作用 | 信息聚合、找相关性 | 非线性变换、特征提炼、存知识 |
| 是否共享权重 | 是 | 是 |
| 参数量 | 较少 | 较多 |

> 记法：**MLP 在 `hidden` 维上混、在 `seq` 维上不混；Attention 正好相反。** 也正因为逐位置独立，MLP 的 `Linear` 才能等价换成 1×1 Conv（见第四节）。

---

## 三、官方 Qwen2 的做法：Qwen2MLP 与 SwiGLU

Qwen/LLaMA 这类模型每层的 MLP 是一个"**三明治**"结构，官方 `Qwen2MLP` 长这样：

```python
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # 升维
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # 升维
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # 降维
        self.act_fn    = ACT2FN[config.hidden_act]  # Qwen2.5 用 SiLU

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 3.1 三个 Linear，两个动作

- **升维**：`gate_proj` 和 `up_proj` 都把 `hidden_size → intermediate_size`（变宽，比如 2048 → 11008，约 5 倍）。
- **降维**：`down_proj` 再把 `intermediate_size → hidden_size`（变回原宽，11008 → 2048）。

维度轨迹：**窄（2048）→ 撑宽（11008）→ 压回窄（2048）**。

> 为什么要先撑宽再压回来？因为在"宽"的空间里，模型有更多"格子"去做非线性组合、筛选特征，表达能力更强。像解题时先在大草稿纸上展开算，再把答案誊回窄格子。

### 3.2 那一行 forward 拆开看，就是 SwiGLU

`forward` 嵌套得深，拆成四步就清楚了：

```python
gate = act_fn(gate_proj(x))   # ① 门控：升维后过激活 SiLU
up   = up_proj(x)             # ② 内容：升维，但不过激活
h    = gate * up              # ③ 逐元素相乘（不是矩阵乘！对应位置各乘各的）
out  = down_proj(h)           # ④ 压回原维度
```

这个"多一个 `gate` 门控、再逐元素相乘"的结构就叫 **SwiGLU**（比传统 MLP 只有"升维→激活→降维"多一路 gate）。

### 3.3 核心直觉：`gate * up` 到底在干嘛

关键：**`gate` 和 `up` 形状完全一样**（都是 11008 维），相乘是**一一对应地乘**：

```
gate = [0.9,  0.1,  0.8,  0.0, ...]   ← 每个数像"水龙头开度"(0~1 附近)
up   = [5.0, -3.0,  2.0,  7.0, ...]   ← 每个数是"实际内容/水量"
------------------------------------
h    = [4.5, -0.3,  1.6,  0.0, ...]   ← 门控决定这条内容放多少过去
```

- `up` 负责"**内容是什么**"；
- `gate` 负责"**这条内容放不放行、放多少**"（0 关掉，1 全放）。

所以叫**门控（Gated）**：模型自己学会**对每个特征通道动态开/关**，而不是一股脑全传下去，比老式 MLP 更灵活、效果更好。

### 3.4 为什么要多一路 gate？直接 `SiLU(up(x))` 不行吗？

一个很自然的疑问：既然要非线性，为什么不用传统 MLP 的 `down(SiLU(up(x)))` 就好，非要多一个 `gate_proj`？把三种写法摆一起对比：

```python
# 写法A：直接对 x 做 SiLU
out = SiLU(x)                              # ❌ 行不通：x 是 hidden 维，没升维、接不上 down_proj

# 写法B：传统 MLP（GPT-2、原始 Transformer）
out = down_proj( SiLU(up_proj(x)) )        # 只有一路，静态激活

# 写法C：SwiGLU（Qwen/LLaMA）
out = down_proj( SiLU(gate_proj(x)) * up_proj(x) )   # 两路，动态门控
```

**先排除写法A**：`x` 是 `hidden_size`（2048），MLP 的意义就是升维到宽空间加工，`down_proj` 的输入必须是 `intermediate_size`（11008）。只写 `SiLU(x)` 既没升维、也接不上，维度就错了。所以**至少要有一个升维投影**。真正该比的是 B vs C。

**B vs C 的区别：静态激活 vs 动态门控。**

- 写法B：激活函数是"**写死的、逐通道固定**"的曲线。同一个通道值 `up=5.0` 过 SiLU 永远得到 ≈4.98，改不了。
- 写法C：`gate` 是**随输入动态算出来的阀门**。同样内容 `up=5.0`，在不同语境下可以被放行、也可以被掐掉：

| 输入语境 | gate | 输出 gate*up | 效果 |
|---------|------|-------------|------|
| 语境甲 | 0.95 | 4.75 | 特征放行 |
| 语境乙 | 0.02 | 0.10 | 同样内容被**关掉** |

**更本质的一层：写法C 多了一次"乘法交互"。** `SiLU(gate(x))` 和 `up(x)` 都是 `x` 的线性投影，两者相乘产生 `x` 的**二阶（乘积）交互**，能表达"特征A存在时才让特征B通过"这种**条件式、上下文相关**的组合；而写法B 的单路逐点激活做不出这种"通道之间互相调制"的效果。这有点像 Attention 的"软选择"，只不过 SwiGLU 是在**特征通道**上做逐通道软路由。

> 实证背书：Noam Shazeer《GLU Variants Improve Transformer》系统对比后发现，同等参数量下 GLU 类（含 SwiGLU）优于传统 `ReLU/GELU` MLP，所以 LLaMA、Qwen、PaLM 全换成了 SwiGLU。代价是多一个 `gate_proj` 矩阵，因此这些模型把 `intermediate_size` 取得比传统整 4 倍略小（约 8/3 倍）来补偿参数预算。

---

## 四、本项目改造后的做法：把 MLP 的 Linear 换成 1×1 Conv

本项目为了在高通端侧（HTP/NPU）跑，通过 Monkey-Patch 给 `Qwen2MLP` 挂了 `prepare_conv` / `forward_conv` 两个方法（见 [附录B](./02-附录B-Linear与Conv算子转换.md) 讲的"为什么端侧偏爱 Conv"）：

```42:56:example1/llm_quant.py
    MLP_prepare_conv,
    ForCausalLM_prepare_conv,
    MLP_forward_conv,
    DynamicCache_update,
    DynamicCache_get_seq_length,
    update_attr
)

# ————————————————Model Adaptation————————————————
modeling_qwen2.QWEN2_ATTENTION_CLASSES['eager'] = QcAttention
assert update_attr(modeling_qwen2.Qwen2Model, '_update_causal_mask', bypass_update_causal_mask) or \
       update_attr(modeling_qwen2.Qwen2Model, '_prepare_decoder_attention_mask', bypass_update_causal_mask), \
    f"neither _prepare_decoder_attention_mask(..) nor _update_causal_mask(..) found, Unknown Qwen2Model definition in {modeling_qwen2.__file__}"
setattr(modeling_qwen2.Qwen2MLP, 'prepare_conv', MLP_prepare_conv)
setattr(modeling_qwen2.Qwen2MLP, 'forward_conv', MLP_forward_conv)
```

`prepare_conv` 把三个 `Linear` 换成三个 `1×1 Conv2d`，并**把权重直接搬过去**（`weight[:, :, None, None]` 就是给权重补两个维度当卷积核）：

```220:243:example1/llm_utils/qcqwen2_adaptation.py
def MLP_prepare_conv(self):
    if not hasattr(self, 'forward_linear'):
        self.gate_proj_conv = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)
        self.down_proj_conv = nn.Conv2d(self.intermediate_size, self.hidden_size, 1, bias=False)
        self.up_proj_conv = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)
        self.forward_linear = self.forward
        self.forward = self.forward_conv

        self.gate_proj_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None])
        self.down_proj_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None])
        self.up_proj_conv.weight.data.copy_(self.up_proj.weight[:, :, None, None])

        del self.gate_proj
        del self.down_proj
        del self.up_proj

def MLP_forward_conv(self, x):
    bsz, _, _ = x.size()
    x = torch.reshape(x, (bsz, -1, 1, self.hidden_size))
    x = x.transpose(1,3) # Transpose right before and after Conv
    x = self.down_proj_conv(self.act_fn(self.gate_proj_conv(x)) * self.up_proj_conv(x))
    x = x.transpose(1,3)
    x = torch.reshape(x, (bsz, -1, self.hidden_size))
    return x
```

对照第三节原版 `forward`，会发现 `forward_conv` 里**核心那行几乎一模一样**：

```
原版：  down_proj     (act_fn(gate_proj     (x)) * up_proj     (x))
端侧：  down_proj_conv(act_fn(gate_proj_conv(x)) * up_proj_conv(x))
```

只是外面多了 `reshape + transpose`，把 `[B, seq, hidden]` 摆成 Conv 要的 `[B, C, H, W]` 形状，算完再摆回来。**SwiGLU 的算法本身没变，只是把算子换成端侧更喜欢的 1×1 Conv。** 加载模型后由这段统一触发：

```95:97:example1/llm_quant.py
for name, module in model.named_modules():
    if hasattr(module, "prepare_conv"):
        module.prepare_conv()
```

> 💡 为什么 1×1 Conv 等价于 Linear？因为 1×1 卷积对每个空间位置就是一次"通道维的全连接"，逐位置独立——这恰好和 MLP"逐 token 独立"的性质完美对上。细节见 [附录B](./02-附录B-Linear与Conv算子转换.md)。

---

## 五、记忆锚点

- MLP = **全连接 + 激活函数**；激活函数是灵魂，没它多层等于一层。
- Transformer 里的 MLP：**升维 → 非线性筛选 → 降维**（窄→宽→窄，中间撑 4~5 倍）。
- Qwen 用的是 **SwiGLU**：`down(act(gate(x)) * up(x))`，`gate` 当阀门、`up` 当水，逐元素相乘做门控。
- gate 为何优于单路激活：把"**写死的静态激活**"升级成"**随输入动态开关的门控**"，并引入**乘法交互**，表达力更强。
- 激活是 **SiLU = x·σ(x)**：逐点夹在 0~x 之间；整体值域 `[-0.278, +∞)`；平滑版 ReLU。
- "逐 token 独立" = **token 之间不混（seq 轴隔离），所有 token 共享同一套权重**；一个 token 内部维度会充分混（hidden 轴）。
- 分工：**Attention 横向（跨 token 交流），MLP 纵向（逐 token 深加工 + 从权重调知识）**；MLP 参数/计算量更大、更好量化。
- 如何理解 MLP：**升维＝一排特征探测器点火，门控＝只留被触发的，降维＝把结论写回**；更硬核地看，**MLP 是模型的"记忆库"**（key 识别模式、value 存知识）。
- 一句话：**Attention 让 token 们"凑到一起"，MLP 让每个 token "想明白 + 查资料"。**

---

## 六、待深入（自己往下填）

- [x] SwiGLU 相比传统 `down(act(up(x)))` 到底好在哪？为什么多一路 gate 值得？→ 见第 3.4 节
- [ ] `intermediate_size` 为什么常取 hidden 的 8/3 倍附近（而不是整 4 倍）？和 SwiGLU 多一个矩阵有没有关系？
- [ ] SiLU / GELU / ReLU 在大模型里的取舍，量化时哪个更友好？
- [ ] 端侧 `forward_conv` 里的 `reshape + transpose` 会不会带来额外开销？导出 ONNX 后这些算子长什么样？
- [ ] "知识存在 MLP 权重里"——对应的探针实验（如 knowledge neurons）是怎么做的？
