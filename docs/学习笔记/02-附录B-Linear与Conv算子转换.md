# 附录 · Linear 与 Conv 算子转换

> **关联**：笔记 [02-模型适配(Monkey-Patch).md](./02-模型适配(Monkey-Patch).md) 里「MLP / lm_head / Attention 转 Conv」的原理出处。
> **前置地基**：先搞懂张量三维含义 → [02-附录C-张量维度(B,seq,hidden).md](./02-附录C-张量维度(B,seq,hidden).md)。
> **一句话本质**：1×1 卷积在数学上等价于 Linear；本项目把所有 Linear 改写成 1×1 Conv2d，**结果不变，纯粹为了端侧硬件(HTP)更高效**。
>
> ⚠️ 状态：这块还没完全吃透，待继续研究（见文末「待深入」）。

> **本篇按四段式组织**（全笔记统一风格）：**① 介绍/为什么 → ② 原理 → ③ 官方 Qwen2 做法 → ④ 本项目改造后做法**。

---

## 一、介绍：为什么要把 Linear 换成 Conv

数学等价、结果不变，**纯粹为了硬件**：

- 高通端侧 **HTP/NPU 对 Conv 算子的支持和优化更成熟**（卷积是视觉硬件的"一等公民"），跑 1×1 Conv 往往比等价的 Linear/MatMul 更快、更省、量化也更友好。
- 所以适配层把所有 `q/k/v/o_proj`、`gate/up/down_proj`、`lm_head` 都改写成 1×1 Conv2d——**模型行为完全一致，只是底层算子更适合端侧执行**。

> 下面 ② 讲清"为什么数学等价"（原理），③ 看官方原本怎么写，④ 看本项目具体怎么改。

---

## 二、原理

### 2.1 Linear（全连接 / 线性层）

最基础的网络层，做一次**矩阵乘法 + 偏置**：

$$y = xW^T + b$$

- 输入 `x`：形状 `[..., in_features]`
- 权重 `W`：形状 `[out_features, in_features]`
- 输出 `y`：形状 `[..., out_features]`

直观理解：**每个输出元素都是所有输入元素的加权和**（"全连接"由此得名）。Transformer 里的 `q_proj`、`gate_proj`、`lm_head` 等都是 Linear。

```python
nn.Linear(in_features, out_features)
# 输入 [B, seq, in]  →  输出 [B, seq, out]
```

---

### 2.2 Conv（卷积）

卷积用一个**小窗口(kernel)在数据上滑动**，每次只对窗口内的局部区域做加权和。以 2D 卷积为例：

$$y[c_{out}, h, w] = \sum_{c_{in}} \sum_{i,j} x[c_{in}, h+i, w+j] \cdot W[c_{out}, c_{in}, i, j] + b$$

- 输入 `x`：形状 `[B, C_in, H, W]`（通道、高、宽）
- 权重 `W`：形状 `[C_out, C_in, kH, kW]`
- 核心特点：**局部连接 + 权重共享**（同一个 kernel 在所有位置复用），擅长处理图像这种有空间结构的数据。

---

### 2.3 关键：1×1 卷积 = Linear

当卷积核大小是 **1×1**（`kH=kW=1`）时，窗口退化成一个点，"滑动"不再聚合任何邻域，只在**通道维度**上做加权和：

$$y[c_{out}, h, w] = \sum_{c_{in}} x[c_{in}, h, w] \cdot W[c_{out}, c_{in}]$$

对比 Linear 的 $y = \sum_{in} x \cdot W$，**数学上完全一样**——只是把"特征维(features)"换名叫"通道维(channels)"，每个空间位置 `(h,w)` 独立地做同一个全连接。

> 一句话：**1×1 Conv 就是把 Linear 套上"通道"的外壳**，对每个像素点各做一次相同的全连接，输出数值完全相同。

#### 维度对应表（最关键，很多人卡在这）

| Linear 世界 | Conv 世界 | 说明 |
|------------|-----------|------|
| 特征维 `hidden`（一个词的若干个数） | **通道 C** | 一个词的特征 = 一个像素的多个通道 |
| 序列里的每个词(token) | **像素位置 (h,w)** | 第几个词 = 第几个像素 |

口诀：**把"一个词的特征向量"看成"一个像素上的多个通道"，把"句子里不同的词"看成"图片上不同的像素"。**

---

### 2.4 带数字的最小例子

设输入特征 `in=3`、输出特征 `out=2`，一个词的向量 `x = [x1, x2, x3]`。

#### Linear 的算法

权重 `W` 形状 `[out=2, in=3]`：
```
W = [[a, b, c],      ← 算 y1 用这一行
     [d, e, f]]      ← 算 y2 用这一行
```
```
y1 = a*x1 + b*x2 + c*x3
y2 = d*x1 + e*x2 + f*x3
```

#### 1×1 Conv 的算法

把这个词当成**一个像素**：它有 3 个输入通道 `x1,x2,x3`，要输出 2 个通道 `y1,y2`。
Conv 权重形状 `[out=2, in=3, 1, 1]`，数值和 Linear 一模一样：
```
W_conv[:, :, 0, 0] = [[a, b, c],
                      [d, e, f]]
```
```
y1 = a*x1 + b*x2 + c*x3
y2 = d*x1 + e*x2 + f*x3
```

**两个公式逐字相同。** 区别只有"权重多了两个 size=1 的维度"，数值没变一个。所以转换只需：

```python
conv.weight.data.copy_(linear.weight[:, :, None, None])
#                                    ↑ [out,in] → [out,in,1,1]，补俩 1
```

#### 多个词怎么办？

Linear 对 `[B, seq, hidden]`：**对每个词独立做一次上面的运算**（seq 个词共用同一个 W）。
Conv 对 `[B, C, H, W]`：**对每个像素独立做一次运算**（所有像素共用同一个 kernel）。
"每个词独立、共用权重" = "每个像素独立、共用 kernel"，所以把 `seq` 塞进空间维：

```python
# [B, seq, hidden]  →  [B, hidden, 1, seq]
#       词↑  特征↑         通道↑    像素位置↑(seq 个词排成 1×seq 的"细长图")
x = x.reshape(B, seq, 1, hidden).transpose(1, 3)
```

---

### 2.5 `[2,3,1,1]` 的"两个 1" vs `[:, :, 0, 0]` 的"两个 0"

这是最容易绕晕的点：同样是数字，**在不同场景里含义不同**。

| 场景 | 写法 | 方括号贴着谁 | 数字含义 |
|------|------|------------|----------|
| 定义形状 | `shape = [2, 3, 1, 1]` | 独立（描述维度） | **个数**：该维度有几个元素 |
| 取数据(indexing) | `W_conv[:, :, 0, 0]` | 紧贴变量 `W_conv` 后 | **下标**：取该维度第几号元素 |

**口诀：方括号紧跟在数组变量后面 → 是用下标取值；独立写的形状 → 是个数。**

为什么这俩能对上：
- kH、kW 两个维度的**个数 = 1**（形状决定）；
- 一个只有 1 个元素的维度，合法**下标只有 `0`**；
- 所以"个数是 1"和"下标取 0"是一回事——**就一个元素，要取它只能取第 0 个**。

`W_conv[:, :, 0, 0]` 拆开看：
```
W_conv[ : , : , 0 , 0 ]
        ↑   ↑   ↑   ↑
       out  in  kH  kW
       全要 全要 取第0个 取第0个   → 得到 [2,3] 的数字表
```

最直白的类比：
```python
a = [10, 20, 30]   # 有 3 个元素(个数=3)
a[0]               # → 10   ← 0 是下标
a = [10]           # 只有 1 个元素
a[0]               # → 10   ← 大小为1，下标只能取 0
```

两个互逆操作记牢：
- `[:, :, None, None]`：`[2,3]` → `[2,3,1,1]`（**加**两个 size=1 维度，用于搬权重进 Conv）
- `[:, :, 0, 0]`：`[2,3,1,1]` → `[2,3]`（**去掉**那两个维度，取唯一下标）

### 2.6 区别与联系总结

| 维度 | Linear | Conv (一般) | 1×1 Conv |
|------|--------|------------|----------|
| 连接方式 | 全连接 | 局部连接 | 全连接（仅通道） |
| 权重共享 | 无 | 有（kernel 滑动复用） | 有（每个空间点共用） |
| 输入形状 | `[..., in]` | `[B,C,H,W]` | `[B,C,H,W]` |
| 擅长 | 通用特征变换 | 空间/图像特征 | 通道变换 = Linear |
| 数学关系 | — | 更一般 | **等价于 Linear** |

**联系**：Linear 是 1×1 Conv 的特例（或反过来说，1×1 Conv 是 Linear 的"4D 包装"）；普通 Conv 则是更一般的局部加权操作。

---

## 三、官方 Qwen2 的做法

官方 `Qwen2` 里这些投影层就是最普通的 `nn.Linear`（`q/k/v/o_proj`、`gate/up/down_proj`、`lm_head`），前向直接 `y = xWᵀ + b`，特征固定在**最后一维**，不做任何形状变换：

```python
self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
# forward: 直接对最后一维 hidden 做矩阵乘 y = xWᵀ + b
```

即"通用矩阵乘 + 特征在最后一维"，是 GPU 训练/推理最自然的写法。本项目的改造前提就是：**数学保持不变，把它们逐一换成等价的 1×1 Conv2d**（见下节）。

---

## 四、本项目改造后的做法（Linear→1×1 Conv2d）

转换分两步：**换层 + 搬权重**。看 `qcqwen2_adaptation.py` 的 MLP 转换：

```python
# 1. 新建 1x1 Conv2d 替代 Linear
self.gate_proj_conv = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)

# 2. 把 Linear 权重拷进 Conv（关键是补两个维度）
self.gate_proj_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None])
```

权重形状的对应关系：

| | Linear 权重 | Conv 权重 |
|---|---|---|
| 形状 | `[out, in]` | `[out, in, 1, 1]` |
| 转换 | — | `weight[:, :, None, None]` 末尾补两个 size=1 的维度 |

数据流上还要配合 reshape/transpose，把 3D 张量 `[B, seq, hidden]` 变成 Conv 要的 4D `[B, hidden, 1, seq]`，算完再变回去（见 `MLP_forward_conv`）：

```python
x = torch.reshape(x, (bsz, -1, 1, self.hidden_size)).transpose(1, 3)  # 变 4D 喂给 Conv
x = self.down_proj_conv(self.act_fn(self.gate_proj_conv(x)) * self.up_proj_conv(x))
x = x.transpose(1, 3).reshape(bsz, -1, self.hidden_size)              # 再变回 3D
```

> Attention 里的 `q/k/v/o_proj`、输出头 `lm_head` 也是同样套路（见适配代码 `QcAttention.prepare_conv` / `ForCausalLM_prepare_conv`）。

### 4.1 为什么非要 reshape+transpose？——两种算子的输入约定不同

很容易误以为"Linear 吃一维向量、Conv 吃二维矩阵"，其实**两者吃的都是多维张量**，真正的差别是 **"特征那一维摆在第几个轴" + "Conv 硬性要求带 H/W 两个空间轴"**：

| | 特征/通道在哪个轴 | 有没有空间轴 | 输入形状 |
|---|-----------------|-------------|----------|
| `Linear` | **最后一轴** | 没有 | `[..., in]`，本项目是 `[B, seq, hidden]` |
| `Conv2d` | **第 1 轴（axis=1）** | 有，必须带 H、W | `[B, C, H, W]` |

`Linear` 直接作用在最后一维 `hidden`；而 `Conv2d` 要求"通道在 axis=1、后面跟 H/W"。约定对不上，所以卷积前后必须"摆正形状再摆回去"。

### 4.2 `MLP_forward_conv` 的形状追踪（一步步走）

以 `hidden=2048, seq=S` 为例，跟着形状走一遍：

```
进来 x            : [B, S, 2048]        ← Linear 风格：特征在最后
reshape(bsz,-1,1,hidden) : [B, S, 1, 2048]    ← 凑成 4 维（插了个 H=1）
transpose(1,3)    : [B, 2048, 1, S]     ← 把特征 2048 换到 axis=1 当通道 ✅ 符合 Conv 约定
gate/up/down 卷积 : [B, 2048, 1, S] → ... → [B, 2048, 1, S]
transpose(1,3)    : [B, S, 1, 2048]     ← 换回去
reshape           : [B, S, 2048]        ← 恢复成 Linear 风格返回给下一层
```

- `reshape(..., 1, hidden)`：把 `[B, S, hidden]` 撑成 4 维，多出来的 `1` 当空间高 H。
- `transpose(1, 3)`：把 `hidden` 从最后一轴挪到 axis=1（Conv 要通道在这），`seq` 挪到 W。
- 算完 `transpose` + `reshape` 原路摆回。

> 💡 `transpose` 默认**不真的搬内存**，只改逻辑形状和步长(stride)，底层数据仍在原地（变成非 contiguous）。所以这些变换**数值一个没变**，只是"换个摆法"给 Conv 看。

### 4.3 那个 `-1` 是什么

`reshape` 要求变形前后**元素总数不变**，其中一维可写成 `-1` 让框架自动反推（总数 ÷ 其他维乘积）：

```python
x = torch.reshape(x, (bsz, -1, 1, self.hidden_size))
#   总数 = B×seq×hidden，已知维乘积 = bsz×1×hidden = B×hidden
#   -1 = (B×seq×hidden) / (B×hidden) = seq
```

所以这里 **`-1` 推出来就是 `seq`**，等价于写 `(bsz, seq, 1, hidden)`。用 `-1` 的好处：

- **省事不易错**：不用手动 `x.size(1)` 取 seq 再传。
- **自动适配变长**：prefill（seq=整段长度）和 decode（seq=1）阶段序列长度不同，`-1` 这行对任意 seq 都成立，不用改。

> ⚠️ 一次 reshape 里**最多只能有一个 `-1`**，写两个会因无法唯一反推而报错。

### 4.4 搬权重那三行怎么读（`.data` / `copy_` / `None`）

新建的 `nn.Conv2d` 权重一开始是**随机值**，所以要把原 Linear 里练好的权重灌进去。三行结构相同，逐片段拆解：

```python
self.gate_proj_conv.weight.data.copy_( self.gate_proj.weight[:, :, None, None] )
└──── 目标：新 Conv 的权重 ────┘         └──── 源：旧 Linear 权重，补两维 ────┘
```

- **`.weight`**：该层里那块可训练权重张量（Conv 的是 `[out, in, 1, 1]`）。
- **`.data`**：绕过 autograd，直接改底层数值。这是"我在初始化权重、不是做前向计算"的标准写法，避免污染计算图/梯度。
- **`.copy_(src)`**：末尾带下划线是**原地(in-place)拷贝**——把 `src` 的数值逐元素填进已有存储，**不新建对象、不换指针**（对比 `a=b` 只是换指针，`a.copy_(b)` 是换内容）。因为 `weight` 这个参数对象要保持不变，只更新数值。
- **`[:, :, None, None]`**：把源权重从 `[out, in]` 补成 `[out, in, 1, 1]`，好和目标形状对齐（`copy_` 要求两边形状一致）。

**重点讲 `None`**：`None`（等价于 numpy 的 `np.newaxis`）的作用是**在该位置插入一个大小为 1 的新维度**：

```
gate_proj.weight              形状 [11008, 2048]
             [ : ,  : , None, None]
               ↓    ↓    ↓     ↓
              out   in  +1维  +1维
gate_proj.weight[:,:,None,None] 形状 [11008, 2048, 1, 1]   ← 对上 Conv 权重
```

- 前两个 `:`：`out`、`in` 两维**原样全取**，不动。
- 后两个 `None`：在末尾**各插一个 size=1 的维度**，充当 `kH=1`、`kW=1`。

数值一个没变，纯粹"从 2D 包装成 4D"。（它和 `[:, :, 0, 0]` 是一对互逆操作：`None` 加维、`0` 取唯一下标去维，见上面「三·补 B」。）

三行的差别只在各层的 `[out, in]` 数字不同：

| 行 | 源 `[out, in]` | 补维后 |
|----|---------------|--------|
| `gate_proj` | `[11008, 2048]` | `[11008, 2048, 1, 1]` |
| `up_proj`   | `[11008, 2048]` | `[11008, 2048, 1, 1]` |
| `down_proj` | `[2048, 11008]` | `[2048, 11008, 1, 1]` |

搬完后代码会 `del` 掉原 Linear、把 `forward` 换成 `forward_conv`，至此彻底切到 Conv 版——**知识不变、算法等价**，因为权重是逐值复制、1×1 Conv 又和 Linear 数学等价。

### 4.5 `prepare_conv` vs `forward_conv`——一次性准备 vs 每次执行

转换涉及**两个函数**，很容易混，其实分工清清楚楚：

| | `MLP_prepare_conv` | `MLP_forward_conv` |
|---|---|---|
| 角色 | **一次性改装（setup）** | **每次推理的计算（runtime）** |
| 执行时机 | 模型加载后**只跑一次** | **每次前向**都跑 |
| 操作对象 | 模块结构 + **权重**（参数）| **激活值**（数据 x）|
| 主要动作 | 建 Conv、搬权重、切 forward、删 Linear | reshape/transpose + 三次 1×1 Conv + SwiGLU |
| 是否算数值 | 否（只搭骨架）| 是（真正出结果）|

两者是**先后依赖**关系：

```
① prepare_conv 先跑（准备）：建好三个 Conv、灌好权重、把 self.forward 接到 forward_conv
        │  （由 llm_quant.py 里 for module ... module.prepare_conv() 触发，每个 MLP 跑一次）
        ▼
② 之后每次 mlp(x) 调用  ──▶  实际执行 forward_conv（用①准备好的 Conv 算数据）
```

- **接头开关**就是 `prepare_conv` 里那句 `self.forward = self.forward_conv`：没有它，`forward_conv` 准备好了也不会被调用。
- **一个管参数、一个管数据**：权重每次推理复用（只搬一次 → `prepare` 跑一次）；数据每次都变（每次都算 → `forward` 每次跑）。
- 若 `prepare_conv` 没先建好 `gate_proj_conv` 等，`forward_conv` 会因找不到这些属性而报错——所以顺序不能反。

> 一句话：**`prepare_conv` 是"装修队"（跑一次，把 Linear 拆了换 Conv、接好线），`forward_conv` 是"日常开车"（每次推理用装修好的 Conv 算数据）。**

---

## 五、待深入（自己往下填）

- [ ] 为什么 reshape 成 `[B, hidden, 1, seq]` 而不是 `[B, hidden, seq, 1]`？维度顺序对 HTP 有讲究吗？
- [ ] 1×1 Conv 和 Linear 在 HTP 上的实测性能/精度差距到底有多大？
- [ ] 量化角度：Conv 比 Linear "更量化友好"具体体现在哪（per-channel scale？算子融合？）
- [ ] `transpose(1,3)` 在端侧是否会引入额外开销？能否避免？
