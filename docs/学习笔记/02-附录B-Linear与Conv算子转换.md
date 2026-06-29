# 附录 · Linear 与 Conv 算子转换

> **关联**：笔记 [02-模型适配(Monkey-Patch).md](./02-模型适配(Monkey-Patch).md) 里「MLP / lm_head / Attention 转 Conv」的原理出处。
> **前置地基**：先搞懂张量三维含义 → [02-附录C-张量维度(B,seq,hidden).md](./02-附录C-张量维度(B,seq,hidden).md)。
> **一句话本质**：1×1 卷积在数学上等价于 Linear；本项目把所有 Linear 改写成 1×1 Conv2d，**结果不变，纯粹为了端侧硬件(HTP)更高效**。
>
> ⚠️ 状态：这块还没完全吃透，待继续研究（见文末「待深入」）。

---

## 一、Linear（全连接 / 线性层）

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

## 二、Conv（卷积）

卷积用一个**小窗口(kernel)在数据上滑动**，每次只对窗口内的局部区域做加权和。以 2D 卷积为例：

$$y[c_{out}, h, w] = \sum_{c_{in}} \sum_{i,j} x[c_{in}, h+i, w+j] \cdot W[c_{out}, c_{in}, i, j] + b$$

- 输入 `x`：形状 `[B, C_in, H, W]`（通道、高、宽）
- 权重 `W`：形状 `[C_out, C_in, kH, kW]`
- 核心特点：**局部连接 + 权重共享**（同一个 kernel 在所有位置复用），擅长处理图像这种有空间结构的数据。

---

## 三、关键：1×1 卷积 = Linear

当卷积核大小是 **1×1**（`kH=kW=1`）时，窗口退化成一个点，"滑动"不再聚合任何邻域，只在**通道维度**上做加权和：

$$y[c_{out}, h, w] = \sum_{c_{in}} x[c_{in}, h, w] \cdot W[c_{out}, c_{in}]$$

对比 Linear 的 $y = \sum_{in} x \cdot W$，**数学上完全一样**——只是把"特征维(features)"换名叫"通道维(channels)"，每个空间位置 `(h,w)` 独立地做同一个全连接。

> 一句话：**1×1 Conv 就是把 Linear 套上"通道"的外壳**，对每个像素点各做一次相同的全连接，输出数值完全相同。

### 维度对应表（最关键，很多人卡在这）

| Linear 世界 | Conv 世界 | 说明 |
|------------|-----------|------|
| 特征维 `hidden`（一个词的若干个数） | **通道 C** | 一个词的特征 = 一个像素的多个通道 |
| 序列里的每个词(token) | **像素位置 (h,w)** | 第几个词 = 第几个像素 |

口诀：**把"一个词的特征向量"看成"一个像素上的多个通道"，把"句子里不同的词"看成"图片上不同的像素"。**

---

## 三·补 A：带数字的最小例子

设输入特征 `in=3`、输出特征 `out=2`，一个词的向量 `x = [x1, x2, x3]`。

### Linear 的算法

权重 `W` 形状 `[out=2, in=3]`：
```
W = [[a, b, c],      ← 算 y1 用这一行
     [d, e, f]]      ← 算 y2 用这一行
```
```
y1 = a*x1 + b*x2 + c*x3
y2 = d*x1 + e*x2 + f*x3
```

### 1×1 Conv 的算法

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

### 多个词怎么办？

Linear 对 `[B, seq, hidden]`：**对每个词独立做一次上面的运算**（seq 个词共用同一个 W）。
Conv 对 `[B, C, H, W]`：**对每个像素独立做一次运算**（所有像素共用同一个 kernel）。
"每个词独立、共用权重" = "每个像素独立、共用 kernel"，所以把 `seq` 塞进空间维：

```python
# [B, seq, hidden]  →  [B, hidden, 1, seq]
#       词↑  特征↑         通道↑    像素位置↑(seq 个词排成 1×seq 的"细长图")
x = x.reshape(B, seq, 1, hidden).transpose(1, 3)
```

---

## 三·补 B：`[2,3,1,1]` 的"两个 1" vs `[:, :, 0, 0]` 的"两个 0"

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

---

## 四、二者怎么转换（项目实际代码）

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

---

## 五、区别与联系总结

| 维度 | Linear | Conv (一般) | 1×1 Conv |
|------|--------|------------|----------|
| 连接方式 | 全连接 | 局部连接 | 全连接（仅通道） |
| 权重共享 | 无 | 有（kernel 滑动复用） | 有（每个空间点共用） |
| 输入形状 | `[..., in]` | `[B,C,H,W]` | `[B,C,H,W]` |
| 擅长 | 通用特征变换 | 空间/图像特征 | 通道变换 = Linear |
| 数学关系 | — | 更一般 | **等价于 Linear** |

**联系**：Linear 是 1×1 Conv 的特例（或反过来说，1×1 Conv 是 Linear 的"4D 包装"）；普通 Conv 则是更一般的局部加权操作。

---

## 六、为什么本项目要把 Linear 换成 Conv？

数学等价、结果不变，**纯粹为了硬件**：

- 高通端侧 **HTP/NPU 对 Conv 算子的支持和优化更成熟**（卷积是视觉硬件的"一等公民"），跑 1×1 Conv 往往比等价的 Linear/MatMul 更快、更省、量化也更友好。
- 所以适配层把所有 `q/k/v/o_proj`、`gate/up/down_proj`、`lm_head` 都改写成 1×1 Conv2d——**模型行为完全一致，只是底层算子更适合端侧执行**。

---

## 七、待深入（自己往下填）

- [ ] 为什么 reshape 成 `[B, hidden, 1, seq]` 而不是 `[B, hidden, seq, 1]`？维度顺序对 HTP 有讲究吗？
- [ ] 1×1 Conv 和 Linear 在 HTP 上的实测性能/精度差距到底有多大？
- [ ] 量化角度：Conv 比 Linear "更量化友好"具体体现在哪（per-channel scale？算子融合？）
- [ ] `transpose(1,3)` 在端侧是否会引入额外开销？能否避免？
