# 02-附录C · 张量维度 [B, seq, hidden]

> **关联**：这是看懂 [02-附录A-Attention注意力机制.md](./02-附录A-Attention注意力机制.md) 和 [02-附录B-Linear与Conv算子转换.md](./02-附录B-Linear与Conv算子转换.md) 里所有 `reshape / transpose / view` 的**地基**。本篇只讲张量的**形状**；权重、激活、KV Cache 三类张量的**角色区分**及各自对应的 encoding，见 [07-附录B · 6.1 三类 Tensor](./07-附录B-Encoding量化参数基础.md)。
> **一句话本质**：模型里流动的文本数据，最常见形状就是三维 `[B, seq, hidden]` —— **几个句子、每句几个词、每个词几维向量**。

> **本篇是基础概念篇**（全笔记统一风格）：只分 **① 介绍 → ② 原理** 两段；张量维度是通用地基，**不涉及"官方 vs 本项目改造"的对比**。

---

## 一、介绍：`[B, seq, hidden]` 三个维度分别是什么

| 维度 | 名称 | 通俗含义 | 在本项目 |
|------|------|----------|----------|
| 第 1 维 `B` | batch | 一次处理 **n 个句子**（样本） | `config.yaml` 里 `batch_size`（端侧常=1） |
| 第 2 维 `seq` | sequence length | 每个句子有 **seq 个词(token)** | 受 `context_length` 限制（如 2048） |
| 第 3 维 `hidden` | hidden size / embedding dim | 每个词用 **hidden 个数** 表示 | = config 的 `hidden_size`（约 2048） |

直观想象一个 `[B=2, seq=4, hidden=3]` 的张量：

```
句子0: [词0:[x,x,x]  词1:[x,x,x]  词2:[x,x,x]  词3:[x,x,x]]
句子1: [词0:[x,x,x]  词1:[x,x,x]  词2:[x,x,x]  词3:[x,x,x]]
        └─ 每个词是一个长度 3 的向量(embedding) ─┘
```

---

## 二、原理

### 2.1 四个关键补充点

#### 1. batch 的作用 = 并行提速
把多个句子打包一起算，硬件能一次并行处理，比逐句算快得多。
- **训练**：常用较大 batch。
- **端侧推理**：往往 `batch_size=1`（一次处理一个请求）→ 这就是 `config.yaml` 里 `batch_size: 1` 的原因。

#### 2. seq / 词 / token 三者的关系（重点）

口语里我们常说"每个句子有 seq 个**词**"，但严格讲，**`seq` 数的是 token，不是"单词"**。三者关系：

```
原始文本(字符串)
   │  tokenizer 切分
   ▼
token 序列  ← seq 数的就是这个：token 的个数
   │  查 embedding 表
   ▼
每个 token → 一个 hidden 维向量
```

**token 是什么**：tokenizer 把文本切成的"子词单位"，不一定等于一个完整单词：

| 原文 | 切成的 token | token 数 |
|------|-------------|----------|
| "playing" | `play` + `ing` | 2 |
| "cat" | `cat` | 1 |
| "你好世界" | `你` `好` `世` `界`（中文常一字一 token） | 4 |
| "ChatGPT" | `Chat` + `G` + `PT`（举例） | 3 |

要点：
- **`seq` = token 数**，所以一个 5 个单词的英文句子，token 数可能是 7、8 个。
- 标点、空格、特殊符号（如句首的 `<bos>`、补齐的 `<pad>`）也算 token。
- 之所以用 token 而非整词：词表(vocab)有限（约 15 万），用子词能用少量单元拼出任意词，还能处理没见过的新词。
- 所以笔记里说"词"时，**心里要换成"token"** 才准确。`config.yaml` 的 `context_length: 2048` 指的也是**最多 2048 个 token**。

#### 3. embedding 向量怎么来的
- 每个 token 先是一个整数 **id**。
- 模型里有一张 **embedding 表**，形状 `[vocab_size, hidden]`。
- 用 id 去查表 → 得到该 token 的 `hidden` 维向量。
- 作用：把"离散的词"变成"连续的向量"，之后才能做矩阵运算。

> 本项目：`hidden = hidden_size`（约 2048），`vocab_size` = 词表大小（约 15 万）。

#### 4. 句子长度不一怎么办 → padding + mask
张量必须是规整的矩形，但真实句子长短不一：
- **padding**：用占位 token 把短句补齐到统一长度。
- **attention mask**：告诉模型"哪些是补出来的，别去关注"。
- 这是 mask 的另一个用途（区别于"因果掩码"屏蔽未来，见笔记 02 第 2 节）。

### 2.2 连回 Attention / Conv 笔记

理解了 `[B, seq, hidden]`，前面那些形状变换就有了根：

- **拆多头**（附录A）：`[B, seq, hidden]` → `[B, heads, seq, head_dim]`，就是把 `hidden` 切成 `heads × head_dim`。
- **转 Conv**（附录B）：`hidden(embedding) → 通道 C`，`seq(词) → 像素位置`，所以 reshape 成 `[B, hidden, 1, seq]`。

一句话：**`B/seq/hidden` 是地基，所有 view/reshape/transpose 都是在这三维（及其拆分）之间搬来搬去。**

---

## 三、记忆锚点

- `[B, seq, hidden]` = 几个句子 / 每句几个 token / 每个 token 几维向量。
- token ≠ 单词（是子词）；id 查 embedding 表得到向量。
- 端侧 `batch=1`；长度不齐用 padding + mask 补齐。

---

## 四、待深入（自己往下填）

- [ ] Qwen2.5-VL-3B 的 `hidden_size` / `num_heads` / `head_dim` / `vocab_size` 具体是多少？（去 model 的 config.json 查）
- [ ] 多模态(VL)里图像是怎么变成 token 拼进 seq 的？
- [ ] padding 在端侧定长推理里具体怎么处理的？
