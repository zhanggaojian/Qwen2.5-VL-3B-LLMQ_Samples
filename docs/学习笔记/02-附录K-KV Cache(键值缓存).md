# 02-附录K · KV Cache（键值缓存）

> **定位**：模型适配 5 处替换里的**最后一块**——重写全局 `DynamicCache.update` / `get_seq_length`（`llm_quant.py` 第 58-61 行）。前 4 块（Attention 换类、掩码透传、MLP/lm_head 转 Conv）见 [总结篇](./02-模型适配总结篇-结构与替换全景.md)。
> **前置**：Q/K/V 与注意力怎么算 → [附录A](./02-附录A-Attention注意力机制.md)；自回归、prefill/decode 两阶段 → [附录D](./02-附录D-自回归与自注意力.md)、[00-基础篇](./00-基础篇-模型与推理两阶段.md)；端侧定长与固定计算图 → [附录E](./02-附录E-端侧定长与计算图导出.md)。
> **一句话本质**：decode 时每生成一个新词，历史 token 的 K/V 每步都算出**一模一样**的结果 → 把它们**缓存**起来别重算，就是 KV Cache；本项目为了**端侧定长导出**，把官方"动态追加"的缓存改写成"**只吐新 K/V、K 转置存储**"的定长版。

> **本篇按四段式组织**（全笔记统一风格）：**① 介绍/为什么 → ② 原理 → ③ 官方 Qwen2 做法 → ④ 本项目改造后做法**。（KV Cache 的"为什么要用"几乎等于它的原理，故 ① 讲动机、② 讲它怎么参与计算。）

---

## 一、介绍：为什么需要 KV Cache

先回忆两件已学的事：

1. **注意力要拿"当前 token 的 Q"去和"所有 token 的 K"打分，再用分数加权"所有 token 的 V"**（[附录A](./02-附录A-Attention注意力机制.md)）。
2. **自回归 = 一次只生成一个新 token，生成完把它接到句尾，再喂回去生成下一个**（[附录D](./02-附录D-自回归与自注意力.md)）。

把这两件事叠在一起会发现一个**巨大的浪费**：

```
已生成: "今天 天气 真"          → 要预测第 4 个词
下一步: "今天 天气 真 好"        → 要预测第 5 个词
```

生成第 5 个词时，"今天 / 天气 / 真"这 3 个 token 的 **K 和 V 和上一步算出来的一模一样**（它们的输入没变、权重没变、位置也没变）。如果每步都把整句重新过一遍 Attention，就等于把前面所有 token 的 K/V **一遍遍重算**，token 越多浪费越夸张（复杂度 ~O(n²)）。

> **KV Cache 的全部意义**：把每个 token 算出来的 **K、V 存进一块缓存**；下一步生成时，历史部分**直接从缓存取**，只需要为**新来的那 1 个 token** 算 Q/K/V。于是每步计算量从"整句"降到"一个词"。

⚠️ **只缓存 K 和 V，不缓存 Q**。因为每一步只关心"**当前新 token** 的 Q 去查询历史"，旧 token 的 Q 后面再也用不到；而旧 token 的 K/V 会被后续每一步反复查询，所以只有 K/V 值得缓存（这也是名字叫 **KV** Cache 的原因）。

---

## 二、原理

### 2.1 新 token 的 Q 到底和谁算？——和历史"全部"K/V 算

这一步想清楚，KV Cache 为什么成立就彻底通了。假设已生成到第 `t` 个位置，新 token 的 query 记作 `Q_t`，它一次注意力做的是：

```
Q_t  与  K_0, K_1, ..., K_{t-1}, K_t   逐个点积  →  t+1 个分数
                                        ↓ softmax
                                     t+1 个权重
                                        ↓ 加权求和
attn_t = w_0·V_0 + w_1·V_1 + ... + w_t·V_t
```

- **和历史所有 K/V 算**：要"回看"全部前文才能理解上下文（生成"好"要参考前面"今天天气真"）。
- **含它自己**（`K_t`、`V_t`）：token 也能关注自身。
- **绝不和"未来"算**：未来 token 此刻还没生成、物理上不存在——这就是"因果/单向"。

**这恰好不和"KV Cache 省算力"矛盾**，因为两者省的是不同的东西：

| 项 | 来源 | 每步变化 | 能不能省 |
|----|------|---------|---------|
| `Q_t` | 只有当前新 token 的，现算 | 每步换新 | —— |
| 历史 `K_0..K_{t-1}` / `V_0..V_{t-1}` | 历史 token | **每步都一模一样** | ✅ 缓存，省"重新生成" |
| 新 `K_t` / `V_t` | 当前新 token，现算 | 每步新增 1 份并追加 | —— |

> 注意力的**打分/加权照样遍历全历史**（这步省不掉）；KV Cache 省的是"把历史 K/V 一次次**重新算出来**"这步（历史值不变，缓存直接取）。
> 补充：标准注意力看**全部**历史；有些模型用**滑动窗口注意力**（只看最近 N 个）等变体则不是全部。Qwen2 有 `sliding_window` 字段，但本项目走标准全注意力。

### 2.2 KV Cache 在 prefill / decode 两阶段里的角色

| 阶段 | 输入 | 干的事 | 和 KV Cache 的关系 |
|------|------|--------|-------------------|
| **prefill**（预填充） | 一整段 prompt（如 100 个 token）| 一次性并行过一遍，理解整段 | **写入**：把这 100 个 token 的 K/V 全存进缓存 |
| **decode**（解码）| 每次只 1 个新 token | 逐词生成 | **读取+追加**：读历史 K/V + 只算新 token 的 K/V 并追加 |

```
prefill:  [t1 t2 ... t100]  ──一次算完──▶  缓存里存好 100 份 K/V，吐出第 101 个词
decode :  [t101]            ──只算1个──▶  读前100份 + 算第101份 → 吐第102个词
          [t102]            ──只算1个──▶  读前101份 + 算第102份 → 吐第103个词
          ...
```

> 没有 KV Cache，decode 每一步都要把越来越长的整句重算；有了它，decode 每步恒定只算 1 个 token，这就是大模型能"流式吐字"还不卡的关键。

#### 2.2.1 ARN 是什么？为什么本项目一次处理 1073 个 token（而不是 1）

看本项目导出配置会发现 `arn: 1073`（`config.yaml`），很容易困惑："decode 不是每次才 1 个 token 吗，1073 哪来的？"——关键：**这张图不是 decode 图，而是 prefill 图**。

- **ARN（Auto-Regressive-N）= 一张图一次前向并行处理的 token 数**，是个**设计选择**，不必是 1。代码里叫 `ARN(BERT) MODE`（`llm_quant.py:152`）——"BERT 模式"= 像 BERT 一样**整段并行**读，不是逐词。
- **为什么 prefill 能一次 1073、decode 只能 1**：区别在这些 token 此刻**存不存在**。

```
prefill：整段 prompt "描述这张图<image>...（共 1073 个 token）" —— 现在全都在手上
         → 一次性并行喂进去、并行算 K/V，快（这就是 ARN=1073 的来源）
decode ：模型吐"这"→拿"这"吐"是"→拿"是"吐"一"…… —— 下一个词依赖上一步输出
         → 物理上只能一次 1 个（自回归），对应 ARN=1
```

> 一句话：**已知的（prompt）能并行 → prefill 一次 1073；未知的（要生成的）只能串行 → decode 一次 1。**

- **端侧真实部署通常是两张图、共享同一份权重**：

| 图 | `num_tokens`(ARN) | `past_size` | 干什么 |
|----|-------------------|-------------|--------|
| **prefill 图**（本项目 `arn:1073`）| 1073（大）| 975 | 一次并行吃完整段 prompt，写满 KV |
| **decode 图**（另配 `arn:1`）| 1 | 2047 | 逐词生成，每次 1 个新 token + 读历史 |

- 真实 prompt 不足 1073？→ **padding 补齐到 1073 + mask 屏蔽**（和历史补零同理），图看到的输入永远 1073。
- 这也呼应 [附录E](./02-附录E-端侧定长与计算图导出.md) 缺点1解法"**prefill 与 decode 分两张图**"。

#### 2.2.2 完整生命周期：一次 prompt = 1 次 prefill + N 次 decode；多轮对话怎么复用历史

**① 单次生成的完整流程**：用户输入 1 段 prompt → **1 次 prefill + N 次 decode**，直到结束符。

```
用户输入 prompt（如 50 个 token）
   │
   ▼
① prefill（跑 1 次）：50 个 token 一次性并行过一遍 → 写满这 50 份 KV → 吐第 1 个新词
   │
   ▼
② decode（跑 N 次，每次 1 个词）：
   decode-1：读 [50 历史] + 算第1个新词的KV → 吐第 2 个词
   decode-2：读 [51 历史] + 算第2个新词的KV → 吐第 3 个词
   ...
   直到吐出 <eos>（结束符）或到达上限（如 2048）→ 停止
```

- **prefill 写、decode 读+追加**：单次生成里 decode 复用的就是 prefill 写下的那份 KV。
- **停止条件**：`<eos>` 或达到 `context_length` 上限。

**② 多轮对话：下一轮 prefill 会复用上一轮的 KV 当历史吗？——分场景**

| 场景 | 复用上轮 KV？| 说明 |
|------|------------|------|
| **连续多轮对话** | ✅ 会 | 第 2 轮 prefill 把 [第1轮 prompt + 回答] 的 KV 当历史(`past_key_i_in`)喂进来，只并行算"新一轮问题"这段，不重算前文 |
| **独立新请求 / 主动清空上下文** | ❌ 不会 | 历史清空，`past_key_i_in` 全补零 + mask 屏蔽，从零开始 |

```
多轮对话（复用）：
  轮1: [空历史]          + prefill(问1) + decode → 缓存=[问1,答1]
  轮2: [问1,答1]         + prefill(问2) + decode → 缓存=[问1,答1,问2,答2]
  轮3: [问1,答1,问2,答2] + prefill(问3) + ...     （超 2048 则滑窗丢最旧）
```

> **这正是 prefill 图也保留 975 历史槽（`past_size=975`）的意义**——就是为了接住"之前几轮已算好的 KV"，避免每轮把整段对话从头重算。它和 decode 复用历史是同一个道理，只是粒度从"上 1 个词"变成"上一整轮"。
> **上限约束**：历史 + 当前不能超 `context_length=2048`；对话越滚越长、超限时用滑动窗口丢最旧（`_do_shift`，见 4.4 / [附录E](./02-附录E-端侧定长与计算图导出.md) 缺点2）。

---

## 三、官方 Qwen2/HF 的做法：`DynamicCache` 动态追加

HuggingFace 的 `DynamicCache` 顾名思义——**缓存长度随生成动态变长**。它内部就是两个 list：`key_cache` 和 `value_cache`，每层一个元素。核心逻辑（官方版）：

```
每次 update(新K, 新V, 第几层):
    把 新K 拼到 key_cache[层] 的末尾（沿 seq 维 cat）
    把 新V 拼到 value_cache[层] 的末尾
    返回"拼接后"的完整 K/V（拿去算注意力）
```

- 第 1 步存 100 份 → 第 2 步变 101 份 → 第 3 步 102 份……**长度一直在涨**。
- 好处：简单、通用、GPU 上无所谓 shape 变化。
- 问题：**shape 一直在变**，这正好踩中端侧的雷区（[附录E](./02-附录E-端侧定长与计算图导出.md)：NPU 要定长、要固定计算图，shape 一变就要重编译甚至跑不了）。

---

## 四、本项目改造后的做法：为什么重写 + 怎么重写

端侧要**定长 + 固定计算图**，所以缓存不能"越拼越长"。项目改写了两个方法（`llm_utils/qcqwen2_adaptation.py`），并靠两个 config 开关驱动行为（`config.yaml` 的 `model_overrides`）：

```29:30:example1/config.yaml
  return_new_key_value_only: true
  transposed_key_cache: true
```

### 4.1 重写 `update`：核心是 `return_new_key_value_only`

```266:295:example1/llm_utils/qcqwen2_adaptation.py
def DynamicCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += value_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    else:
        return_new_key_value_only = cache_kwargs.get('return_new_key_value_only', False)
        transposed_key_cache = cache_kwargs.get('transposed_key_cache', False)
        key_cat_dim = -1 if transposed_key_cache else -2

        key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=key_cat_dim)
        value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if return_new_key_value_only:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = key_cache
            self.value_cache[layer_idx] = key_cache
        return key_cache, value_cache
```

逐段拆：

| 代码 | 干什么 | 直觉 |
|------|--------|------|
| `if layer_idx == 0: self._seen_tokens += ...` | 只在第 0 层累计"已见 token 数" | 每层都会调一次 update，只让第 0 层记账，避免重复计数 |
| `if len(self.key_cache) <= layer_idx:` | 该层**第一次**存（还没有缓存槽）| 对应 prefill 首次写入：直接 append |
| `else:` 分支 | 该层**后续**再存 | 对应 decode 每步追加 |
| `key_cat_dim = -1 if transposed_key_cache else -2` | 决定沿哪个维拼 K | K 转置存储时 seq 维在 `-1`，否则在 `-2`（见 4.2）|
| `key_cache = torch.cat([旧, 新])` / `value_cache = ...` | 拼出"完整历史 + 新"的 K/V | **这份拼接结果是返回值**，拿去算注意力（注意力需要看全历史）|
| `if return_new_key_value_only:` → 存 `key_states`（新的）| **缓存里只留刚算的新 K/V** | ✅ 本项目开关为 `true` 走这里 |
| `else:` → 存 `key_cache`（拼接后的）| 缓存里留完整历史 | 官方式的"越存越长"|
| `return key_cache, value_cache` | **返回的永远是拼接后的完整 K/V** | 算注意力要全历史，但**存进缓存的**只有新的 |

#### 4.1·精读 · 逐行走读（表格看不够时看这里）

> 读这段的**总纲**：全程盯住一句——**"返回给注意力的东西" 和 "存进缓存的东西" 是分开的两回事**，就不会绕晕。

**① 签名（`def ... (self, key_states, value_states, layer_idx, cache_kwargs)`）**
- `self`：缓存对象，内部两个 list `self.key_cache` / `self.value_cache`（每层一个元素）。
- `key_states`/`value_states`：**本层本次刚算出的新 K/V**（来自 `k_proj_conv`/`v_proj_conv`）。
- `layer_idx`：**第几层**在调用——模型 28 层，每层 self_attn 各调一次，靠它区分存到哪个槽。
- 返回 `Tuple[K, V]`：**拿去算注意力的那份**。

**② `if layer_idx == 0: self._seen_tokens += value_states.shape[-2]`**
- `_seen_tokens` = 整个模型累计处理的 token 数；`shape[-2]` 是本次新增 token 数（V 没转置，seq 维稳定在 `-2`）。
- ⚠️ **为什么限定第 0 层**：一次前向里 28 层**各调一次** update，若每层都加会把 token 数**重复计 28 倍**，所以只让第 0 层记一次账。

**③ `if len(self.key_cache) <= layer_idx:`（首次填充分支）**
- 条件成立 = **这层还没有缓存槽**（prefill 首跑时 list 为空）→ `append` 把新 K/V 落为这层第一份历史，直接原样返回。
- **对应 prefill**：整段 prompt 第一次进来，无历史可拼。

**④ `else:`（后续追加分支）——decode 每步走这里**
- `key_cat_dim = -1 if transposed_key_cache else -2`：K 若转置存过，形状是 `[.., head_dim, seq]`，**seq 跑到最后 `-1`** → 沿 `-1` 拼；V 永不转置，固定 `-2`。
- `key_cache = cat([旧, 新])` / `value_cache = cat([旧, 新])`：拼出**完整历史 ⊕ 新**——⭐ 这两个局部变量是**返回值**（第 ⑥ 行 return），给注意力看全历史用。
- `if return_new_key_value_only:`（本项目 `True`）→ **存进 `self` 缓存的只有本次新的** `key_states`/`value_states`，**不是**完整历史 → 模型内部不累积历史，历史甩给外部管，shape 恒定可定长导出。
- `else:`（官方式）→ 存完整拼接。⚠️ 此分支第 133 行 `self.value_cache[layer_idx] = key_cache` 疑似笔误（应为 `value_cache`），但本项目 `return_new_key_value_only=True` **永不走这里**，故无影响（见第七节待深入）。

**⑤/⑥ `return key_cache, value_cache`**
- **无论上面往缓存里存了什么，返回的永远是"完整历史 ⊕ 新"**——因为调用方（注意力）必须遍历全历史打分加权。

**执行路径速记**：`prefill → 走③ append`；`decode → 走④ 拼完整当返回值、只把新的存进缓存`。两个"分开"：`_seen_tokens` 只在 layer0 累计；**返回值(完整) ≠ 存储值(只新)**。

**关键理解 `return_new_key_value_only=True` 的分工**（这是端侧定长的精髓）：

```
返回值(给注意力用)  = 旧历史 ⊕ 新token   ← 完整，注意力才能看全上下文
存进 self 缓存的    = 只有新 token 的 K/V  ← 不在模型内部累积历史
                     ↑
             历史由"外部"管理：作为模型的 past_key_value 输入喂进来，
             输出的新 K/V 再由外部拼接/滚动 → 模型内部 shape 恒定 → 可定长导出
```

> 对比 [附录E](./02-附录E-端侧定长与计算图导出.md) 的掩码改造，思路**完全一致**：都是把"随长度变化的东西"从**模型图内部**挪到**外部输入/输出**，让图本身保持定长、静态。掩码是"外部喂入"，KV Cache 是"外部喂入历史 + 模型只吐新 KV"。

### 4.2 `transposed_key_cache`：K 转置存储

在 `QcAttention.forward_conv` 里，存 K 之前会按开关做一次转置：

```133:143:example1/llm_utils/qcqwen2_adaptation.py
        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, 
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

**为什么转置？** 注意力打分是 `Q · Kᵀ`：

- 不转置：K 形状 `[b, head, seq, head_dim]`，算分时要在**运行时**做一次 `key.transpose(2,3)` 再和 Q 相乘。
- 转置存储：直接把 K 存成 `[b, head, head_dim, seq]`，算分时 `Q @ K` **不用再转置**，少一个运行时算子。

**转置就是"最后两维行列互换"**（前面的 `batch`、`head` 不动，`transpose(2,3)` 只换第 2、3 维）：

```
K 原形:  [b, head, seq,      head_dim]
                     └──互换第2、3维──┘
K 转置:  [b, head, head_dim,  seq     ]
```

**为什么这么换刚好能省转置？** 矩阵乘 `A @ B` 的铁律是"A 的列数 = B 的行数"（中间维必须对齐）。看 `Q @ K` 的末两维：

```
   Q(末两维)      @    K转置(末两维)   =   分数矩阵
[seq, head_dim]   @  [head_dim, seq]  =  [seq, seq]
        └────── head_dim 对齐 ──────┘
                （能直接乘，不用再转）
```

若 K **不**转置（末两维是 `[seq, head_dim]`），`Q @ K` 的中间维是 `head_dim` vs `seq`，对不齐 → 必须先 `K.transpose(2,3)` 把它掰成 `[head_dim, seq]` 才能乘。转置存储等于**把这一步提前在存缓存时做好**，算注意力时白捡一个"免转置"。

呼应 `forward_conv` 里算注意力那段——转置存储时直接 `matmul(query, key)`，否则才 `matmul(query, key.transpose(2,3))`：

```148:157:example1/llm_utils/qcqwen2_adaptation.py
        if advance_attention_div:
            if transposed_key_cache:
                attn_weights = torch.matmul(query_states, key_states)
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) 
        else:
            if transposed_key_cache:
                attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
```

> **端侧动机**：少一次转置 = 少一个算子、少一次数据搬运（[附录E](./02-附录E-端侧定长与计算图导出.md) 说过端侧的大头开销在"搬数据"而非"算"）；而且转置后的内存布局对 HTP 的 matmul 更友好。这也是 `update` 里 K 要沿 `-1` 维拼、V 仍沿 `-2` 维拼的原因——**K 的 seq 维因转置跑到了最后一维**。

#### 4.2·深入 · transpose 到底改了什么？为什么端侧才真省搬运

这里有个极易搞反的点：**`transpose` 本身不搬物理内存**，它只改"怎么解读这块内存"的元信息（`stride`）。省搬运省的是**端侧那一次物理重排**，不是 transpose 这行代码。

**① 张量 = 一条 1D 内存 + 元信息（shape / stride）**

内存永远是一条连续长条，`stride` 是一张换算表：**沿第 i 维走一步，要在这条长条上跨几个元素**。以 2×3 为例：

```
物理内存:  a   b   c   d   e   f
地址:      0   1   2   3   4   5      ← 这条永远不变

shape=[2,3]  stride=[3,1]
  dim0(行) stride=3：换到下一行要跨过一整行(3个)
  dim1(列) stride=1：右移一个元素就挨着
取元素公式：地址 = Σ 下标[i]×stride[i]
  [1,2]=f → 1×3+2×1 = 5 ✓
```

**② 连续张量的 stride = "右边所有维度 size 的乘积"**

```
stride[最后一维] = 1
stride[i]       = shape[i+1] × stride[i+1]
shape=[2,3] → stride[1]=1, stride[0]=3×1=3 → [3,1]
```

**③ `transpose(0,1)` 只是把两维的 (shape, stride) 对调 → O(1)、零拷贝**

```
转置前:  shape=[2,3]  stride=[3,1]
            └ 交换 dim0/dim1 ┘
转置后:  shape=[3,2]  stride=[1,3]   ← 物理内存 a b c d e f 一字节没动
```

验证（物理没变，只是换 stride 解读）：

```
转置后逻辑     地址 = i×1 + j×3
 a d    [0,1] → 0×1+1×3=3 → d ✓
 b e    [1,0] → 1×1+0×3=1 → b ✓
 c f
```

**④ 从 stride 一眼看"连不连续"：看最后一维 stride 是不是 1**

```
转置前 stride=[3,1] → 末维=1 → 连续 ✓（按逻辑读 a b c d e f，地址 0 1 2 3 4 5 递增）
转置后 stride=[1,3] → 末维=3 → 不连续 ✗（按逻辑读 a d b e c f，地址 0 3 1 4 2 5 跳跃）
```

> "不连续" = **逻辑相邻的元素（a 和它下面的 d）在物理内存里隔得远**（地址 0 与 3）。物理内容没改，改的只是"读的跳法"。

**⑤ 端侧为什么"存成转置的"能省搬运**

关键：**PyTorch 的 matmul 能忍受带 stride 的"跳着读"，但端侧 NPU 的 matmul 通常要求输入物理连续**（它成块 tile 搬进来算，跳跃地址会让搬运崩）。所以喂给 NPU matmul 前，不连续的张量必须先**物理复制一份摆连续**——这次复制就是"数据搬运"。

| 存法 | 缓存里 K 的布局 | 算 `Q@Kᵀ` 时 |
|------|--------------|-------------|
| **A 不转置** | `[.., seq, head_dim]` | 需要 `[.., head_dim, seq]` → K 是不连续转置视图 → **每步物理复制重排一次**（真搬运）|
| **B 转置存**（本项目）| `[.., head_dim, seq]` | 正好是 matmul 要的连续布局 → **直接读，不复制** |

省的就是 A 里"每算一次注意力就把 K 物理重排一遍"那次复制。

> 追问：B 写缓存时新 k 不也要转一下吗？会，但只对**本次新增的一小段**（decode 1 个 / prefill 1073）转；而**已累积的大段历史**一直躺在连续布局里，后续每步直接用、再不重转。A 则每步都要把"历史+新"整段重排，越往后越亏。

**一句话**：`transpose` = 只对调 (shape, stride) 的零拷贝视图，不动物理内存；"不连续"是末维 stride≠1 导致的读法跳跃；端侧省的是"因 NPU 只吃连续数据、不转置存就得每步物理重排 K"那次真复制。

### 4.3 重写 `get_seq_length`：按缓存实际长度返回

```298:303:example1/llm_utils/qcqwen2_adaptation.py
def DynamicCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.value_cache) <= layer_idx:
        return 0
    return self.value_cache[layer_idx].shape[-2]
```

- 用 **V**（而不是 K）去读长度：因为 V **没被转置**，它的 seq 维稳定在 `-2`，读长度不受 `transposed_key_cache` 影响，最稳妥。
- 缓存还没建（`len <= layer_idx`）返回 0，表示"没有历史"。

### 4.4 只存新 KV，为什么反而能"定长导出"？

这是本篇最容易绕晕的地方，先立一条分界线（这条线不画清楚，后面全是糊的）：

```
┌──────────────────────────────┐     ┌────────────────────────────────────────┐
│  模型 self.model             │     │  外部包装器 LLMForwardPassManager(fpm)   │
│  = 真正被导出成 ONNX 的静态图  │◀───▶│  = 模拟"端侧运行时"，本身不会被导出        │
│  内部所有张量形状全部写死      │     │  负责维护那块"会变长"的历史缓冲区          │
└──────────────────────────────┘     └────────────────────────────────────────┘
        被导出/编译的只有左边              纯 Python 外壳，端侧由高通运行时(Genie/QNN)承担
```

> ⚠️ 先分清两个词：这里说的**"模型内部" = 左边那张"前向计算图"**（网络结构 + 一组具体输入形状 trace 出来的算子/张量流），**不是**抽象的网络结构本身——网络结构对序列长度是"无所谓"的，是**导出 trace 那一刻**才把长度写死进图里。详见 [附录E · 2.1 网络结构 vs 计算图](./02-附录E-端侧定长与计算图导出.md#21-网络结构-vs-计算图先分清模型内部到底指谁)。

#### （4.4-a）"shape 恒定"到底指谁恒定——三个编译期常数

"只存新 KV" 说的是**模型的输出**；"shape 恒定、可定长导出"说的是**模型这张图里所有张量的形状**。这俩别混。图里所有长度都来自三个**导出前就定死的常数**（下面直接用本项目 `config.yaml` 的**真实值**）：

| 常数 | 代码 | 含义 | 本项目值 |
|------|------|------|---------|
| `max_tokens` | `tokenizer.model_max_length = context_length`（`llm_quant.py:99`）| KV 容量**上限** | **2048** |
| `num_tokens`(ARN) | `ARN = _quant['arn']`（`llm_quant.py:83`）| 每次 forward **并行处理几个新 token** | **1073** |
| `past_size` | `max_tokens - num_tokens` | 喂进来的**历史 KV 槽位长度** | **975** |

```19:20:example1/config.yaml
  context_length: 2048
  arn: 1073
```

于是模型图里每层 K/V 的输入输出形状**永远是同一套数字**（`2048 = 975 历史 + 1073 新`）：

```
past_key_i_in   [b, head, head_dim, 975]    ← past_size，永远 975
      │  模型内部拼上本次新算的 1073 份 → [.., 2048] 算注意力（也是定值）
      ▼
past_key_i_out  [b, head, head_dim, 1073]   ← num_tokens(ARN)，永远 1073
```

**关键①**：`past_key_i_in` 的长度是 `past_size=975`（固定），**不是"实际已处理多少词"**。历史实际不足 975（如刚开始只有 300 份）时补零 padding 到 975、再用 mask 屏蔽。所以图形状永不变。

**关键②（易错点）**：**"只存最新的" ≠ "只存 1 份"**。"最新的"数量 = `num_tokens` = ARN，本项目 = **1073**，不是 1。只有 ARN=1（纯逐词 decode）时才等于 1 份。"只存新的"真正的含义是**不在模型内部累积历史**（历史交给外部），而不是"只留一个 token"。

**这一趟前向的进 / 出全景**（把"1073 是新词、975 是历史 KV"分清）：

| 方向 | 张量 | 长度 | 是什么 |
|------|------|------|--------|
| 进 | `input_ids` / `inputs_embeds` | **1073** | 本次要处理的**新 token 本身（词）**——不是 KV |
| 进 | `past_key_i_in` / `past_value_i_in` | **975** | **历史 token 的 KV**（唯一从外部拿的 KV cache）|
| 出 | `logits` | 1073 | 每个新 token 的预测 |
| 出 | `past_key_i_out` / `past_value_i_out` | **1073** | 本趟**自己新算的 KV**（"只存最新的"指这个）|

> ⚠️ 别把 1073 当成"从外部拿的 KV"：1073 是喂进来的**新词**，它们的 KV 是**图内自己算**的，算完作为 `past_key_i_out` 吐回外部；外部真正喂进来的 KV 只有 975 份历史。

> 反推官方 `DynamicCache`：它把历史越拼越长存在**模型内部**（100→101→102…），这个长度是**图的一部分** → 每步 trace 出的图都不一样 → 编译失败。所以"**只吐新的**"是定长的因，"**历史当定长输入喂进来**"是定长的果。

#### （4.4-b）"丢给外部"是什么、在哪——`LLMForwardPassManager`

"外部" = 包着模型的 `LLMForwardPassManager`（代码里叫 `fpm`）。它每次调用干三件事，变长的脏活全在 `prepare_*` 里：

```620:624:example1/llm_utils/forward_pass_wrapper.py
    def __call__(self, *args, **kwargs):
        prepared_inputs, kvcache_info_bundle = self.prepare_inputs(*args, **kwargs)
        outputs = self.model(**prepared_inputs)
        prepared_outputs = self.prepare_outputs(outputs, prepared_inputs, kvcache_info_bundle)
        return prepared_outputs
```

**① `prepare_inputs`：把变长历史"补齐成定长"再喂给模型。** 真实历史可能只有 300 长，模型却只吃固定长度，于是算差多少、补多少零：

```477:493:example1/llm_utils/forward_pass_wrapper.py
        desired_kv_length = self.max_tokens - self.num_tokens
        kv_padding_length = max(desired_kv_length - kv_length, 0)
        kvcache_info_bundle['kv_padding_length'] = kv_padding_length

        past_key_values_extension = get_padded_kv_values(past_size=kv_padding_length,
                                                         num_layers=self.num_layers,
                                                         hidden_size=self.embed_dim,
                                                         num_attention_heads=self.num_heads,
                                                         num_kv_heads=self.num_kv_heads,
                                                         transposed_key_cache=self.transposed_key_cache,
                                                         device=self.device,
                                                         dtype=self.dtype)
        past_key_values = self._update_kv_cache(past_key_values_extension, past_key_values, desired_kv_length)

        attention_mask_extension = torch.zeros((batch_size, kv_padding_length), dtype=torch.long,
                                               device=self.device)
        attention_mask = torch.cat((attention_mask_extension, attention_mask), dim=1)
```

- `kv_length` 真实历史长度（变，如 300）；`desired_kv_length` 模型要的固定长度（如 1023）；`kv_padding_length = 1023-300 = 723` → 造 723 长的零缓存拼到前面。
- 同时把 mask 对应位置也填 0 → **补零 padding + mask 屏蔽**，模型永远只看到 1023 长的输入。

**② `prepare_outputs`：把模型吐的"新 KV"拼回历史缓冲区，超长就滚掉最旧的。**

```601:616:example1/llm_utils/forward_pass_wrapper.py
        new_past_key_values = _get_past_kv_from_outputs(outputs)   # 模型吐的新 KV
        ...
        old_past_key_values = _get_past_kv_from_prepared_inputs(prepared_inputs)  # 老历史
        ...
        past_key_values = self._update_kv_cache(
            old_past_key_values,
            new_past_key_values,
            current_kv_length_with_padding_removed
        )
```

`_update_kv_cache` 里就是"拼接 + 超长裁剪"两步，`_do_shift` 切掉最前面的即"丢最旧历史"，这就是那块**固定大小缓冲区的滚动（滑动窗口）**：

```318:324:example1/llm_utils/forward_pass_wrapper.py
            next_key_value = _do_concat(prev_key_value, new_key_value, key_dim, value_dim)  # 老 ⊕ 新

        shift_size = next_key_value[0][1].shape[-2] - max_cache_size
        if shift_size > 0:
            next_key_value = _do_shift(next_key_value, key_dim, value_dim, shift_size)  # 超长丢最旧
```

**③ 外部循环：把上一轮历史喂进下一轮。** `slice_inputs_and_run_successive_kvcache_inference` 每次切 `num_tokens` 个词跑一遍，并把上轮输出的 `past_key_values` 塞进下轮输入：

```644:659:example1/llm_utils/forward_pass_wrapper.py
        if input_ids is not None:
            cur_outputs = fpm(input_ids=input_ids[:, max(0, idx - fpm.num_tokens):idx], **kwargs)
        ...
        kwargs['past_key_values'] = outputs['past_key_values'] = cur_outputs['past_key_values']
```

最后一行就是"外部持有历史、循环回喂"的字面体现。**真实端侧上，`prepare_inputs/outputs` + 这个循环由高通运行时承担**，所以它们不需要、也不能被编进那张定长图里。

#### （4.4-c）整条链（把上面两段串起来）

```
              ┌────────── 外部 fpm（不导出；端侧由运行时干）───────────┐
真实历史(变长) │ prepare_inputs：补零到 975 + mask 屏蔽                │
  ≤975 ─────▶│         │  （另外喂进 1073 个新词 input_ids）          │
              │         ▼  喂进定长历史 [.., 975]                     │
              │   ┌──────────────────────────────┐                  │
              │   │ self.model 静态图              │ ← 只有这里被导出   │
              │   │ 形状全写死                     │                  │
              │   │ 吃975历史KV + 算1073新→拼2048算注意力→吐[..,1073]新KV│ │
              │   └──────────────────────────────┘                  │
              │         │ 新 KV [.., 1073]                          │
              │         ▼                                           │
              │ prepare_outputs：老历史 ⊕ 新KV，超975丢最旧(滚动)     │
              │         │                                           │
              └─────────┼─────────────────────────────────────────┘
                        ▼  下一轮回喂
```

从模型这张图的视角看：历史输入永远 `[.., 975]`、新 KV 输出永远 `[.., 1073]`、内部注意力永远按 2048 算，**从头到尾没变过** → 可定长导出；真正"变长"的历史管理被完全推到外部那块固定缓冲区里。这和 [附录E](./02-附录E-端侧定长与计算图导出.md) 掩码改造是**同一套"内部搬外部"**的思路。

> 上图的 1073 是本项目 **prefill（ARN/BERT）图**的值；decode 图是另一份配置（`arn:1`），此时新 KV 输出才是 `[.., 1]`、历史槽为 2047。两张图共享权重、共用同一套改写机制，只是 `num_tokens` 不同。为什么 prefill 能一次 1073、decode 只能 1 → 见 **2.2.1 节**。

### 4.5 和"结构树 / dummy 输入"的衔接

KV Cache 不是孤立的，它和端侧导出的输入输出直接挂钩。`llm_quant.py` 造 dummy 输入时，会**预分配一块定长的 past_key_values** 喂给模型：

```295:302:example1/llm_quant.py
    inputs['past_key_values'] = get_padded_kv_values(past_size=max_tokens - num_tokens,
                                                     num_layers=num_layers,
                                                     hidden_size=hidden_size,
                                                     num_attention_heads=num_attention_heads,
                                                     num_kv_heads=num_kv_heads,
                                                     transposed_key_cache=config.transposed_key_cache,
                                                     device=device,
                                                     dtype=dtype)
```

导出的 ONNX 图里，每层的 K/V 都是**成对的输入和输出**（`past_key_i_in` / `past_value_i_in` 进，`past_key_i_out` / `past_value_i_out` 出）：

```339:357:example1/llm_quant.py
def _get_past_key_values_names(sfx, n_layers):
    all_kvs = []
    for i in range(n_layers):
        all_kvs.append(f'past_key_{i}_{sfx}')
        all_kvs.append(f'past_value_{i}_{sfx}')
    return all_kvs
...
    input_names += _get_past_key_values_names('in', llm_config.num_hidden_layers)
    output_names = ['logits'] + _get_past_key_values_names('out', llm_config.num_hidden_layers)
```

**这就把 3.1 的"外部管理历史"落地了**：

```
外部(端侧运行时)                模型(定长静态图)
  持有全部历史 K/V   ──past_key_i_in──▶   读历史 + 算新token的K/V
  拼接/滚动更新历史  ◀──past_key_i_out──   只吐出新算的 K/V(return_new_key_value_only)
```

模型内部 shape 永远固定 → 满足端侧定长导出；历史的"变长"完全甩给外部循环去管。

---

## 五、因果链（串起来记）

```
自回归逐词生成
      │  历史 token 的 K/V 每步都一样 → 重算浪费(O(n²))
      ▼
缓存 K/V，只算新 token → KV Cache（decode 每步恒定算 1 个）
      │  但官方 DynamicCache 缓存越拼越长 → shape 动态
      ▼
端侧要定长/固定计算图（附录E）
      │
      ▼
重写 update：return_new_key_value_only → 模型只吐新 KV，历史交给外部输入输出
重写 update：transposed_key_cache      → K 转置存，省一次运行时转置
重写 get_seq_length：用未转置的 V 读长度
      │
      ▼
每层 K/V 成对做 ONNX 输入输出 → 模型内部 shape 恒定 → 可定长导出、端侧高效跑
```

---

## 六、记忆锚点

- **KV Cache 本质**：缓存历史 token 的 **K、V**（不缓存 Q），让 decode 每步只算 1 个新 token，从 O(n²) 降到每步 O(n)。
- **Q 和谁算**：新 token 的 `Q_t` 和历史**全部** K/V（含自己、不含未来）打分加权；注意力遍历全历史这步省不掉，KV Cache 省的是"重新生成历史 K/V"这步。
- **两阶段**：prefill 一次性写满整段 prompt 的 K/V；decode 逐词读历史 + 追加 1 份。
- **生命周期**：一次 prompt = **1 次 prefill + N 次 decode**，直到 `<eos>` 或到上限；**多轮对话**下一轮 prefill 会复用上一轮 (prompt+回答) 的 KV 当历史（这就是 prefill 也留 975 历史槽的意义），独立新请求则清空从零；总长受 `context_length=2048` 约束，超限滑窗丢最旧。
- **官方版**：`DynamicCache` 用两个 list 存，每步沿 seq 维 `cat`，**越存越长**（shape 动态，端侧不友好）。
- **本项目改 2 个方法**：`update`（`return_new_key_value_only` + `transposed_key_cache`）、`get_seq_length`。
- **`return_new_key_value_only=True`**：返回值给注意力用的是"历史⊕新"的完整 K/V，但**存进缓存/吐给外部的只有新 K/V** → 历史由外部管，模型内部定长。
- **`transposed_key_cache=True`**：K 存成 `[.., head_dim, seq]`，算 `Q@K` 免去运行时转置；所以 K 沿 `-1` 拼、V 沿 `-2` 拼，`get_seq_length` 用未转置的 V 读长度。
- **transpose 是零拷贝视图**：只对调两维的 (shape, stride)，**不搬物理内存**；`stride[i]`=沿第 i 维走一步跨几个元素，末维 stride≠1 即"不连续"。省搬运省的是**端侧那一次**——NPU matmul 只吃连续数据，不转置存就得每步把 K **物理复制重排**一次（见 4.2·深入）。
- **为什么"只存新的"反而定长**：图里长度全来自三个编译期常数 `max_tokens` / `num_tokens`(ARN) / `past_size = max_tokens-num_tokens`；本项目真实值 **2048 / 1073 / 975**。`past_key_i_in` 恒为 `past_size=975`（不够补零 + mask 屏蔽），与"实际处理多少词"无关。
- **"只存最新的" ≠ size=1**：吐出的新 KV 长度 = ARN = 1073（只有 ARN=1 才是 1 份）；"只存新的"= 不累积历史，非"只留一个 token"。
- **进/出别混**：从外部拿的 KV 只有 **975 历史**；**1073 是新词（input_ids）**，其 KV 图内自算、算完吐 1073 份出去；内部拼 `975+1073=2048` 算注意力。
- **1073 从哪来**：来自 **prefill/ARN(BERT) 图**——整段 prompt 已知可并行处理 1073 个；decode 图 ARN=1 逐词。两图共享权重（见 2.2.1）。
- **"模型内部" = 计算图，不是网络结构**：结构对长度无所谓，是 trace 那一刻把长度写死进图；只需保证每次 trace 出的图形状是同一套常数。
- **"外部"在哪**：`LLMForwardPassManager`（`prepare_inputs` 补零喂入、`prepare_outputs` 拼接+滚动、`slice_..._inference` 循环回喂）；端侧上这层由高通运行时承担，不进定长图。
- **落地**：每层 K/V 做成 ONNX 成对输入输出（`past_key_i_in/out`），把"变长"甩给外部，图本身定长。
- **一句话**：KV Cache 省重算；端侧改写把"变长的历史"挪到模型外部，让内部保持定长可导出——和掩码改造是同一套"内部搬外部"的思路。

---

## 七、待深入（自己往下填）

- [x] 外部（端侧运行时）到底怎么拼接/滚动 K/V？超过最大长度时怎么滑窗？→ 已答，见 **第四节 4.4**：`prepare_inputs` 补零喂入、`prepare_outputs` 用 `_update_kv_cache`（`_do_concat` 拼接 + `_do_shift` 丢最旧）滚动、`slice_..._inference` 循环回喂。（对照 [附录E](./02-附录E-端侧定长与计算图导出.md) 缺点2、3 的解法）
- [ ] `update` 的 `else` 分支里 `self.value_cache[layer_idx] = key_cache` 看着像笔误（把 key 赋给了 value）；因本项目 `return_new_key_value_only=true` 不走该分支所以无影响——确认是否确为无用分支的遗留 bug。
- [ ] `cache_kwargs` 里传了 `sin`/`cos`/`cache_position` 但当前 `update` 没用到，它们原本给谁用？（对比官方静态缓存 `StaticCache`）
- [ ] K/V 缓存被量化到 8bit（`llm_quant.py` 里 `set_matmul_second_input_producer_to_8bit_symmetric` 的注释提到"reduce data I/O costs associated with KV-cache"），这一步和本篇的关系？
- [~] prefill 图和 decode 图的 `num_tokens`（ARN）分别是多少？两张图怎么共享权重？→ 已答一半，见 **2.2.1 节**：本项目 prefill `arn=1073`（`past_size=975`）、decode `arn=1`（`past_size=2047`），两图共享同一份权重、共用同一套 KV 改写。**"怎么共享权重"的导出/加载细节仍待深入**。
