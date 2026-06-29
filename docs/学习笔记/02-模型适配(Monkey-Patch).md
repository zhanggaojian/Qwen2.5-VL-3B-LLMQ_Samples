# 02 · 模型适配（Monkey Patch）

> **流程位置**：读完 `config.yaml` 之后、加载模型之前（`llm_quant.py` 第 50-61 行）。
> **一句话本质**：在**不修改官方模型源码**的前提下，运行时把官方 Qwen2 的几个"零件"**替换成高通量化专用版本**。这种"运行时替换"叫 Monkey Patch（猴子补丁）。

---

## 一、为什么需要适配？

官方 `modeling_qwen2.py` 是为 **GPU 训练/推理** 写的。而本项目要把模型跑在**高通端侧 NPU(HTP)** 上并做量化，硬件有特殊偏好：

| 端侧硬件的需求 | 对应的适配改动 |
|----------------|----------------|
| Conv 算子比 Linear 更高效 | 把 `Linear` 全换成 `1x1 Conv2d` |
| 输入定长、掩码由外部传入 | 绕过动态生成因果掩码 |
| KV Cache 要可控（定长/转置） | 重写 `DynamicCache.update` |
| 量化要能感知加法等算子 | 用 `aimet_torch` 的 `Add` 替代 `+` |

> 关键点：**官方文件一行没改**（所以它仍是"官方原版"），所有改动都在 `qcqwen2_adaptation.py` 里以"补丁"形式注入。

---

## 二、核心机制：Monkey Patch 的两种手法

### 手法 A：直接赋值 / setattr 替换

```python
modeling_qwen2.QWEN2_ATTENTION_CLASSES['eager'] = QcAttention   # 换整个类
setattr(modeling_qwen2.Qwen2MLP, 'prepare_conv', MLP_prepare_conv)  # 给类加方法
```

### 手法 B：`update_attr(...)` —— 带"备份+校验"的替换

```python
# qcqwen2_adaptation.py 第 306-313 行
def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):                       # 1. 原方法必须存在
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))  # 2. 先备份原版
            setattr(cls, attr_name, new_attr)         # 3. 再替换
        return True
    return False                                       # 找不到就返回 False
```

它比 setattr 多了两层保险：
- **校验**：原方法不存在就返回 `False`，配合外层 `assert` 提示"transformers 版本对不上"。
- **备份**：把原方法存成 `_original_xxx`，需要时能还原。

> 第 52-54 行用 `A or B`：因为不同 transformers 版本里因果掩码方法可能叫 `_update_causal_mask`，也可能叫 `_prepare_decoder_attention_mask`，两个都试一遍，有一个成功即可。

---

## 三、逐行拆解（第 50-61 行）

| 行 | 代码 | 替换了什么 | 作用 |
|----|------|-----------|------|
| 51 | `QWEN2_ATTENTION_CLASSES['eager'] = QcAttention` | 注意力类 | 换成量化版注意力 |
| 52-54 | `update_attr(..., bypass_update_causal_mask)` | 因果掩码生成方法 | 改成"直接用外部传入的掩码" |
| 55 | `setattr(Qwen2MLP,'prepare_conv',...)` | 给 MLP 加方法 | 用于把 Linear 转 Conv |
| 56 | `setattr(Qwen2MLP,'forward_conv',...)` | 给 MLP 加方法 | Conv 版前向计算 |
| 57 | `setattr(Qwen2ForCausalLM,'prepare_conv',...)` | 给输出头加方法 | lm_head 转 Conv |
| 58-59 | `update_attr(DynamicCache,'update',...)` | KV 缓存更新逻辑 | 定长/转置缓存 |
| 60-61 | `update_attr(DynamicCache,'get_seq_length',...)` | KV 缓存长度查询 | 适配新缓存结构 |

---

## 四、四个被替换的零件，原理说明

### 1. QcAttention（注意力）— `qcqwen2_adaptation.py` 第 62-212 行

> 📎 Attention 本身是什么、QKV/多头/RoPE/KVCache 怎么回事、官方源码逐段对照，见独立笔记：[02-附录A-Attention注意力机制.md](./02-附录A-Attention注意力机制.md)。

继承官方 `Qwen2Attention`，主要改动：
- **Linear → 1x1 Conv2d**：`prepare_conv()` 里新建 `q/k/v/o_proj_conv`，并把原 Linear 的权重 `[:, :, None, None]` 拷进去（升 2 维以适配 Conv），然后删掉原 Linear。
- **量化感知加法**：`self.attn_add = Add()`（来自 `aimet_torch`），把 `attn_weights + attention_mask` 换成可被量化工具识别的算子。
- **逐层掩码缩放**（第 162-173 行）：第 0 层 `mask*2`、第 27 层 `mask*10`、其余正常。这是为应对某些层注意力数值范围过大、量化易溢出而做的**经验性微调**。
- **支持外部 RoPE**：`position_ids` 若是 tuple，直接用传入的 cos/sin（`_apply_rope_single`），适配端侧导出。
- **transposed_key_cache**：可把 K 转置后再算 `Q·K`，省掉一次 transpose。

### 2. bypass_update_causal_mask（因果掩码）— 第 215-217 行

#### 背景：什么是因果掩码（Causal Mask）？

因果掩码是 Transformer 在**文本生成类任务**中用的"遮挡规则"，作用是：**模型在预测某个位置时，只能看到它前面的词，看不到后面的词。**

- **为什么需要**：语言模型是**自回归**的（一个词一个词往外蹦），预测第 5 个词时只能依据第 1-4 个词。但注意力机制天生是"全局"的——每个位置能同时看到所有位置。若训练时让它偷看到后面的"答案"，就学不会真正预测，等于作弊。因果掩码就是用来"挡住未来"。
- **长什么样**：注意力会算出一个 `序列长度 × 序列长度` 的分数矩阵（第 i 行 = 第 i 个词对其他词的关注度）。因果掩码把**上三角（未来位置）置成负无穷**：

```
        词1    词2    词3    词4
词1  [  0    -inf  -inf  -inf ]
词2  [  0     0    -inf  -inf ]
词3  [  0     0     0    -inf ]
词4  [  0     0     0     0   ]
```

- `0`：允许关注（自己和前面的词）；`-inf`：禁止关注（后面的词）。
- 这个矩阵**加到注意力分数上**再过 softmax，加了 `-inf` 的位置概率变成 0，相当于"看不见"。
- 允许的部分正好是**下三角（含对角线）**，所以也叫"下三角掩码"。

> **连回 config**：FP16 数值范围有限，不能真用 `-inf`，得用"足够小"的负数代替——这正是 `config.yaml` 里 `mask_neg_fp16: -50 / mask_neg_fp32: -100` 的来历。

#### 这里的适配改动

```python
def bypass_update_causal_mask(self, attention_mask, *args, **kwargs):
    return attention_mask   # 原样返回，不再动态生成
```
官方会在 forward 里**动态构造**下三角因果掩码（`_update_causal_mask`）；端侧改成**掩码由模型输入直接喂进来**（对应 config 的 `use_combined_mask_input`），所以这里直接透传。

> ⚠️ **别误解**：改的是掩码的"**来源**"（动态生成 → 外部输入），**不是"有没有掩码"**。原始模型有、训练时更必须有，这里只是换了供货渠道。

> **为什么这么改**：端侧推理输入定长、掩码预先算好由外部喂入，比每次在模型里动态构造更高效，也更利于导出成固定计算图(ONNX)。
>
> 📎 三个概念（动态生成 / 端侧定长 / 固定计算图导出）+ "为什么端侧要定长" 详解见独立笔记：[02-附录E-端侧定长与计算图导出.md](./02-附录E-端侧定长与计算图导出.md)。

### 3. MLP / lm_head 转 Conv — 第 220-263 行

`MLP_prepare_conv` / `MLP_forward_conv`：把 `gate/up/down_proj` 三个 Linear 换成 1x1 Conv2d；前向时 `reshape + transpose` 成 4D 喂给 Conv，算完再变回来。`ForCausalLM_prepare_conv` 同理处理最后的 `lm_head`。

> 数学上 1x1 Conv 与 Linear 等价，但 HTP 硬件对 Conv 支持更好/更快。
>
> 📎 原理详解（Linear/Conv 是什么、为何等价、怎么搬权重）见独立笔记：[02-附录B-Linear与Conv算子转换.md](./02-附录B-Linear与Conv算子转换.md)（这块还在深入研究中）。

### 4. DynamicCache 改写（KV 缓存）— 第 266-303 行

- `return_new_key_value_only=True` 时，缓存里**只保留新算的 K/V**（端侧 KVCache 模式：历史 KV 由外部管理）。
- `transposed_key_cache` 决定 K 在哪个维度拼接（`-1` vs `-2`）。
- `get_seq_length` 改成按新结构取序列长度。

---

## 五、和 config.yaml 的对应关系

适配代码里读的这些开关，都来自 `config.yaml` 的 `model_overrides`（在第 80-88 行 setattr 到 `llm_config`）：

| config 开关 | 在适配代码里的作用 |
|-------------|--------------------|
| `use_combined_mask_input` | 配合 `bypass_update_causal_mask`，掩码外部传入 |
| `return_new_key_value_only` | 控制 KV Cache 只存新值 |
| `transposed_key_cache` | 控制 K 是否转置存储 |
| `use_position_embedding_input` | 配合外部 RoPE（cos/sin 外部传入） |
| `attn_implementation: eager` | 决定用 `QWEN2_ATTENTION_CLASSES['eager']`，即 QcAttention |

---

## 六、怎么"学"这一段（方法建议）

1. **先记住本质**：这就是"运行时换零件"，原版不动。不用通读 1552 行官方文件。
2. **建立 4 大块概念**：Attention / MLP / Causal Mask / KV Cache 各是干嘛的。
3. **用三段式读 Qc 实现**：对每个零件问"原版怎么做 → 改成什么 → 为什么改"。
4. **官方文件当字典查**：只在"看不懂改了什么"时，去 `modeling_qwen2.py` 查对应原版实现做对比。

---

## 七、待确认 / 疑问（自己往下填）

- [ ] `prepare_conv` 是在哪一步被实际调用的？（提示：搜 `prepare_conv(` 在 llm_quant.py 的调用处）
- [ ] 第 162-173 行逐层 `mask*2 / mask*10` 的倍数是怎么定出来的？
- [ ] `aimet_torch` 的 `Add` 和普通 `+` 在量化时差别具体是什么？
