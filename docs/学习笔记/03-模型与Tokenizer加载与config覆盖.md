# 03 · 模型 / Tokenizer 加载 与 config 覆盖

> **关联**：主线脚本 `example1/llm_quant.py` 第 64-104 行；承接 [02-模型适配(Monkey-Patch)](./02-模型适配(Monkey-Patch).md)（挂补丁）之后、[04 PPL 评估] 之前。
> **一句话本质**：加载官方权重 → 用 `config.yaml` 的开关覆盖模型 config（总开关面板）→ 遍历触发 `prepare_conv` 真正执行 Linear→Conv。核心思想是**"配置驱动行为"**。

> **本篇按四段式组织**：**① 介绍/为什么 → ② 原理 → ③ 官方/通用做法 → ④ 本项目改造后做法**。
> **背景附录**：`from_pretrained` 加载的 `.safetensors` 及各类权重格式 → [03-附录A-模型权重文件格式](./03-附录A-模型权重文件格式.md)。

---

## 一、介绍：这一步在流程里的位置 & 为什么需要

整条主线：

```
读配置(config.yaml)          ✅
  → 模型适配(挂补丁 L50-61)    ✅ 上一站：把 QcAttention / prepare_conv / DynamicCache 改写"挂"到官方类上
  → 加载模型/Tokenizer + config覆盖   ← 本篇 03
  → PPL 评估 / prepare / 量化 / 导出
```

**为什么需要单独这一步**：上一站的适配只是"改了官方类的定义"，还没有真正的模型对象。这一步要做三件事，缺一不可：

1. **加载官方权重**，得到一个真正能跑的 `model`；
2. **把端侧开关写进 config**，让适配代码运行时知道该走哪条分支（只存新 KV？K 转置？掩码/RoPE 外部化？）；
3. **真正执行 Linear→Conv 转换**（适配只是"挂上方法"，这里才落地）。

---

## 二、原理

### 2.1 `from_pretrained` 加载机制

`AutoConfig.from_pretrained` / `Qwen2ForCausalLM.from_pretrained` 会：
1. 读模型仓库里的 `config.json`（结构参数）→ 得到 `config` 对象；
2. 按 `config` 里的类型去**实例化网络结构**（各层、各子模块）；
3. 加载 `*.safetensors` 权重、填进对应模块。

**关键**：第 2 步实例化时用的"类"，是从 `modeling_qwen2` 模块里查的。而这些类**已经在上一站被 Monkey Patch 替换过了**，所以建出来的网络天然是改造版。

### 2.2 配置驱动行为（config → 分支）

适配代码里大量出现 `self.config.xxx` 判断，例如：

```105:107:example1/llm_utils/qcqwen2_adaptation.py
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False
        advance_attention_div = self.config.advance_attention_div if hasattr(self.config, 'advance_attention_div') else False
```

这些属性官方 `config.json` 里**没有**，是本项目在 03 这一步用 `setattr` 注入的。所以：

```
config.yaml (model_overrides)
    │ setattr 写进
    ▼
llm_config.xxx
    │ from_pretrained(config=llm_config) 传给每个子模块的 self.config
    ▼
forward_conv 运行时读取 → 决定走哪条端侧分支
```

这就是"改一个 yaml 开关 = 改变模型运行行为"的完整链路。

### 2.3 Monkey Patch 的时序（为什么"加载出来就是适配版"）

```
L50-61 先打补丁：
    QWEN2_ATTENTION_CLASSES['eager'] = QcAttention
    Qwen2MLP.prepare_conv = MLP_prepare_conv
    Qwen2ForCausalLM.prepare_conv = ForCausalLM_prepare_conv
    DynamicCache.update / get_seq_length = 改写版
        ↓
L101 才 from_pretrained：内部按（已被替换的）类实例化
        ↓
每个注意力层天生是 QcAttention、MLP 天生带 prepare_conv、Cache 天生是改写版
```

**顺序不能反**：必须先补丁、后加载。若先加载再补丁，已经实例化的层不会自动变成改造版。

---

## 三、官方 / 通用做法

标准 HuggingFace 加载模型就三行，**不覆盖任何 config、不做结构转换**：

```python
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, config=config)
# 直接就能 model.generate(...)
```

- 用官方 `config.json` 原样，不注入端侧开关。
- 权重是 `nn.Linear`，不转 Conv。
- 注意力用官方 `Qwen2Attention`，KV Cache 用官方 `DynamicCache`。

即：官方加载出来是**通用 GPU 推理版**。

---

## 四、本项目改造后的做法（逐段）

### 4.1 加载官方 config + 打印结构（L78-81）

```78:81:example1/llm_quant.py
llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
context_length = _quant['context_length']
print(f'num_layer: {llm_config.num_hidden_layers}, context_length : {context_length},'
      f'num_hidden_size :{llm_config.num_attention_heads},  num_kv_heads: {llm_config.num_key_value_heads}')
```

- `trust_remote_code=True`：允许加载 Qwen 仓库自带的自定义建模代码。
- 本项目真实结构参数（来自 `config.json`）：

| 参数 | 值 | 含义 |
|------|----|----|
| `num_hidden_layers` | **36** | Transformer 层数（层索引 0~35）|
| `hidden_size` | 2048 | 隐藏维度 |
| `num_attention_heads` | 16 | Q 头数 |
| `num_key_value_heads` | 2 | K/V 头数（GQA，`n_rep=16/2=8`）|
| `intermediate_size` | 11008 | MLP 中间维 |
| `vocab_size` | 151936 | 词表大小 |
| `rope_theta` | 1e6 | RoPE 频率底数 |
| `rope_scaling` | mrope, `[16,24,24]` | mRoPE 分段 |

> ⚠️ 注意：模型是 **36 层**，代码里对 `layer_idx==0` / `layer_idx==27` 的掩码特殊放大，第 27 层是**中间偏后的某一层**（实测数值范围异常），并非最后一层（最后是第 35 层）。

### 4.2 覆盖端侧开关：总开关面板（L85-95）⭐核心

```85:95:example1/llm_quant.py
setattr(llm_config, 'return_new_key_value_only', _overrides['return_new_key_value_only'])
setattr(llm_config, 'transposed_key_cache', _overrides['transposed_key_cache'])
setattr(llm_config, 'use_combined_mask_input', _overrides['use_combined_mask_input'])
setattr(llm_config, 'use_position_embedding_input', _overrides['use_position_embedding_input'])
setattr(llm_config, "use_cache", _overrides['use_cache'])
setattr(llm_config, '_attn_implementation', _overrides['attn_implementation'])
setattr(llm_config, '_attn_implementation_internal', _overrides['attn_implementation'])
setattr(llm_config, 'mask_neg', _overrides['mask_neg_fp16'] if enable_fp16 else _overrides['mask_neg_fp32'])
setattr(llm_config, 'pretraining_tp', _overrides['pretraining_tp'])
setattr(llm_config, 'use_input_embeddings', _overrides['use_input_embeddings'])
setattr(llm_config, 'use_mrope', _overrides['use_mrope'])
```

`_overrides` = `config.yaml` 的 `model_overrides` 段。逐个开关对应的适配行为：

| 开关 | 本项目值 | 驱动的适配行为 | 关联 |
|---|---|---|---|
| `return_new_key_value_only` | true | KV Cache 只吐新 KV、历史交外部管 | [附录K](./02-附录K-KV%20Cache(键值缓存).md) |
| `transposed_key_cache` | true | K 转置存储、打分免运行时转置 | [附录K](./02-附录K-KV%20Cache(键值缓存).md) |
| `use_combined_mask_input` | true | 因果掩码外部合并好当输入喂入 | [附录E](./02-附录E-端侧定长与计算图导出.md) |
| `use_position_embedding_input` | true | RoPE 的 cos/sin 外部预算好喂入 | [附录G](./02-附录G-RoPE位置编码.md) |
| `use_cache` | true | 启用 KV Cache | [附录K](./02-附录K-KV%20Cache(键值缓存).md) |
| `attn_implementation` | eager | 选 `QWEN2_ATTENTION_CLASSES['eager']`（已被换成 `QcAttention`）| [Monkey-Patch](./02-模型适配(Monkey-Patch).md) |
| `mask_neg` | -50 / -100 | 掩码负值大小（fp16 用 -50、fp32 用 -100）| [附录A](./02-附录A-Attention注意力机制.md) |
| `pretraining_tp` | 1 | 张量并行度=1（关闭并行切分）| — |
| `use_input_embeddings` | true | 用外部 embedding 作输入（多模态需要）| — |
| `use_mrope` | false | 是否走 mRoPE 分段（此处关）| [附录G](./02-附录G-RoPE位置编码.md) |

**两个细节**：
1. `_attn_implementation='eager'` + L51 `QWEN2_ATTENTION_CLASSES['eager']=QcAttention` → "选 eager"= "选中改造版注意力"。同时写了 `_attn_implementation` 和 `_attn_implementation_internal` 两个属性，覆盖 transformers 不同版本的读取名。
2. `mask_neg` 按精度分档：fp16 表示范围小，-50 够压；fp32 用 -100。呼应"掩码要够负才能屏蔽未来"。

### 4.3 加载 Tokenizer（L97-99）

```97:99:example1/llm_quant.py
os.environ['TOKENIZERS_PARALLELISM'] = '0'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
tokenizer.model_max_length = context_length
```

- Tokenizer 做"文字 ↔ token id"互转，后面 PPL 评估、造标定数据都要用。
- `model_max_length = context_length`（2048）：限制最长序列，和定长导出对齐。
- `TOKENIZERS_PARALLELISM='0'`：关分词器多线程，避免和后面 DataLoader 多进程冲突刷警告。

### 4.4 加载模型（L101）：权重官方、结构改造、行为由开关控制

```101:101:example1/llm_quant.py
model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(model_id, config=llm_config)
```

三者在这一行合体：
- **权重**：官方预训练权重（`from_pretrained` 加载）；
- **结构**：改造版（因补丁在前，实例化时用的是 `QcAttention` 等）；
- **行为**：由 `config=llm_config` 里那批开关控制。

### 4.5 触发 prepare_conv：真正执行 Linear→Conv（L102-104）

```102:104:example1/llm_quant.py
for name, module in model.named_modules():
    if hasattr(module, "prepare_conv"):
        module.prepare_conv()
```

- 适配只是给类"挂上 `prepare_conv` 方法"，**这里遍历所有子模块、谁有就调谁的**，才真正：搬 Linear 权重进 1×1 Conv、切换 `forward→forward_conv`、删掉旧 Linear。
- 覆盖：每个 `QcAttention`（q/k/v/o）、每个 `Qwen2MLP`（gate/up/down）、顶层 `Qwen2ForCausalLM`（lm_head）。
- `named_modules()` 递归遍历整棵模型树，36 层每层都被处理。
- 这就是"`prepare_conv` 在哪被调用"的答案。

### 4.6 （附带）fp16 转换（L107-149）

```148:149:example1/llm_quant.py
if (enable_fp16):
    convert_model_to_fp16(model)
```

可选精度转换：开 `enable_fp16` 时把模型转半精度，但对 RMSNorm 的 `Pow`（平方）用 `PreCast` 升回 fp32 算、`Mul_1` 后 `PostCast` 降回 fp16。原因：**RMSNorm 的平方/求均值对精度敏感**，低精度易溢出/掉精度，故局部保 fp32。本项目 `enable_fp16=false`，默认不走。属数值稳定小技巧，了解即可。

---

## 五、记忆锚点

- **本篇本质**：加载官方 config → `setattr` 覆盖端侧开关（总开关面板）→ 加载 tokenizer → `from_pretrained` 加载权重（补丁在前，建出即改造版）→ 遍历触发 `prepare_conv`。
- **配置驱动行为**：yaml 开关 → `llm_config` 属性 → `from_pretrained(config=...)` → 各模块 `self.config` → `forward_conv` 运行时读取选分支。
- **时序铁律**：先打补丁（L50-61）后加载（L101），顺序不能反。
- **prepare_conv 落地点**：就在 L102-104 的遍历，适配"挂方法"、这里"真转换"。
- **真实结构**：36 层、hidden 2048、Q 头 16 / KV 头 2（GQA n_rep=8）、词表 151936。

---

## 六、待深入（自己往下填）

- [ ] `use_input_embeddings=true` 时，输入的 embedding 从哪来、怎么替代 token id？（多模态：图像/视频 token）
- [ ] `pretraining_tp` 张量并行在官方代码里具体影响哪段计算？
- [ ] `from_pretrained` 如何把 `config.json` 的 `architectures` 映射到具体类？（`AutoModel` 的注册机制）
- [ ] 36 层里为什么恰好第 27 层数值异常？是否和某种结构位置有关？
