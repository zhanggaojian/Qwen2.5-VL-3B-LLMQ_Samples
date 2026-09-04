# 04 · PPL 困惑度评估

> **关联**：主线脚本 `example1/llm_quant.py`：`ppl_eval` / `ppl_eval_embedding`（L157-212）、浮点模型评估（L245-249）、量化后评估（L574）。承接 [03-模型与Tokenizer加载与config覆盖](./03-模型与Tokenizer加载与config覆盖.md) 之后。
> **一句话本质**：PPL（Perplexity，困惑度）= **交叉熵损失取 exp**，衡量"模型对真实文本的预测有多不确定"。本项目用它**对比量化前后精度掉了多少**——量化后 PPL 涨得越少越好。
> **面试复习**：简短问答与 `切分函数 → FPM → ppl_eval` 三层职责见 [附录A · PPL 面试速答](./04-附录A-PPL面试速答.md)。

> **本篇按四段式组织**：**① 介绍/为什么 → ② 原理 → ③ 官方/通用做法 → ④ 本项目改造后做法**。

---

## 一、介绍：PPL 是什么、为什么要用

### 1.1 直觉

PPL 衡量"模型读到真实的下一个词时有多**意外**"：

- 模型对真实文本预测得越准（给正确的下一个 token 越高的概率）→ 越不意外 → **PPL 越低**。
- 预测得越差 → 越意外 → **PPL 越高**。

可以粗略理解成"模型在每一步平均要在多少个词里犹豫"。PPL=10，约等于每步都在 10 个候选里纠结；PPL=1 表示完全确定（理想）。

### 1.2 为什么量化流程要用它

量化会把权重压到低比特（本项目权重 4bit、激活 16bit），必然带来精度损失。需要一个**客观数字**回答"损失大不大"：

- 只看单条输出好坏太主观；
- PPL 用一批文本算平均，稳定、可复现、行业通用；
- **同一份测试集**上比"浮点模型 PPL" vs "量化模型 PPL"，差距小说明量化成功。

所以 PPL 是本项目的**精度守门员**：改造/量化每动一步，都用它验证没把模型搞坏。

---

## 二、原理

### 2.1 数学定义：PPL = exp(交叉熵)

语言模型是自回归的：给定前文，预测下一个 token 的概率分布。对一段真实文本 $x_1, x_2, \dots, x_N$：

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i \mid x_{<i})\right)$$

括号里就是**平均负对数似然**，也正是**交叉熵损失（CrossEntropyLoss）**。所以：

$$\boxed{\text{PPL} = \exp(\text{平均交叉熵 loss})}$$

代码里就是这两行：

```181:182:example1/llm_quant.py
    loss = loss / num_batches
    ppl = loss.exp()
```

#### 2.1.1 训练和 PPL 都使用平均 loss 吗？

**通常都使用平均 loss，但用途不同**：

- **训练**：对当前 batch 中所有有效 token 的 loss 求平均，然后用这个标量执行 `backward()`，计算梯度并更新权重。
- **PPL 评估**：对整个评估范围内所有有效 token 的负对数概率求平均，然后执行 `exp`，得到困惑度；不做反向传播，也不更新权重。

PyTorch 的 `CrossEntropyLoss()` 默认参数是 `reduction="mean"`。假设错位后共有 $M$ 个有效预测位置，每个位置的 loss 是 $l_i$，那么：

$$\text{average loss} = \frac{1}{M}\sum_{i=1}^{M}l_i$$

训练时：

```text
当前 batch 的平均 loss
        ↓ backward()
计算梯度
        ↓ optimizer.step()
更新模型权重
```

PPL 评估时：

```text
全部有效 token 的平均 loss
        ↓ exp
得到 PPL
```

顺序不能交换：

```text
正确：PPL = exp(mean(token_loss))
错误：PPL = mean(exp(token_loss))
```

如果 label 使用默认忽略值 `-100`，对应位置不会计入 `CrossEntropyLoss` 的平均值。当前项目的 PPL label 通常都是普通 token id，因此被保留下来的 token 基本都会参与 loss。

本项目先得到每个 batch 的平均 loss，再把这些 batch loss 相加并除以 `num_batches`。当前 DataLoader 是 `batch_size=1`，样本通常又被整理成相同长度，所以这种“batch 均值再平均”通常等价于对所有 token 统一平均；如果各 batch 的有效 token 数不同，则应按有效 token 数加权，而不能直接平均各 batch 的均值。

### 2.2 Teacher forcing + 错位对齐（shift）

评估时用**真实前文**去预测下一个词（不是用模型自己生成的，叫 teacher forcing）。关键是"第 i 个位置的输出，要去预测第 i+1 个 token"，所以要把 logits 和 labels **错开一位**：

```173:174:example1/llm_quant.py
        shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
        shift_labels = batch['input_ids'][..., 1:].contiguous().to(shift_logits.device)
```

- `shift_logits`：取前 N-1 个位置的输出（每个位置预测"下一个"）→ 去掉最后一个（它没有下一个可对）。
- `shift_labels`：取第 1~N 个真实 token 作为答案 → 去掉第 0 个。
- 对齐后：`shift_logits[i]` 的预测目标就是 `shift_labels[i]`。

```text
文本:   [A,   B,   C,   D]
logits:  ↓predict B  ↓predict C  ↓predict D  ↓(丢弃)
labels:      B        C        D
```

- 转 fp32 算 loss：交叉熵含 `log`/`exp`，用 fp32 更稳（呼应前面 softmax 也升 fp32）。

### 2.3 CrossEntropyLoss

```175:179:example1/llm_quant.py
        loss_fct = CrossEntropyLoss()
        loss += loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
```

- `view(-1, vocab)`：把 `[B, seq-1, vocab]` 摊平成 `[B*(seq-1), vocab]`，每行是一个位置对整个词表的打分。
- `shift_labels` 中保存的是每个位置的**真实 token id**，不是 one-hot 概率，也不一定对应一个完整的自然语言“词”。
- `CrossEntropyLoss` 内部先 `log_softmax`，再根据 label 取出真实 token 对应的负对数概率，最后对所有有效位置求平均。
- 每个 batch 的 loss 累加，最后 `/num_batches` 求平均再 `exp` → PPL。

#### 2.3.1 Label 如何从 vocab_size 个 logits 中确定 loss

假设词表只有 5 个 token，某个位置的模型输出为：

```text
logits = [1.0, 3.0, 0.5, 2.0, -1.0]
label  = 1
```

`label=1` 表示这个位置的真实答案是词表中 ID 为 1 的 token。Softmax 后假设得到：

```text
token 0：0.08
token 1：0.59  ← label 指向的真实 token
token 2：0.05
token 3：0.27
token 4：0.01
```

这个位置的交叉熵就是：

$$l=-\log p(\text{label})=-\log(0.59)\approx0.528$$

因此可以直观理解为：

```text
根据 label 找到真实 token
        ↓
取得模型分给真实 token 的概率
        ↓
计算 -log(真实 token 概率)
        ↓
得到该位置的 loss
```

- 真实 token 概率越高，loss 越小；例如 $-\log(0.9)\approx0.105$。
- 真实 token 概率越低，loss 越大；例如 $-\log(0.01)\approx4.605$。
- 其他 token 虽然不是 label，但它们的 logits 会进入 Softmax 分母，所以仍会影响真实 token 的概率和 loss。

对于某个位置的 logits $z_1,\dots,z_V$ 和真实 token id $y$，交叉熵为：

$$
l=-\log\frac{e^{z_y}}{\sum_j e^{z_j}}
  =-z_y+\log\sum_j e^{z_j}
$$

实际实现通常不会先显式保存完整 Softmax 概率。PyTorch 把 `log_softmax` 和负对数似然组合起来，通过数值稳定的 `logsumexp` 直接从 logits 计算结果：

```text
CrossEntropyLoss = LogSoftmax + NLLLoss
```

最后，`CrossEntropyLoss(reduction="mean")` 再对所有有效位置的 $l$ 求平均，得到训练或 PPL 使用的平均 loss。

### 2.4 关键澄清：PPL 评估用 teacher forcing，是"测量"不是"生成"（无 decode）

最容易混的一点：**PPL 评估 ≠ 推理生成**。它用真实 token 当上下文（teacher forcing），和训练一样，不走 decode 循环。三个场景要分清：

| 场景 | 每步喂什么 | 方式 | 目的 |
|---|---|---|---|
| **训练** | 真实 token（teacher forcing）| 并行一次前向 | 算 loss 更新权重 |
| **PPL 评估**（本篇）| **真实 token（teacher forcing）**| 并行一次前向 | **量**模型对真实文本的概率 |
| **推理生成**（prefill+decode）| 模型**自己生成**的 token | decode 循环 | **产**新内容 |

**为什么评估能用真实 token**：评估时测试集里有**完整真实文本**（标准答案已知），于是直接问"给真实前文，模型给真实的下一个词打多少概率"——需要真实 token 当上下文，不需要模型自己生成。而真实生成时没有标准答案，模型只能把自己的输出喂回去（decode）。

**为什么不能用生成的 token 算 PPL**：
- PPL 定义就是"真实文本的概率" $p(x_i \mid x_{<i})$，前文和答案都必须是真实的；
- 若用生成 token 当上下文 → 一步错步步错、结果不可复现，也答不出"模型对原文有多确定"。

**怎么做到"用真实上下文预测下一个"**：靠**因果掩码 + 一次并行前向**（不是循环生成）。把整段真实 `input_ids` 一次喂入，因果掩码保证位置 i 只看得到 0~i 的真实 token（看不到未来），所以每个位置天然是"在真实前文下预测下一个"，且所有位置并行算完。这正是 2.2 的 shift 能成立的前提——**输入端喂的整段都是真实 token**。

> ⚠️ 别把 4.2 的分块误会成 decode：`slice_..._kvcache_inference` 是把**真实长文本**切成 `num_tokens=ARN` 的块，**每块内部并行 teacher forcing**（一次吃 ARN 个真实 token），块间用 KV Cache 接历史——这更像**连续的 prefill**，不是"一次一个自回归生成"的 decode。用 KV Cache 只是为了历史接得上、路径和端侧一致，不代表在生成。

### 2.5 为什么用交叉熵？有没有别的办法？（面试深追向）

**先破一个错觉**：不是"算 PPL 时挑了交叉熵这个工具"，而是 **PPL 的定义本身就是 `exp(交叉熵)`**——2.1 的公式里，括号中的"平均负对数似然"在信息论里就叫交叉熵。所以"PPL 为什么用交叉熵"≈"正方形面积为什么用边长平方"，它就是这么定义的。真正该问的是下一层：**为什么用交叉熵定义"语言模型好不好"？**

**交叉熵凭什么合理（三个硬道理）**：

1. **信息论意义直观**：交叉熵 = "用模型分布编码真实文本，平均每 token 花多少 nat/bit"；取 `exp` 后的 PPL 就是"**有效候选数**"（PPL=10 ≈ 每步在 10 个词里犹豫），可解释。
2. **只依赖"模型给真实词的概率"、无需生成**：一次 teacher forcing 前向即可算，**稳定可复现**（不像生成一步错步步错）。
3. **和训练目标一致**：训练最小化的就是交叉熵，评估用同一把尺 → 分数直接反映训练目标达成度。本质是**最大似然**："让真实文本在模型眼里概率最大"，交叉熵越小=似然越大=PPL 越低。

**有没有其他办法——分清你要衡量"什么"**：

| 类别 | 指标 | 说明 |
|------|------|------|
| **A. 同源变体**（换汤不换药）| NLL、BPC / BPB | NLL 就是交叉熵不取 exp；BPC/BPB 换底为 2、按字符/字节归一（跨 tokenizer 更公平）。本质仍是交叉熵。 |
| **B. 语言模型质量（任务层面）** | 下游准确率(MMLU/VQA)、BLEU/ROUGE、人工评测 | 直接测"能不能把事做对"，但要真实**生成**、依赖标注、主观/昂贵、方差大。 |
| **C. 衡量"量化掉多少"（★本项目更实用）** | **KL 散度**、**MSE/L2**、余弦相似度、**SQNR** | 直接比"量化输出 vs 浮点输出"的差异，**不需标签、可逐层定位哪层坏**。本项目 **SeqMSE 标定就是用 MSE**。 |

> **为什么本项目仍主用 PPL**：PPL 用**行业通用、可复现、和训练目标一致**的单一数字，衡量"改造/量化后语言建模能力掉多少"，且只要一次前向。而 KL/MSE/SQNR 更多是**量化内部调试**用——两者是**配合**关系：**底层用 MSE 标定（SeqMSE）、顶层用 PPL 验收**。

---

## 三、官方 / 通用做法

标准的 HuggingFace PPL 评估通常是：

```python
model.eval()
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)  # HF 内部自动 shift
    loss = outputs.loss
ppl = torch.exp(loss)
```

- 直接把 `labels=input_ids` 传给模型，HF 内部自动做 shift + CrossEntropyLoss。
- 一次前向吃完整个序列（长文本用滑动窗口切）。
- 模型是标准 GPU 版，不涉及定长/KV Cache 外部管理。

即：通用做法"一把梭"，模型自己算 loss。

---

## 四、本项目改造后的做法

本项目模型是**端侧定长改造版**（KV Cache 外部管理、掩码/RoPE 外部化、定长 ARN 模式），不能像官方那样"一把梭"，所以 PPL 计算包了两层：**外层算 loss + 内层分块喂模型**。

### 4.1 两个评估函数：`ppl_eval` vs `ppl_eval_embedding`

```157:166:example1/llm_quant.py
def ppl_eval(data_loader, forward_pass_manager, num_batches=10):
    ...
        outputs = slice_inputs_and_run_successive_kvcache_inference(forward_pass_manager, input_ids=batch['input_ids'])
        lm_logits = outputs["lm_logits"].cpu()
```

两者逻辑几乎一样，区别只在**喂什么输入**：

| 函数 | 输入 | 用途 |
|---|---|---|
| `ppl_eval` | `input_ids`（token 序列）| 纯文本模型 |
| `ppl_eval_embedding` | `input_embeddings`（embedding 向量）| 多模态（图像/视频已转成 embedding）|

本项目 `use_input_embeddings=true`（Qwen2.5-VL 多模态），走 `ppl_eval_embedding`；若是纯文本，则代码直接把它别名成 `ppl_eval`：

```243:244:example1/llm_quant.py
if not llm_config.use_input_embeddings:
    ppl_eval_embedding = ppl_eval
```

### 4.2 内层：定长分块跑 KV Cache 推理

`slice_inputs_and_run_successive_kvcache_inference` 把长序列切成 `num_tokens`（=ARN=1073）大小的块，**逐块喂模型、KV Cache 依次往下传、logits 拼接**：

```637:659:example1/llm_utils/forward_pass_wrapper.py
    for idx in range(0, input_length, fpm.num_tokens)[::-1]:
        idx = input_length - idx
        ...
        cur_outputs = fpm(input_ids=input_ids[:, max(0, idx - fpm.num_tokens):idx], **kwargs)
        ...
        outputs['lm_logits'] = torch.cat(
            (outputs.get('lm_logits', ...), cur_outputs['lm_logits']), dim=1)
        kwargs['past_key_values'] = outputs['past_key_values'] = cur_outputs['past_key_values']
```

- **为什么要分块**：模型是定长图，一次只能吃 `num_tokens` 个 token（见 [附录E](./02-附录E-端侧定长与计算图导出.md)）。长文本必须切成块，模拟端侧"一段一段处理、KV Cache 累积历史"的真实运行方式。
- **KV 往下传**：每块输出的 `past_key_values` 作为下一块的历史输入（呼应 [附录K](./02-附录K-KV%20Cache(键值缓存).md) 的"历史交外部管"）。
- 最终把每块的 `lm_logits` 拼成完整序列的输出，交给外层算 PPL。

这样评估用的前向路径，和真实端侧推理**完全一致**——所以 PPL 能真实反映端侧表现。

### 4.3 `LLMForwardPassManager`：定长前向的统一入口

```237:241:example1/llm_quant.py
orig_fpm = LLMForwardPassManager(cfg=llm_config,
                                 model=model,
                                 tokenizer=tokenizer,
                                 separate_tuple_input_output=False,
                                 num_tokens=ARN)
```

`fpm`（forward pass manager）封装了定长前向要准备的一切：外部 KV 缓冲、合并掩码、RoPE 的 cos/sin、`num_tokens=ARN` 等。`ppl_eval` 只管调 `fpm(...)`，不用关心定长细节。

### 4.4 在主线里被调用三次：量化前后对比

PPL 是全流程的"精度探针"，在三个关键节点各测一次：

| 节点 | 代码位置 | 测什么 |
|---|---|---|
| 浮点模型 | L245-249 | 改造后、量化前的 fp 模型基线 PPL |
| prepare 后 | L429 | 模型准备阶段后是否掉精度 |
| 量化 sim 后 | L574 | 量化模拟后的 PPL（和基线比差多少）|

```245:249:example1/llm_quant.py
with torch.no_grad():
    with orig_fpm.place_on_device("cuda"):
        orig_ppl = ppl_eval_embedding(test_dataloader, orig_fpm, num_batches=_eval_cfg['ppl_num_batches'])

print(f"ppl score of original fp model: {orig_ppl}")
```

- `torch.no_grad()`：评估不需要梯度，省显存。
- `place_on_device("cuda")`：临时把模型搬 GPU 上算，算完释放。
- `num_batches`：由 `config.yaml` 的 `ppl_num_batches=10` 控制，只跑 10 个 batch 快速估计。

**判读**：`量化后 PPL` 相比 `浮点 PPL` 只小幅上升 → 量化成功；若暴涨 → 说明量化把模型搞坏了，需要回头调（混合精度、SeqMSE 等）。

### 4.5 数据从哪来

```214:233:example1/llm_quant.py
if not llm_config.use_input_embeddings:
    from llm_utils.wikitext_dataloader import get_wiki_dataset
    train_dataloader, test_dataloader, _ = get_wiki_dataset(context_length, tokenizer, cache_dir)
else:
    from llm_utils.qwen2_5_vl_dataloader import get_qwen_dataset
    ...
    train_dataloader, test_dataloader, dataset = get_qwen_dataset(model.model, llava_dataset_setting, ...)
```

- 纯文本：WikiText 数据集（经典 PPL 基准）。
- 多模态（本项目）：LLaVA 数据集，图像先过视觉编码器转成 embedding。
- `test_dataloader` 供 PPL 评估，`train_dataloader` 后面供量化标定（compute_encodings / SeqMSE）用。

---

## 五、记忆锚点

- **PPL = exp(平均交叉熵 loss)**，越低越好；直觉="模型每步在多少个词里犹豫"。
- **为什么用交叉熵**：PPL 本就定义为 exp(交叉熵)，交叉熵是信息论上衡量语言模型最自然、和训练目标一致的尺子；替代品=同源变体(NLL/BPC) 或换衡量对象(下游准确率、量化专用的 KL/MSE/SQNR)。量化里 SeqMSE 用 MSE 标定、PPL 做顶层验收（见 2.5）。
- **作用**：量化流程的精度守门员，比"浮点 PPL vs 量化 PPL"，差距小=量化成功。
- **shift 错位**：`logits[:-1]` 预测 `labels[1:]`，teacher forcing 逐 token 对齐；loss 用 fp32 算。
- **评估 ≠ 生成**：PPL 用 teacher forcing（真实 token 当上下文）、因果掩码 + 并行前向，是"测量"不是"生成"，**没有 decode 循环**；decode（喂自己生成的 token）只属于真实推理生成。分块是"连续 prefill"，非 decode。
- **两个函数**：`ppl_eval`（input_ids，纯文本）/ `ppl_eval_embedding`（embedding，多模态），纯文本时前者别名后者。
- **内层分块**：`slice_...kvcache_inference` 把长序列切 `num_tokens=ARN` 块、KV 往下传、logits 拼接——和端侧真实前向一致。
- **测三次**：浮点(L247)→ prepare后(L429)→ 量化sim后(L574)，全程盯精度。

---

## 六、待深入（自己往下填）

- [ ] 严格 PPL（全 token 统一平均）与本项目"batch 均值再平均"的数值差异有多大？
- [x] `LLMForwardPassManager` 与切分/PPL 函数如何分工？→ 见 [04-附录A · PPL 面试速答](./04-附录A-PPL面试速答.md)；内部细节见 [05 · 通用前向处理流程](./05-通用前向处理流程.md)。
- [ ] 多模态 PPL 里图像 embedding 与文本 token 如何拼接、mask 怎么处理？
- [ ] `num_batches=10` 够不够稳？增大对 PPL 估计的方差影响多大？
- [ ] prepare 前后 PPL 若有变化，通常是哪一步引入的（算子替换 / dtype）？
