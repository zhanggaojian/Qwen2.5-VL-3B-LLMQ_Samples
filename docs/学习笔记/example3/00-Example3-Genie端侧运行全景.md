# E3-00 · Example3：Genie 端侧运行全景

> **在整个流程中的位置**：Example1 量化与导出 → Example2 编译模型 → Context Binary → **Example3 在设备上加载并推理**。
>
> **一句话本质**：Example3 不再改模型，而是把已经编译好的模型、运行时和一次图文请求的输入准备好，交给 Genie 在 HTP/NPU 上完成自回归生成。

## 一、Example3 分成哪几步

| 阶段 | 做什么 | 主要产物或输入 |
|---|---|---|
| 1. 准备 Embedding | 导出整张词向量表，并生成本次图文请求的融合向量 | `embedding_weights_151936x2048.raw`、`inputs_embeds.bin` |
| 2. 准备运行时 | 把 QNN、Genie 库和可执行程序推到设备 | `libQnn*.so`、`libGenie.so`、`genie-t2t-run` |
| 3. 准备模型与配置 | 推送 Context Binary、Genie 配置、Tokenizer 和 HTP 配置 | `*.serialized.bin`、JSON 文件、`tokenizer.json` |
| 4. 端侧运行 | 设置库路径并启动 Genie | 逐 token 生成的文本 |

当前先学第 1 阶段。设备库、JSON 配置和实际运行留到后面的分篇。

## 二、两个 Embedding 文件不是一回事

### 1. `embedding_weights_151936x2048.raw`

脚本从 `llm.get_input_embeddings().weight` 导出完整的 token Embedding 权重表：

```text
151936 个 token × 每个 token 2048 个浮点数
```

它相当于一本模型级的“token ID → 2048 维向量”字典：

- 对同一个模型通常可以重复使用；
- 不依赖本次图片和问题；
- 脚本显式以 FP32 导出。

### 2. `inputs_embeds.bin`

它是**本次请求**的输入序列向量。换一张图片或换一句问题，都应重新生成。

脚本的真实流程是：

```text
图片 + prompt
  → Processor 生成聊天模板、input_ids、pixel_values、image_grid_thw
  → 根据 input_ids 查 Embedding 表，得到初始序列向量
  → ViT 根据图片得到 image_embeds
  → 找到序列中的图片占位 token
  → 用 image_embeds 替换这些占位位置的向量
  → inputs_embeds.bin
```

因此，它不是简单的：

```text
[文本向量][图像向量]
```

而更接近：

```text
[聊天模板文本][图像向量所在的位置][用户问题][等待回答的位置]
```

最终张量可理解为 `[1, 本次序列长度, 2048]`；序列长度会随聊天模板、图片 token 数量和问题长度变化。

## 三、`genie-t2t-run` 三个参数

仓库 README 给出的意图是：

```bash
./genie-t2t-run \
  -c qwen25vl3B_os.json \
  -e inputs_embeds.bin \
  -t embedding_weights_151936x2048.raw
```

| 参数 | 交给 Genie 的内容 | 回答的问题 |
|---|---|---|
| `-c` | Genie 配置 JSON | 模型怎么加载、Context Binary 在哪、上下文长度和 HTP 参数是什么？ |
| `-e` | 当前请求的融合 Embedding | 这一次要让模型看什么图片、理解什么问题？ |
| `-t` | 完整 token Embedding 权重表 | 生成出一个 token ID 后，下一轮解码怎样把它变成 2048 维输入向量？ |

为什么 `-e` 和 `-t` 都需要：

```text
Prefill：先用 -e 提供整段图文问题的向量
   ↓
模型生成第 1 个 token ID
   ↓
Decode：用 -t 把这个 token ID 查成向量，作为下一轮输入
   ↓
继续生成后续 token
```

也就是说：

- `-e` 是“一次请求的数据”；
- `-t` 是“整个模型可复用的查表权重”。

## 四、哪些是固定的，哪些每次会变

| 类型 | 内容 | 什么时候变化 |
|---|---|---|
| 模型级静态文件 | Context Binary、Embedding 权重表 | 更换模型或重新编译时 |
| 运行配置 | Genie JSON、HTP JSON | 更换模型结构、切分方式、SoC 或运行策略时 |
| 文本规则 | `tokenizer.json` | 更换模型或 Tokenizer 时 |
| 请求级输入 | `inputs_embeds.bin` | 图片、prompt 或聊天模板变化时 |

## 五、当前仓库需要先知道的四个问题

这些问题不会妨碍理解流程，但会妨碍直接运行：

1. README 写的是 `qwen2.5vl.json`，仓库实际文件是 `qwen25vl3B_os.json`。
2. README 写的是 `input_embeds.bin`，导出脚本实际生成 `inputs_embeds.bin`。
3. Genie 配置列出 6 个 `*_of_6.serialized.bin`，但 HTP 图名和当前 Example2 配置是 `1_of_1`，切分数尚未统一。
4. JSON 声明输入 Embedding 为 FP32，但脚本导出 `inputs_embeds.bin` 前没有显式 `.float().astype(np.float32)`，实际运行前要核对 dtype 和文件大小。

另外，脚本中的模型路径、图片路径和 `cuda:1` 都是示例作者的硬编码，不能原样照搬。

## 六、本节学到哪里

本节只需要记住：

```text
Example3 = 准备一次图文 Embedding
         + 加载已经编译好的 Context Binary
         + Genie 在设备上执行 prefill/decode
```

下一节再进入 `qwen25vl3B_os.json`：重点看它怎样把 AR128 的 prefill 图和 AR1 的 decode 图组织成一次完整生成。

## 小结

Example3 的第一个关键不是再次量化模型，而是区分模型级的完整 Embedding 表和请求级的 `inputs_embeds.bin`；前者供 token 查表复用，后者已经融合本次文字与图片，作为 prefill 的起始输入。
