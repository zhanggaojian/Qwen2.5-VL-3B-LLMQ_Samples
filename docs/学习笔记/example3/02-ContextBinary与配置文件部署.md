# E3-02 · Context Binary 与配置文件部署

> **在整个流程中的位置**：设备端 Genie/QNN 运行库准备好 → **放入模型与四类配置数据** → 设置环境变量并运行。
>
> **一句话本质**：Context Binary 提供“执行什么模型”，Genie 主配置负责把模型、Tokenizer 和 HTP 配置连接起来。

## 一、这一部分需要哪四类文件

| 文件 | 谁读取 | 主要作用 |
|---|---|---|
| `*.serialized.bin` | Genie → QNN HTP | 已针对 HTP 编译的模型图和权重，即真正执行的模型 |
| `qwen25vl3B_os.json` | `libGenie.so` | Genie 主配置，描述对话、采样、Tokenizer、后端和模型路径 |
| `tokenizer.json` | Genie Tokenizer | token 字符串与 ID 的转换规则及特殊 token 定义 |
| `htp_backend_ext_config.json` | QNN HTP 扩展 | 指定图、SoC、DSP 架构、核数、内存与性能配置 |

它们的关系是：

```text
genie-t2t-run -c qwen25vl3B_os.json
                   │
                   ├─ tokenizer.path ───────→ tokenizer.json
                   ├─ backend.extensions ──→ htp_backend_ext_config.json
                   └─ model.ctx-bins ──────→ *.serialized.bin
```

主配置本身不包含模型权重，只保存参数和文件路径。

## 二、Context Binary 从哪里来

Example2 中被注释的生成函数表达了以下设计：

```text
第 i 份 AR128 Quantized DLC
             +
第 i 份 AR1 Quantized DLC
             ↓
qnn-context-binary-generator
             ↓
weight_sharing_model_i_of_N.serialized.bin
```

因此，若模型切成 `N` 份，Example3 通常加载 `N` 个 Context Binary，而不是分别加载 `N` 个 AR128 和 `N` 个 AR1 文件。每个 Binary 按脚本意图包含同一 split 的 prefill/decode 图，并启用权重共享。

Context Binary 与普通 DLC 的区别：

| DLC | Context Binary |
|---|---|
| 模型的 QNN/DLC 表示 | HTP 可恢复和执行的序列化上下文 |
| 相对更接近模型中间产物 | 更接近设备运行产物 |
| 仍需进一步创建 HTP Context | 已保存创建后的图结构与设备相关优化信息 |

## 三、Genie 主配置分成什么

仓库实际文件是 `qwen25vl3B_os.json`，可分成五组理解。

### 1. 输入与上下文规格

| 字段 | 当前值 | 必须匹配什么 |
|---|---:|---|
| `embedding.size` | 2048 | `inputs_embeds.bin` 的最后一维和 Embedding Table |
| `embedding.datatype` | `float32` | 输入 Embedding 的实际 dtype |
| `context.size` | 2048 | 模型编译时的 `cl2048` 和 KV Cache 设计 |
| `n-vocab` | 151936 | Tokenizer 和 Embedding Table 的词表大小 |
| `bos-token` | 151643 | Tokenizer 中的特殊 token ID |
| `eos-token` | 151645 | Tokenizer 中的停止 token ID |

### 2. 采样规则

当前配置是：

```text
temp = 0
top-k = 1
greedy = true
```

这表示使用确定性较强的贪心生成：每一步主要选择概率最高的 token。

### 3. Tokenizer 路径

```json
"tokenizer": {
  "path": "tokenizer.json"
}
```

`tokenizer.json` 不包含模型的 Embedding 权重。它保存的是：

- 文本切分规则；
- token 字符串与 ID 的映射；
- 特殊 token；
- 输出 token ID 如何还原为文本。

本仓库 Tokenizer 中，ID `151643` 是 `<|endoftext|>`，ID `151645` 是 `<|im_end|>`，与 Genie 主配置相互对应。

### 4. 推理后端

```json
"type": "QnnHtp",
"extensions": "htp_backend_ext_config.json"
```

这表示 Genie 选择 QNN HTP/NPU 后端，并把更底层的 HTP 参数交给扩展配置。

`pos-id-dim`、`kv-dim`、`rope-theta` 等字段属于模型结构约束，不能因为更换设备就随意修改；`cpu-mask`、轮询和 mmap 等字段更偏运行策略。

### 5. Context Binary 列表

```json
"ctx-bins": [
  "models/weight_sharing_model_1_of_6.serialized.bin",
  "...",
  "models/weight_sharing_model_6_of_6.serialized.bin"
]
```

Genie 按这份列表找到并恢复全部模型 split。文件顺序、数量和内部图名必须与实际编译产物一致。

## 四、HTP 扩展配置管什么

`htp_backend_ext_config.json` 可以分成四组：

| 分组 | 当前配置示例 | 作用 |
|---|---|---|
| `graphs` | AR128、AR1 图名，`O=3`、VTCM、核数 | 指定图和图级优化/资源参数 |
| `context` | `weight_sharing_enabled=true` | 控制上下文和权重共享方式 |
| `devices` | `soc_id=72`、`dsp_arch=v81`、4 核 burst | 选择目标设备和 HTP 执行策略 |
| `memory` | `shared_buffer` | 指定 CPU/HTP 之间的内存方式 |

它不包含模型权重；它告诉 QNN HTP 应当怎样加载和执行 Context Binary。

## 五、LLM Binary 与 ViT Binary 的关系

当前仓库展示的是两段式多模态路径：

```text
ViT：图片 → vision embedding
LLM Genie：融合 embedding → 文本回答
```

当前 `qwen25vl3B_os.json` 的 `ctx-bins` 只列出 LLM 的权重共享 Binary，没有列出 ViT/VEG Binary。导出脚本默认在 GPU 上运行 ViT；若改为设备 ViT，则脚本注释要求用另一条 `qnn-net-run + veg.serialized.bin` 流程先获得 `vision_embedding.raw`。

所以 README 所说的“推送 LLM binary 或 ViT binary”并不代表当前 Genie 配置会自动把两者一起运行。

## 六、当前仓库还不能直接运行的原因

1. `example3/Qwen2.5-VL-3B/` 中没有 `models/` 目录，也没有任何 Context Binary。
2. Example2 当前 `num_splits = 1`，理论上对应一个 `*_1_of_1.serialized.bin`。
3. Example3 主配置却列出六个 `*_of_6.serialized.bin`。
4. HTP 扩展配置中的图名仍是 `ar128-cl2048_1_of_1` 和 `ar1-cl2048_1_of_1`，与六份 Binary 再次不一致。
5. README 写 `qwen2.5vl.json`，仓库实际文件是 `qwen25vl3B_os.json`。
6. `soc_id=72`、`dsp_arch=v81` 尚未根据真实设备确认。
7. Example2 脚本选择的 `NspTargets.Android.GEN4` 在本仓库映射为 `soc_id=69`、`v79`，又与已提交的 `72/v81` 配置不一致；恢复 Context 生成前必须先选定真实目标。

因此，真正部署前必须统一：

```text
模型版本
→ split 数量与文件顺序
→ AR128/AR1 图名
→ hidden size / vocab / Tokenizer / 特殊 token
→ Context Length / KV / RoPE 参数
→ SoC / DSP 架构 / QAIRT 版本
→ 所有相对文件路径
```

## 七、本节完成标准

本节只需要掌握：

```text
Context Binary = 模型本体
Genie JSON      = 总装配说明书
tokenizer.json  = 文字与 token ID 的翻译规则
HTP JSON        = 目标硬件与执行策略
```

下一节再学习设置环境变量并执行 `genie-t2t-run`，以及如何从启动日志判断失败发生在哪一层。

## 八、参考位置

- 本项目：[example3/README.md](../../../example3/README.md)
- 本项目：[qwen25vl3B_os.json](../../../example3/Qwen2.5-VL-3B/qwen25vl3B_os.json)
- 本项目：[htp_backend_ext_config.json](../../../example3/Qwen2.5-VL-3B/htp_backend_ext_config.json)
- Qualcomm 官方：[Genie 配置与 Qwen2.5-VL Bundle 示例](https://github.com/qualcomm/qai-appbuilder/blob/main/docs/genie_guide_en.md)

## 小结

Example3 第三部分把模型本体、文本规则和硬件运行参数交给 Genie：主配置负责连接所有文件，Context Binary 提供模型，Tokenizer 解释 token，HTP 扩展配置决定模型怎样在目标 NPU 上运行。
