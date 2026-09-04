# example1 学习笔记 · 总索引

> 本目录用「一个主题一个 md」的方式记录 example1（Qwen2.5-VL-3B LLM 量化）的学习过程。
> 这一篇是总地图，串起所有分篇笔记；每学完一块就更新对应分篇和下面的进度表。

## 一、整体流程地图

example1 做的事：**把官方 Qwen2.5-VL-3B 模型，改造并量化成可在高通端侧 (HTP/NPU) 运行的格式**。

主线脚本：`example1/llm_quant.py`，大致阶段：

```
读配置(config.yaml)
  → 模型适配(Monkey Patch，替换零件)
  → 加载模型 / tokenizer
  → 浮点模型评估(PPL)
  → 通用定长前向处理(FPM)
  → prepare(模型准备 / 导出中间结构)
  → 量化(QuantSim → SeqMSE → compute_encodings)
  → 量化后评估
  → 导出 Test Vector + ONNX + Encoding
  → example2 主机编译(AR 变体 → Split → MHA2SHA → 普通 DLC → Quantized DLC)
  → Context Binary（衔接内容；具体生成暂缓到实操前）
  → example3 端侧运行（核心概念已完成）
```

视觉分支并行主线：

```text
图片 → Qwen Processor → ViT/VEG → vision_embedding
  → 融合成 inputs_embeds.bin → 接入已学习的 example3 / LLM
```

> 配套项目文档：[docs/PIPELINE.md](../PIPELINE.md)（流程）、[example1/TROUBLESHOOTING.md](../../example1/TROUBLESHOOTING.md)（排错）、[example1/README.md](../../example1/README.md)。

## 二、分篇笔记

| 序号 | 主题 | 文件 | 状态 |
|------|------|------|------|
| 00 | 基础篇 · 模型与推理两阶段(prefill/decode) | [00-基础篇-模型与推理两阶段.md](./00-基础篇-模型与推理两阶段.md) | 学习中 |
| 01 | 配置文件 config.yaml 详解 | [01-配置文件config.yaml详解.md](./01-配置文件config.yaml详解.md) | 学习中 |
| 02 | 模型适配（Monkey Patch，第 50-61 行） | [02-模型适配(Monkey-Patch).md](./02-模型适配(Monkey-Patch).md) | 学习中 |
| 02-总结 | 模型适配总结篇 · 结构与替换全景 | [02-模型适配总结篇-结构与替换全景.md](./02-模型适配总结篇-结构与替换全景.md) | 学习中 |
| 03 | 模型/Tokenizer 加载与 config 覆盖 | [03-模型与Tokenizer加载与config覆盖.md](./03-模型与Tokenizer加载与config覆盖.md) | 学习中 |
| 03-附录A | 模型权重文件格式 | [03-附录A-模型权重文件格式.md](./03-附录A-模型权重文件格式.md) | 学习中 |
| 03-附录B | mmap 与数据搬运路径 | [03-附录B-mmap与数据搬运路径.md](./03-附录B-mmap与数据搬运路径.md) | 学习中 |
| 04 | PPL 困惑度评估是什么、怎么算 | [04-PPL困惑度评估.md](./04-PPL困惑度评估.md) | 学习中 |
| 04-附录A | PPL 面试速答 | [04-附录A-PPL面试速答.md](./04-附录A-PPL面试速答.md) | 学习中 |
| 05 | 通用前向处理流程 · LLMForwardPassManager | [05-通用前向处理流程.md](./05-通用前向处理流程.md) | 学习中 |
| 06 | Prepare 模型准备阶段 | [06-Prepare模型准备阶段.md](./06-Prepare模型准备阶段.md) | 已完成 |
| 06-附录A | Prepare Dummy Input 输入模具 | [06-附录A-Prepare-Dummy-Input输入模具.md](./06-附录A-Prepare-Dummy-Input输入模具.md) | 已完成 |
| 06-附录B | QAIRT、QNN、AIMET 与 QuantSim 概念关系 | [06-附录B-QAIRT-QNN-AIMET-QuantSim概念与关系.md](./06-附录B-QAIRT-QNN-AIMET-QuantSim概念与关系.md) | 已完成 |
| 06-附录C | QAIRT model_preparer 内部流程 | [06-附录C-QAIRT-model_preparer内部流程.md](./06-附录C-QAIRT-model_preparer内部流程.md) | 已完成 |
| 06-附录D | Prepare 面试速答 | [06-附录D-Prepare面试速答.md](./06-附录D-Prepare面试速答.md) | 已完成 |
| 07 | 量化主流程 · QuantSim 到 Encoding | [07-量化主流程-QuantSim到Encoding.md](./07-量化主流程-QuantSim到Encoding.md) | 学习中 |
| 07-附录A | QuantSim 模型骨架与 QDQ | [07-附录A-QuantSim模型骨架与QDQ.md](./07-附录A-QuantSim模型骨架与QDQ.md) | 学习中 |
| 07-附录B | Encoding 量化参数基础 | [07-附录B-Encoding量化参数基础.md](./07-附录B-Encoding量化参数基础.md) | 学习中 |
| 07-附录C | MatMul、Concat 与混合精度规则 | [07-附录C-量化规则配置-MatMul-Concat与混合精度.md](./07-附录C-量化规则配置-MatMul-Concat与混合精度.md) | 学习中 |
| 07-附录D | SeqMSE 权重量化优化 | [07-附录D-SeqMSE权重量化优化.md](./07-附录D-SeqMSE权重量化优化.md) | 已建立 |
| 07-附录E | `compute_encodings()` 激活标定 | [07-附录E-compute_encodings激活标定.md](./07-附录E-compute_encodings激活标定.md) | 已建立 |
| 07-附录F | 量化方法总览与选型 | [07-附录F-量化方法总览与选型.md](./07-附录F-量化方法总览与选型.md) | 已建立 |
| 08 | ONNX 导出与测试向量 | [08-ONNX导出与测试向量.md](./08-ONNX导出与测试向量.md) | 已建立 |
| 09 | 工程运行指南 · LLM 与 ViT 双链路 | [09-工程运行指南-LLM与ViT双链路.md](./09-工程运行指南-LLM与ViT双链路.md) | 已建立 |
| 10 | example1 产物总览与清理 | [10-example1产物总览与清理.md](./10-example1产物总览与清理.md) | 已建立 |
| 02-附录A | Attention 注意力机制 | [02-附录A-Attention注意力机制.md](./02-附录A-Attention注意力机制.md) | 学习中 |
| 02-附录B | Linear 与 Conv 算子转换 | [02-附录B-Linear与Conv算子转换.md](./02-附录B-Linear与Conv算子转换.md) | 深入中 |
| 02-附录C | 张量维度 [B, seq, hidden]（地基） | [02-附录C-张量维度(B,seq,hidden).md](./02-附录C-张量维度(B,seq,hidden).md) | 学习中 |
| 02-附录D | 自回归 与 自注意力 | [02-附录D-自回归与自注意力.md](./02-附录D-自回归与自注意力.md) | 学习中 |
| 02-附录E | 端侧定长输入 与 固定计算图导出 | [02-附录E-端侧定长与计算图导出.md](./02-附录E-端侧定长与计算图导出.md) | 学习中 |
| 02-附录F | Attention 分类大全（面试向） | [02-附录F-Attention分类大全(面试向).md](./02-附录F-Attention分类大全(面试向).md) | 学习中 |
| 02-附录G | RoPE 旋转位置编码（从零到懂） | [02-附录G-RoPE位置编码.md](./02-附录G-RoPE位置编码.md) | 学习中 |
| 02-附录H | MLP 前馈网络（从零到懂） | [02-附录H-MLP前馈网络.md](./02-附录H-MLP前馈网络.md) | 学习中 |
| 02-附录I | 层结构：Norm 与残差（Transformer 骨架） | [02-附录I-层结构-Norm与残差.md](./02-附录I-层结构-Norm与残差.md) | 学习中 |
| 02-附录J | 采样策略：从 logits 到下一个 token | [02-附录J-采样策略(logits到下一个token).md](./02-附录J-采样策略(logits到下一个token).md) | 学习中 |
| 02-附录K | KV Cache（键值缓存，第 58-61 行改写） | [02-附录K-KV Cache(键值缓存).md](./02-附录K-KV%20Cache(键值缓存).md) | 学习中 |

### Example2 主机编译分篇

| 编号 | 主题 | 文件 | 状态 |
|------|------|------|------|
| E2-00 | example2 主机编译全景 | [example2/00-example2主机编译全景.md](./example2/00-example2主机编译全景.md) | 已建立 |
| E2-01 | AR 图适配：从 AR1073 到 AR1/AR128 | [example2/01-AR图适配-change_hardcoding.md](./example2/01-AR图适配-change_hardcoding.md) | 已完成 |
| E2-02 | Split ONNX 与测试向量 | [example2/02-Split-ONNX与测试向量.md](./example2/02-Split-ONNX与测试向量.md) | 已完成 |
| E2-03 | MHA2SHA 图结构转换 | [example2/03-MHA2SHA图结构转换.md](./example2/03-MHA2SHA图结构转换.md) | 已完成 |
| E2-04 | ONNX 到 DLC：qairt-converter | [example2/04-ONNX到DLC-qairt-converter.md](./example2/04-ONNX到DLC-qairt-converter.md) | 已完成 |
| E2-05 | 量化 DLC：qairt-quantizer | [example2/05-量化DLC-qairt-quantizer.md](./example2/05-量化DLC-qairt-quantizer.md) | 核心概念已完成 |
| E2-06 | example2 产物总览与清理 | [example2/06-example2产物总览与清理.md](./example2/06-example2产物总览与清理.md) | 已建立 |
| E2-07 | Example2 → Example3 衔接：Context Binary | [example2/07-ContextBinary编译与HTP后端.md](./example2/07-ContextBinary编译与HTP后端.md) | 已了解，实操前再学 |

### Example3 Genie 端侧运行分篇

| 编号 | 主题 | 文件 | 状态 |
|------|------|------|------|
| E3-00 | Example3：Genie 端侧运行全景与 Embedding 输入 | [example3/00-Example3-Genie端侧运行全景.md](./example3/00-Example3-Genie端侧运行全景.md) | 已完成 |
| E3-01 | Genie 与 QNN/HTP 运行时准备 | [example3/01-Genie与QNN-HTP运行时准备.md](./example3/01-Genie与QNN-HTP运行时准备.md) | 已完成 |
| E3-02 | Context Binary 与配置文件部署 | [example3/02-ContextBinary与配置文件部署.md](./example3/02-ContextBinary与配置文件部署.md) | 已完成 |
| E3-03 | 设备端启动与日志分层 | [example3/03-设备端启动与日志分层.md](./example3/03-设备端启动与日志分层.md) | 已完成 |

### ViT/VEG 视觉分支

| 编号 | 主题 | 文件 | 状态 |
|------|------|------|------|
| V-00 | ViT/VEG 视觉分支全景与五类输入 | [vit/00-ViT-VEG视觉分支全景与输入.md](./vit/00-ViT-VEG视觉分支全景与输入.md) | 学习中 |

> **最近学习记录（2026-09-03）**：Example1 → Example2 → Example3 的 LLM 主线核心概念已经学完，现已开始 `vit/qwen2_5_vl` 视觉编码器分支 V-00，先学习图片预处理、五个固定 VEG 输入和 `784 → 196 → 2048` 的形状链。这里记录的是学习进度，不代表模型、SDK 或设备验收通过。

> 🧭 **新手入口**：建议先读 [00-基础篇](./00-基础篇-模型与推理两阶段.md)（搞懂"模型是一套权重、推理分 prefill/decode 两阶段、改造作用在模型本身"），再看其余分篇。
> 🎯 **收敛/面试入口**：觉得学得凌乱、想抓重点或准备面试时，看 [学习·面试地图](./面试地图.md)——按"主线 + 面试优先级"重排所有笔记，含高频问答清单和项目自述。
> 🚀 **实际运行入口**：先看 [仓库根 README](../../README.md) 选择 LLM 或视觉分支；逐步原理与验收命令见 [09 · 工程运行指南](./09-工程运行指南-LLM与ViT双链路.md)，只执行 `llm_quant.py` 时再看 [example1/README.md](../../example1/README.md)。

## 三、需要先建立的背景概念（学一次终身受用）

- **Transformer 四大块**：Attention（注意力，[附录A](./02-附录A-Attention注意力机制.md)）、MLP（前馈，[附录H](./02-附录H-MLP前馈网络.md)）、Causal Mask（因果掩码，[附录E](./02-附录E-端侧定长与计算图导出.md)）、KV Cache（键值缓存，[附录K](./02-附录K-KV%20Cache(键值缓存).md)）。
- **量化(Quantization)**：把 FP32/FP16 权重和激活压成低位宽（本项目权重 4bit、激活 16bit），减小体积、加速端侧推理。
- **端侧/HTP/NPU**：手机等设备上的 AI 加速硬件，对算子形态有特殊偏好（如更喜欢 Conv 而非 Linear、要求定长输入）。
- **AIMET / QNN(QAIRT)**：高通的离线量化优化工具和端侧部署 SDK；概念边界与完整关系见 [06-附录B](./06-附录B-QAIRT-QNN-AIMET-QuantSim概念与关系.md)。

## 四、记笔记的小约定

- 每篇开头写「这块在整个流程中的位置」+「一句话本质」。
- **统一四段式结构**：`一 介绍（是什么/为什么）→ 二 原理 → 三 官方 Qwen2 的做法 → 四 本项目改造后的做法`；不适用的段落写"不涉及"，纯概念/参考篇可轻量化（只分「介绍 / 原理」）。
- 四段式里「官方做法 → 本项目改造」这一对，仍沿用「原版怎么做 → 改成什么 → 为什么改」的对照写法。
- 代码引用尽量标注文件名和行号，方便回看。
