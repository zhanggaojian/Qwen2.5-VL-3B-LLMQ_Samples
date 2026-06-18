# example1 学习笔记 · 总索引

> 本目录用「一个主题一个 md」的方式记录 example1（Qwen2.5-VL-3B LLM 量化）的学习过程。
> 这一篇是总地图，串起所有分篇笔记；每学完一块就更新对应分篇和下面的进度表。

## 一、整体流程地图

example1 做的事：**把官方 Qwen2.5-VL-3B 模型，改造并量化成可在高通端侧 (HTP/NPU) 运行的格式**。

主线脚本：`example1/llm_quant.py`，大致阶段：

```
读配置(config.yaml)
  → 模型适配(Monkey Patch，替换零件)        ← 当前学习到这里
  → 加载模型 / tokenizer
  → 浮点模型评估(PPL)
  → prepare(模型准备 / 导出中间结构)
  → 量化(QuantSim：compute_encodings / SeqMSE)
  → 量化后评估
  → 导出 ONNX
```

> 配套官方文档：`example1/PIPELINE.md`（流程）、`example1/TROUBLESHOOTING.md`（排错）、`example1/README.md`。

## 二、分篇笔记

| 序号 | 主题 | 文件 | 状态 |
|------|------|------|------|
| 01 | 配置文件 config.yaml 详解 | [01-配置文件config.yaml详解.md](./01-配置文件config.yaml详解.md) | 学习中 |
| 02 | 模型适配（Monkey Patch，第 50-61 行） | [02-模型适配(Monkey-Patch).md](./02-模型适配(Monkey-Patch).md) | 学习中 |
| 03 | 模型/Tokenizer 加载与 config 覆盖 | （待建） | 未开始 |
| 04 | PPL 评估是什么、怎么算 | （待建） | 未开始 |
| 05 | prepare 模型准备阶段 | （待建） | 未开始 |
| 06 | 量化 QuantSim / SeqMSE / compute_encodings | （待建） | 未开始 |
| 07 | ONNX 导出 | （待建） | 未开始 |

## 三、需要先建立的背景概念（学一次终身受用）

- **Transformer 四大块**：Attention（注意力）、MLP（前馈）、Causal Mask（因果掩码）、KV Cache（键值缓存）。
- **量化(Quantization)**：把 FP32/FP16 权重和激活压成低位宽（本项目权重 4bit、激活 16bit），减小体积、加速端侧推理。
- **端侧/HTP/NPU**：手机等设备上的 AI 加速硬件，对算子形态有特殊偏好（如更喜欢 Conv 而非 Linear、要求定长输入）。
- **AIMET / QNN(QAIRT)**：高通的量化工具链（`aimet_torch`）和端侧推理 SDK。

## 四、记笔记的小约定

- 每篇开头写「这块在整个流程中的位置」+「一句话本质」。
- 多用「原版怎么做 → 改成什么 → 为什么改」三段式。
- 代码引用尽量标注文件名和行号，方便回看。
