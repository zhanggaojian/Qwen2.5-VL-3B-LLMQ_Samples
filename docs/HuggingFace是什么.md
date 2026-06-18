# HuggingFace 是什么？是模型吗？

结论：**HuggingFace 本身不是一个模型**，它是一个平台 / 工具库。在本项目语境里，"huggingface" 有两层含义，需要分开理解。

## 一、HuggingFace（🤗）是平台 / 公司，不是模型

HuggingFace 是目前最主流的 AI 开源社区和工具平台，主要提供三样东西：

| 组成 | 说明 |
|------|------|
| Model Hub（模型仓库） | 托管模型的网站（huggingface.co），有几十万个开源模型，如 Qwen、LLaMA、BERT、Stable Diffusion 等。`Qwen2.5-VL-3B-Instruct` 就是从这里下载的 |
| `transformers` 库 | Python 库，用统一接口（`AutoConfig`、`AutoTokenizer`、`from_pretrained` 等）加载和运行模型。`llm_quant.py` 里大量用到 |
| 数据集 / 配套工具 | `datasets`、`accelerate`、`tokenizers` 等 |

所以「HuggingFace 模型」通常指**从 HuggingFace 平台下载的、符合 transformers 格式的模型**（即 `config.json` + `.safetensors` + tokenizer 那一套目录结构）。

## 二、代码里的 `huggingface` 是项目内的本地文件夹

注意 `llm_quant.py` 第 37 行的 import：

```python
from huggingface.baseline_models.qwen2 import modeling_qwen2
```

这里的 `huggingface` 不是那个网站/库，而是**本项目里的一个本地目录**，结构如下：

```
example1/huggingface/baseline_models/qwen2/
├── configuration_qwen2.py   # Qwen2 的配置类定义
└── modeling_qwen2.py        # Qwen2 的模型结构实现（核心）
```

它其实是把 HuggingFace `transformers` 官方的 Qwen2 模型实现**拷贝 / 改造**到本地，目的是方便量化时对模型结构做适配修改（脚本第 50-61 行的一堆 `setattr`、替换 Attention 类、改 KV cache 等操作，就是在改这份本地代码里的类）。

## 三、小结

| "huggingface" | 含义 |
|--------------|------|
| HuggingFace（平台） | AI 模型托管网站 + `transformers` 工具库，**它不是模型**，是放模型、跑模型的地方 |
| HuggingFace 模型 | 从该平台下载的、transformers 格式的开源模型（如 Qwen2.5-VL-3B） |
| 代码里的 `huggingface/` 文件夹 | 本项目里的本地目录，存放 Qwen2 的基线模型实现，供量化适配改造用 |

所以本项目真正在量化的模型是 **Qwen2.5-VL-3B-Instruct**（阿里通义千问的多模态大模型），HuggingFace 只是它的来源平台和加载工具。
