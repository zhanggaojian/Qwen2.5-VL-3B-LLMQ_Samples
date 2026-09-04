# V-00 · ViT/VEG 视觉分支全景与五类输入

> **这块在整个流程中的位置**：LLM 分支已经能接收 `inputs_embeds.bin` 并运行；视觉分支负责把一张图片变成其中的视觉 token embedding。
>
> **一句话本质**：图片先被切成 patch，ViT 理解这些 patch，再把 784 个 patch 合并成 196 个、每个 2048 维的视觉 token，交给 LLM。

> 当前状态：**学习中**。以下结论来自仓库代码、Notebook、配置和 RAW 文件的静态核对；尚未实际运行模型、QAIRT SDK 或设备。

## 一、视觉分支的完整位置

```text
图片
  ↓ AutoProcessor：缩放、归一化、切 patch
pixel_values + image_grid_thw
  ↓ 构造固定网格所需的 RoPE 与 Attention Mask
五个 VEG 输入
  ↓ Qwen2.5-VL Visual Encoder + Patch Merger
vision_embedding [196, 2048]
  ↓ 替换文本序列中的 image-token 槽位
inputs_embeds [221, 2048]
  ↓ 已学习的 Genie / LLM 分支
文字输出
```

仓库里的部署产物链是：

```text
vit/qwen2_5_vl/example1/veg.ipynb
  → veg.onnx + veg.encodings + 五个 RAW 测试输入
vit/qwen2_5_vl/example2/qnn_model_prepare_for_veg.ipynb
  → MHA→SHA → DLC → Quantized DLC → veg.serialized.bin
example3
  → qnn-net-run 得到 vision_embedding.raw
  → 融合为 inputs_embeds.bin
  → genie-t2t-run 执行 LLM
```

这里的 **VEG** 是 **Visual Embedding Generator**。它不是另一种模型，而是项目为端侧导出封装的“视觉塔 + Patch Merger”。

## 二、原始 Qwen 接口：两个关键输入

原始 Hugging Face 视觉模型的调用近似为：

```python
vision_embedding = model.visual(pixel_values, image_grid_thw)
```

处理器还会生成 `input_ids`，但它属于完整多模态 Prompt，**不是视觉编码器本身的输入**。

### 1. `pixel_values`

它不是原始 RGB 图片，也不是一张普通的 `[3, H, W]` 张量，而是经过缩放、归一化、时间维复制和 patch 展平后的数值：

```text
原始形状：[784, 1176]
适配 Conv 后：[784, 1, 1, 1176]

1176 = 3 个颜色通道 × 2 个时间 patch × 14 × 14
784  = 1 × 28 × 28 个 patch
```

单张静态图片没有真正的两帧运动；`temporal_patch_size=2` 是模型统一处理图像和视频时采用的输入组织方式。

### 2. `image_grid_thw`

它描述 patch 网格，而不是图像内容：

```text
image_grid_thw = [T, H, W] ≈ [1, 28, 28]
```

含义分别是时间、高度、宽度方向上有多少个 patch。官方文档也将其定义为每张图像的 temporal、height、width feature grid：
[Hugging Face Qwen2.5-VL 文档](https://huggingface.co/docs/transformers/v5.13.0/en/model_doc/qwen2_5_vl)。

配置虽然写了 `384 × 384`，处理器需要按 `patch_size × merge_size = 28` 的倍数对齐；结合仓库 RAW 大小，当前样例形成的是 `28 × 28` 网格，对应约 `392 × 392` 的有效输入尺寸。

## 三、本项目为什么变成五个 VEG 输入

原始模型会在运行时根据 `image_grid_thw` 动态计算位置、窗口划分和 Mask。这个项目面向固定形状的 QNN 图，把这些计算提前到主机侧，并将计算结果直接作为图输入：

| 最终 VEG 输入 | 当前形状 | 谁决定它 | 用途 |
|---|---:|---|---|
| `pixel_values` | `[784, 1, 1, 1176]` | 图片内容 + 固定预处理尺寸 | 784 个图像 patch 的数值 |
| `position_ids_cos` | `[784, 40]` | patch 空间位置 | 对 Attention 的 Q/K 做视觉 RoPE |
| `position_ids_sin` | `[784, 40]` | patch 空间位置 | 与 cos 配合完成旋转位置编码 |
| `window_attention_mask` | `[1, 784, 784]` | 窗口划分 | 普通层只允许同一局部窗口内互看 |
| `full_attention_mask` | `[1, 784, 784]` | 整张图的 token 范围 | 指定层允许所有 patch 全局互看 |

仓库五个 `.raw` 都以 `float32` 保存，文件大小与这些形状一致。

### `image_grid_thw` 去哪里了？

它没有消失，而是从“导出图的运行时输入”变成了“构造导出图时的上游信息”：

```text
image_grid_thw
  ├─→ window_index / reverse index（烘焙进包装模型）
  ├─→ cumulative sequence lengths（构图时使用）
  ├─→ position_ids_cos / sin（成为图输入）
  └─→ window/full attention mask（成为图输入）
```

所以，最终 ONNX/VEG 图接收五个输入，却不再直接接收 `image_grid_thw`。

对当前固定的 `[1, 28, 28]` 网格来说：

- `pixel_values` 会随图片内容变化；
- cos/sin 和两个 Mask 只由网格与模型规则决定，通常可以复用；
- 如果分辨率、图片数量或网格发生变化，就必须重新生成辅助输入，并且当前固定形状图通常也要重新导出、编译。

这与 Qwen2.5-VL 原版支持动态分辨率并不矛盾：原版运行时保留了动态计算，本项目为了端侧固定图部署主动收紧了动态性。Qwen 官方说明了动态分辨率和窗口注意力；Qualcomm 的端侧适配源码也采用了预计算视觉位置与窗口信息的思路：
[Qwen2.5-VL 官方博客](https://qwenlm.github.io/blog/qwen2.5-vl/)、[Qualcomm AI Hub 视觉编码器适配](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/_shared/qwen2_vl/vision_encoder.py)。

## 四、两种 Mask 为什么都要有

32 层 ViT 并非全部使用相同注意力范围：

- 大多数层使用 `window_attention_mask`，只在局部窗口内计算，减少 Attention 的计算量；
- 第 `7、15、23、31` 层使用 `full_attention_mask`，让整张图的信息周期性地全局交流。

Mask 中：

```text
0      = 允许关注
-1000  = 近似禁止关注（Softmax 后接近 0）
```

可以把它理解为：大部分时间在“小组内讨论”，隔几层再开一次“全体会议”。

## 五、784 为什么最终变成 196

视觉 Transformer 先处理 784 个 patch token，随后 Patch Merger 按 `2 × 2` 空间合并：

```text
28 × 28 = 784 个 patch
每 2 × 2 合为 1 个 token
14 × 14 = 196 个 vision token
```

项目中视觉塔内部维度为 1280，Merger/Projector 最终输出 LLM 所需的 2048 维：

```text
VEG 输出：vision_embedding [196, 2048]
完整样例：inputs_embeds [221, 2048]
          = 196 个视觉 token + 25 个模板/文本 token
```

视觉 embedding 会替换 Prompt 中预留的 image-token 位置；LLM 看到的是一串统一的 2048 维 embedding，不再直接看到图片像素。

## 六、不要混淆两套位置编码

本节的 `position_ids_cos/sin` 是 **ViT 内部的视觉空间 RoPE**，帮助视觉 patch 表达上下左右的位置关系。

图片进入 LLM 后，语言模型还有自己的多模态 MRoPE，用于统一文本、时间、高度和宽度位置。二者位于不同模型阶段，不能把这里的五输入 RoPE 当成 LLM 的 MRoPE。

## 七、VEG 视觉 BIN 与 LLM BIN 的职责差异

视觉 BIN 和 LLM BIN 都经过量化与 QNN 编译，但它们不是两个职责相同的模型。

| 组件 | 输入 | 输出 | 运行方式 | 职责 |
|---|---|---|---|---|
| VEG 视觉 BIN | 图片预处理后的五个张量 | `vision_embedding [196,2048]` | 每张新图片通常运行一次 | 把图片编码成视觉 token |
| 文本 Embedding 表 | Tokenizer 产生的 token ID | `text_embedding [N,2048]` | 每个文本 token 查表 | 把文字 token ID 转成 LLM 内部向量 |
| LLM BIN | 融合后的 `inputs_embeds`、位置和缓存等 | 下一个 token 的预测结果 | Prefill 一次，Decode 反复运行 | 理解图文并生成答案 |

因此，文字变成 Embedding 并不是由六个 LLM Context Binary 完成的。本项目会单独导出：

```text
embedding_weights_151936x2048.raw
```

运行时的职责链是：

```text
图片 → veg.serialized.bin → vision_embedding ─┐
                                               ├→ inputs_embeds → LLM serialized.bin → 答案
文字 → Tokenizer → Embedding 表 → text_embedding ┘
```

这里合并的是两种 **Embedding 数据**，不是两个模型的权重或 Context Binary。视觉和文本 Context Binary 始终分开：按照当前配置，视觉侧目标产物是一个 `veg.serialized.bin`，LLM 侧由六个 `weight_sharing_model_*_of_6.serialized.bin` 组成。

还要区分“量化编译完成”和“推理完成”：

```text
量化、编译：制作一个可以在 HTP 上运行的 VEG
运行 VEG：让这个 VEG 处理某一张具体图片，得到该图片的 Embedding
```

因此，每换一张图片通常都要重新运行视觉 BIN；如果同一张图片连续提问，可以缓存并复用它的 `vision_embedding`。

按照仓库给出的目标设备流程，主机程序先通过 `qnn-net-run` 调用 `veg.serialized.bin`，取得视觉 Embedding 并完成融合，再由 `genie-t2t-run` 根据配置加载 LLM Context Binary。当前 Example3 脚本的 `use_on_device=False`，默认仍使用 GPU 上的浮点视觉模型，端侧 VEG 调用尚未完整自动化。

## 八、本节学习边界

本节先掌握三件事：

1. VEG 的任务是把图片变成 LLM 能接收的视觉 embedding。
2. 原版的 `image_grid_thw` 在固定图适配时被拆解为位置和 Mask 信息。
3. 当前样例的核心形状链是 `784 patches → 196 vision tokens → 每个 2048 维`。

下一节再进入 **VEG 模型适配**：为什么要改 Attention、为什么 `Linear → Conv`、窗口重排如何进入固定图，以及哪些只是导出改写、哪些会影响数学结果。

## 简短总结

当前视觉分支把图片预处理成 784 个 patch，并把网格派生的 RoPE 和两类 Mask 显式送进固定 VEG 图；VEG 再合并为 196 个 2048 维视觉 token。视觉 BIN 只负责图片编码，文字通过独立 Embedding 表变成向量，融合后的 Embedding 才交给 LLM BIN 理解并生成答案。
