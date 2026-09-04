# Example2 · Qwen2.5-VL-3B LLM 主机编译全景

> **流程位置**：`example1` 完成模型适配、量化设计和 ONNX 导出以后，`example2` 负责在 x86 Linux 主机上将其编译成 Qualcomm DLC。
>
> **一句话本质**：`example2` 是一条“固定 Shape 适配 → 切图 → Attention 图改写 → Qualcomm 图转换 → 低比特封装”的主机编译流水线。
>
> **当前代码真实终点**：AR1 和 AR128 两份 `*_quantized.dlc`；Context Binary 代码尚未启用。

---

## 零、学完这一章应该会什么

学完 `example2` 以后，应该能够：

1. 说清 `example1 → example2 → example3` 的职责边界。
2. 说清 `example2` 的三类核心输入和两份当前最终产物。
3. 画出五个执行阶段，并说清每一步的输入、处理和输出。
4. 解释 AR1、AR128、Context Length 和 Past-KV Length 的关系。
5. 区分 ONNX、Encoding、RAW、普通 DLC、Quantized DLC 和 Context Binary。
6. 知道为什么日志出现 `done` 仍不能判定成功。
7. 能根据中间文件所在目录，反推当前运行到了哪个阶段。

---

## 一、先记住十个结论

1. `example2` 不训练模型，也不重新执行 AIMET 的 SeqMSE 和 `compute_encodings()`。
2. 它使用 `example1` 已经生成的 ONNX、外置权重、Encoding 和 Test Vector。
3. 主入口是 `example2/host_linux/qnn_compile_deploy.py`，不是上游 README 遗留描述中的 Notebook。
4. 完整脚本可按“一个前置 AR 图适配 + 四个核心编译阶段”理解，合计五个部分。
5. AR1 通常用于逐 Token Decode；AR128 通常用于分块 Prefill。
6. 当前 Context Length 是 2048，因此 `Past-KV Length = 2048 - AR`。
7. `qairt-converter` 生成的是尚未走完最终 Quantizer 的中间 DLC。
8. `qairt-quantizer` 才生成当前脚本的最终 `*_quantized.dlc`。
9. 权重是静态常量，可在编译时量化；Activation 和 KV 的具体数值在运行时才动态产生。
10. 当前 Context Binary 生成调用被注释，所以“生成 Quantized DLC”不等于“已经可以在设备上直接运行”。

---

## 二、`example2` 在整个项目中的位置

本篇只讨论仓库根目录下的 **LLM `example2`**，不讨论 `vit/qwen2_5_vl/example2` 视觉编码器分支。

```text
原始 Qwen2.5-VL-3B LLM
        │
        │ example1：结构适配 + AIMET 量化设计与验证
        ▼
ONNX + External Weight/Bias + Encoding + qt_0.pkl
        │
        │ example2：x86 Linux 主机编译
        ▼
AR1 / AR128 Quantized DLC
        │
        │ Context Binary 生成（当前需另行启用）
        ▼
HTP Context Binary
        │
        │ example3：Genie + Snapdragon HTP 运行
        ▼
端侧文本生成结果
```

三个 example 的职责可以缩写为：

| 阶段 | 核心问题 | 典型产物 |
|---|---|---|
| `example1` | 模型应该如何适配和量化，精度是否可接受？ | ONNX、外置权重、Encoding、Test Vector |
| `example2` | 如何把上述产物转成 Qualcomm 部署格式？ | AR1/AR128 Quantized DLC |
| `example3` | 如何在 Snapdragon 设备上调用模型完成推理？ | 实际文本输出 |

---

## 三、输入：`example2` 从 `example1` 接过什么

主脚本用 `LLAMA_MODELS` 指向 `example1` 输出根目录，当前是硬编码绝对路径（`qnn_compile_deploy.py:15`）。

最少需要以下文件：

| 输入 | 保存什么 | 在 `example2` 中的作用 |
|---|---|---|
| `onnx/qwen25llm.onnx` | 计算图、Tensor 名称、Shape、外置数据引用 | AR Shape 适配、切图、MHA2SHA、DLC 转换 |
| ONNX 外置 `*.weight/*.bias/*.data` | 大模型真实参数字节 | 随 ONNX 计算图被后续工具读取 |
| `onnx/qwen25llm.encodings` | Weight、Activation、KV 等 Tensor 的位宽、scale、offset、对称性和粒度 | 保留 `example1` 已验证的量化规则 |
| `test_vectors/qt_0.pkl` | QuantSim 开启时的一套模型输入、输出和部分中间结果 | 生成 RAW、`input_list` 和 Golden Output |

另外，还需要一套工具而非模型数据：

- `QNN_SDK_ROOT`：QAIRT/QNN SDK 根目录（`qnn_compile_deploy.py:16`）；
- `qairt-converter`：ONNX 到 DLC 转换器；
- `qairt-quantizer`：DLC 量化和封装工具；
- `mha2sha-onnx-converter`：仓库中附带的 Attention 图改写工具。

---

## 四、五个阶段的全局地图

### 4.1 一张表先看懂

| # | 阶段 | 核心入口 | 输入 | 本质 | 主要输出 |
|---:|---|---|---|---|---|
| 1 | AR 图适配 | `gen_ar()` / `change_hardcoding.execute()` | AR1073、CL2048 的 ONNX 和 Test Vector | 重分配“当前 Token 槽位”与“历史 KV 槽位” | `ar1-cl2048/`、`ar128-cl2048/` |
| 2 | Split ONNX | `thread_split()` / `utils.split_onnx()` | AR 适配后 ONNX、Encoding、`qt_0.pkl` | 按切分点准备子图和对拍数据 | Split ONNX、RAW、`input_list`、Golden |
| 3 | MHA → SHA | `thread_g2g()` / `mha2sha-onnx-converter` | Split ONNX + Encoding | 将 Attention 图等价改写为更适合 HTP 的表示 | SHA ONNX + 对应 Encoding |
| 4 | ONNX → DLC | `thread_convert()` / `qairt-converter` | SHA ONNX + SHA Encoding | 转成 Qualcomm 图容器/中间模型表示 | 普通/中间 DLC |
| 5 | Quantized DLC | `thread_genlib()` / `qairt-quantizer` | 中间 DLC + `input_list` + RAW | 将量化设计落实为低比特 DLC | `*_quantized.dlc` |

### 4.2 数据流串起来

```text
example1/output
├── onnx/qwen25llm.onnx
├── onnx/外置权重
├── onnx/qwen25llm.encodings
└── test_vectors/qt_0.pkl
          │
          ▼  1. change_hardcoding
assets/models_ar_n/
├── ar1-cl2048/
└── ar128-cl2048/
          │
          ▼  2. split_onnx
assets/artifacts/ar{1,128}-cl2048/
├── split_onnx/*_1_of_1.onnx
├── test_inputs_*/*.raw
├── input_list_*.txt
└── test_golden_outputs_*/*.raw
          │
          ▼  3. MHA2SHA
1_of_1/sha_output/
├── *.onnx
└── *.encodings
          │
          ▼  4. qairt-converter
1_of_1/converted_model/*.dlc
          │
          ▼  5. qairt-quantizer
1_of_1/compiled_model/*_quantized.dlc
```

---

## 五、四个关键参数先学会

主脚本当前配置（`qnn_compile_deploy.py:29-37`）：

```python
CL = 2048
ARNs = [1, 128]
EXPORT_AR = 1073
EXPORT_CONTEXT_LENGTH = 2048
num_splits = 1
```

### 5.1 AR 是什么

在本项目中，可以先把 AR 理解为：

> **一张固定计算图一次前向并行处理的新 Token 数量。**

- AR1：一次处理 1 个新 Token，通常服务于 Decode；
- AR128：一次处理 128 个新 Token，通常服务于分块 Prefill；
- AR1073：`example1` 当前原始导出图的 Current Token 槽位数。

AR128 是本项目的设计选择，不是“所有 Prefill 必须固定等于 128”。

### 5.2 Context Length 与 Past-KV Length

当总 Context Length 保持 2048 时：

```text
Current Token Length + Past-KV Length = Context Length

Past-KV Length = Context Length - AR
```

| 图 | Current/AR | Past-KV | 总长度 |
|---|---:|---:|---:|
| `example1` 原始导出图 | 1073 | 975 | 2048 |
| AR1 | 1 | 2047 | 2048 |
| AR128 | 128 | 1920 | 2048 |

### 5.3 `num_splits=1` 是什么意思

代码具备切成多个子图的能力，但当前只产生 `1_of_1`。

这不代表 Split 阶段没有作用：它仍会统一产物命名，并准备 RAW、`input_list` 和 Golden Output。

---

## 六、阶段一：AR 图适配

### 6.1 为什么需要

端侧 HTP 偏好固定 Shape 计算图，而 Prefill 与 Decode 一次处理的 Token 数量不同。因此，需要从同一套模型参数语义上生成不同固定 Shape 的图。

### 6.2 项目中如何做

`gen_ar()` 调用 `change_hardcoding.execute()`（`qnn_compile_deploy.py:47-53`）。

生成 AR1 时的核心替换：

```text
1073 → 1
-1073 → -1
975 → 2047
```

生成 AR128 时的核心替换：

```text
1073 → 128
-1073 → -1
975 → 1920
```

`change_hardcoding.py` 会检查并修改：

- ONNX 输入、输出和 `value_info` 中的 Shape；
- 部分节点 Attribute 中的常量；
- 部分与 Shape 相关的 Initializer；
- `qt_0.pkl` 等 Test Vector 中的 Tensor Shape；
- 原样复制 Encoding、JSON 和 YAML 等辅助文件。

它不会重新训练模型，也不会重新搜索 Weight/Activation Encoding。

### 6.3 输出

```text
assets/models_ar_n/ar1-cl2048/
assets/models_ar_n/ar128-cl2048/
```

### 6.4 关键风险

- 它根据数值匹配替换 Shape，必须保证 `EXPORT_AR` 和 `EXPORT_CONTEXT_LENGTH` 与原导出图一致。
- 当前复制逻辑没有显式列出所有 ONNX 外置权重后缀，运行后要确认新 ONNX 的 External Data 仍可解析。

---

## 七、阶段二：Split ONNX 与测试数据

### 7.1 为什么需要

大模型可以按层或指定 Tensor 拆分，便于后续分段编译、资源管理和数值定位。

当前 `num_splits=1`，所以主要学习重点是“统一准备编译输入和数值对拍产物”。

### 7.2 `qt_0.pkl` 如何变成 RAW

```text
example1 真实图文样本
  → QuantSim 开启时前向
  → qt_0.pkl
  → Split 工具按 ONNX 输入名展开
  → test_inputs_*/0/*.raw
  → input_list_*.txt
```

本项目 `use_input_embeddings=true`，所以 RAW 主要对应：

- `inputs_embeds`；
- `attention_mask`；
- `position_ids_cos` / `position_ids_sin`；
- 36 层 `past_key_i_in` / `past_value_i_in`。

`.raw` 是没有文件头的连续 Tensor 字节；Shape、dtype 和 Tensor 名称需由模型图及 `input_list` 配合解释。

`input_list` 每行大致如下：

```text
inputs_embeds:=.../inputs_embeds.raw attention_mask:=.../attention_mask.raw ...
```

### 7.3 RAW 与 Golden 不要混淆

| 产物 | 内容 | 作用 |
|---|---|---|
| `test_inputs_*/*.raw` | 模型输入 Tensor | Quantizer 输入、运行测试 |
| `input_list_*.txt` | Tensor 名称到 RAW 路径的映射 | 告诉工具如何绑定多输入 |
| `test_golden_outputs_*/*.raw` | 上游路径产生的参考输出 | 数值对拍，不作为 Quantizer 输入 |

---

## 八、阶段三：MHA → SHA

### 8.1 为什么需要

`mha2sha-onnx-converter` 将 Multi-Head Attention 的计算图改写为等价的 Single-Head Attention 表示，目的是获得更适合 HTP 后续编译和执行的图形态。

这里不是把多个注意力头删到只剩一个，而是把多头计算改写、展开成多个更适合后端处理的单头分支；所有注意力头的语义仍然保留。

这里的重点是“计算图等价改写”，不是重新训练 Attention，也不是把已训练模型真的改成另一套语义不同的权重。

### 8.2 输入和输出

```text
Split ONNX
  + example1 导出并随图变换的 Encoding
          │
          ▼ mha2sha-onnx-converter
SHA ONNX
  + 与新 Tensor 名称/新图结构对齐的 Encoding
```

脚本当前还启用了 RoPE、Past-KV、GQA、NCHW 对齐等处理选项（`qnn_compile_deploy.py:157-169`）。

### 8.3 输出目录

```text
assets/artifacts/ar1-cl2048/1_of_1/sha_output/
assets/artifacts/ar128-cl2048/1_of_1/sha_output/
```

初学阶段不需要立即逐行读完 `G2G/MHA2SHA/src/`。先掌握它的输入、输出、图改写目标和 Encoding 同步映射；逐层调用链和验收边界见 [03 · MHA2SHA 图结构转换](./03-MHA2SHA图结构转换.md)。

---

## 九、阶段四：ONNX → 普通 DLC

### 9.1 DLC 是什么

DLC 可以先理解为 Qualcomm 工具链使用的模型图容器/中间表示。

ONNX 是跨框架图格式；DLC 则已经进入 Qualcomm 工具链的表达体系。

### 9.2 项目命令

`thread_convert()` 构造的核心参数（`qnn_compile_deploy.py:216-220`）：

```text
qairt-converter
  --input_network <sha_model.onnx>
  --quantization_overrides <sha_model.encodings>
  -o <converted_model/model.dlc>
```

其中：

- `--input_network` 提供图结构和权重；
- `--quantization_overrides` 将 `example1` 已经确定的位宽、scale、offset 等规则带入 Qualcomm 图表示；
- `utils.get_input_layout()` 根据 ONNX 输入补充 Layout 参数。

### 9.3 “普通 DLC”的准确边界

这一阶段的 DLC 已可携带 Encoding 等量化元数据，但还没有通过下一步 `qairt-quantizer` 生成当前项目所需的最终 Quantized DLC。

因此，把它记成“中间 DLC”比简单理解成“什么量化信息都没有的纯浮点模型”更准确。

### 9.4 输出目录

```text
assets/artifacts/ar{1,128}-cl2048/1_of_1/converted_model/*.dlc
```

---

## 十、阶段五：Quantized DLC

### 10.1 为什么还需要 Quantizer

`example1` 主要完成“量化方案设计与验证”：

```text
SeqMSE
  → 优化 Weight Encoding

compute_encodings()
  → 统计 Activation/KV Encoding

QuantSim / PPL
  → 模拟并验证低比特误差
```

`qairt-quantizer` 负责在 Qualcomm DLC 中落实这套方案，不再执行模型训练、反向传播或 SeqMSE。

### 10.2 项目命令

`thread_genlib()` 构造的命令（`qnn_compile_deploy.py:259-265`）：

```text
qairt-quantizer
  --input_dlc <converted_model/model.dlc>
  --input_list <input_list_model.txt>
  --output_dlc <compiled_model/model_quantized.dlc>
  --act_bitwidth 16
  --bias_bitwidth 32
  --keep_weights_quantized
```

### 10.3 权重、Activation 和 KV 最终各保存什么

| 对象 | 能否离线提前确定具体值 | Quantized DLC 中的内容 | 运行时 |
|---|---|---|---|
| Weight | 能，权重是静态常量 | 启用量化的权重以低比特常量表示，并携带 Encoding | 每次推理重用 |
| Activation | 不能，它由当前输入决定 | 量化计算图、dtype/bitwidth 和 Encoding | 每次推理动态产生量化 Buffer |
| KV Cache | 不能，它是有状态 Activation | KV 端口、Shape、执行规则和 Encoding | Prefill/Decode 动态产生、保存并回灌 |

不能笼统地说“所有权重都是 INT4”：当前是混合精度设计，主要权重默认 W4，部分可能是 W8、W16 或未量化。

同样不能笼统地说“整条 KV 都是 8 bit”。本项目已导出 Encoding 的实际分析表明：

```text
外部 past_key/value_i_in 与 Concat 结果 → A16
展开后进入 Attention MatMul 的内部 K/V → A8
```

量化边界的详细证据见 [08 · ONNX 导出与测试向量](../08-ONNX导出与测试向量.md)。

### 10.4 当前最终输出

```text
assets/artifacts/ar1-cl2048/1_of_1/compiled_model/
└── ar1-cl2048_1_of_1_quantized.dlc

assets/artifacts/ar128-cl2048/1_of_1/compiled_model/
└── ar128-cl2048_1_of_1_quantized.dlc
```

---

## 十一、六类产物不要混淆

| 名称 | 可以先记成 | 是否已是当前脚本最终产物 |
|---|---|---|
| ONNX | 跨框架计算图 + 参数引用 | 否 |
| Encoding | 量化“尺子”，不是另一份模型 | 否 |
| Input RAW | 一套模型输入 Tensor 的原始字节 | 否 |
| Golden RAW | 用于对拍的参考输出 | 否 |
| 中间 DLC | Qualcomm 图容器，尚未完成当前项目的最终 Quantizer 阶段 | 否 |
| Quantized DLC | 已落实量化设计的 Qualcomm DLC | **是** |
| Context Binary | 针对 HTP Backend/SoC 生成的序列化执行上下文 | 当前脚本不会自动生成 |

---

## 十二、Context Binary 的当前边界

`qnn_compile_deploy.py:290-311` 中存在 `qnn-context-binary-generator` 示例命令，但它们位于三引号字符串中。

`qnn_compile_deploy.py:324-425` 中的配置生成和 Context Binary 自动调用代码也全部被注释。

因此：

```text
直接运行当前 qnn_compile_deploy.py
  → 生成 AR1 / AR128 Quantized DLC
  → 不会自动生成 HTP Context Binary
```

启用该阶段以前还必须核对：

- 目标芯片 `soc_id`；
- `dsp_arch`；
- HTP Backend Extension 配置；
- AR1/AR128 图名称；
- Weight Sharing 组合；
- SDK 与设备运行时版本。

AR1 与 AR128 来自同一套模型权重语义，但不应仅凭这一点就断言当前两个 DLC 文件在物理存储上已自动共享权重；真正的 HTP Weight Sharing 还属于后续 Context Binary 阶段。

---

## 十三、运行环境与启动方式

### 13.1 运行环境

- x86_64 Linux 主机，项目说明以 Ubuntu 22.04 为主；
- Python 3.10；
- QAIRT/QNN SDK；
- 不需要 GPU，工具主要在 CPU 上执行；
- 需要较大的 CPU RAM 和磁盘空间。

项目历史日志曾记录 MHA2SHA 峰值 RAM 超过 50 GB，中间产物也可达几十 GB。这意味着“不需要 GPU”不等于“对主机资源要求低”。

### 13.2 必须从正确目录启动

脚本将 `os.getcwd()` 记为 `workfolder`（`qnn_compile_deploy.py:13`），所以必须先进入：

```bash
cd example2/host_linux
```

再运行：

```bash
PYTHONUNBUFFERED=1 python qnn_compile_deploy.py 2>&1 | tee qnn_compile.log
```

如果从仓库根目录直接执行，模块搜索路径、MHA2SHA 路径和 `assets/` 输出位置都可能错误。

### 13.3 当前需人工核对的硬编码

```python
LLAMA_MODELS = "/root/autodl-tmp/zgj/Qwen25/outputs/output"
QNN_SDK_ROOT = "/root/autodl-tmp/zgj/tools/qairt/2.42.0.251225"
CL = 2048
ARNs = [1, 128]
EXPORT_AR = 1073
EXPORT_CONTEXT_LENGTH = 2048
```

前两项必须改成当前机器真实路径；后四项必须与 `example1` 导出 Shape 和目标推理配置一致。

---

## 十四、不要只看日志：每阶段如何验收

正式运行时，应以“日志无真实错误 + 目标文件存在且非空”为成功标准。

```bash
# 1. AR 适配后的模型
du -sh assets/models_ar_n/ar1-cl2048 assets/models_ar_n/ar128-cl2048

# 2. Split ONNX / RAW / input list
ls -lh assets/artifacts/ar{1,128}-cl2048/split_onnx/
ls -lh assets/artifacts/ar{1,128}-cl2048/input_list_*.txt

# 3. MHA2SHA
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/sha_output/

# 4. 中间 DLC
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/converted_model/*.dlc

# 5. Quantized DLC
ls -lh assets/artifacts/ar{1,128}-cl2048/1_of_1/compiled_model/*_quantized.dlc
```

还应核对：

- ONNX 外置权重是否能被完整加载；
- Encoding 是否与变换后 Tensor 名称对齐；
- RAW 数量、Shape 和 dtype 是否符合图输入；
- Golden 对拍是否在可接受误差内；
- AR1 与 AR128 两条路径是否都完成。

---

## 十五、当前脚本的工程风险

### 15.1 `done` 不一定代表成功

当前脚本存在以下控制流问题：

- `executor.map()` 的结果没有被遍历，子进程异常可能没有在父进程中重新抛出；
- 多个异常分支使用 `exit(0)`，失败可能呈现成功退出码；
- MHA2SHA、Converter 和 Quantizer 没有统一严格检查子进程 `returncode`；
- 后续仍可能打印 `All ... done.`。

所以排错时要找“第一个真实错误”，不要只看日志最后一行。

### 15.2 环境与配置不一致

高频问题包括：

- 未从 `example2/host_linux` 启动；
- `LLAMA_MODELS` 或 `QNN_SDK_ROOT` 路径错误；
- ONNX 外置权重不完整；
- 缺失 `.encodings`、`qt_0.pkl`、`psutil` 或 MHA2SHA 可执行文件；
- `example1` 与 `example2` 的模型名、ARN、Context Length 不一致；
- Python、ONNX、NumPy 或 QNN SDK 版本不匹配；
- 目标芯片的 `soc_id` / `dsp_arch` 配置不一致；
- RAM 不足、磁盘写满或进程被 OOM Killer 终止。

---

## 十六、最容易形成的错误理解

### 16.1 “`example2` 又重新做了一遍 AIMET 量化”

不对。`example1` 搜索、统计并验证 Encoding；`example2` 将这套规则映射到 Qualcomm 图并生成 Quantized DLC。

### 16.2 “RAW 就是用户的原始图片或文本”

不对。这里的 RAW 是已经过预处理、可直接绑定到 LLM 图输入的 Tensor 字节。

### 16.3 “Quantized DLC 里保存了所有未来输入的 Activation 和 KV”

不对。DLC 保存它们的计算图和量化规则；具体 Activation/KV 值由每次推理动态产生。

### 16.4 “中间 DLC 已经是当前最终产物”

不对。当前脚本还会执行 `qairt-quantizer`，以 `compiled_model/*_quantized.dlc` 作为当前终点。

### 16.5 “AR1 和 AR128 是两个单独训练的模型”

不对。它们来自同一套模型权重语义，主要差异是固定输入/KV Shape；但当前 DLC 文件是否已在物理存储上共享权重，需要与后续 Weight Sharing Context 生成分开判断。

### 16.6 “日志打印 `done` 就是成功”

不对。必须检查首个真实错误、子进程返回码以及目标文件是否存在且非空。

### 16.7 “得到 Quantized DLC 就已经完成端侧部署”

不对。还需要针对目标 HTP/SoC 生成并验证 Context Binary，再由 `example3` 准备运行库、模型、Tokenizer 和输入执行推理。

---

## 十七、后续分篇笔记如何展开

本篇只负责全局认知。后续按以下顺序深入：

1. [01 · AR 图适配：从 AR1073 到 AR1/AR128](./01-AR图适配-change_hardcoding.md)
   - AR/CL/Past-KV 公式；
   - ONNX Shape 与常量替换；
   - Test Vector Shape 同步；
   - External Data 风险。

2. [02 · Split ONNX 与测试向量](./02-Split-ONNX与测试向量.md)
   - 切图原理；
   - `qt_0.pkl → RAW → input_list`；
   - Golden Output 和数值对拍。

3. [03 · MHA2SHA 图结构转换](./03-MHA2SHA图结构转换.md)
   - MHA/GQA/SHA 关系；
   - HTP 图改写目的；
   - RoPE、Past-KV、Layout 与 Encoding 映射。

4. [04 · ONNX 到 DLC：qairt-converter](./04-ONNX到DLC-qairt-converter.md)
   - DLC 容器；
   - `quantization_overrides`；
   - Input Layout；
   - 转换后验收。

5. [05 · 量化 DLC：qairt-quantizer](./05-量化DLC-qairt-quantizer.md)
   - Weight/Activation/KV 的量化时机；
   - `input_list` 在 Quantizer 中的作用；
   - 混合精度；
   - Quantized DLC、Context Binary 与端侧执行边界。

6. [06 · example2 产物总览与清理](./06-example2产物总览与清理.md)
   - 各阶段落盘文件与下游依赖；
   - 当前最终 DLC 与 Context Binary 的边界；
   - 分里程碑清理、验收和备份策略。

7. [07 · Context Binary 编译与 HTP 后端](./07-ContextBinary编译与HTP后端.md)
   - 量化 DLC 与执行上下文的区别；
   - 后端、配置和输入输出；
   - 当前未启用代码与后续实操边界。

---

## 十八、总览自测

学完本篇后，尝试不看文档回答：

1. `example2` 的输入与当前最终输出分别是什么？
2. 五个阶段的顺序是什么？
3. 为什么同时需要 AR1 和 AR128？
4. 当 AR=128、Context Length=2048 时，Past-KV Length 是多少？
5. `qt_0.pkl`、Input RAW 和 Golden RAW 有什么不同？
6. MHA2SHA 为什么还要同步处理 Encoding？
7. 普通 DLC 与 Quantized DLC 有什么区别？
8. 为什么权重可以离线量化，Activation/KV 的具体值却不能预先写死？
9. 为什么脚本打印 `done` 仍然要检查文件？
10. 为什么当前得到 Quantized DLC 仍不等于完成设备部署？

---

## 十九、相关项目文件

- 主入口：[`example2/host_linux/qnn_compile_deploy.py`](../../../example2/host_linux/qnn_compile_deploy.py)
- AR Shape 适配：[`example2/G2G/change_hardcoding.py`](../../../example2/G2G/change_hardcoding.py)
- Split 主工具：[`example2/G2G/split_onnx_utils/utils.py`](../../../example2/G2G/split_onnx_utils/utils.py)
- MHA2SHA 说明：[`example2/G2G/MHA2SHA/README.md`](../../../example2/G2G/MHA2SHA/README.md)
- `example2` 环境说明：[`example2/host_linux/README.md`](../../../example2/host_linux/README.md)
- 全工程运行指南：[09 · 工程运行指南](../09-工程运行指南-LLM与ViT双链路.md)
- 上游 ONNX/Encoding/Test Vector：[08 · ONNX 导出与测试向量](../08-ONNX导出与测试向量.md)

---

## 二十、一句话总结

> **`example2` 从 `example1` 接收 ONNX、外置权重、Encoding 和 Test Vector，先生成 AR1/AR128 固定 Shape 图，再经过 Split、MHA2SHA、ONNX→DLC 和 DLC 量化，当前最终得到两份 Quantized DLC；HTP Context Binary 仍需按目标芯片另行启用和验证。**
