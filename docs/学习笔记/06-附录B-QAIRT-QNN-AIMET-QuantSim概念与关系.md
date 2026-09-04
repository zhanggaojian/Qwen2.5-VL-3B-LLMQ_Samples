# 06 · 附录B · QAIRT、QNN、AIMET 与 QuantSim 概念关系

> **关联主篇**：[06 · Prepare 模型准备阶段](./06-Prepare模型准备阶段.md)。这篇不深挖某一行代码，而是建立高通端侧量化工具链的“概念地图”。
>
> **一句话本质**：AIMET/QuantSim 在主机上决定“怎么量化、量化参数是多少”，QAIRT/QNN 把模型转换、编译并运行到目标后端，HTP 是最终执行计算的硬件，Genie 则在端侧负责大模型推理编排。
>
> **先记最重要的边界**：`QuantSim ≠ QNN Runtime`，`AIMET ≠ HTP`，`.onnx + .encodings ≠ 最终设备模型`。它们是同一条流水线上的不同层。

---

## 〇、30 秒速记版（先建框架，再读细节）

> 本篇后面很长、概念很多（11 个）。**第一次读或日后回看，先只看这一节**：先用 5 个核心概念搭好骨架，再往下深挖。分不清这些名字，几乎都是因为没先建立这个框架。

### 0.1 三问法：哪台机器 / 什么时候 / 干什么

区分它们最有效的办法，是问"这东西跑在**哪台机器**上"：

| 概念 | 跑在哪台机器 | 什么时候 | 干什么 |
|------|------------|---------|--------|
| **AIMET** | 你的**开发机**(x86+GPU) | 离线 | **决定怎么量化**（算出参数）|
| **QuantSim** | 同上（AIMET 里的一个对象）| 离线 | **模拟**量化误差、产出 `encodings` |
| **QAIRT** | 你的**开发机**(x86 CPU) | 离线**编译期** | **转换 + 编译**成设备格式 |
| **QNN** | **手机** | **运行时** | 在设备上**执行**模型图 |
| **HTP** | **手机芯片里** | 运行时 | **硬件**真正算数 |

> **一条线记住**：前三个在**你的电脑**上（离线准备），后两个在**手机**上（真跑）。两边的交界产物就是 `ONNX + encodings`。

### 0.2 类比：中央厨房 → 生产线 → 门店

```text
AIMET/QuantSim = 中央厨房试菜：反复试"盐放多少、火候多大" → 写出【配方参数】(encodings)
                 ⚠️ QuantSim 是"模拟试吃"，还不是真产品
QAIRT          = 把配方翻译成【工厂生产线指令】（转换、编译、打包）
QNN            = 门店里的【操作系统/微波炉控制器】，按指令加热
HTP            = 微波炉里真正发热的【加热元件】（硬件）
```

### 0.3 混淆的两个真正根源

#### 根源1：AIMET 和 QAIRT **都能说"量化"**，但完全不是一回事

| | AIMET / QuantSim | QAIRT / QNN quantizer |
|---|---|---|
| 做什么 | **模拟量化**（fake quant）| **真正转成整数** |
| 模型还是浮点吗 | ✅ **还是浮点**，只是插了量化模拟节点 | ❌ 真的变成 INT4/INT8 |
| 产出 | `.encodings`（**一张参数表**：每个张量的 scale/offset）| DLC / context binary（**真·设备模型**）|
| 目的 | 在电脑上**预演**"压缩后会掉多少精度" | 生成能在芯片上跑的东西 |

> **最大的坑**：`QuantSim` **不是** INT4 模型，它是"拿浮点模拟整数误差"的**沙盘推演**，用来提前测 PPL 掉多少（详见 7.3 / 18.2）。真正的整数化在 QAIRT 那一步。

#### 根源2：QAIRT 和 QNN 分不清，是因为**它俩不是并列关系**

```text
QAIRT  = 整个 SDK 的品牌名/工具包（新名字）
  └─ QNN = 里面那套面向图、backend、端侧执行的 API/runtime（老名字，库名仍是 Qnn*）
```

**是包含关系，不是互斥的两个产品**（详见 4.3）。本项目里按职责记即可：

- `qairt-converter`、`model_preparer` → **主机侧工具**
- `libQnnHtp`、context binary、runtime → **设备侧执行**

### 0.4 按出场顺序串一遍

```text
① 改造模型(你的代码)          ← Linear→Conv、外部mask、定长KV
② QAIRT model_preparer        ← 规范化图（prepare，此时还没量化！）
③ AIMET QuantSim              ← 插模拟量化器
④ SeqMSE + compute_encodings  ← 用【真实数据】算出 encodings 参数表
⑤ 导出 ONNX + encodings       ← 【离线阶段到此结束，交接产物】
─────────── 以上在你的电脑，以下换机器 ───────────
⑥ QAIRT converter/quantizer   ← 真正整数化，转 DLC
⑦ context binary generator    ← 编译成特定芯片的二进制
⑧ 手机上 QNN Runtime + HTP    ← 真正执行
```

> **重要定位**：本项目 `example1/llm_quant.py` 全部在 **①~⑤**（"中央厨房"阶段），**根本没碰到 QNN 运行时**。认清这条边界，能消掉一大半困惑。

### 0.5 一句话总纲

> **AIMET/QuantSim 在你电脑上"算出怎么压"（产出 encodings，模型仍是浮点）；QAIRT 在你电脑上"编译成设备格式"（真整数化）；QNN 在手机上"执行"；HTP 是手机里干活的硬件。**

---

## 一、为什么这些名字容易混在一起

本项目同时出现：

```text
QAIRT
QNN
AIMET
QuantSim
HTP
Genie
ONNX
DLC
encodings
SeqMSE
compute_encodings
```

它们混乱的原因主要有三个：

1. 都服务于“把模型部署到高通设备”这一目标；
2. QAIRT SDK 中同时能看到 `qairt-*` 工具、`Qnn*` API/库和 `qti.aisw.*` Python 包；
3. 量化、编译和运行虽然连续发生，但实际上属于不同软件层。

正确的分层方法是：

```text
模型算法层：PyTorch / Hugging Face Qwen2
          ↓
离线优化层：AIMET / QuantSim / SeqMSE / compute_encodings
          ↓
交换产物层：ONNX + weights + encodings
          ↓
编译工具层：QAIRT/QNN converter、quantizer、context generator
          ↓
端侧运行层：Genie + QNN Runtime + HTP Backend
          ↓
硬件执行层：Hexagon HTP/NPU
```

---

## 二、先看一张总关系图

```text
┌─────────────────────────────────────────────────────────────┐
│ example1：x86 Linux + GPU，离线模型准备与量化                │
│                                                             │
│  PyTorch FP 模型                                             │
│      │                                                      │
│      ├─ 项目适配：Linear→Conv / 外部 mask / 定长 KV          │
│      │                                                      │
│      ├─ QAIRT model_preparer                                │
│      │    torch → ONNX → QuIR → QNNIR → prepared torch      │
│      │                                                      │
│      └─ AIMET QuantSim                                      │
│           ├─ 插入量化模拟器                                  │
│           ├─ mixed precision                                │
│           ├─ SeqMSE：优化权重 encodings                      │
│           └─ compute_encodings：统计激活 encodings           │
│                                                             │
│  输出：ONNX + external weights + encodings                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ example2：x86 Linux + CPU，QNN/QAIRT 主机编译                 │
│                                                             │
│  split ONNX → MHA2SHA → qairt-converter → DLC               │
│              → qairt-quantizer → quantized DLC              │
│              → qnn-context-binary-generator（当前为注释模板）│
│                                                             │
│  设计输出：面向具体 HTP/SoC 的 context binary                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ example3：高通设备，端侧运行                                 │
│                                                             │
│  Genie / 应用层                                              │
│      ↓                                                      │
│  QNN Runtime + libQnnHtp Backend                             │
│      ↓                                                      │
│  HTP/NPU 执行定长量化计算图                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、QAIRT 是什么

### 3.1 定位

QAIRT 是 Qualcomm AI Runtime SDK 的名称，可以把它理解为高通 AI 模型转换、编译、运行相关能力的 SDK/工具集合。

在当前 SDK 文档中，Qualcomm AI Engine Direct 仍常被称为 **QNN SDK**；QAIRT 文档和安装目录中同时保留大量 QNN API、QNN backend 与 QNN 命令。官方 Linux Setup 也直接将 Qualcomm AI Engine Direct 称为 “QNN SDK”。

本项目配置很能说明这种关系：

```yaml
qnn_sdk_root: /root/.../qairt/2.42.0.251225
```

变量名叫 `QNN_SDK_ROOT`，实际路径却位于 `qairt/`。因此代码和文档中交替出现 QNN/QAIRT，不代表安装了两套完全无关的软件。

### 3.2 本项目用到的 QAIRT 能力

在 example1 中：

```python
from qti.aisw.preparer_api import model_preparer
```

用来执行项目注释中的：

```text
torch graph → ONNX → QuIR → QNNIR → 重建 torch graph
```

这条链的逐阶段输入、输出、真实产物和排错重点见 [附录C · QAIRT model_preparer 内部流程](./06-附录C-QAIRT-model_preparer内部流程.md)。

在 example2 的设计流程和主机脚本中，还能看到 QAIRT/QNN 主机工具：

```text
qairt-converter
qairt-quantizer
qnn-context-binary-generator
mha2sha-onnx-converter
```

其中 converter 与 quantizer 是当前脚本的执行步骤；`qnn-context-binary-generator` 命令块目前处于注释状态，需要按目标设备配置并启用。也就是说，QAIRT 在本项目既参与前期 Prepare，也参与后期主机编译，但“生成 context binary”目前应理解为设计流程，而非脚本无条件执行的结果。

### 3.3 `qairt-converter` 做什么

官方文档将 `qairt-converter` 描述为把 ONNX、TensorFlow、TFLite 或 PyTorch 等来源模型转换为 DLC；DLC 内保存 Qualcomm 图格式，供后续运行/编译流程使用。

本项目 example2 中：

```text
量化 ONNX + encodings
       ↓ qairt-converter
DLC
```

### 3.4 QAIRT 不是什么

QAIRT 不是：

- 某一种量化算法；
- 一个 PyTorch 模型类；
- 一块 NPU 硬件；
- PPL/精度指标。

它是贯穿模型转换、编译、运行的一套 SDK/工具环境。

---

## 四、QNN 是什么

### 4.1 定位

QNN 是 QAIRT / Qualcomm AI Engine Direct 中面向神经网络图的底层 C API、runtime 与 backend 体系；SDK 文档、库名和命令中仍大量使用 `Qnn*`。与其死记缩写全称，更重要的是记住它在工具链中的职责。

QNN 的核心抽象可以简化为：

```text
QNN Context
   └─ QNN Graph
        └─ QNN Ops / Tensors
             ↓
          QNN Backend
             ├─ CPU
             ├─ GPU
             └─ HTP
```

官方 QNN API 中，许多操作都需要先创建 backend。不同 backend 决定图最终在哪类处理器上执行。

### 4.2 本项目中的 QNN 职责

QNN 主要负责：

- 接收已经转换/量化描述好的模型图；
- 检查目标 backend 支持哪些算子、dtype 和布局；
- 针对具体 HTP 架构优化/编译；
- 生成或加载序列化 context binary；
- 在设备运行时调度图执行。

### 4.3 QNN 与 QAIRT 的关系

最实用的理解是：

```text
QAIRT = 更上层/更完整的 SDK 与工具包装
QNN   = 其中面向图、backend 和端侧执行的核心 API/runtime 体系
```

但不同 SDK 版本、文档页面和脚本仍可能把整套包称作 QNN SDK，因此不必把两个名字机械地理解为互斥产品。

在本项目语境中：

```text
“QAIRT 工具” 侧重 qairt-converter / preparer 等主机工具
“QNN”       侧重 backend、context binary、设备 runtime/API
```

这是一种帮助阅读项目的职责划分，不是严格的公司产品组织边界。

---

## 五、HTP 是什么

### 5.1 定位

HTP 是 Hexagon Tensor Processor，可以把它理解为高通 SoC 中面向张量/神经网络计算的 NPU 类硬件执行单元。

本项目的目标不是“运行在 QNN 上”就结束，而是：

```text
通过 QNN Runtime 的 HTP Backend
把模型图实际放到 HTP 上执行
```

### 5.2 软件与硬件不要混淆

```text
QAIRT/QNN：软件 SDK、编译器、API、runtime
HTP：目标硬件/backend
```

配置中的：

```text
htp_v68
htp_v73
htp_v81
```

表示不同 HTP 架构/目标能力配置。项目注释中给出的对应关系为：

| 项目目标 | HTP 配置 |
|---|---|
| 8 Gen 3 | `htp_v73` |
| SA8295P | `htp_v68` |
| SA8797 | `htp_v81` |

> 这张表只是仓库 `config.yaml` / 代码注释给出的配置值，不是通用芯片规格对照表。SDK 版本、目标 SoC 和样例预设可能不同；真正部署时必须以当前 QAIRT 支持表及设备实际 `soc_id` / `dsp_arch` 为准，不能仅凭这张表决定 HTP 架构。

目标配置会影响：

- 支持的量化位宽和算子组合；
- supergroup/融合规则；
- backend 编译参数；
- context binary 对目标芯片的兼容性。

### 5.3 为什么项目要做端侧适配

为适配 HTP 的静态图和数据通路，本项目提前完成：

- Linear → 1×1 Conv；
- 变长输入 → ARN 定长；
- causal/padding mask 外部生成；
- RoPE cos/sin 外部化；
- KV Cache 固定槽位与 K 转置存储；
- 嵌套 KV tuple → 独立 Tensor 输入输出。

这些动作发生在模型和图层面，目的是让后续 QNN/HTP 编译与运行更稳定高效。

---

## 六、AIMET 是什么

### 6.1 定位

AIMET 是 AI Model Efficiency Toolkit，是用于已训练模型压缩、量化和精度恢复的工具库。本项目使用的是 PyTorch 接口：

```python
aimet_torch
```

AIMET 主要运行在主机侧，用来回答：

```text
哪些 Tensor 要量化？
用多少 bit？
对称还是非对称？
每张量、每通道还是每块？
scale/offset 应该是多少？
哪些敏感层需要更高精度？
量化后精度损失有多大？
```

### 6.2 AIMET 与 QNN 的分工

```text
AIMET：离线“设计和模拟量化”
QNN：  编译并在目标 backend 执行量化模型
```

AIMET 不负责在 HTP 上真正执行模型；QNN 也不会替你完成完整的 PyTorch 量化精度调优实验。

二者通过以下产物衔接：

```text
ONNX 模型 + quantization encodings
```

### 6.3 AIMET 的目标感知配置

创建 QuantSim 时，本项目传入：

```python
config_file=htp_config_file
```

它的作用是让模拟量化规则尽量符合目标 HTP runtime 支持的精度、量化器启停和算子组合约束。

因此 QuantSim 不是随意插入 Q/DQ，而是尽量模拟将来 QNN/HTP 会执行的量化边界。

---

## 七、QuantSim 是什么

### 7.1 QuantSim 是 AIMET 中的一个核心对象

项目代码：

```python
from aimet_torch.v2.quantsim import QuantizationSimModel

quantsim = QuantizationSimModel(
    model=sim_fpm.model,
    quant_scheme=...,
    dummy_input=dummy_input,
    default_output_bw=16,
    default_param_bw=4,
    in_place=True,
    config_file=htp_config_file,
)
```

关系是：

```text
AIMET 是工具库
QuantizationSimModel 是 AIMET 提供的量化模拟模型类
quantsim 是本项目创建出的具体实例
quantsim.model 是插入量化模拟器后的 PyTorch 模型
```

### 7.2 它如何模拟整数误差

QuantSim 会在权重或激活周围加入 Quantize-Dequantize（Q/DQ）行为。概念上：

```text
浮点 x
  ↓ 除 scale、舍入、加 zero-point、截断
整数格点 q
  ↓ 减 zero-point、乘 scale
近似浮点 x_hat
```

公式可以简化为：

```text
q     = clip(round(x / scale) + zero_point)
x_hat = (q - zero_point) × scale
```

这里采用标准 `zero_point` 记法；项目所用旧版 AIMET encoding 若写成 `offset`，符号约定可能相反，见第九节。

后续 PyTorch 算子通常仍在 GPU/CPU 浮点硬件上计算 `x_hat`，但输入已经带有量化舍入和截断误差。因此它能近似预测端侧低比特模型的精度。

### 7.3 QuantSim 不等于真正的端侧整数模型

QuantSim：

- 仍是主机上的 PyTorch 模型；
- 插入的是量化模拟器；
- 重点是模拟量化噪声和验证精度；
- 不会直接生成可在 HTP 上加载的 context binary。

真正的端侧模型还需要：

```text
QuantSim export
   → ONNX + encodings
   → QAIRT/QNN compile
   → context binary
```

### 7.4 dummy input 在 QuantSim 中的作用

构造 QuantSim 时传入 dummy input，主要用于识别计算图、输入结构和 shape，从而判断在哪里插入量化器。

它不负责得到正确的激活范围：

```text
dummy input           → 建图/插量化器
真实 calibration data → 计算 encodings
```

---

## 八、Quantizer 是什么

Quantizer 是 QuantSim 中绑定在参数或激活 Tensor 边界上的量化模拟组件。

常见分类：

```text
Parameter quantizer：量化权重/参数
Input quantizer：    量化模块输入激活
Output quantizer：   量化模块输出激活
```

一个 quantizer 通常包含或决定：

- 是否启用；
- bitwidth；
- signed/unsigned；
- symmetric/asymmetric；
- per-tensor/per-channel/per-block；
- scale/delta；
- offset 或 zero-point（随 encoding schema 而异）；
- 当前 encoding 是否已经初始化/冻结。

本项目默认：

```text
参数/权重：4 bit
输出激活：16 bit
部分 MatMul 第二输入（KV 路径）：8 bit 对称
敏感层：按 exceptions.json 做混合精度覆盖
```

---

## 九、Encodings 是什么

### 9.1 Encodings 不是模型权重

Quantization encoding 描述“浮点值如何映射到有限整数格点”。常见信息包括：

```text
bitwidth
min/max
scale（也常叫 delta）
offset 或 zero-point（取决于 encoding 文件版本）
对称性
量化粒度（per-tensor/per-channel/per-block）
```

它回答的问题是：

```text
浮点数 0.137 应映射到哪个整数？
整数 23 在反量化后代表哪个浮点值？
超出范围的值如何截断？
```

注意：新版 AIMET encoding 采用 `y_scale/y_zero_point`，旧版格式常见 `scale/offset`；二者表达的是同一类仿射量化映射，但字段名、符号约定不能脱离具体版本机械等同。

### 9.2 为什么 ONNX 和 encodings 要一起交付

```text
ONNX       = 算子、连接关系、Tensor 名称、权重
encodings  = 各参数/激活 Tensor 的量化规则
```

只有 ONNX 而没有 encodings，QNN 工具不知道本项目调好的 W4/A16/KV8 量化范围；只有 encodings 而没有 ONNX，也不知道这些规则应该套到哪个计算节点和 Tensor 上。

因此 example1 输出：

```text
qwen25llm.onnx
*.weight / *.bias
qwen25llm.encodings
```

一起交给 example2。

### 9.3 Encodings 从哪里来

本项目有两条主要来源：

```text
权重 encodings：SeqMSE 搜索并冻结一部分最优候选
激活 encodings：compute_encodings 用真实校准数据统计
```

另外还有：

- 默认 min-max/TF 量化方案；
- HTP config 的目标约束；
- `exceptions.json` 的人工混合精度覆盖；
- Concat encoding 传播与 MatMul KV 8bit 特殊处理。

---

## 十、Quantization Scheme 是什么

项目配置：

```yaml
quant_scheme: post_training_tf
```

这里的 `TF` 在 AIMET 中表示一种 min-max 范围估计方式，其名字来源于历史算法命名，不表示本项目改用了 TensorFlow。

Min-Max 的基本思想：

```text
观察 Tensor 的最小值和最大值
        ↓
用有限的 2^bitwidth 个整数格点覆盖该范围
```

优点：覆盖完整观察范围。

缺点：对离群值敏感；极端值会拉大 scale，使大多数普通值的舍入误差上升。

所以本项目还需要 SeqMSE、混合精度等手段进一步降低量化误差。

---

## 十一、`compute_encodings()` 是什么

### 11.1 它做的事

项目代码：

```python
quantsim.compute_encodings(_forward_fn, kwargs)
```

AIMET 会调用 `_forward_fn`，让真实、有代表性的校准数据通过 QuantSim 模型；量化器观察各层真实激活分布，并据此初始化 encodings。

当前配置：

```yaml
compute_encodings_num_batches: 20
```

### 11.2 它不做的事

`compute_encodings()` 通常不是：

- 反向传播训练；
- 更新语言模型权重；
- 计算 PPL；
- 编译 QNN context binary。

它的主要产出是 QuantSim 内部各量化器的范围/scale/offset 状态。

### 11.3 为什么必须用真实数据

激活分布取决于真实输入。若用随机 dummy embedding 校准：

```text
观察到的 min/max ≠ 真实推理激活范围
        ↓
scale/offset 错误
        ↓
量化模型精度严重下降
```

所以：

```text
QuantSim(dummy_input=...)       使用 dummy 建图
quantsim.compute_encodings(...) 使用真实 train_dataloader 校准
```

---

## 十二、SeqMSE 是什么

### 12.1 定位

SeqMSE（Sequential MSE）是 AIMET 的后训练量化优化方法。它逐层搜索候选参数 encodings，使该层量化输出与对应浮点输出之间的误差尽量小。

项目调用：

```python
apply_seq_mse(
    fp_prepared_fpm.model,
    quantsim,
    train_dataloader,
    params,
)
```

它同时需要：

```text
浮点 prepared model
QuantSim model
代表性数据
候选搜索参数
```

### 12.2 它优化什么

对于某一层，概念上比较：

```text
y_fp = 浮点层输出
y_q  = 使用候选 weight encoding 后的模拟量化输出
```

搜索使：

```text
MSE(y_fp, y_q)
```

尽可能小的候选 encoding，然后冻结该层参数量化器的结果，再继续下一层。

### 12.3 当前参数

```yaml
num_batches: 20
num_candidates: 20
inp_symmetry: symqt
loss_fn: mse
```

含义：

- 使用 20 个 batch；
- 每层尝试 20 组候选量化范围；
- `inp_symmetry=symqt` 表示比较候选时，浮点权重和量化权重两侧都使用 QuantSim 路径采集到的输入；它属于 SeqMSE 的输入配对策略，不是量化器的 `is_symmetric` 开关；
- 以 MSE 作为候选评分函数。

### 12.4 SeqMSE 和 compute_encodings 的区别

| 对比 | SeqMSE | `compute_encodings()` |
|---|---|---|
| 核心目标 | 搜索每层更优的参数/权重 encoding | 统计并初始化各量化器 encoding，重点是激活 |
| 方法 | 候选搜索 + 浮点/量化输出误差 | 代表性数据前向 + 分布统计 |
| 是否需要 FP 对照模型 | 需要 | 通常只运行 QuantSim model |
| 本项目顺序 | 先执行 | 后执行 |
| 数据 | 真实 calibration data | 真实 calibration data |

官方推荐流程同样是：

```text
create QuantSim
  → apply SeqMSE（冻结支持层的参数 encodings）
  → compute_encodings（补齐激活和其余未初始化 encodings）
  → evaluate
  → export
```

---

## 十三、Mixed Precision 是什么

“量化”不一定要求所有层用同一个 bitwidth。

本项目默认：

```text
W4A16
```

但某些层可能对 4 bit 权重或低位激活特别敏感，因此使用：

```python
ManualQuantsimMixedPrecisionConfig
```

读取：

```text
config/mixed_precision_config/exceptions.json
```

对特定层覆盖：

- bitwidth；
- encoding；
- 对称性；
- 量化器开关。

本质是：

```text
大多数层低比特，换取速度/内存收益
少数敏感层高精度，守住模型精度
```

---

## 十四、ONNX、DLC、Context Binary 分别是什么

### 14.1 ONNX

ONNX 是跨框架计算图交换格式。本项目 example1 导出的 ONNX：

- 描述算子与连接关系；
- 保留稳定 Tensor 名称；
- 大模型权重可以外置为 `.weight/.bias`；
- 与 AIMET encodings 一起交给 QNN 编译。

ONNX 仍不是针对某颗具体 HTP 已完成编译的最终产物。

### 14.2 DLC

DLC 是 `qairt-converter` 生成的 Qualcomm 图容器/中间模型表示。本项目 example2 先将 ONNX 转成 DLC，再执行量化和后续 context 生成。

### 14.3 Context Binary

Context binary 是针对目标 backend/SoC 生成的序列化执行产物，可以包含已优化、已编译的 QNN graph/context 信息。

概念关系：

```text
ONNX + encodings
   ↓ qairt-converter / quantizer
DLC / quantized DLC
   ↓ qnn-context-binary-generator + HTP config
HTP context binary
```

可以类比：

```text
ONNX              ≈ 平台无关工程图
DLC               ≈ 高通工具链中间图
context binary    ≈ 针对目标设备编好的可加载产物
```

---

## 十五、Genie 是什么

在 example3 中，端侧通过：

```text
libGenie.so
genie-t2t-run
```

运行大模型。

在本项目语境中，Genie 位于 QNN Runtime 之上，负责更高层的生成式 AI/LLM 编排，例如：

- 加载模型和配置；
- 组织 embedding 输入；
- 管理多张/多段 LLM 图；
- 调用 QNN backend 执行；
- 组织推理循环与输出。

底层关系可以简化为：

```text
用户/应用
   ↓
Genie
   ↓
QNN Runtime
   ↓
QNN HTP Backend
   ↓
HTP 硬件
```

Genie 不是量化器，也不负责用校准集计算 encodings。

---

## 十六、Prepare、QuantSim、编译、运行的边界

| 阶段 | 主要工具 | 输入 | 输出 | 是否真实低比特硬件运行 |
|---|---|---|---|---|
| 模型适配 | PyTorch + 项目 Monkey Patch | HF 浮点模型 | QNN 友好浮点模型 | 否 |
| Prepare | QAIRT `model_preparer` | 适配模型 + dummy | prepared 浮点模型 | 否 |
| 量化模拟 | AIMET QuantSim | prepared 模型 + dummy | 带量化模拟器的 PyTorch 模型 | 否 |
| SeqMSE | AIMET | FP/QuantSim + 校准数据 | 优化后的参数 encodings | 否 |
| Calibration | AIMET `compute_encodings` | QuantSim + 校准数据 | 激活/其余 encodings | 否 |
| 精度评估 | PyTorch + PPL | QuantSim model + test data | PPL 指标 | 否 |
| Export | AIMET | QuantSim + dummy | ONNX + encodings | 否 |
| 主机编译 | QAIRT/QNN 工具 | ONNX + encodings | DLC；启用 context 生成步骤后得到 context binary | 否 |
| 端侧执行 | Genie + QNN HTP backend | context binary + 输入 | 推理输出 | 是 |

这张表最适合用来回答：“当前代码到底是在模拟，还是已经跑在 NPU 上？”

`example1/llm_quant.py` 中即使调用 QuantSim，也仍是在主机 GPU 上模拟；真正 HTP 执行发生在 example3。

---

## 十七、本项目代码逐项对应

| 概念 | 本项目代码/配置 | 作用 |
|---|---|---|
| QAIRT SDK 根目录 | `config.yaml:qnn_sdk_root` | 提供 Python 包、动态库与主机工具 |
| QAIRT Prepare | [`model_preparer.prepare_model`](./06-附录C-QAIRT-model_preparer内部流程.md) | 转换并重建 prepared model |
| QuantSim | `QuantizationSimModel(...)` | 插入量化模拟组件 |
| HTP 规则 | `config_file=htp_config_file` | 让量化边界/能力贴近目标 backend |
| KV 8bit | `set_matmul_second_input_producer_to_8bit_symmetric` | 降低 KV 数据 I/O 成本 |
| Encoding 传播 | `propagate_output_encodings(..., Concat)` | 让 Concat 输入输出保持兼容 encoding |
| 混合精度 | `ManualQuantsimMixedPrecisionConfig` | 给敏感层设置例外 |
| SeqMSE | `apply_seq_mse(...)` | 优化支持层的参数 encodings |
| Calibration | `quantsim.compute_encodings(...)` | 用真实数据初始化激活/其余 encodings |
| 量化验收 | `ppl_eval_embedding(..., sim_fpm)` | 比较量化前后 PPL |
| Export | `quantsim.export(...)` | 产生 ONNX、外置权重与 encodings |
| QNN 编译 | example2 | 当前脚本生成 DLC；注释模板展示 HTP context binary 生成步骤 |
| 端侧运行 | example3 Genie | 调用 QNN HTP backend 真正推理 |

---

## 十八、最常见的误解

### 18.1 “AIMET 就是 QNN”

不是。AIMET负责离线优化/量化模拟；QNN负责编译与运行。二者通过 ONNX + encodings 衔接。

### 18.2 “QuantSim 已经是 INT4 模型”

不准确。QuantSim 在浮点硬件上模拟 INT4/INT16 等量化误差；真正目标编译和整数执行还在后面。

### 18.3 “dummy input 就是 calibration data”

不是。dummy 只用于建图和确定接口；calibration 必须使用代表性真实数据。

### 18.4 “SeqMSE 和 compute_encodings 是同一件事”

不是。SeqMSE通过逐层候选搜索优化参数 encodings；`compute_encodings()`通过真实前向统计激活并补齐未初始化 encodings。

### 18.5 “有了 `.encodings` 就能直接上设备”

不能。还需要对应 ONNX/权重，并通过 QAIRT/QNN 工具编译成目标 HTP 可加载的 context binary。

### 18.6 “QAIRT 是硬件”

不是。QAIRT是 SDK/工具环境；HTP才是目标硬件执行单元。

### 18.7 “Prepare 已经量化了模型”

不是。Prepare 后首先验证的仍是浮点 PPL；QuantSim 创建后才进入量化模拟流程。

### 18.8 “PPL 是 calibration”

不是。calibration 用来确定 encodings；PPL 用测试集评价模型质量。可以使用相似数据来源，但目的与计算完全不同。

---

## 十九、面试版回答

### 19.1 QAIRT、QNN、AIMET 的关系

> AIMET 是主机侧模型优化与量化模拟工具，本项目用 QuantSim、SeqMSE 和 compute_encodings 得到低比特模型的量化参数；QAIRT 是高通模型转换、编译与运行工具环境，QNN 是其中面向图、backend 和设备执行的核心 API/runtime。AIMET 导出 ONNX + encodings，QAIRT/QNN 再将其编译成 HTP context binary，最后由 Genie/QNN Runtime 在设备 HTP 上执行。

### 19.2 QuantSim 是什么

> QuantSim 是 AIMET 提供的量化模拟模型。它在 PyTorch 图中加入 Q/DQ 行为，在浮点 GPU/CPU 上模拟低比特舍入和截断误差，便于校准、优化和评估精度；它本身不是最终 HTP 可执行模型。

### 19.3 SeqMSE 与 compute_encodings 的区别

> SeqMSE逐层搜索参数量化范围，使量化层输出逼近浮点层输出；compute_encodings则用代表性数据跑模型，统计并初始化激活和其他未完成量化器的 scale/offset。项目中先 SeqMSE，再 compute_encodings。

---

## 二十、记忆锚点

> 若只想快速回忆概念关系（而非逐个定义），直接回看 [〇、30 秒速记版](#〇30-秒速记版先建框架再读细节)：三问法表 + 两个混淆根源 + 出场时间线。

**先记这三条边界**（分不清时回到这里）：

- **按机器分**：AIMET / QuantSim / QAIRT 在**你的电脑**（离线）；QNN / HTP 在**手机**（运行时）。
- **两种"量化"**：AIMET 是**模拟量化**（模型仍是浮点，产出 encodings 参数表）；QAIRT/QNN quantizer 才**真正整数化**。
- **QAIRT ⊃ QNN**：包含关系，不是并列的两个产品。

```text
AIMET：决定怎么量化
QuantSim：模拟量化误差
SeqMSE：优化权重 encoding
compute_encodings：统计激活 encoding
encodings：保存量化映射规则
ONNX：保存平台无关计算图
QAIRT：转换/编译/运行工具环境
QNN：图、backend 与设备 runtime
HTP：真正执行张量计算的硬件
Genie：端侧大模型推理编排层
```

再压缩成一条链：

```text
PyTorch FP
  → QAIRT Prepare
  → AIMET QuantSim/SeqMSE/calibration
  → ONNX + encodings
  → QAIRT/QNN compile
  → HTP context binary
  → Genie/QNN/HTP inference
```

---

## 二十一、官方资料与项目入口

### 官方资料

- [Qualcomm AI Runtime / QNN Linux Setup](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/linux_setup.html?product=1601111740009302)：QAIRT/QNN SDK 安装、支持平台与 backend 概览。
- [Qualcomm `qairt-converter`](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/qairt_converter.html?product=1601111740010412)：来源模型到 DLC 的转换工具。
- [Qualcomm AI Engine Direct SDK 文档入口](https://docs.qualcomm.com/bundle/publicresource/topics/80-87189-1/overview.html?product=1601111740009302)：QAIRT API、QNN、HTP backend 等文档树。
- [AIMET Quantization Simulation Guide](https://quic.github.io/aimet-pages/releases/latest/tutorials/quantsim.html)：QuantSim、calibration、encodings 与 export 流程。
- [AIMET Encoding Format Specification](https://quic.github.io/aimet-pages/releases/latest/techniques/encoding_spec.html)：scale/offset 与 encoding 文件结构。
- [AIMET Sequential MSE](https://quic.github.io/aimet-pages/releases/2.10.0/ptq_techniques/seq_mse.html)：SeqMSE 的目标、工作流与 API。
- [AIMET GitHub](https://github.com/quic/aimet)：AIMET 开源仓库与支持的 PTQ 技术概览。

### 本项目入口

- `example1/llm_quant.py`：Prepare、QuantSim、SeqMSE、compute encodings、PPL 与 export。
- `example1/config.yaml`：SDK、HTP、位宽、SeqMSE 与评估参数。
- `docs/PIPELINE.md`：example1 完整量化流水线。
- `docs/EXAMPLES_OVERVIEW.md`：example1/2/3 的职责和产物流转。
- [06 · Prepare 模型准备阶段](./06-Prepare模型准备阶段.md)
- [06-附录A · Prepare Dummy Input](./06-附录A-Prepare-Dummy-Input输入模具.md)
- [06-附录C · QAIRT model_preparer 内部流程](./06-附录C-QAIRT-model_preparer内部流程.md)
- [05 · 通用前向处理流程](./05-通用前向处理流程.md)
