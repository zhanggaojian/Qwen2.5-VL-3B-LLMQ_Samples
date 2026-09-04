# 03 · MHA2SHA 图结构转换

> **学习位置**：Example2 五阶段中的第三阶段。
>
> **上一阶段**：[02 · Split ONNX 与测试向量](./02-Split-ONNX与测试向量.md)
>
> **下一阶段**：[04 · ONNX 到 DLC：qairt-converter](./04-ONNX到DLC-qairt-converter.md)
>
> **流程总览**：[00 · Example2 主机编译全景](./00-example2主机编译全景.md)
>
> **AR Shape 基础**：[01 · AR 图适配](./01-AR图适配-change_hardcoding.md)
>
> **Attention 基础**：[Attention 注意力机制](../02-附录A-Attention注意力机制.md)
>
> **产物与清理**：[06 · example2 产物总览与清理](./06-example2产物总览与清理.md)
>
> **一句话本质**：识别每层 GQA Attention，把原来“合并计算多个 Head”的图展开成逐 Query Head 的 SHA/Conv 分支，同时保留两组共享 K/V、RoPE、Past-KV 和原量化 Encoding，最后再把 16 个 Head 的结果拼回原接口。

---

## 零、先记住十个结论

1. `SHA` 不是把模型裁成只剩一个注意力头，而是把每个头显式展开成一条 **Single-Head Attention 分支**。
2. 当前 Qwen2.5-VL-3B 的 Attention 语义是 **GQA**：16 个 Query Head、2 个 KV Head、`head_dim=128`。
3. 每 8 个 Query Head 共享 1 组 K/V；转换后这层共享关系仍然保留。
4. 这是 ONNX **计算图等价改写**，不训练模型，也不重新执行 Calibration、SeqMSE 或 `compute_encodings()`。
5. 当前图已经是 Conv 投影和 NCHW/BD1L 布局，因此走的是工具里的 SHA-Conv + RoPE 路径。
6. 图中的 Tensor 和权重被拆分、改名后，旧 Encoding 不能原封不动地直接套用，必须同步映射和切片。
7. 本阶段为 AR1、AR128 各生成一套 `SHA ONNX + SHA Encoding + External Data`。
8. MHA2SHA 自己会用 ONNX Runtime 做一次随机输入对拍，但它不使用 Split 阶段的 `qt_0.pkl` 或 RAW。
9. 文件已经生成、进程返回 0、日志打印 `done`，都不能单独证明转换正确；必须看到明确的 `Verification Status ... OK`。
10. 本阶段还没有生成 DLC；下一阶段才由 `qairt-converter` 消费 SHA ONNX 和 SHA Encoding。

先用一张图建立直觉：

```text
转换前：一层 GQA 的合并表达

大 Q Conv ─┐
大 K Conv ─┼─ Reshape / Repeat / Batched Attention ─ Head 合并 ─ O-Proj
大 V Conv ─┘
              │
              ▼ MHA2SHA

转换后：16 条显式 Query 单头分支，仍共享 2 组 K/V

Q0  ─┐
Q1  ─┤
...  ├── 使用 K0 / V0 ── SHA0 ... SHA7  ─┐
Q7  ─┘                                    │
                                          ├─ Concat 16 Head ─ 恢复原输出接口
Q8  ─┐                                    │
Q9  ─┤                                    │
...  ├── 使用 K1 / V1 ── SHA8 ... SHA15 ─┘
Q15 ─┘
```

最容易记错的地方是：

> **“Single Head”描述的是新图中每条计算分支的形态，不是说整个模型最终只剩一个 Head。**

---

## 一、本阶段在完整流程中的位置

```text
阶段一：AR 图适配
  AR1073 → AR1 / AR128
        │
        ▼
阶段二：Split ONNX + RAW / Input List / Golden
  当前 num_splits=1，得到两份 1_of_1.onnx
        │
        ▼
阶段三：MHA2SHA                         ← 本篇
  GQA 合并图 → 显式逐 Query Head SHA 图
  原 Encoding → SHA Tensor 对应 Encoding
        │
        ▼
阶段四：qairt-converter
  SHA ONNX + SHA Encoding → 普通 DLC
        │
        ▼
阶段五：qairt-quantizer
  普通 DLC + RAW → Quantized DLC
```

### 1.1 当前实际会跑几个任务

主脚本当前配置为：

```python
ARNs = [1, 128]
num_splits = 1
go_parallel = False
```

由 `arn_list` 和 `split_idxs` 展开后，本阶段有两个任务：

```text
ar1-cl2048_1_of_1
ar128-cl2048_1_of_1
```

因为 `go_parallel=False`，`ProcessPoolExecutor` 只有一个 Worker，两项任务按顺序执行。

### 1.2 单个任务的输入

以 AR1 为例：

```text
example2/host_linux/assets/artifacts/ar1-cl2048/
├── split_onnx/
│   ├── ar1-cl2048_1_of_1.onnx
│   └── 该 ONNX 引用的全部 External Data
└── src -> assets/models_ar_n/ar1-cl2048
    └── onnx/
        └── qwen25llm.encodings
```

两类输入职责不同：

| 输入 | 保存什么 | 本阶段怎样使用 |
|---|---|---|
| Split ONNX + External Data | 当前 AR 固定 Shape 的图和真实参数 | 识别并重写 Attention 子图 |
| `qwen25llm.encodings` | 原 Tensor/参数的量化 scale、offset、位宽等规则 | 映射到新 SHA Tensor 和切片后的权重 |

这里没有读取：

- `qt_0.pkl`；
- `test_inputs_*/*.raw`；
- `input_list_*.txt`；
- `test_golden_outputs_*/*.raw`。

它们属于 Split 和后续 Quantizer/真实对拍链，不是当前 MHA2SHA 内置验证的输入。

### 1.3 本阶段做什么、不做什么

| 会做 | 不会做 |
|---|---|
| 找出每层 Attention Pattern | 不训练模型 |
| 把合并 Head 图改成逐 Head 分支 | 不改变模型的层数和隐藏维度 |
| 保留 GQA 的 KV 共享关系 | 不把 GQA 改成普通 MHA |
| 重建 RoPE、Past-KV 和相关布局 | 不重新生成 AR1/AR128 Shape |
| 映射原 AIMET Encoding | 不重新标定 scale/offset |
| 保存 SHA ONNX 模型束 | 不生成 DLC |
| 用 ORT 做随机输入图等价验证 | 不自动做 HTP 真机验证 |

---

## 二、MHA、GQA、SHA 到底是什么关系

### 2.1 三个词不在完全相同的分类维度上

| 名称 | 回答的问题 | 当前项目中的含义 |
|---|---|---|
| MHA | Q/K/V 的 Head 如何组织 | 每个 Query Head 有自己的一组 K/V |
| GQA | 多个 Query Head 如何共享 K/V | 16 个 Q Head 共享 2 个 KV Head |
| SHA 图 | 后端计算图怎样展开 | 每条 Attention 计算分支只处理一个 Query Head |

所以，本工具虽然叫 `MHA2SHA`，当前输入模型严格说是 GQA；这个名字是工具的通用名称。

转换前后，模型的 Attention 语义仍然是 GQA：

```text
语义结构：16Q + 2KV，8 个 Q 共享一组 KV
图表达：  Batched/合并 Head → 16 条显式单 Q-Head 分支
```

### 2.2 当前模型的具体数字

```text
hidden_size          = 2048
num_attention_heads  = 16
num_key_value_heads  = 2
head_dim              = 2048 / 16 = 128
每组 Query Head 数量 = 16 / 2 = 8
```

因此：

```text
Query 投影输出通道 = 16 × 128 = 2048
Key 投影输出通道   =  2 × 128 = 256
Value 投影输出通道 =  2 × 128 = 256
```

映射关系是：

| Query Head | 使用的 Key Head | 使用的 Value Head |
|---|---|---|
| Q0～Q7 | K0 | V0 |
| Q8～Q15 | K1 | V1 |

工具源码中的核心选择逻辑可概括为：

```python
head_num_per_group = num_query_heads // num_kv_heads  # 16 // 2 = 8
kv_group_index = head_num // head_num_per_group
```

### 2.3 为什么这不是“删除 15 个头”

一层中仍然会为 `head_num in range(num_heads)` 循环 16 次，每次创建一条 Query Head 的 Attention 分支。随后将 16 个分支的结果 `Concat`，再 `Reshape/Transpose` 回原 Attention 输出接口。

当前完整 `1_of_1` 图预期包含 36 层 Attention，因此结构上会形成：

```text
36 层 × 每层 16 条 Query SHA 分支 = 576 条 Query 单头分支
```

这只是帮助理解规模；是否真的完整转换了 36 层，必须以本次日志中的 Pattern 匹配结果为准。

MHA、GQA 的数学原理见：

- [Attention 注意力机制](../02-附录A-Attention注意力机制.md)
- [Attention 分类大全](../02-附录F-Attention分类大全(面试向).md)

---

## 三、为什么要为 HTP 改写成 SHA 图

原始/框架友好的 Attention 图常把 Head 维放在一个大 Tensor 里统一计算：

```text
大 Q/K/V Projection
  → Reshape 出 Head 维
  → 多次 Transpose
  → 带 Head 维的批量 QKᵀ / Softmax / AV
  → 再拼回 Hidden
```

MHA2SHA 将 Head 维显式展开：

```text
逐 Head Projection
  → 固定 4D Layout
  → 逐 Head QKᵀ / Softmax / AV
  → Concat
```

仓库内置工具的 README 将目标描述为：

- 为 HTP Backend 生成更适合处理的 SHA/Conv 图形态；
- 减少冗余 Layout 变换；
- 改善端侧延迟。

这里要保持一个工程上的边界：

> **SHA 是针对当前 Qualcomm HTP 工具链的部署图优化，不代表它在所有 CPU、GPU 或其他 NPU 上都必然更快。最终性能必须在目标 SoC 上实测。**

### 3.1 为什么当前不是“现在才把 Linear 变成 Conv”

Example1 的 `QcAttention` 已经把 Q/K/V/O Projection 准备成 1×1 Conv，并把输入对齐到 NCHW/BD1L 布局。

当前命令使用的是：

```text
--mha-conv
--nchw-aligned
```

而没有使用：

```text
--replace-linear-with-conv
```

所以本阶段的含义是：

```text
识别“已经是 Conv 的合并 Head Attention”
                  ↓
生成“逐 Head 的 SHA-Conv Attention”
```

不是对全模型再做一次 Linear→Conv。

### 3.2 BD1L 是什么

工具 README 对 LLM NCHW 输入的描述是：

```text
[N, vector_dim, 1, context/current length]
```

本笔记缩写为：

```text
[B, D, 1, L] = BD1L
```

- `B`：Batch；
- `D`：通道/向量维；
- `1`：占位空间维；
- `L`：当前 Token 长度或相关序列长度。

`--nchw-aligned` 是对真实图布局的声明，不只是一个性能提示。声明与真实图不一致，可能导致 Pattern、Transpose 或 Shape 错误。

---

## 四、外层代码 `thread_g2g()` 做了什么

主入口位于：

[`example2/host_linux/qnn_compile_deploy.py`](../../../example2/host_linux/qnn_compile_deploy.py)

### 4.1 先把内置工具加入搜索路径

```python
mha2sha_root = workfolder + "/../G2G/MHA2SHA"

g2g_env["PYTHONPATH"] = ... + mha2sha_root + "/src/python"
g2g_env["PATH"]       = ... + mha2sha_root + "/bin"
```

两者作用不同：

| 环境变量 | 找什么 | 本项目中找到什么 |
|---|---|---|
| `PATH` | 可执行命令 | `bin/mha2sha-onnx-converter` |
| `PYTHONPATH` | Python 包/模块 | `src/python/mha2sha/` |

这也串起了“Python import 到底是什么”的问题：

```text
PATH 找到一个无 .py 后缀、但带 Python shebang 的命令脚本
    ↓
脚本执行 from mha2sha.converter import MHA2SHAConverter
    ↓
PYTHONPATH 帮 Python 找到 mha2sha 包
    ↓
再加载包内的 converter.py 模块
```

需要注意，代码把项目路径追加在已有 `PATH/PYTHONPATH` 的后面。若环境中已有同名命令或同名 Python 包，前面的版本可能先被命中。正式运行应记录：

```bash
which mha2sha-onnx-converter
python -c "import mha2sha, inspect; print(inspect.getfile(mha2sha))"
```

### 4.2 为什么必须从 `example2/host_linux` 启动

脚本将 `os.getcwd()` 直接赋给 `workfolder`。因此预期启动方式是：

```bash
cd example2/host_linux
python qnn_compile_deploy.py
```

这时：

```text
workfolder/../G2G/MHA2SHA
= example2/G2G/MHA2SHA
```

若从仓库根目录直接运行，工具根目录和 `assets/` 落盘位置都会错。

### 4.3 每个 AR/Split 的目录和名字

核心代码是：

```python
model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
split_work_dir = os.path.join(model_artifact, f"{split}_of_{num_splits}")
sha_folder = f"{split_work_dir}/sha_output/"
name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
```

因此 AR1 的真实字符串是：

```text
目录：1_of_1/sha_output/
模型名：ar1-cl2048_1_of_1
```

如果聊天或 Markdown 中看起来像 `1*of*1`，只是 `_` 被 Markdown 解释成斜体标记，源代码实际使用的是下划线 `_of_`。

### 4.4 子进程调用链

```text
qnn_compile_deploy.py::thread_g2g()
  │
  ├─ subprocess.Popen(["mha2sha-onnx-converter", ...])
  │
  ▼
bin/mha2sha-onnx-converter
  │
  ├─ 解析 CLI 参数
  ├─ 创建 MHA2SHAConverter
  │
  ▼
MHA2SHAConverter.convert()
  │
  ├─ 加载/检查原图
  ├─ 生成临时 Golden
  ├─ 匹配 Attention
  ├─ MHA/GQA → SHA
  ├─ 映射 Encoding 并保存模型
  └─ ORT 数值对拍
```

---

## 五、当前命令参数逐个解释

外层最终拼出的核心命令等价于：

```bash
mha2sha-onnx-converter \
  --sha-export-path <1_of_1/sha_output/> \
  --model-name <ar*-cl2048_1_of_1> \
  --exported-model-encoding-path <src/onnx/qwen25llm.encodings> \
  --exported-model-path <split_onnx/ar*-cl2048_1_of_1.onnx> \
  --llm-model \
  --handle-rope-ops \
  --handle-past-key-value \
  --mha-conv \
  --gqa-model \
  --nchw-aligned \
  --log-level verbose
```

| 参数 | 当前作用 | 不应误解成 |
|---|---|---|
| `--sha-export-path` | SHA 模型束和映射 JSON 的输出目录 | DLC 输出目录 |
| `--model-name` | 决定输出 ONNX/Encoding 基名 | 改模型架构名称 |
| `--exported-model-path` | 输入 Split ONNX | Example1 的原始 AR1073 ONNX |
| `--exported-model-encoding-path` | 输入原 AIMET Encoding | 重新 Calibration 的数据集 |
| `--llm-model` | 启用 LLM 特有接口逻辑，也是 Past-KV 处理前提 | 自动选择 Qwen 模板 |
| `--handle-rope-ops` | 识别并逐分支重建 RoPE | 重新计算训练时的位置知识 |
| `--handle-past-key-value` | 处理历史 KV 输入、拼接和新 KV 输出 | 将 Cache 永久存进 ONNX |
| `--mha-conv` | 声明输入 Attention Projection 已是 Conv，并生成 SHA-Conv | 本阶段全局 Linear→Conv |
| `--gqa-model` | Q Head 数与 KV Head 数不同，保留分组共享 | 把 GQA 扩成 16 套 KV 权重 |
| `--nchw-aligned` | 声明 Projection 输入已是 NCHW/BD1L | 一个无关紧要的格式标签 |
| `--log-level verbose` | 输出详细匹配、转换和误差信息 | 自动提高数值精度 |

参数间还有依赖：

```text
--handle-rope-ops
        └── 要求 --handle-past-key-value
                    └── 要求 --llm-model

--mha-conv + --handle-rope-ops
        └── 当前实现要求 --nchw-aligned
```

当前没有启用的几个参数也很重要：

| 未启用参数 | 当前结果 |
|---|---|
| `--no-verification` | 未启用，所以会执行 MHA/SHA ORT 对拍 |
| `--create-input-lists` | 未启用，所以 MHA2SHA 的随机输入和 Golden 不落盘 |
| `--replace-linear-with-conv` | 未启用，因为输入已经是 Conv 图 |
| `--optimize-o-proj` | 未启用，不应把可选的 Head-Concat→O-Proj 清理当作当前流程 |
| `--build-ar` | 未启用，AR Shape 已在阶段一完成 |
| `--base-llm` | 未启用，脚本逐个显式传入需要的特性开关 |

---

## 六、`convert()` 内部完整六步

源码主流程位于 `converter.py:201-309`。

### 第 1 步：加载 Split ONNX 和 External Data

```text
split_onnx/<name>.onnx
      + ONNX 中 external_data.location 指向的参数文件
                        ↓
                 内存 ModelProto
```

运行环境至少要求：

- Python `>=3.10`；
- ONNX `>=1.14.1`。

### 第 2 步：检查原始模型

因为没有传 `--no-verification`，工具会执行原模型检查。大模型的某些内部检查会受 2 GiB 限制，不能把“跳过大模型 checker 的 Warning”当成完整验收。

### 第 3 步：生成随机输入和原图 Golden

工具固定：

```python
np.random.seed(42)
```

然后用 ONNX Runtime 跑原始 Split ONNX，把全部输出暂存在内存里，作为稍后 SHA 图的临时 Golden。

这套数据与阶段二生成的真实/候选测试资产完全独立。

### 第 4 步：寻找 Attention Pattern

工具优先运行 Auto Attention Finder；当前有 `--gqa-model`，会走针对 GQA/LoRA 场景的快速查找路径。若自动查找没有得到 Pattern，再回退到预定义 Pattern。

它需要识别的关键结构包括：

```text
Q/K/V Projection
    ↓
RoPE / Reshape / Transpose
    ↓
QK MatMul
    ↓
Scale + Mask + Softmax
    ↓
Attention × V MatMul
    ↓
Head 合并
```

当前 `1_of_1` 是完整 36 层图，预期日志应出现：

```text
found_matched_pattern: 36
```

如果不是 36，不能直接继续编译；要先判断漏匹配、误匹配还是输入图结构发生了变化。

### 第 5 步：改图、映射 Encoding、保存结果

这一步又分为：

```text
5.1 PreQuantAdaption
    ├─ 做必要的前置清理/改名
    └─ 写 prequant_encodings_map.json

5.2 MHA2SHAOptimizer.optimize()
    ├─ 对每个 Attention 层读取 Q/K/V 信息
    ├─ 检测 16 个 Q Head、2 个 KV Head
    ├─ 创建逐 Query Head SHA 分支
    ├─ 重建 RoPE、Past-KV、Layout
    └─ 拼回原 Attention 输出接口

5.3 MHA2SHAEncodingMapper
    ├─ 映射 Activation Encoding
    ├─ 切分/复制 Parameter Encoding
    └─ 写 SHA Encoding 和映射 JSON

5.4 保存 SHA ONNX + External Data
```

### 第 6 步：用同一输入对拍原图与 SHA 图

```text
同一组 seed=42 随机输入
      ├─ 原 MHA/GQA 图 → Golden Outputs
      └─ 新 SHA 图     → Converted Outputs
                              ↓
                    逐输出 np.allclose()
```

成功时需要看到明确日志：

```text
Verification Status ----- OK -----
```

---

## 七、一层 Attention 的图究竟怎样改

### 7.1 先拆 Q/K/V Projection 的输出通道

当前输入是 Conv 图，可以按输出通道理解权重拆分：

```text
Q 大 Conv：2048 个输出通道
  → Q0 ... Q15
  → 每条 Q Head Conv 取 128 个输出通道

K 大 Conv：256 个输出通道
  → K0、K1
  → 每条真实 K Head Conv 取 128 个输出通道

V 大 Conv：256 个输出通道
  → V0、V1
  → 每条真实 V Head Conv 取 128 个输出通道
```

若有 Bias，也必须按相同 Head 通道范围切片。

这并不是把参数随意复制 16 份：

- Q 本来就有 16 个 Head 对应的不同通道；
- K/V 本来只有 2 个真实 Head；
- 工具只把合并存储的通道拆成显式小权重。

### 7.2 GQA 为什么先构建 K/V，再循环 16 个 Q Head

优化器在进入 Query Head 循环前，会调用 GQA Extension：

```text
先构建 K0、V0
再构建 K1、V1
应用 K 的 RoPE
处理 K/V 的 Layout 和 Past-KV Concat
把两组结果缓存到 key_groups_list / value_groups_list
```

随后才执行：

```python
for head_num in range(num_query_heads):  # 0...15
    创建 Q_head
    选中所属的 K/V Group
    创建这一条 SHA Attention
```

这种顺序保证：

```text
16 个 Q 分支存在
2 组 K/V 投影存在
同组的 8 个 Q 分支引用同一组 K/V 结果
```

而不是错误地生成 16 套独立 K/V 权重。

### 7.3 当前实际选择的是哪条实现路径

优化器的分发条件是：

```python
if mha_conv and handle_rope_ops and nchw_aligned:
    create_sha_func = create_sha_conv_with_rope
```

当前三个条件全部为真，所以执行的是 **Conv + RoPE + NCHW 的专用 SHA 路径**，不是普通 Linear/MatMul Projection 路径。

### 7.4 每条 Query Head 分支的核心计算

忽略 Layout 辅助节点后，一条分支可以抽象为：

```text
Hidden
  │
  ├─ Q_head Conv → Q RoPE ─────────────────────┐
  │                                             │
  ├─ 所属 K_group Conv → K RoPE → Past K Concat├─ QKᵀ
  │                                             │
  └─ 所属 V_group Conv → Past V Concat ────────┘

QKᵀ
  → Scale
  → Add Causal/Combined Mask
  → Softmax
  → × V
  → 当前 Head Output
```

Attention 的数学本身没有改变：

```text
Attention(Q, K, V) = softmax(QKᵀ / √head_dim + Mask) V
```

变化的是“一个大图一次表达所有 Head”还是“每个 Head 一条显式图分支”。

### 7.5 最后怎样恢复原接口

16 条分支结束后：

```text
head_output_0
head_output_1
...
head_output_15
       │
       ▼
Concat
       │
       ▼
Reshape / Transpose
       │
       ▼
替换原 Attention 最终输出的消费者
```

因此下游 Decoder Layer 仍然看到与原 Attention 对应的 Hidden 接口，后续残差连接和 O-Proj 不需要理解“这是 SHA 展开图”。

---

## 八、RoPE 在改图时怎样保留

RoPE 只作用于 Q 和 K，不作用于 V。

当前模型将 `position_ids_cos` 和 `position_ids_sin` 作为输入。工具需要在新的逐 Head 图中重建对应关系：

```text
Q_head → 分半/旋转 → 与 cos、sin 做 Mul/Add → Q_rope_head
K_head → 分半/旋转 → 与 cos、sin 做 Mul/Add → K_rope_head
```

数学上可简写为：

```text
[x1, x2]
  → [x1·cos - x2·sin,
     x1·sin + x2·cos]
```

GQA 场景中：

- 16 个 Q Head 分别处理自己的 Q RoPE；
- 只有 2 个真实 K Head 需要生成 K RoPE；
- 8 个 Query Head 共享处理后的同一个 K Group。

RoPE 的完整原理见：[RoPE 位置编码](../02-附录G-RoPE位置编码.md)。

---

## 九、Past-KV 在改图时怎样保留

一轮推理中有两条不同的数据路径：

```text
用于本轮 Attention：
Past K/V + Current New K/V → Concat → 参与 QKᵀ 和 Attention×V

返回给外部 Runtime：
Current New K/V → 模型输出 → 外部更新固定长度 Cache
```

本项目配置 `return_new_key_value_only=true`，所以 ONNX 不负责永久保存完整 KV Cache。外部运行管理器负责：

1. 接收本轮 New K/V；
2. 与历史 Cache 更新/滑动；
3. 下一轮再通过 `past_key_*_in`、`past_value_*_in` 回灌模型。

MHA2SHA 需要完成的是：

- 找到每层原来的 Past-Key/Past-Value 输入；
- 为逐 Head/GQA Group 重建 Transpose 与 Concat；
- 保持 New-KV 输出接口；
- 保持 AR1/AR128 已经确定的 Current/Past 长度语义。

工具注释中的常见布局是：

```text
Key   用于 Attention：[B, 1, D, L_total]
Value 用于 Attention：[B, 1, L_total, D]
```

其中：

```text
L_total = Past Length + Current AR = CL = 2048
```

AR/CL/Past 长度的完整推导见 [01 · AR 图适配](./01-AR图适配-change_hardcoding.md)，KV Cache 的通用原理见 [KV Cache](../02-附录K-KV%20Cache(键值缓存).md)。

---

## 十、为什么 Encoding 必须同步改

### 10.1 旧 Encoding 的名字已经对不上新图

Example1 导出的 Encoding 通过 Tensor/参数名称绑定量化规则。MHA2SHA 后发生了：

- 一个大 Q 权重被切成 16 个 Head 权重；
- 一个大 K/V 权重分别被切成 2 个 KV Head 权重；
- 原合并 Activation 变成多个逐 Head Activation；
- RoPE、Past-KV、Transpose、Concat 等 Tensor 名称改变；
- 原图部分节点和参数被删除。

如果只改 ONNX、不改 Encoding：

```text
旧 Encoding 名称 ──X── 新 SHA Tensor 名称
```

下一阶段 `qairt-converter --quantization_overrides` 就无法正确把量化规则应用到新图。

### 10.2 Encoding 映射链

```text
src/onnx/qwen25llm.encodings
        │ JSON load
        ▼
PreQuantAdaption
        ├─ 必要的前置名称调整
        └─ prequant_encodings_map.json
        │
        ▼
MHA/GQA → SHA 图改写
        │
        ▼
MHA2SHAEncodingMapper
        ├─ Activation Encoding 一对多复制/改名
        ├─ Per-channel 参数 Encoding 按 Head 通道切片
        ├─ Per-tensor 参数 Encoding 复制到对应新权重
        ├─ 映射 RoPE 和 Past-KV Encoding
        └─ 删除新图中已不存在的旧 Encoding
        │
        ├─ <name>.encodings
        ├─ mha_to_sha_encodings_names.json
        └─ all_stages_encodings_mapping.json
```

概念例子：

```text
原 Q 大权重 Encoding
        │
        ├─ 通道 0...127     → Q0 Weight Encoding
        ├─ 通道 128...255   → Q1 Weight Encoding
        ├─ ...
        └─ 通道 1920...2047 → Q15 Weight Encoding
```

### 10.3 这不是重新量化

本阶段不会重新收集 Activation 分布，也不会重新搜索 scale/offset。

```text
Example1：决定“用什么量化尺子”
MHA2SHA：图拆开以后，把原尺子正确分配给新 Tensor
```

如果原 Encoding 错了，MHA2SHA 不会自动把它修好；它只负责映射已有规则。

### 10.4 三份 Mapping JSON 和最终 Encoding 的区别

| 文件 | 内容 | 当前下游是否直接读取 |
|---|---|---|
| `prequant_encodings_map.json` | PreQuant 前后 Tensor 名称关系 | 否，主要用于追踪 |
| `mha_to_sha_encodings_names.json` | 原 MHA Tensor → 新 SHA Tensor 列表 | 否，主要用于追踪/调试 |
| `all_stages_encodings_mapping.json` | 合并两阶段名称映射链 | 普通路径不读；LoRA 路径可再读取 |
| `<name>.encodings` | 新 SHA 图真正使用的量化规则 | **是，下一阶段必需** |

三份 Mapping JSON 记录“谁变成了谁”；`<name>.encodings` 才保存新图实际要交给 QAIRT 的量化规则。

当前没有启用 LoRA，所以 `prequant_encodings_map.json` 通常只包含空的 `activation_encodings` / `param_encodings` 映射，但仍会生成。

---

## 十一、本阶段产物最终存在哪里

### 11.1 AR1

```text
example2/host_linux/assets/artifacts/ar1-cl2048/1_of_1/sha_output/
├── ar1-cl2048_1_of_1.onnx
├── 该 ONNX 的 external_data.location 引用的全部文件
├── ar1-cl2048_1_of_1.encodings
├── prequant_encodings_map.json
├── mha_to_sha_encodings_names.json
└── all_stages_encodings_mapping.json
```

### 11.2 AR128

```text
example2/host_linux/assets/artifacts/ar128-cl2048/1_of_1/sha_output/
├── ar128-cl2048_1_of_1.onnx
├── 该 ONNX 的 external_data.location 引用的全部文件
├── ar128-cl2048_1_of_1.encodings
├── prequant_encodings_map.json
├── mha_to_sha_encodings_names.json
└── all_stages_encodings_mapping.json
```

当前本地工作区的 `example2/host_linux/assets/` 没有实际文件，因此上面的树是根据源码推导的预期结构；生成服务器上仍需按真实产物验收。

### 11.3 ONNX 与 External Data 是一个不可拆分的模型束

`save_model()` 在模型达到 2 GiB 时会启用 ONNX External Data，并设置：

```python
save_as_external_data=True
all_tensors_to_one_file=False
```

因此不能假设输出永远只有一个 `<name>.data`。正确判断方法是：

```text
读取 SHA ONNX
  → 枚举每个 Initializer/Attribute 的 external_data.location
  → 每一条引用都必须存在且非空
```

只复制小的 `.onnx` 主文件，模型一定可能不完整。

### 11.4 下一阶段真正读取什么

`thread_convert()` 明确使用：

```text
--input_network
  1_of_1/sha_output/<name>.onnx

--quantization_overrides
  1_of_1/sha_output/<name>.encodings
```

所以本阶段向阶段四交付的最小逻辑组合是：

```text
SHA ONNX
+ SHA ONNX 引用的全部 External Data
+ SHA Encoding
```

三份 Mapping JSON 虽不是 `qairt-converter` 的直接输入，但体积小、排错价值高，建议保留到最终 DLC 完整验收以后。

---

## 十二、MHA2SHA 内置数值验证到底验证了什么

### 12.1 当前验证确实开启

当前命令没有传 `--no-verification`，所以：

1. 用随机输入运行原始 MHA/GQA ONNX；
2. 保存全部原图输出到内存；
3. 改写并保存 SHA ONNX；
4. 用同一输入运行 SHA ONNX；
5. 对每个输出执行：

```python
np.allclose(original_output, sha_output, atol=1e-4)
```

`rtol` 没有显式传入，使用 NumPy 默认值 `1e-5`。

对 Past-KV 输出，代码会尝试用 Transpose 对齐转换前后的 Cache Layout，然后再比较。

### 12.2 当前验证不会落盘测试数据

因为没有传 `--create-input-lists`，随机输入和原图 Golden 只停留在内存。

当前主脚本不应新增以下必选产物：

```text
mha_input_vectors/
golden_output_from_mha/
sha_test_vectors/
on_device_input_list.txt
```

若目录里出现这些文件，需要判断是不是旧运行残留，或人工运行时额外启用了参数。

### 12.3 内置随机输入的局限

启用 Past-KV 后，工具调用 `_generate_llama_test_data()`。它把“第一个模型输入”当作 `input_ids`，使用 1～499 的随机整数再转成目标 dtype。

但本项目使用 `inputs_embeds` 作为第一个输入，因此实际会得到：

```text
形状正确、dtype 正确，但数值像整数 1～499 的 FP32 Embedding
```

其他输入也只是随机数，并不保证是：

- 合法的 Causal/Combined Attention Mask；
- 成对、满足三角函数关系的 RoPE cos/sin；
- 来自真实 Prompt 的 Past-KV。

因此它能发现一部分改图错误，但不能替代真实 AR1/AR128 输入对拍。

### 12.4 `allclose` 还有两个 Shape 边界

1. 比较前没有统一强制断言两个输出 Shape 完全一致；若 NumPy 可以广播，Shape 错误可能不一定立即报错。
2. AR128 的 New-Key 某些末两维可能同为 `128 × 128`，仅凭维度大小无法辨别 `seq` 和 `head_dim` 是否互换。

所以真实验收应先按 Tensor 语义和名称对齐 Layout，再显式断言 Shape，最后计算误差。

### 12.5 ORT 对拍没有验证 Encoding 的数值效果

ONNX Runtime 在这里运行的是转换后的 ONNX 浮点计算图，不会像后续 QAIRT Quantizer/HTP 那样真正执行 `<name>.encodings` 中描述的低比特量化。

因此：

```text
MHA2SHA ORT 对拍 OK
  → 说明图结构在这组随机输入上基本等价

不等于
  → Encoding 一定覆盖完整
  → qairt-converter 一定接受
  → Quantized DLC 一定数值正确
```

---

## 十三、为什么 `done`、返回码 0、文件存在仍不够

这是本阶段最重要的工程风险。

### 13.1 `verification_status=False` 仍可能返回 0

`converter.convert()` 会返回：

```python
(mha2sha_model, verification_status)
```

但 CLI 入口只是调用：

```python
converter.convert()
```

没有接收或检查 `verification_status`，随后执行 `sys.exit(0)`。

因此可能出现：

```text
Verification Status ----- FAIL -----
进程退出码仍然是 0
```

即使以后给外层补上 `proc.returncode` 检查，也仍需同时解析/确认验证状态。

### 13.2 外层也没有检查子进程返回码

`thread_g2g()` 当前：

```python
output, error = proc.communicate()
print(output.decode(), error.decode())
print("... done.")
```

它没有：

```python
if proc.returncode != 0:
    raise ...
```

随后 `executor.map()` 的结果又没有被消费，异常传播也不够可靠。最后仍会打印：

```text
All mha2sha convert done.
```

### 13.3 模型先保存，最后才对拍

执行顺序是：

```text
写 Mapping JSON
写 SHA Encoding
保存 SHA ONNX / External Data
最后才跑 SHA ORT 和 np.allclose
```

所以即使 Step 6 失败，文件也可能已经齐全。这解释了源码注释中“Shape mismatch 但不影响 generated files”的现象，但：

> **不影响文件落盘，不等于不影响模型正确性。Shape mismatch 必须定位，不能作为可长期忽略的正常日志。**

### 13.4 输出目录不会自动清空

代码只执行：

```python
os.makedirs(sha_folder, exist_ok=True)
```

新运行失败时，上一轮的旧 ONNX、Encoding 或 External Data 仍可能留在目录里。后续 Converter 甚至可能误用陈旧文件。

因此每轮应记录：

- 运行开始时间；
- 输入/输出文件的 mtime；
- 输入 ONNX、Encoding 和输出模型束的 Hash；
- 完整命令和工具版本。

---

## 十四、资源、版本和工具边界

### 14.1 内存需求很高

本阶段需要同时处理大 ONNX、权重、原图 ORT 输出和新图。项目历史 `example2/host_linux/logfile.log` 在 MHA2SHA 结束处记录过约：

```text
50.6 GB
50.8 GB
52.3 GB
```

这说明“CPU 工具”不等于“低资源工具”。当前 `go_parallel=False` 是合理的保守配置；若并行跑 AR1 和 AR128，内存需求可能大幅上升，应先实测单任务峰值。

### 14.2 内置 MHA2SHA 已标记 Deprecated

仓库内 `example2/G2G/MHA2SHA/README.md` 明确写着：

```text
This Project Is Now Deprecated
development moved into ONNX G2G
```

本项目仍然直接调用这份内置代码，因此应：

- 固定并记录本项目使用的提交版本；
- 不随意把另一套 ONNX G2G/MHA2SHA 包混入环境；
- 确认 `which` 和 `inspect.getfile()` 指向项目预期版本；
- 升级 ONNX、ORT、NumPy 或工具代码后重新完成全部对拍。

这不是说当前代码一定不能用，而是说明它的兼容性和问题修复不能假设仍由上游持续维护。

---

## 十五、怎样验收这一阶段

### 15.1 第一层：确认本轮身份

- [ ] 从 `example2/host_linux` 启动；
- [ ] `which mha2sha-onnx-converter` 指向项目内置 `bin/`；
- [ ] Python 导入的 `mha2sha` 指向项目内置 `src/python/`；
- [ ] 记录 AR、CL、Split、输入文件 Hash 和开始时间；
- [ ] 本轮产物 mtime 晚于开始时间，不是旧文件。

### 15.2 第二层：确认 Attention 转换覆盖

- [ ] AR1、AR128 分别运行完成；
- [ ] 每份 `1_of_1` 日志均为 `found_matched_pattern: 36`；
- [ ] 日志识别 `head_dim: 128`、Query Head 16、KV Head 2；
- [ ] 没有未解释的 RoPE Pattern、Past-KV 或 Shape 错误；
- [ ] 16Q→2KV 的 8:1 共享关系仍正确。

### 15.3 第三层：确认模型束闭包

- [ ] SHA `.onnx` 存在且非空；
- [ ] 枚举出的每个 `external_data.location` 都存在且非空；
- [ ] `onnx.load(..., load_external_data=True)` 成功；
- [ ] 不把目录里的未引用旧文件误认为当前模型组成部分。

可用下面的只读脚本枚举引用：

```bash
python - <<'PY'
from pathlib import Path
import onnx
from onnx import AttributeProto

model_path = Path("assets/artifacts/ar1-cl2048/1_of_1/sha_output/ar1-cl2048_1_of_1.onnx")
model = onnx.load(model_path.as_posix(), load_external_data=False)

def all_tensors(graph):
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                yield attr.t
            elif attr.type == AttributeProto.TENSORS:
                yield from attr.tensors
            elif attr.type == AttributeProto.GRAPH:
                yield from all_tensors(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    yield from all_tensors(subgraph)

locations = set()
for tensor in all_tensors(model.graph):
    for item in tensor.external_data:
        if item.key == "location":
            locations.add(item.value)

print("external file count:", len(locations))
for location in sorted(locations):
    path = model_path.parent / location
    print(path, "OK" if path.is_file() and path.stat().st_size > 0 else "MISSING")
PY
```

AR128 需要把路径替换成对应模型，不能只检查 AR1。

### 15.4 第四层：确认 Encoding

- [ ] `<name>.encodings` 和三份 Mapping JSON 均能解析；
- [ ] Encoding 版本符合当前 Mapper 支持范围；
- [ ] Activation Encoding 目标名存在于新图；
- [ ] Param Encoding 目标名对应真实 Initializer；
- [ ] Q 参数映射覆盖 16 个 Head，K/V 参数映射符合 2 个 KV Head；
- [ ] Per-channel Encoding 数量与实际通道切片一致；
- [ ] scale/offset 数值合法，无 NaN/Inf；
- [ ] 不只依赖工具 Warning 判断 Encoding 是否完整。

### 15.5 第五层：确认数值

- [ ] 每个 AR/Split 日志明确出现 `Verification Status ----- OK -----`；
- [ ] 不能只看 `done` 或进程返回码；
- [ ] 使用阶段二的合法 Mask、RoPE、Past-KV、`inputs_embeds` 再做一次真实输入对拍；
- [ ] 对拍前先断言名称、语义、Shape 和 Layout 完全对齐；
- [ ] 单独确认 AR128 New-Key 的 `seq/head_dim` 轴；
- [ ] 保存每个输出的最大/平均绝对误差，而不是只保存一个布尔值。

### 15.6 第六层：确认能交给下一阶段

- [ ] `qairt-converter` 真正返回成功；
- [ ] Converter 日志无未解释 Error/Warning；
- [ ] `--input_network` 指向本轮 SHA ONNX；
- [ ] `--quantization_overrides` 指向同一模型名的 SHA Encoding；
- [ ] 后续 Quantized DLC 仍要与可信 Golden 对拍。

---

## 十六、常见误解

### 16.1 “SHA 就是把 16 个头删成 1 个头”

不对。新图有 16 条单 Query Head 分支，最后仍拼回 16 个 Head。

### 16.2 “转换后 GQA 变成了普通 MHA”

不对。仍然是 16Q、2KV，每 8 个 Q 共享一组 K/V。

### 16.3 “每条 SHA 分支都有一套独立 K/V 权重”

不对。GQA Extension 只预构建 2 组真实 K/V，Query 分支按组引用。

### 16.4 “`--mha-conv` 会在这里把全模型 Linear 改成 Conv”

不对。它声明输入 Attention 已是 Conv，并选择 SHA-Conv 生成路径。

### 16.5 “`--nchw-aligned` 只是文件保存顺序”

不对。它声明 Q/K/V Projection 接口的真实 4D Layout，直接影响节点生成和 Transpose。

### 16.6 “原 Encoding 文件直接复制一下就可以”

不对。权重被切片、Tensor 被一对多展开和改名，Encoding 也必须同步映射。

### 16.7 “MHA2SHA 会重新 Calibration”

不对。它复用并重映射 Example1 已得到的量化规则，不重新采样统计分布。

### 16.8 “它会使用 Split 阶段的 RAW 做验证”

不对。当前内置验证自己生成随机输入，Split RAW 属于另一条验证/Quantizer 数据链。

### 16.9 “生成了 ONNX 和 Encoding 就说明验证通过”

不对。文件在数值对拍以前就已经保存，而且目录还可能残留旧文件。

### 16.10 “退出码 0 就说明 allclose 成功”

不对。CLI 忽略 `verification_status=False`，仍可能正常退出 0。

### 16.11 “MHA2SHA 已经生成了 DLC”

不对。本阶段只到 SHA ONNX；下一阶段才执行 `qairt-converter`。

---

## 十七、把整个阶段压缩成八步

```text
1. 读取 AR1/AR128 的 Split ONNX、External Data 和原 Encoding
2. 用 ORT 跑原图，生成内存中的随机输入 Golden
3. 找到完整图中的 36 层 Attention Pattern
4. 识别 16 个 Query Head、2 个 KV Head、head_dim=128
5. 切分 Q/K/V Conv，并按 8:1 的 GQA 关系创建逐 Query Head SHA 分支
6. 重建 RoPE、Past-KV、Mask、Softmax 和 NCHW Layout
7. 拼回 16 个 Head，同时映射并保存 SHA Encoding
8. 保存 SHA 模型束，再运行 SHA ORT，与原图逐输出 allclose
```

如果只记一句：

```text
16Q + 2KV 的 GQA 合并图
          ↓
16 条 Query 单头分支 + 2 组共享 KV
          ↓
RoPE / Past-KV / Encoding 一起迁移
          ↓
拼回原接口并做数值对拍
```

---

## 十八、自测题

1. SHA 为什么不是“只剩一个注意力头”？
2. 当前模型为什么是 16 条 Query 分支，却只有 2 组 K/V？
3. Q0～Q7 和 Q8～Q15 分别使用哪组 K/V？
4. `--mha-conv` 与 `--replace-linear-with-conv` 有什么区别？
5. `--nchw-aligned` 声明的 BD1L 分别是什么维度？
6. 为什么改 ONNX 的同时必须改 Encoding？
7. 三份 Mapping JSON 与最终 `<name>.encodings` 有什么区别？
8. 当前 MHA2SHA 验证为什么不使用 `qt_0.pkl`？
9. 为什么 `Verification Status OK` 仍不能证明 Quantized DLC 正确？
10. 为什么文件存在和退出码 0 都不能证明本阶段成功？
11. AR1 与 AR128 的 SHA 产物分别存在哪里？
12. 为什么检查 External Data 时不能只找一个 `.data` 文件？

---

## 十九、相关源码与笔记

- 第三阶段外层入口：[`qnn_compile_deploy.py`](../../../example2/host_linux/qnn_compile_deploy.py)
- CLI 可执行入口：[`mha2sha-onnx-converter`](../../../example2/G2G/MHA2SHA/bin/mha2sha-onnx-converter)
- 转换六步主流程：[`converter.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/converter.py)
- Attention 改图主逻辑：[`optimizer.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/optimizer.py)
- Conv/SHA 专用路径：[`mha_conv_extension.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/optimizer_extension/mha_conv_extension.py)
- GQA 分组与 KV 共享：[`gqa_extension.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/optimizer_extension/gqa_extension.py)
- RoPE 图改写：[`rope_extension.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/optimizer_extension/rope_extension.py)
- Past-KV 图改写：[`past_key_value_extension.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/optimizer_extension/past_key_value_extension.py)
- Encoding 映射：[`encoding_mapper.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/encoding_mapper.py)
- ONNX/External Data 保存：[`utils/onnx.py`](../../../example2/G2G/MHA2SHA/src/python/mha2sha/utils/onnx.py)
- 内置工具说明与 Deprecated 声明：[`MHA2SHA/README.md`](../../../example2/G2G/MHA2SHA/README.md)
- 历史资源日志：[`example2/host_linux/logfile.log`](../../../example2/host_linux/logfile.log)
- 当前模型结构参数：[`models/Qwen2.5-VL-3B-Instruct/config.json`](../../../models/Qwen2.5-VL-3B-Instruct/config.json)
- 上一阶段：[02 · Split ONNX 与测试向量](./02-Split-ONNX与测试向量.md)
- Attention 基础：[Attention 注意力机制](../02-附录A-Attention注意力机制.md)
- GQA 对照：[Attention 分类大全](../02-附录F-Attention分类大全(面试向).md)
- RoPE 原理：[RoPE 位置编码](../02-附录G-RoPE位置编码.md)
- KV Cache 原理：[KV Cache](../02-附录K-KV%20Cache(键值缓存).md)
- 产物保留与清理：[06 · example2 产物总览与清理](./06-example2产物总览与清理.md)

---

## 二十、本篇总结

> **MHA2SHA 阶段读取 AR1/AR128 的 Split ONNX 和原 AIMET Encoding，把每层 16Q、2KV、每 8 个 Q 共享一组 KV 的 GQA 合并图，等价改写为 16 条显式 Query 单头 SHA-Conv 分支；转换同时重建 RoPE、Past-KV 和 NCHW/BD1L Layout，将切片和改名后的 Tensor 同步映射到新的 SHA Encoding，最后拼回原 Attention 输出接口并用 ONNX Runtime 对拍。输出位于各自 `1_of_1/sha_output/`，下一阶段真正消费的是 SHA ONNX、其全部 External Data 和同名 `.encodings`。验收不能只看 `done`、返回码或文件存在，而必须确认 36 层匹配、GQA 共享关系、External Data/Encoding 完整，以及每个 AR 的明确 `Verification Status OK` 和真实输入数值对拍。**
