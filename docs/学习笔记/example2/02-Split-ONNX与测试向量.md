# 02 · Split ONNX 与测试向量

> **学习位置**：Example2 五阶段中的第二阶段。
>
> **上一阶段**：[01 · AR 图适配](./01-AR图适配-change_hardcoding.md)
>
> **流程总览**：[00 · Example2 主机编译全景](./00-example2主机编译全景.md)
>
> **上游 Test Vector**：[08 · ONNX 导出与测试向量](../08-ONNX导出与测试向量.md)
>
> **一句话本质**：分别把 AR1、AR128 ONNX 按 `num_splits` 组织成 `N_of_M` 模型束，同步处理各片的权重、Hidden/KV 接口和测试数据；当前 `num_splits=1`，因此不产生中间切点，只重建两份 `1_of_1` 整图并生成 QNN 测试资产。

本文按六个问题展开：为什么切、切在哪里、怎样切图、哪些对象要同步处理、Hidden 如何交接，以及第一个 Hidden 如何识别。

---

## 零、先看结论

### 0.1 一张图看懂本阶段

```text
第一阶段产物
├── AR1 ONNX 模型束 + qt_0.pkl
└── AR128 ONNX 模型束 + qt_0.pkl
              │
              ▼
       thread_split(arn)
              │
              ├── 按 num_splits 选择层末 Hidden 切点
              ├── 从目标输出反向遍历，重建 ONNX 子图
              ├── 同步筛选权重并重新保存 External Data
              ├── 建立相邻分片的 Hidden 输出/输入接口
              ├── 按层分配 Past-KV 输入与 New-KV 输出
              └── 按每片接口生成 RAW、Input List、候选 Golden
              │
              ▼
AR1：ar1-cl2048_N_of_M.onnx + .data + 测试资产
AR128：ar128-cl2048_N_of_M.onnx + .data + 测试资产
```

### 0.2 切分时必须同步处理什么

| 对象 | 必须同步做什么 |
|---|---|
| ONNX 计算节点 | 每片只保留其目标输出真正依赖的节点 |
| 权重 / Initializer | 每片只保留自身节点使用的完整权重 Tensor |
| External Data | 将各片需要外置的权重重新保存为配套 `.data` |
| Hidden 接口 | 上一片层末 Hidden 变成输出，下一片把同一 Tensor 变成输入 |
| KV 接口 | 每片保留自己层号对应的 Past-KV 输入和 New-KV 输出 |
| 公共输入 | 根据依赖保留 `attention_mask`、RoPE `cos/sin` 等输入 |
| Shape / dtype | 为新增的 Hidden 输入输出补充 `value_info` |
| Test Vector | 按各片真实接口生成 Input RAW、Input List 和候选 Golden RAW |

最容易混淆的一点是：

```text
Hidden：分片 1 → 分片 2 → 分片 3
        在同一次前向中横向传递

New KV：当前分片 → 外部 KV Cache → 下一次前向的同层 Past-KV 输入
        不会直接传给当前前向的下一分片
```

### 0.3 本阶段不做什么

- 不重新训练或微调；
- 不改变权重数值；
- 不改变 AR 和 Context Length；
- 不重新执行 AIMET Calibration；
- 不重新计算量化 Scale/Offset；
- 不执行 MHA2SHA；
- 不运行 Split ONNX 重新计算 Golden。

### 0.4 AR 与 Split 不要混淆

| 概念 | 切什么 | 本项目示例 |
|---|---|---|
| AR | Token / Current Length 方向 | AR1、AR128 |
| Split | Transformer 层深度方向 | `1_of_1`，以后也可以是 `1_of_6...6_of_6` |

AR1 和 AR128 是两张独立固定 Shape 模型。若 `num_splits=6`，它们会分别产生 6 片，不是两张模型合起来一共 6 片。

---

## 一、为什么需要分片

当 `num_splits>1` 时，分片通常有四个目的：

1. 降低单次 ONNX 转换、QNN 编译和加载的资源压力；
2. 允许运行时按顺序组织多个较小模型段；
3. 逐片定位转换误差或量化误差；
4. 为分段 DLC、Context Binary 和权重共享流程提供组织结构。

分片不会减少：

```text
模型总层数
模型总权重
完整推理的理论计算量
```

它还会增加段间 Buffer 传递、模型调度、测试数据准备和接口管理成本。因此 `num_splits` 是工程取舍，不是越大越好。

### 1.1 当前项目为什么仍执行 Split 阶段

当前配置为：

```python
num_splits = 1
```

因此没有真正拆开 36 层，但仍会：

- 从原始输出反向遍历并重建完整 ONNX；
- 筛选实际使用的 Initializer 和 `value_info`；
- 重新封装 External Data；
- 统一生成 `1_of_1` 名称；
- 从 `qt_0.pkl` 生成 RAW、Input List 和候选 Golden。

所以当前阶段更准确的名字是：

> **ONNX 标准化重建与 QNN 测试资产准备阶段。**

---

## 二、切分单位：完整 Transformer 层

### 2.1 每层的基本结构

本项目的 36 个 LLM Decoder Layer 都遵循同一主结构：

```text
输入 Hidden
   │
   ├──────────── residual ①
   ▼
RMSNorm → Self-Attention → Add
                              │
                              ├──────── residual ②
                              ▼
                         RMSNorm → MLP → Add
                                           │
                                           ▼
                                    下一层 Hidden
```

写成公式：

```text
x' = x  + Attention(Norm(x))
y  = x' + MLP(Norm(x'))
```

每层有两个主要残差 Add：

1. Attention 后的 Add；
2. MLP 后的 Add。

第二个 Add 的输出是一整层完成后的 Hidden，通常是最干净的层间切点。

### 2.2 程序如何寻找疑似层尾 Hidden

入口是 `get_split_tensors()`：

```text
收集 ONNX 中所有 Add / Cast 节点
        ↓
根据 Tensor 生产者—消费者关系寻找最长 Add/Cast 链
        ↓
过滤掉 Cast，只保留 Add 输出
        ↓
每两个 Add 取第二个
        ↓
得到每层结束后的 Hidden 候选
```

假设识别正确，结果可表示为：

```python
output_tensor_list = [
    H0,   # Layer 0 结束后的 Hidden
    H1,   # Layer 1 结束后的 Hidden
    ...
    H35,  # Layer 35 结束后的 Hidden
]
```

这里不是根据节点名称中的 `layer.0` 判断，而是依赖残差主干拓扑。

> **重要边界**：这是启发式识别，不是真正的 Transformer 语义解析。代码中虽然定义了残差 Add 检查函数，但当前筛选逻辑没有启用它。若图优化融合、重写或插入 Add/Cast，候选层数可能识别错误。

### 2.3 “第一个 Hidden”有三种含义

| 说法 | 当前模型中的含义 |
|---|---|
| Layer 0 输入 Hidden | `inputs_embeds`，因为 Embedding 已在 ONNX 外部 |
| 第一个层末 Hidden 候选 | Layer 0 第二个残差 Add 的输出 `H0` |
| 36 层切 6 片时的第一个实际切点 | Layer 5 结束后的 `H5` |

当前 `num_splits=1` 时虽然会识别候选，但不会选择任何实际 Hidden 切点。

---

## 三、`num_splits` 如何决定切点

### 3.1 基本规则

```text
M 个分片需要 M-1 个中间切点
```

例如：

```text
1 片 → 0 个切点
2 片 → 1 个切点
6 片 → 5 个切点
```

### 3.2 36 层切 6 片

```python
num_layers = 36
num_splits = 6
num_layers_per_split = 36 // 6  # 6
```

循环选择：

```text
layer_end = 6、12、18、24、30
```

实际切点为：

| 切点 | 含义 |
|---|---|
| `H5` | Layer 0～5 完成，准备进入 Layer 6 |
| `H11` | Layer 6～11 完成，准备进入 Layer 12 |
| `H17` | Layer 12～17 完成，准备进入 Layer 18 |
| `H23` | Layer 18～23 完成，准备进入 Layer 24 |
| `H29` | Layer 24～29 完成，准备进入 Layer 30 |

每个 `names_to_split` 条目不是只有 Hidden，还包括本片各层的 New KV 输出：

```text
H5  + New KV 0～5
H11 + New KV 6～11
H17 + New KV 12～17
H23 + New KV 18～23
H29 + New KV 24～29
```

这些 KV 被放进当前片的正式输出，但不会成为下一片输入。

### 3.3 当前 `num_splits=1`

```python
num_layers_per_split = 36 // 1  # 36
range(36, 36, 36)              # 空
names_to_split = []
```

断言成立：

```text
1 == len([]) + 1
```

随后 `OnnxSplitter.split([])` 直接以全部原始模型输出为目标，重建一个完整 `1_of_1` 图。

### 3.4 分片数量的实现约束

当前算法使用整数除法和 `range()` 选点，因此：

- `num_splits` 不应大于识别出的层数；
- 层数最好能被 `num_splits` 整除；
- 候选识别错误会使分片数量断言失败；
- 当前输入已经是 `inputs_embeds`，所以 `split_embedding` 会被设为 `False`，Embedding 不单独占一片。

---

## 四、真正的切分方法：反向依赖遍历

Split 不是按文件字节切，也不是把节点数组按序号截成几段，而是：

> **以分片目标输出为起点，沿 Tensor 的生产者关系向前反向遍历，构造该输出的完整计算依赖闭包。**

### 4.1 一个最小例子

原图：

```text
输入 → A → B → Hidden_cut → C → D → logits
```

分片 1 的目标输出是 `Hidden_cut`：

```text
Hidden_cut ← B ← A ← 输入
```

得到：

```text
分片 1：输入 → A → B → Hidden_cut
```

分片 2 的目标输出是 `logits`，并把 `Hidden_cut` 当作新的叶子输入：

```text
logits ← D ← C ← Hidden_cut（到这里停止）
```

得到：

```text
分片 2：Hidden_cut → C → D → logits
```

### 4.2 代码中的五个步骤

1. 建立 `Tensor 名 → 生产它的 Node` 映射；
2. 把原始图输入、Initializer、上一片 Hidden 放入 `leaf_tensors`；
3. 从指定输出开始，用队列反向查找生产者；
4. 只保留访问到的 Node、Initializer、Sparse Initializer 和 `value_info`；
5. 用筛选结果重新创建 `GraphProto` 和 `ModelProto`。

调用链：

```text
split_onnx()
  ├── get_split_tensors()       # 找候选层末 Hidden
  ├── 生成 names_to_split       # 选择 M-1 个切点
  └── split_onnx_by_names()
      └── OnnxSplitter.split()
          └── partition_subgraph()  # 反向依赖构图
```

### 4.3 多片如何避免重复包含前面的层

生成前一片后，代码把“不是原图最终输出”的切口加入 `additional_input_tensors`。

因此：

```text
H5  成为第 2 片输入
H11 成为第 3 片输入
H17 成为第 4 片输入
...
```

下一片反向遍历到该 Hidden 时停止，不再继续追入前面的层。

最后一片不需要新的中间切点，它以尚未被前面分片覆盖的原始输出为目标，一般包括 `logits` 和最后若干层的 New KV。

---

## 五、各分片的接口如何组织

定义公共输入：

```text
G = attention_mask + position_ids_cos + position_ids_sin
```

36 层切 6 片时，预期接口为：

| 分片 | 负责计算 | 主要输入 | 主要输出 |
|---|---|---|---|
| 1 | Layer 0～5 | `inputs_embeds` + G + Past KV 0～5 | `H5` + New KV 0～5 |
| 2 | Layer 6～11 | `H5` + G + Past KV 6～11 | `H11` + New KV 6～11 |
| 3 | Layer 12～17 | `H11` + G + Past KV 12～17 | `H17` + New KV 12～17 |
| 4 | Layer 18～23 | `H17` + G + Past KV 18～23 | `H23` + New KV 18～23 |
| 5 | Layer 24～29 | `H23` + G + Past KV 24～29 | `H29` + New KV 24～29 |
| 6 | Layer 30～35 + Final Norm + LM Head | `H29` + G + Past KV 30～35 | `logits` + New KV 30～35 |

### 5.1 Hidden 交接接口如何建立

以 `H5` 为例：

1. 给 `H5` 补充 `FLOAT [batch, AR, hidden_size]` 的 `value_info`；
2. 将 `H5` 写入第 1 片 `graph.output`；
3. 将同名、同 Shape、同 dtype 的 `H5` 写入第 2 片 `graph.input`；
4. 实际运行时由外部调度器把第 1 片输出 Buffer 绑定给第 2 片输入。

Split 脚本只建立接口，不负责设备端顺序调度。

### 5.2 KV 为什么不横向传给下一片

Layer 5 的 New KV 属于 Layer 5，Layer 6 当前前向需要的是自己的 `past_key_6_in/past_value_6_in`。

所以：

```text
当前同一次前向：
H5 → Layer 6

下一次 Token 前向：
Layer 5 New KV → 外部 Cache → Layer 5 Past KV 输入
```

当前配置 `return_new_key_value_only=true`，虽然输出名称仍是 `past_key_*_out/past_value_*_out`，其语义是本轮新产生的 KV。

### 5.3 权重怎样随分片处理

“权重切分”不是把一个矩阵按字节或维度切开，而是按层归属分配：

```text
分片 1保留Layer 0～5使用的完整权重Tensor
分片 2保留Layer 6～11使用的完整权重Tensor
...
```

`partition_subgraph()` 会根据已访问节点筛选真正使用的 Initializer。

---

## 六、External Data 如何处理

大模型 ONNX 通常由两部分组成：

```text
模型.onnx   # 图结构与外置参数引用
模型.data   # 实际大权重数据
```

达到外置阈值的大 Tensor 会进入 `.data`，小 Initializer 仍可能保存在 `.onnx` 内；因此 `.data` 不一定包含全部参数。只要模型引用了 External Data，`.onnx` 和被引用的 `.data` 就是不可拆分的模型束。

切图时：

1. 最初用 `load_external_data=False` 读取图结构，避免立即加载全部大权重；
2. 子图确定后，只保留其使用的 Initializer；
3. 保存前从源目录加载这些外置权重；
4. 为每个子图重新保存对应 `.onnx + .data`。

因此分片后的权重数值不变，但物理文件按各子图重新封装。

> 仅看到 `.onnx` 文件存在，不能证明模型完整。必须确认其 `external_data.location` 指向的文件存在、非空且能实际加载。

---

## 七、Test Vector 如何同步处理

### 7.1 不是把 PKL 文件机械切几块

程序先取得每个子图的真实输入输出名称，再从 `qt_0.pkl` 中按 Tensor 名称挑出对应数据：

```text
qt_0.pkl
   │
   ├── 子图输入 Tensor  → Input RAW + Input List
   └── 子图输出 Tensor  → 候选 Golden RAW
```

多分片时，后一片的层末 Hidden 输入也必须能在 `qt_0.pkl` 中找到。Example1 当前配置记录了层末 `Add_1` 激活，但名称映射仍需验收。

### 7.2 当前只读取 `qt_0.pkl`

当前实现：

- 只遍历 `source=['qt']`；
- 文件发现逻辑使用 `range(1)`；
- 因此只读取 `qt_0.pkl`；
- 不读取 `fp_0.pkl`，也不会自动读取 `qt_1.pkl`。

### 7.3 Input RAW

对每个输入 Tensor：

- 主 `.raw` 始终转换为 FP32 裸字节；
- 若原 dtype 不是 FP32，还会额外保存 `<tensor>.<dtype>.raw`；
- Input List 默认引用不带 dtype 后缀的 FP32 主 RAW。

RAW 本身不保存名称、Shape、dtype 或维度顺序，这些必须由模型接口和 Input List 共同解释。

### 7.4 Input List

每一项是：

```text
Tensor名称:=RAW文件路径
```

它保存绑定关系，不保存 Tensor 数据。当前通常写入绝对路径，因此移动整个目录后可能失效。

### 7.5 候选 Golden

Golden RAW 直接来自 `qt_0.pkl` 中已有的输出或中间 Tensor：

```text
qt_0.pkl 中的值 → _dump() → Golden RAW
```

Split 阶段没有运行 Split ONNX 重新计算结果。

而且第一阶段对 Test Vector 做的是机械 Shape Resize，不是针对 AR1/AR128 重新生成 Mask、RoPE、KV 和前向结果。因此这里统一称为：

> **候选 Golden / PKL 参考输出，而不是已经证明可信的绝对真值。**

### 7.6 缺少 Tensor 时的风险

如果所需 Tensor 在 PKL 中找不到，代码会打印：

```text
Adding dummy test vector
```

并尝试根据原图 `value_info` 生成 dummy。但这不是无条件兜底：缺失名称必须能在 `value_info` 中找到，而且当前实现要求 dtype 为 FLOAT，否则会直接断言失败。即使 dummy 生成成功，量化输入和 Golden 的数值语义也已经不可信，不能忽略该日志。

另外，Pickle 可以执行反序列化代码，只能加载可信来源文件。

---

## 八、当前项目的真实执行结果

### 8.1 AR1、AR128 分别执行

主脚本分别调用：

```text
thread_split(1)
thread_split(128)
```

每个任务：

1. 创建 `assets/artifacts/ar{arn}-cl2048/`；
2. 创建 `src` 符号链接，指向对应 AR 导出目录；
3. 调用 `utils.split_onnx(..., num_splits=1, split_embedding=False)`；
4. 重建 `1_of_1` ONNX 模型束；
5. 生成 Input RAW、Input List 和候选 Golden。

### 8.2 当前 `1_of_1` 的接口

因为没有中间切点：

- 输入仍是原模型全部输入：`inputs_embeds`、Mask、RoPE 和 36 层 Past KV；
- 输出仍是 `logits` 和 36 层 New KV；
- 不会出现 `H5/H11...` 等新增 Hidden 输入输出。

### 8.3 产物树

AR1 大致产生：

```text
assets/artifacts/ar1-cl2048/
├── src -> assets/models_ar_n/ar1-cl2048
├── split_onnx/
│   ├── ar1-cl2048_1_of_1.onnx
│   └── ar1-cl2048_1_of_1.data
├── input_list_ar1-cl2048_1_of_1.txt
├── test_inputs_ar1-cl2048_1_of_1/
│   └── 0/*.raw
└── test_golden_outputs_ar1-cl2048_1_of_1/
    └── Result_0/*.raw
```

AR128 目录结构相同，只把名称换成 `ar128-cl2048`。

### 8.4 下游消费者

| 产物 | 下游用途 |
|---|---|
| Split ONNX + `.data` | 交给 MHA2SHA |
| Input RAW + Input List | 交给 Quantizer，也可用于 Host/QNN 运行测试 |
| 候选 Golden RAW | 留给数值对拍，不被当前 Quantizer 命令直接读取 |
| 原 `.encodings` | 第三阶段继续读取；当前 `split_embedding=False` 不生成新的 Split Encoding |

---

## 九、主要风险与验收

### 9.1 风险与检查动作

| 风险 | 必须检查什么 |
|---|---|
| 层边界识别只是启发式 | 候选数是否为 36，切点是否落在各层第二个残差 Add 后 |
| 边界 Shape/dtype 被假设为 `FLOAT [B,AR,H]` | 子图新增 Hidden 接口是否与真实 Tensor 一致 |
| 分片数量不合理 | `num_splits` 是否不大于层数并能合理整分 |
| 分片可能重叠或重复计算 | 每片节点和权重范围是否符合预期 |
| External Data 缺失 | `.onnx` 引用的所有 `.data` 是否存在并可加载 |
| Test Vector 名称未匹配 | 日志中不能出现未经处理的 `Adding dummy test vector` |
| Golden 数值不可信 | 使用真实 AR 输入重新前向生成参考结果并对拍 |
| Worker 失败被完成日志掩盖 | 不能只看 `All onnx model splitted.`，要检查返回码、错误日志和新文件时间 |
| 旧文件混入本次结果 | 运行前后核对时间戳、数量和输出目录 |
| 路径不可移植 | 检查 `src` 符号链接和 Input List 中的绝对路径 |

当前代码还有两项扩展限制：

- 文件发现只读取一个 `qt_0.pkl`；
- 多 Batch 写 Input List 时没有显式换行，扩展样本数前需要修正并验证。

### 9.2 最小验收清单

- [ ] AR1、AR128 分别生成了预期数量的 `N_of_M.onnx`；
- [ ] 每份 `.onnx` 都有可解析的 External Data；
- [ ] 分片输入输出名称、Shape、dtype 与设计一致；
- [ ] 多分片时只有 Hidden 横向交接，KV 按层归属正确；
- [ ] 每份 Input List 中的 RAW 路径存在且文件大小合理；
- [ ] 日志没有未解释的 dummy、Traceback 或保存失败；
- [ ] Golden 来源已记录为 PKL 参考或重新前向真值；
- [ ] 至少完成一次相邻分片串联和端到端数值对拍。

---

## 十、把完整逻辑压缩成七步

```text
1. 对 AR1、AR128 分别处理
2. 根据 num_splits 确定分片数量
3. 从残差 Add 主链寻找疑似层尾 Hidden
4. 选择 M-1 个切点，从目标输出反向遍历构造子图
5. 同步筛选权重、External Data、Hidden/KV/公共输入接口
6. 按各片输入输出从 qt_0.pkl 生成 RAW、Input List、候选 Golden
7. 分别验收模型束、接口、路径和数值
```

如果是 36 层切 6 片，记成：

```text
片间只传 Hidden：H5 → H11 → H17 → H23 → H29
各片 New KV：输出给外部 Cache，下一轮回到对应层
```

如果是当前 `num_splits=1`，记成：

```text
没有中间 Hidden 切口
只有完整 1_of_1 ONNX 重建 + External Data 重封装 + 测试资产生成
```

---

## 十一、相关源码与笔记

- Example2 第二阶段入口：[`qnn_compile_deploy.py`](../../../example2/host_linux/qnn_compile_deploy.py)
- Split 编排、边界选择与测试资产：[`split_onnx_utils/utils.py`](../../../example2/G2G/split_onnx_utils/utils.py)
- 反向依赖构图与 External Data 保存：[`split_onnx_utils/split_onnx.py`](../../../example2/G2G/split_onnx_utils/split_onnx.py)
- 上游 Test Vector 生成：[`example1/llm_utils/test_vectors.py`](../../../example1/llm_utils/test_vectors.py)
- 第一阶段 AR Shape 适配：[01 · AR 图适配](./01-AR图适配-change_hardcoding.md)
- 下一阶段：[03 · MHA2SHA 图结构转换](./03-MHA2SHA图结构转换.md)
- Example1 ONNX 与 Test Vector：[08 · ONNX 导出与测试向量](../08-ONNX导出与测试向量.md)
- Example2 产物保留与清理：[06 · example2 产物总览与清理](./06-example2产物总览与清理.md)

---

## 十二、本篇总结

> **Split 阶段先从残差 Add 主链寻找疑似层尾 Hidden，再按 `num_splits` 选择 `M-1` 个切点，并以目标输出为起点反向遍历计算依赖，重建每个 ONNX 子图。切图时必须同步筛选各片节点、完整权重和 External Data，建立相邻分片的 Hidden 输出/输入接口，并按层保留 Past-KV 输入与 New-KV 输出；只有 Hidden 在同一次前向中横向传片，New KV 交给外部 Cache 并在下一轮回到同一层。随后程序按每片接口从 `qt_0.pkl` 生成 FP32 Input RAW、Input List 和候选 Golden。当前 `num_splits=1`，因此 AR1、AR128 都没有中间切点，只重建各自的 `1_of_1.onnx + .data` 并准备测试资产。**
