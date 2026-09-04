# 04 · ONNX 到 DLC：qairt-converter

> **学习位置**：Example2 五阶段中的第四阶段，概念学习已完成（2026-08-28）；不代表实际转换验收通过。
>
> **上一阶段**：[03 · MHA2SHA 图结构转换](./03-MHA2SHA图结构转换.md)
>
> **流程总览**：[00 · Example2 主机编译全景](./00-example2主机编译全景.md)
>
> **下一阶段**：[05 · 量化 DLC：qairt-quantizer](./05-量化DLC-qairt-quantizer.md)。
>
> **一句话本质**：读取 SHA ONNX、它引用的权重数据和配套 SHA Encoding，补充输入布局声明，转换成 Qualcomm 工具链使用的普通／中间 DLC，交给后续 Quantizer。
>
> **核对日期**：2026-08-26。本文依据当前仓库代码和官方部署文档整理；本次没有执行 SDK 转换，也没有验收实际 DLC。

---

## 一、介绍：这一阶段是什么、为什么需要

### 1.1 先区分四种产物

| 产物 | 在当前项目中的作用 | 谁接着使用 |
|---|---|---|
| SHA ONNX + External Data + SHA Encoding | 已完成注意力图改写的模型束，以及对应量化规则 | `qairt-converter` |
| 普通／中间 DLC | 转换到 Qualcomm 图表示，尚未走完本项目后续量化阶段 | `qairt-quantizer` |
| Quantized DLC | 完成后续 Quantizer 处理的模型产物 | Context Binary 生成阶段 |
| HTP Context Binary | 面向目标 HTP 后端准备的序列化上下文 | 本项目设备端运行链路 |

ONNX 与 DLC 都可以描述模型，但属于不同的工具链表示。Converter 的工作涉及图、算子、张量与参数的转换，不能通过把 `.onnx` 后缀改成 `.dlc` 来完成。

官方 AIMET 部署指南将 Conversion、Quantization、Compilation 分开说明：先转换到 DLC，再使用 Quantizer，随后生成 HTP Context Binary。本项目采用这条分阶段路径。[官方流程说明](https://qualcomm.github.io/aimet-pages/releases/2.13.0/tutorials/on_target_inference.html#qualcomm-ai-engine-direct-sdk)

### 1.2 与上一阶段的区别

MHA2SHA 主要解决「Attention 在 ONNX 图中怎样表达」；本阶段解决「怎样把这张图交接给 Qualcomm 的后续工具」。

本阶段不会再次负责把 AR1073 改成 AR1／AR128，也不负责重新拆分模型或重新展开 Attention Head。相关工作已经由前面三个阶段承担。

### 1.3 本节先掌握三个问题

1. Converter 的三个核心参数分别提供什么？
2. 为什么需要 SHA Encoding，而不能直接使用改图前的 Encoding？
3. 为什么得到普通 DLC 后，当前项目还会调用 Quantizer？

---

## 二、原理：图、参数与量化规则如何交接

### 2.1 三类输入各自负责什么

| 输入 | 内容 | 注意事项 |
|---|---|---|
| SHA `.onnx` | 算子连接、张量名称、Shape、静态参数或外部数据引用 | 必须是本 AR／Split 的改写后图 |
| ONNX 引用的 External Data | 存在外部文件中的静态权重等数据 | 按 ONNX 的引用收齐，不能只复制 `.onnx` |
| 同名 SHA `.encodings` | 与新图张量／参数对应的量化规则 | 规则可能包括位宽、scale、offset 等；不能与旧图混配 |

这些 External Data 与推理用的输入 RAW 是两回事：前者属于模型参数，后者属于样本输入。

### 2.2 为什么还需要 Encoding

本项目将模型图与量化规则作为两个输入交接：

- 模型图描述要执行的运算及其参数。
- Encoding 描述相关张量／参数的量化设置。

例如一个权重原本对应一条 Encoding，MHA2SHA 把权重按 Head 切片并改变相关名称后，需要使用同步映射后的规则。第四阶段源码正是从 `sha_output/` 同时读取 ONNX 和 Encoding。

`--quantization_overrides` 是把这份量化规则交给 Converter 的入口。它不是校准数据集，也不是让 Converter 重新执行 AIMET 的 SeqMSE 或 `compute_encodings()`。

### 2.3 「普通 DLC」不等于「完全没有量化信息」

当前 Converter 已经接收 Encoding，因此不能把中间 DLC 理解为一份丢弃了量化规则的文件。

但它也不能代替本项目后面的 Quantizer 阶段。下一步仍会读取它，生成另一个文件：`<name>_quantized.dlc`。

源码把中间文件变量命名为 `float_dlc_file`，这不构成「文件内部所有数据都一定是 FP32」的证据。实际参数存储类型、规则是否完整带入，应通过所用 SDK 的工具和真实 DLC 检查；这里先用**Quantizer 之前的中间 DLC**来定位它。

### 2.4 当前 Converter 是否用 RAW 做标定

从本项目的调用可以明确看到：

- `thread_convert()` 没有读取 RAW。
- Converter 命令没有传 `--input_list`。
- 没有调用数据集前向、SeqMSE 或 `compute_encodings()`。

因此，当前调用没有把 RAW 样本交给 Converter 做数据驱动的标定。至于 SDK 内部如何处理图和量化元数据，不能仅凭 Python 外层调用推断全部实现细节。

---

## 三、官方 Qwen2／Qwen2.5 的做法

这一阶段不是 Qwen 注意力结构本身的一部分。

在本仓库中，原始模型从 Hugging Face 模型目录加载，经过 Example1 的适配、量化与导出，再进入 Example2。DLC 是本项目部署到 Qualcomm 工具链时引入的表示；无需把它理解为 Qwen 模型定义必须包含的文件。

因此，本节的「原版做法 → 项目改造」可以概括为：从原始模型及导出图，接入目标设备所需的格式转换流程，保持模型计算含义并交接配套量化规则。

---

## 四、本项目的做法：顺着 thread_convert() 阅读

主入口：[qnn_compile_deploy.py](../../../example2/host_linux/qnn_compile_deploy.py)，`thread_convert()` 从第 190 行开始。

### 4.1 当前会转换几份模型

当前配置是 `ARNs = [1, 128]`、`num_splits = 1`，对应两个任务：

| AR | 模型名 |
|---|---|
| AR1 | `ar1-cl2048_1_of_1` |
| AR128 | `ar128-cl2048_1_of_1` |

第 234 行进程池在 `go_parallel=False` 时只有一个 Worker。这是主机上的工具调用，不是设备端推理。

### 4.2 三个核心参数

下面摘出当前源码第 206～222 行的核心逻辑，省略目录准备和注释；用于阅读，不要作为完整脚本直接运行：

```python
input_onnx = f"{split_work_dir}/sha_output/{name}.onnx"
quantization_overrides = f"{split_work_dir}/sha_output/{name}.encodings"

args = [
    QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qairt-converter",
    "--input_network", input_onnx,
    "--quantization_overrides", quantization_overrides,
    "-o", f"{out_dir}/{name}.dlc",
]

options = utils.get_input_layout(input_onnx, using_qairt_workflow=True)
for entry in options:
    args += entry
```

| 参数 | 本项目传什么 | 作用 |
|---|---|---|
| `--input_network` | `sha_output/<name>.onnx` | 指定要转换的图；引用的权重数据也必须可读 |
| `--quantization_overrides` | `sha_output/<name>.encodings` | 提供与 SHA 图配套的量化规则 |
| `-o` | `converted_model/<name>.dlc` | 指定中间 DLC 输出路径 |

代码的 `out_dir` 在第 196 行设为 `converted_model/`。官方文档使用输出参数的长写法 `--output_path`；本项目命令使用 `-o`，实际可用选项以所安装版本的 `qairt-converter --help` 为准。

### 4.3 输入布局参数是怎么来的

辅助函数位于 [utils.py](../../../example2/G2G/split_onnx_utils/utils.py) 第 725 行。

当前 QAIRT 分支的关键逻辑是：

```python
layout = os.getenv("INPUT_LAYOUT", "NONTRIVIAL")
onnxmodel = _load_model(onnxfile, load_external_data=False)

input_info = [
    ("--source_model_input_layout", i.name, layout)
    for i in onnxmodel.graph.input
]
```

这里省略了白名单校验，以及当前为空的排除名单。准确行为如下：

| 问题 | 当前代码的答案 |
|---|---|
| 默认 Layout 是什么？ | 未设置环境变量 `INPUT_LAYOUT` 时为 `NONTRIVIAL` |
| 是否按二维／三维／四维 Shape 自动选择布局？ | 不会；所有输入使用同一个 Layout 值 |
| Mask、Position、KV 会被单独处理吗？ | 当前排除名单为空，没有按输入名分别判断 |
| 实际追加哪个参数？ | 每个输入一组 `--source_model_input_layout <输入名> <layout>` |
| 是否追加 `--input_dim`？ | 当前 `using_qairt_workflow=True` 分支不追加 |
| 是否重新排列 Tensor 数据？ | 这个 helper 只读图并生成参数，没有重排数据 |

`NONTRIVIAL` 在这里是默认布局声明，不是动态 Shape 开关，也不是自动推断布局的算法。不要凭它推断 SDK 内部一定不调整任何布局。

上一阶段的 `--nchw-aligned` 描述 MHA2SHA 如何处理其投影布局；本阶段的参数描述交给 Converter 的模型输入。这两个工具选项作用不同，不能因为上一阶段采用 NCHW，就盲目把这里所有输入统一改成 NCHW。

此外，helper 读取 ONNX 时使用 `load_external_data=False`。能够列出输入名称，并不能证明外置权重已经完整、可读。

### 4.4 为什么代码还创建 Input List 和 RAW 目录的软链接

第 200～205 行在 Split 工作目录下准备两个链接：

- `input_list_<name>.txt`；
- `test_inputs_<name>`。

链接指向 AR 产物目录中的对应资源。它们用于准备工作目录中的数据路径；创建链接不等于 Converter 已消费这些数据，更不等于已经执行标定。

后续 `thread_genlib()` 会进入 Split 目录，并把 Input List 传给 Quantizer。当前 Split 调用向 `utils.py` 第 927～929 行传入绝对 `output_dir`，因此生成的列表包含绝对 RAW 路径。不能据此断言 Quantizer 必须依靠这些软链接定位 RAW；实际依赖应检查生成的 Input List，移动产物目录后尤其需要核对路径。

### 4.5 本阶段输出到哪里

路径均相对 `example2/host_linux/`：

```text
assets/artifacts/ar1-cl2048/1_of_1/converted_model/ar1-cl2048_1_of_1.dlc
assets/artifacts/ar128-cl2048/1_of_1/converted_model/ar128-cl2048_1_of_1.dlc
```

下一阶段读取这些文件，并将量化 DLC 写入各自的 `compiled_model/`。目录名不是验收结果，也不表示已经生成 HTP Context Binary。

### 4.6 运行与验收边界

本节是阅读笔记；以下是以后实际执行时需要检查的事项，不表示本次已经验证通过：

- [ ] 使用配置匹配的 Linux x86 QAIRT 环境，不能在当前 Windows PowerShell 中直接运行 Linux SDK 工具。
- [ ] 从 `example2/host_linux` 启动，因为主脚本第 13 行使用 `os.getcwd()`。
- [ ] 先检查脚本语法；2026-08-26 阅读时第 178 行仍有残缺文本，本次没有修复代码。
- [ ] ONNX、全部 External Data、Encoding 属于同一个 AR／Split 和同一次转换。
- [ ] 布局声明与输入接口相符；不要把所有模型输入都当成图像输入。
- [ ] 检查本轮 Converter 的真实退出码及完整错误信息。
- [ ] 检查新生成 DLC 的路径、非空状态和时间，排除旧文件造成的误判。
- [ ] 使用所安装 SDK 提供的 DLC 检查工具检查输入输出接口和量化信息；参数以该版本帮助为准。
- [ ] 后续继续完成数值验证，不能用格式转换成功代替精度验收。

当前源码有三个值得记住的风险：

1. `Popen().communicate()` 后没有检查 `proc.returncode`，所以子进程失败后仍可能打印 `done`。
2. `executor.map()` 的结果迭代器没有被消费，Worker 异常可能没有被显式传回主流程；这不等于任务没有执行。
3. 代码只会先删除同名软链接；同名位置若是普通文件／目录，创建链接可能失败；目标缺失也可能形成悬空链接。

Context Binary 的示例调用仍在三引号字符串中，自动生成部分也未启用。当前学习完这一节，不代表设备部署已经完成。

---

## 五、自测：能说清这五句就可以往下学

1. **为什么 Converter 输入选 `sha_output/`？** 因为它必须接收上一阶段改写完成的图。
2. **为什么 Encoding 也从同一目录读取？** 因为图中的名称和权重切片已经变化，规则必须与新图对应。
3. **三个核心参数是什么？** `--input_network`、`--quantization_overrides`、`-o`。
4. **当前 Converter 是否用 RAW 标定？** 当前命令没有传 RAW 或 Input List；下一阶段才显式使用 Input List。
5. **下一步是什么？** `qairt-quantizer` 读取中间 DLC 与 Input List，生成本项目的 Quantized DLC。

学会这些后，再进入「为什么有了 AIMET Encoding，仍需要 qairt-quantizer」；暂时不必同时展开 Context Binary 和设备运行。

## 六、本篇总结

**本阶段把 SHA 模型束和配套量化规则交接为 Qualcomm 中间 DLC。核心是图与 Encoding 配对、输入布局声明正确、输出验收可靠；RAW 数据驱动处理和最终 Quantized DLC 属于下一阶段。**
