# 07 · Context Binary 编译与 HTP 后端（Example2 → Example3 衔接）

> **学习位置**：Example2 当前五阶段之后、Example3 设备部署之前的衔接内容。核心作用已了解（2026-08-31），具体配置与生成验收留到实操前再学。
>
> **章节边界**：Context Binary 的生成示例在 `example2/host_linux/qnn_compile_deploy.py`，但尚未启用；[Example3 README](../../../example3/README.md) 负责准备输入、部署已有模型 Binary，并用 Genie 在设备上运行。本篇不是 Example2 当前启用流程中已经执行的第六阶段。
>
> **上一节**：[05 · 量化 DLC：qairt-quantizer](./05-量化DLC-qairt-quantizer.md)。[06 · 产物总览与清理](./06-example2产物总览与清理.md) 是配套查阅篇，笔记编号不等于执行步骤编号。
>
> **本次重点**：为什么已经有量化 DLC，还要生成 Context Binary？
>
> **一句话本质**：将量化模型交给目标后端准备，并把准备后的执行上下文保存成文件，供后续运行时加载。
>
> **核对日期**：2026-08-28。只核对代码与公开文档；没有执行 SDK，没有生成或验收 Context Binary。

## 一、介绍：这一节与上一节有什么区别

| 阶段 | 主要工作 | 输出 |
|---|---|---|
| Quantizer | 将模型按量化设计处理为部署用量化表示 | Quantized DLC |
| Context 生成 | 通过指定后端准备模型，保存其序列化上下文 | Context Binary |
| 测试推理 | 加载已准备的模型，对本次输入执行计算 | logits、New K/V 等实际结果 |

量化 DLC 包含模型图、静态参数和量化信息；它与已经为目标后端生成的 Context Binary 属于不同阶段。AIMET 官方流程将 Quantization、Compilation、Execution 分开，并使用 `qnn-context-binary-generator` 为 HTP 生成上下文。[官方部署说明](https://qualcomm.github.io/aimet-pages/releases/2.13.0/tutorials/on_target_inference.html#compilation)

## 二、原理：先把输入与输出记清

本项目示例中的主要输入是：

- 量化 DLC：要准备的模型。
- HTP 后端库与配置：指定处理模型的后端及相关选项。

输出是可由匹配的 QNN 运行时／后端加载的 Context Binary。它不是独立可执行程序，也不等于分词、采样、对话管理等完整应用。

当前 Context 示例没有传入 `input_list`、Input RAW 或 Golden Output。这一步不负责对样本做数值对拍，也不把某次推理的激活写进模型。

这里的 **Context 是执行上下文**，不要与 `CL=2048` 的上下文长度或某一轮对话的 KV Cache 混为一谈。

## 三、官方 Qwen2／Qwen2.5 的做法

本节属于目标设备部署流程，不是修改 Qwen 模型的数学定义，也不重新训练模型。是否需要哪种上下文产物，要由实际选用的运行时和后端决定；不能推广成所有平台都必须生成这种文件。

## 四、本项目的位置与当前边界

入口参考 [qnn_compile_deploy.py](../../../example2/host_linux/qnn_compile_deploy.py)：

| 位置 | 当前状态 |
|---|---|
| 第 290～298 行 | AR1 单份 DLC 的 Context 命令示例 |
| 第 300～308 行 | 同时传入 AR128、AR1 两份 DLC 的命令示例 |
| 第 289～310 行 | 上述示例包在三引号字符串中，不会执行 |
| 第 323～424 行 | 配置生成和自动调用代码处于注释状态 |

先认识四个参数，不把下面的对应关系当成可直接运行的命令：

| 参数 | 当前示例中的作用 |
|---|---|
| `--dlc_path` | 指向一份或多份量化 DLC |
| `--backend` | 指定 HTP 后端库 `libQnnHtp.so` |
| `--config_file` | 提供相关配置；文件内容及配置层级必须另行核对 |
| `--output_dir`、`--binary_file` | 指定生成位置和名称 |

当前脚本还有此前记录的语法残缺，因此不能把「示例写在文件里」理解为已经能完整运行。目标芯片、SDK 兼容性、配置引用、实际权重共享效果和设备加载结果均需在实操时核实。

## 五、接下来按这个顺序学

1. Context Binary 与量化 DLC 的区别。
2. `libQnnHtp.so`、DLC 配合使用的模型库、后端配置分别负责什么。
3. 目标 SoC／HTP 架构与配置如何对应。
4. AR1、AR128 为什么可能一起生成 Context，以及权重共享的目的。
5. 编译产物如何加载，再用 RAW 和 Golden 做实际验证。

当前先掌握第 1 项即可；其余内容暂缓，在拥有目标设备、明确 SoC／HTP 架构并准备实际生成时再学习和验收。

## 六、总结

**Quantizer 生成量化模型；Context 编译为目标后端准备并保存执行上下文；推理阶段才接收本次输入并产生激活与输出。三者不是同一个步骤。**
