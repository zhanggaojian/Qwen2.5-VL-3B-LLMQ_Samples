# E3-01 · Genie 与 QNN/HTP 运行时准备

> **在整个流程中的位置**：准备好 `inputs_embeds.bin` 和 Embedding Table → **把推理程序与运行库部署到设备** → 再推送模型、配置并运行。
>
> **一句话本质**：这一部分不是再次处理模型，而是在设备上搭好一条“Genie 调度 → QNN 后端 → HTP 执行”的软件通路。

## 一、先区分程序、运行库和模型

Example3 README 的第二部分要求推送：

```text
genie-t2t-run
libGenie.so
libQnnHtp.so
libQnnSystem.so
libQnnHtpNetRunExtensions.so
与目标 DSP 架构匹配的运行库
```

它们都不是模型权重：

> README 给出的是组件类别，不是适用于所有 QAIRT 版本的完整文件清单。某些版本还会需要 `libQnnHtpPrepare.so`、C++ 运行库或 DSP `.cat` 文件，应以实际安装的 QAIRT SDK 和目标平台文档为准。

| 类别 | 例子 | 作用 |
|---|---|---|
| 可执行程序 | `genie-t2t-run` | 接收命令行参数，启动一次生成任务 |
| 高层推理库 | `libGenie.so` | 组织 Tokenizer、采样、prefill/decode 循环和 QNN 模型调用 |
| QNN CPU 侧运行库 | `libQnnHtp.so`、`libQnnSystem.so` | 加载 HTP 后端和 Context Binary，并向 HTP 提交执行 |
| HTP 扩展库 | `libQnnHtpNetRunExtensions.so` | 应用 HTP 专用的设备、图、内存和性能配置 |
| CPU↔DSP 通信配套库 | `*Stub.so`、`*Skel.so` 等 | 让 CPU 侧程序通过 RPC 调用 HTP/DSP 侧实现 |
| 模型文件 | `*.serialized.bin` | 真正被执行的编译后模型；属于下一部分 |

## 二、调用关系

可以把运行栈理解成五层：

```text
用户命令
  ↓
genie-t2t-run                命令行入口，运行在 CPU
  ↓
libGenie.so                  管理 LLM/VLM 生成流程，运行在 CPU
  ↓
libQnnSystem.so + libQnnHtp.so + HTP Extensions
  ↓
HTP Stub  ── FastRPC ── HTP Skeleton
  CPU 侧                  DSP/HTP 侧
  ↓
HTP 执行 Context Binary
```

所以 `genie-t2t-run` 本身不负责矩阵乘法；真正的大量模型计算由 HTP/NPU 完成。

## 三、每个核心文件负责什么

### 1. `genie-t2t-run`

这是一个命令行测试程序。它主要负责：

- 解析 `-c`、`-e`、`-t` 等参数；
- 加载 `libGenie.so`；
- 创建对话并触发生成；
- 把生成文本和日志输出到终端。

它更像“启动按钮”，不是模型，也不是 HTP 驱动。

### 2. `libGenie.so`

Genie 位于应用与 QNN 之间，负责 LLM 特有的控制流程，例如：

- 创建并维护推理上下文；
- 组织 prefill 与逐 token decode；
- 根据配置选择、切换相应的计算图；
- 执行采样并管理生成结束条件；
- 调用 QNN HTP 后端执行真正的模型图。

### 3. `libQnnHtp.so`

这是 QNN 的 HTP 后端库。它把通用的 QNN 调用转换成 HTP 可以执行的后端操作，包括创建设备、加载上下文和提交图执行。

### 4. `libQnnSystem.so`

它提供 QNN System 相关能力。对于本项目，重点理解为：协助读取、识别和恢复已经序列化的 Context Binary 信息。

### 5. `libQnnHtpNetRunExtensions.so`

它让运行时可以应用 HTP 专用扩展配置。本项目的 `htp_backend_ext_config.json` 中包含：

- `soc_id`、`dsp_arch`；
- HTP 核数和 `perf_profile`；
- VTCM、共享内存；
- 权重共享等选项。

JSON 是“参数”，扩展库是“识别并应用这些参数的代码”。

## 四、为什么还需要 Stub 和 Skeleton

Android 上的 `genie-t2t-run` 通常运行在 ARM CPU，而模型计算运行在 HTP/DSP，二者不是同一个处理器。

### DSP 与 HTP 的关系

两者高度相关，但不能完全画等号：

| 名称 | 更侧重什么 |
|---|---|
| DSP | Hexagon 的处理器架构、标量控制能力和 DSP/CDSP 执行域 |
| HTP | Hexagon Tensor Processor，由标量、HVX 向量、HMX 张量及共享内存共同组成/协作的 AI 计算体系，也是 QNN 后端目标 |

因此不要把 HTP 想成“塞在 DSP 核心内部的一块小硬件”，也不要把它想成与 DSP 完全无关的另一颗芯片。现代 Hexagon NPU/HTP 把 DSP 标量控制、HVX 向量和 HMX 张量加速等能力组织在同一个子系统内，并使用 Hexagon DSP/CDSP 的运行环境、架构版本和 FastRPC 通路，所以代码中会同时看到：

```text
NPU（通用类别）
└─ Qualcomm Hexagon NPU / HTP（高通的具体 AI 处理器体系）
   ├─ Hexagon DSP 标量/控制能力
   ├─ HVX 向量计算单元
   ├─ HMX 张量/矩阵计算单元
   └─ 共享内存与调度机制
```

这是一张便于理解的软件/体系结构图，不是具体芯片的晶体管级框图；不同 SoC 代际的单元数量和组织方式会变化。

```text
QNN 后端名称：QnnHtp
硬件架构配置：dsp_arch = v81
DSP 库目录：hexagon-v81/unsigned
```

可以暂时记成：**DSP/CDSP 表示底层架构和执行域，HTP/Hexagon NPU 表示整套 AI 加速子系统与 QNN 的计算目标**。不同代际的 Qualcomm 文档会使用 Hexagon DSP、Hexagon NPU 或 HTP 等不同粒度的名称，因此日志里的称呼会有重叠。

```text
ARM CPU                              HTP/DSP
libQnnHtpVxxStub.so  ←── RPC ──→  libQnnHtpVxxSkel.so
```

- **Stub**：CPU 侧代理，把调用打包后发给 DSP。
- **Skeleton**：DSP 侧入口，接收调用并执行对应实现。

FastRPC 的完整但简化的传递过程是：

```text
CPU 应用调用一个看似本地的函数
  → Stub 封送函数名和参数
  → CPU 侧 FastRPC 用户库
  → FastRPC 内核驱动
  → rpmsg 把请求送到 DSP
  → DSP 侧 FastRPC 驱动
  → Skeleton 解封送参数
  → 调用 DSP/HTP 中的真实实现
  → 结果沿原路返回 CPU
```

这里的“远程”不是通过互联网，而是指同一颗 SoC 内的另一个处理器。FastRPC 负责跨处理器调用和缓冲区映射，本身不负责 Transformer 计算；真正的模型图仍由 HTP 执行。

其中 `Vxx` 必须匹配真实 HTP 架构。本仓库配置写的是 `dsp_arch: v81`，所以不能混用 v73、v75 或 v79 的配套库。

## 五、两个主要环境变量

### `LD_LIBRARY_PATH`

供 CPU 侧动态链接器查找 `.so`：

```text
libGenie.so
libQnnHtp.so
libQnnSystem.so
libQnnHtpNetRunExtensions.so
HTP Stub
```

### `ADSP_LIBRARY_PATH`

供 DSP/FastRPC 侧查找 HTP Skeleton、`.cat` 等 DSP 文件。

```text
libQnnHtpVxxSkel.so
```

仓库 README 还提到 `CDSP_LIBRARY_PATH`；具体使用哪个变量取决于设备平台和 QAIRT 版本。较新的 Qualcomm Genie Android 示例主要设置 `ADSP_LIBRARY_PATH`。

## 六、为什么不能随便复制一组库

下面几项必须成套匹配：

| 检查项 | 要求 |
|---|---|
| CPU ABI | Android ARM64 应使用对应的 `aarch64-android` 程序和库，不能推送 Example2 主机使用的 `x86_64-linux-clang` 库 |
| QAIRT 版本 | 运行库尽量与生成 Context Binary 的 QAIRT 版本一致 |
| HTP 架构 | 配置、Stub、Skeleton 必须对应同一个 v73/v75/v79/v81 |
| SoC 配置 | `soc_id`、`soc_model` 必须对应目标设备 |
| DSP 域 | `signed/unsigned` 目录和 `pd_session` 应与设备支持方式一致 |

只要其中一项错配，就可能出现“CPU 程序能启动，但 HTP 设备创建或 Context Binary 加载失败”。

## 七、这一部分最终得到什么

这一部分不会生成新的 DLC 或 Context Binary。它的完成标准只是：

```text
设备目录中已经有正确架构、正确版本的 Genie/QNN 程序与动态库，
CPU 侧和 DSP 侧都能通过环境变量找到自己的运行库。
```

仓库没有提供真实设备型号、QAIRT 安装目录或完整的 `adb push` 脚本，所以目前只学习运行栈，不能把“文件已推送并能加载”视为已验证。

## 八、参考位置

- 本项目：[example3/README.md](../../../example3/README.md)
- Qualcomm 官方：[LLM on Genie 教程](https://github.com/quic/ai-hub-apps/blob/main/tutorials/llm_on_genie/README.md)
- Qualcomm 官方：[FastRPC 的 Stub/Skeleton 架构](https://github.com/qualcomm/fastrpc)
- Qualcomm 官方：[AI Hub 部署 FAQ](https://dev.aihub.qualcomm.com/docs/hub/faq.html)

## 小结

Example3 第二部分是在设备上搭建推理运行栈：`genie-t2t-run` 负责启动，`libGenie.so` 负责生成流程，QNN HTP 库负责加载和提交模型，Stub/Skeleton 负责 CPU 与 HTP/DSP 之间的通信。
