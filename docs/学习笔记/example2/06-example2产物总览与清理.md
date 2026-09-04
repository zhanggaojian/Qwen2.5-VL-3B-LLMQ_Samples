# 06 · example2 产物总览与清理

> **上游交接**：[10 · example1 产物总览与清理](../10-example1产物总览与清理.md)
>
> **完整流程**：[00 · example2 主机编译全景](./00-example2主机编译全景.md)
>
> **第一阶段细节**：[01 · AR 图适配：从 AR1073 到 AR1/AR128](./01-AR图适配-change_hardcoding.md)
>
> **本篇范围**：只总结 `example2` 各阶段的落盘文件、下游依赖、验收门槛和清理边界，不重复讲 AR/CL、MHA2SHA、Converter 和 Quantizer 的原理。
>
> **一句话本质**：example2 是一条逐级派生链；只有下游替代产物完成结构检查、功能对拍和备份后，上游大文件才真正具备安全清理条件。

---

## 零、先看结论

1. 当前 `qnn_compile_deploy.py` 的实际终点是两份 Quantized DLC：
   - `ar1-cl2048_1_of_1_quantized.dlc`
   - `ar128-cl2048_1_of_1_quantized.dlc`
2. **Context Binary 生成代码目前没有进入活动主流程**。所以这两份 Quantized DLC 仍是后续生成共享权重 Context Binary 的必需输入，暂时不能因为“DLC 已生成”就删除。
3. `split_onnx/*.onnx` 与它引用的 `.data`、`sha_output/*.onnx` 与它引用的全部 External Data 都是不可拆分的模型束；不能只保留小的 `.onnx` 主文件。
4. 从技术依赖看，某一级下游成功生成后，上一级通常不再被继续读取；从工程安全看，最好等 AR1、AR128 两路 Quantized DLC 都通过结构检查、Host/QNN 数值对拍并完成备份后，再批量清理大体积中间图。
5. `input_list_*.txt` 中写入的是生成时的完整路径。移动或恢复 `assets/` 后，即使 RAW 文件还在，旧 `input_list` 也可能失效，需要改写或重新生成。
6. `artifacts/ar*/src`、`1_of_1/input_list_*` 和 `1_of_1/test_inputs_*` 是符号链接，本身几乎不占空间；删除它们不能有效释放磁盘，删除其目标反而会留下断链。
7. 当前脚本没有可靠检查 Converter/Quantizer 子进程返回码，日志中的 `done` 不能单独作为删除上游的依据。
8. ViT 链路与这里的 LLM 编译中间产物相互独立。ViT 尚未运行，不妨碍在 LLM 产物验收后清理 example2 中间文件；但不能因此删除 LLM 的两份 Quantized DLC。

---

## 一、输出根目录与产物全景

`qnn_compile_deploy.py` 使用 `os.getcwd()` 作为 `workfolder`。按项目预期，应先进入：

```bash
cd example2/host_linux
```

再运行脚本。此时下文用 `$ASSETS` 表示：

```text
example2/host_linux/assets
```

在当前配置 `CL=2048`、`ARNs=[1, 128]`、`num_splits=1` 下，完整产物可以按下面的结构理解：

```text
$ASSETS/
├── models_ar_n/
│   ├── ar1-cl2048/
│   │   ├── onnx/
│   │   │   ├── qwen25llm.onnx
│   │   │   ├── qwen25llm.encodings
│   │   │   └── ONNX external_data.location 引用的全部文件  # 模型束必需；当前脚本未自动复制，可能缺失
│   │   └── test_vectors/
│   │       ├── fp_0.pkl
│   │       └── qt_0.pkl
│   └── ar128-cl2048/
│       └── 与 AR1 相同的目录结构
│
└── artifacts/
    ├── ar1-cl2048/
    │   ├── src -> $ASSETS/models_ar_n/ar1-cl2048              # 实际创建为绝对路径链接
    │   ├── split_onnx/
    │   │   ├── ar1-cl2048_1_of_1.onnx
    │   │   └── ar1-cl2048_1_of_1.data
    │   ├── input_list_ar1-cl2048_1_of_1.txt
    │   ├── test_inputs_ar1-cl2048_1_of_1/
    │   │   └── 0/*.raw
    │   ├── test_golden_outputs_ar1-cl2048_1_of_1/
    │   │   └── Result_0/*.raw
    │   └── 1_of_1/
    │       ├── sha_output/
    │       │   ├── ar1-cl2048_1_of_1.onnx
    │       │   ├── 该 ONNX 引用的全部 External Data
    │       │   ├── ar1-cl2048_1_of_1.encodings
    │       │   ├── prequant_encodings_map.json
    │       │   ├── mha_to_sha_encodings_names.json
    │       │   └── all_stages_encodings_mapping.json
    │       ├── input_list_ar1-cl2048_1_of_1.txt -> $ASSETS/artifacts/ar1-cl2048/同名文件
    │       ├── test_inputs_ar1-cl2048_1_of_1 -> $ASSETS/artifacts/ar1-cl2048/同名目录
    │       ├── converted_model/
    │       │   └── ar1-cl2048_1_of_1.dlc
    │       └── compiled_model/
    │           └── ar1-cl2048_1_of_1_quantized.dlc
    │
    └── ar128-cl2048/
        └── 与 AR1 对应的 Split、RAW、Golden、SHA、普通 DLC 和 Quantized DLC
```

> **External Data 提醒**：上面列出的 `.data` 是当前 Split 保存方式的常见结果；判断模型束是否完整时，不能只匹配某个后缀，而要以 ONNX 中每一项 `external_data.location` 的真实引用为准。SHA ONNX 也必须用同样方法检查。

上述三个 `->` 都表示符号链接关系，当前实现写入的是生成机器上的绝对目标路径；目录整体搬家后可能变成断链。

还有三点容易在实际目录中造成疑惑：

- AR 适配会保持相对目录递归复制发现的 `.json/.yaml/.encodings`，但不会复制 `.pth`，这些辅助文件可能因 example1 实际输出而增减。
- 非 FP32 Test Tensor 可能同时出现 `<tensor>.<dtype>.raw` 和 `<tensor>.raw`；应由 `input_list` 判断 Quantizer 真正引用哪一份，不能按文件名猜测后删重。
- 当前 MHA2SHA 命令没有启用 `--create-input-lists`，因此通常不会产生 `mha_input_vectors/`、`golden_output_from_mha/`、`sha_test_vectors/` 或 `on_device_input_list.txt`。Split 返回的产物映射也只存在内存中，不会额外落盘为 manifest。

> **写作时的工作区快照**：截至 2026-08-15，本地 `example2/host_linux/assets/` 为空。上面的完整树来自代码路径和已有运行记录，不代表本机当前已经生成这些文件。实际清理远程机器或其他运行目录前，必须重新扫描当时的 `assets/`。

如果以后启用脚本末尾目前被注释的 Context Binary 阶段，还会出现类似下面的产物：

```text
$ASSETS/artifacts/
├── ar128-ar1-cl2048_conf_files/
│   ├── HtpConfigFile_API_1.json
│   └── PerfSetting_API_1.conf
└── ar128-ar1-cl2048/
    └── weight_sharing_model_1_of_1.serialized[.bin]
```

`[.bin]` 表示后缀取决于实际 QAIRT/QNN 工具输出；当前主流程尚未生成这项，验收时应以真实文件为准。

---

## 二、各阶段产物说明

| 阶段 | 主要产物 | 当前下游消费者 | 重建成本 | 技术上最早失去硬依赖的节点 |
|---|---|---|---|---|
| AR 图适配 | `models_ar_n/ar1-cl2048/`、`ar128-cl2048/` | Split；MHA2SHA 还会通过 `src` 读取原 Encoding | 中到高，需重新改图和改 Test Vector | 两路 MHA2SHA 模型束完整生成后 |
| Split | `split_onnx/*.onnx` + External Data | MHA2SHA | 高，需重新加载/切分大 ONNX | 对应 SHA 模型束完整生成后 |
| Test Vector 展开 | `input_list_*.txt`、`test_inputs_*/*.raw` | `qairt-quantizer`；也可用于 Host/QNN 推理 | 中，需从 `qt_0.pkl` 重新展开 | Quantized DLC 完成且不再重跑/验证后 |
| Golden Output | `test_golden_outputs_*/*.raw` | 当前编译主流程不直接读取；用于数值验收 | 中，失去后难判断结果是否正确 | 数值对拍报告已经保存后 |
| MHA2SHA | `sha_output` 中的 ONNX、External Data、Encoding 和映射 JSON | `qairt-converter` | 很高，图变换记录显示峰值内存很大 | 普通 DLC 成功转换后 |
| Converter | `converted_model/*.dlc` | `qairt-quantizer` | 高，需重新执行 Converter | Quantized DLC 完成结构和功能验收后 |
| Quantizer | `compiled_model/*_quantized.dlc` | 后续 Context Binary 生成 | 高，是当前活动脚本的最终产物 | Context Binary 生成并完成目标端验收后 |
| Context Binary（未来） | `*.serialized` 或 `*.serialized.bin` | `example3` 端侧推理 | 很高，且绑定 QNN/HTP/SoC 配置 | 一般作为部署交付物长期保留 |

当前 `num_splits=1`，所以 Split 阶段并没有把模型拆成多个分片，而是生成统一命名的 `1_of_1` 模型束；以后改成多分片时，表中的每项都要按 `N_of_M` 成组验收和清理。

主依赖链可以压缩成：

```text
example1 ONNX/Encoding/qt_0.pkl
          │
          ▼
AR1 / AR128 模型束
          │
          ├──► Split ONNX + External Data ──► SHA ONNX + External Data + Encoding
          │                                      │
          │                                      ▼
          │                                  普通 DLC
          │                                      │
          └──► input_list + RAW ─────────────────┤
                                                 ▼
                                      AR1 / AR128 Quantized DLC
                                                 │
                                                 ▼
                                    共享权重 Context Binary（尚未启用）

Golden Output ─────────────────────► 功能/数值验收，不是编译硬输入
```

这张图也解释了为什么“下游文件存在”不等于“上游立刻可删”：如果下游未经功能验证，删掉上游就失去了回滚和重新编译的入口。

---

## 三、当前最终产物与下一阶段最小集合

### 3.1 当前活动脚本的最终产物

```text
$ASSETS/artifacts/ar1-cl2048/1_of_1/compiled_model/
└── ar1-cl2048_1_of_1_quantized.dlc

$ASSETS/artifacts/ar128-cl2048/1_of_1/compiled_model/
└── ar128-cl2048_1_of_1_quantized.dlc
```

这两份 DLC 分别服务于：

- `AR=1`：逐 token 解码阶段；
- `AR=128`：当前项目配置下的批量输入/Prefill 路径。

二者不是重复备份，不能只留其中一份。

### 3.2 生成完整 AR1+AR128 共享权重 Context 的最小模型集合

从模型文件依赖看，生成 AR1+AR128 共享权重 Context 的最小集合就是上面的两份 Quantized DLC；如果只生成脚本示例中的 AR1-only Context，则只需要 AR1 DLC。除此之外，执行机器还需要匹配的：

- HTP/QNN Context 配置；
- 目标 SoC、DSP/HTP 架构设置；
- 同版本 QAIRT/QNN SDK、backend 和相关运行库。

Split ONNX、SHA ONNX、普通 DLC、RAW 和 Golden Output 都不是 Context 生成命令的模型硬输入。

### 3.3 工程上更建议保留的归档集合

只留两个 DLC 可以继续向下走，但不利于验证和复现。更稳妥的归档集合是：

- AR1、AR128 两份 Quantized DLC；
- 两套 `input_list`、Test Input RAW 和 Golden Output；
- Context/HTP 配置与生成后的 Context Binary；
- 本次脚本配置、完整日志、软件版本、目标芯片信息；
- DLC 和 Context Binary 的文件大小及 SHA-256；
- 若磁盘允许，保留普通 DLC 或将其转移到低成本存储，便于重新量化。

---

## 四、什么可以删，什么暂时不能删

### 4.1 可优先考虑的低风险清理

| 对象 | 可清理前提 | 说明 |
|---|---|---|
| `models_ar_n/ar*/test_vectors/fp_0.pkl` | 不做 FP/QT/QNN 对拍 | 当前 Split 主流程使用 `qt`，不使用 `fp`；删除后失去浮点参考 |
| 已明确属于失败轮次的残留 | 已通过路径、时间、ONNX 引用和当前配置精确识别 | 不要用宽泛通配符误删当前轮次 |
| 不再需要的旧版本归档 | 当前有效版本已有 checksum 和至少一份可恢复备份 | 先确认不是唯一可回滚副本 |

符号链接不是空间回收重点。`src`、`1_of_1/input_list_*`、`1_of_1/test_inputs_*` 只保存链接信息；真正占空间的是它们指向的模型、RAW 或其他目标文件。

### 4.2 满足条件后才能清理的大体积中间产物

| 对象 | 技术上最早可删节点 | 更稳妥的建议 | 删除后失去什么 |
|---|---|---|---|
| `models_ar_n/` | 两路 SHA ONNX 束及 Encoding 都完整可读 | 等两份 Quantized DLC 数值验收、备份后 | 不能直接重跑 Split/MHA2SHA；`src` 变成断链 |
| `split_onnx/` | 对应 SHA 模型束完整可读 | 等 Converter 或最终 DLC 验收后 | 不能直接重跑 MHA2SHA |
| `sha_output/` | 普通 DLC 转换成功 | 等 Quantized DLC 数值验收后 | 不能直接重跑 Converter，丢失图变换排错现场 |
| `converted_model/*.dlc` | Quantized DLC 已生成 | 等 Quantized DLC 结构和功能验收、备份后 | 不能直接重新量化 |
| `input_list_*` + `test_inputs_*` | Quantizer 返回码为 0，Quantized DLC 可读 | 等 Host/QNN 推理和数值对拍完成后 | 不能直接重跑 Quantizer/同一批测试输入 |
| `test_golden_outputs_*` | 对拍结果和容差报告已保存 | 文件通常比模型图小，建议随最终产物保留 | 失去最直接的数值正确性基线 |

### 4.3 当前不能删除的产物

| 对象 | 原因 |
|---|---|
| 两份 `*_quantized.dlc` | 当前尚未生成并验证 Context Binary，它们仍是下一阶段硬输入 |
| 已生成但尚未端侧验证的 Context Binary | 文件存在不代表能在目标 HTP/SoC 正确执行 |
| 唯一一份配置、日志、版本记录和 checksum | 体积很小，却是复现与排错依据 |

### 4.4 不要与生成产物一起误删

| 对象 | 为什么要保留 |
|---|---|
| `example2/G2G/` 与 `example2/host_linux/*.py` | 这是生成脚本和图变换源码，不是 `assets/` 中间产物 |
| `example2/host_linux/*.json`、`*.yaml`、`*.conf` | 其中包含 QNN/HTP、量化和图变换配置；删除会破坏复现 |
| 原始 Hugging Face 模型 | ViT 链路和 `example3` 生成 embedding/test vector 仍可能读取 |
| `example1/` 源码 | `example3` 的部分脚本会导入 `example1.llm_utils` |

---

## 五、按里程碑制定清理策略

| 里程碑 | 必须保留 | 此时可考虑清理 |
|---|---|---|
| AR1/AR128 改图刚完成 | 两套 AR 模型束、Encoding、`qt_0.pkl` | 不做 FP 调试时，可先处理两套 `fp_0.pkl` |
| Split/MHA2SHA 完成 | SHA 模型束、Encoding、RAW、input list、Golden | Split 束已无下游硬依赖，但建议先等 Converter 成功 |
| 普通 DLC 完成 | 普通 DLC、RAW/input list、Golden，建议暂留 SHA | SHA 束技术上可删；磁盘紧张且普通 DLC 已验证可读时才提前处理 |
| 两份 Quantized DLC 生成但未对拍 | 两份 Quantized DLC、所有验证输入和 Golden | 只清理明确的失败轮次，不建议批量删除可重编源束 |
| 两份 Quantized DLC 通过结构检查、Host/QNN 对拍并完成备份 | 两份 Quantized DLC、验证记录、配置、日志、checksum | 若不再重编，可清理 `models_ar_n`、Split、SHA、普通 DLC 和 Test Input；Golden 建议归档 |
| Context Binary 完成目标端验收 | Context Binary、端侧配置、运行库清单、测试向量和 checksum | 两份 Quantized DLC 可转移到归档存储；只有接受以后重生成 Context 成本时才彻底删除 |

当磁盘非常紧张时，可以采用“生成一级、验收一级、备份一级、再清上一级”的滚动策略。这里的验收必须包含真实子进程返回码和模型可读性；若尚未做数值对拍，需明确接受后续难以回滚的风险。

---

## 六、删除前的验收和备份

### 6.1 先确认扫描的是实际运行目录

```bash
cd example2/host_linux

pwd
du -ah assets | sort -h | tail -n 40
find assets -type l -ls
```

由于脚本以当前工作目录定位 `assets/`，若当初不是在 `example2/host_linux` 启动，产物可能位于另一个工作目录。不要只因仓库内 `assets/` 为空就判定远程产物不存在。

### 6.2 验证 ONNX 与 External Data 是完整模型束

对下面三类阶段产物都应用相同规则；AR1、AR128 各一份，一共是六份 ONNX：

- AR1/AR128 的 `models_ar_n/.../qwen25llm.onnx`；
- AR1/AR128 的 `split_onnx/*.onnx`；
- AR1/AR128 的 `sha_output/*.onnx`。

完整性应优先满足第一种检查；内存不足时，第二种只能作为结构核对方案：

1. `onnx.load(path, load_external_data=True)` 成功；这种方式会实际装载权重，需要预留较大内存。
2. 使用 `load_external_data=False` 读取主图，枚举每个 Tensor 的全部 `external_data.location`，再相对于 ONNX 所在目录逐一确认文件存在且非空；若引用带有 `offset/length`，还要确认 `offset + length` 没有超过文件大小。若源文件有可信 checksum，还应逐一比对。

仅执行 `onnx.load(path, load_external_data=False)` 只能证明图主文件可解析，不能证明外置权重齐全。即使路径、大小范围都通过，也主要属于结构检查；没有成功加载或可信 checksum 时，仍不能排除文件内容损坏。

尤其要注意：当前 `change_hardcoding.py` 用 `load_external_data=False` 读图，却没有按每个 `external_data.location` 自动复制权重文件。因此 `models_ar_n/ar*/onnx/qwen25llm.onnx` 存在，并不能证明 AR 模型束完整。

### 6.3 验证 input list 与 RAW 的路径关系

```bash
cd example2/host_linux

head -n 2 assets/artifacts/ar1-cl2048/input_list_ar1-cl2048_1_of_1.txt
head -n 2 assets/artifacts/ar128-cl2048/input_list_ar128-cl2048_1_of_1.txt
```

逐项确认每一行引用的 RAW 文件都存在且非空。如果 `assets/` 被移动、重新挂载或恢复到不同路径，应改写 input list 中的旧绝对路径，或重新运行 Test Vector 展开阶段。

### 6.4 验证当前两份最终 DLC

```bash
cd example2/host_linux

test -s assets/artifacts/ar1-cl2048/1_of_1/compiled_model/ar1-cl2048_1_of_1_quantized.dlc
test -s assets/artifacts/ar128-cl2048/1_of_1/compiled_model/ar128-cl2048_1_of_1_quantized.dlc
```

验收要分两层：

1. **结构完整性**：Converter 和 Quantizer 的实际返回码为 `0`；两个文件存在、非空，并能被同版本 QAIRT/QNN 的 DLC 检查工具读取。
2. **功能/数值验收**：分别用 AR1、AR128 对应 RAW 做 Host/QNN 推理，将输出与各自 Golden Output 按项目容差或 SQNR 规则对拍。

当前 `qnn_compile_deploy.py` 启动子进程后没有检查 `proc.returncode`，并且并行任务的结果未被显式消费。因此日志出现 `done` 只是一条流程日志，不是成功凭证。

### 6.5 做 checksum 与可恢复备份

```bash
cd example2/host_linux

sha256sum \
  assets/artifacts/ar1-cl2048/1_of_1/compiled_model/ar1-cl2048_1_of_1_quantized.dlc \
  assets/artifacts/ar128-cl2048/1_of_1/compiled_model/ar128-cl2048_1_of_1_quantized.dlc
```

同时记录：

- DLC 文件大小和 SHA-256；
- `CL`、`ARNs`、`num_splits`、量化配置和 HTP/SoC 配置；
- QNN/QAIRT SDK、ONNX、Python 及系统版本；
- Converter/Quantizer 的完整命令、返回码和日志；
- 数值对拍使用的输入、Golden Output、容差和结果。

备份应先做恢复抽查，确认归档文件真的能读，再删除唯一的本地副本。

---

## 七、ViT、Context Binary 与 example3 的边界

### 7.1 ViT 尚未运行时，能否清理 example2

可以清理**已被验收下游替代的 LLM 中间产物**，因为两条链路相互独立：

```text
LLM：example1 → example2 AR/Split/SHA/DLC → Context Binary → example3
ViT：原始 HF 模型 → vit/qwen2_5_vl → 视觉编码器部署产物 → example3
```

但 ViT 是否运行，与两份 LLM Quantized DLC 是否仍被 Context 生成需要没有关系。当前 Context 阶段尚未完成，所以两份 DLC 仍需保留。

### 7.2 当前 `num_splits=1` 与 example3 配置需要先对齐

当前 `qnn_compile_deploy.py` 设置的是 `num_splits=1`，预期每条 AR 路径各生成一个 DLC，并进一步生成一组 `1_of_1` 共享权重 Context。

另外，脚本中的两段手工 Context 示例使用了相同的 `--output_dir` 和 `--binary_file`。如果先生成 AR1-only Context、再生成 AR128+AR1 共享权重 Context，应改成不同名称或先确认工具的覆盖行为，避免后一次覆盖前一次而误判产物。

但现有 `example3/Qwen2.5-VL-3B/qwen25vl3B_os.json` 记录的是六个：

```text
weight_sharing_model_1_of_6.serialized.bin
...
weight_sharing_model_6_of_6.serialized.bin
```

这说明 example2 当前配置与 example3 的既有部署配置并未天然对齐。必须先确认最终采用 `1_of_1` 还是 `6_of_6`，再生成并验收 Context Binary。完成这一步以前，不要把两份 Quantized DLC 当成“已经被 Context 完全替代”。

---

## 八、清理检查表

- [ ] 已确认真实运行时的工作目录和 `$ASSETS`，没有把本地空目录误认为远程产物状态。
- [ ] AR1、AR128 两条路径都完成到预定里程碑，没有只检查其中一路。
- [ ] 所有待保留 ONNX 都能以 `load_external_data=True` 加载；若只能做轻量检查，全部 `external_data.location` 均存在、非空、引用范围未越界，并已尽可能比对可信 checksum。
- [ ] 待保留 Encoding/JSON 可以解析，不是空文件或失败轮次残留。
- [ ] 两次 Converter、Quantizer 的实际返回码都是 `0`，没有只依据日志中的 `done`。
- [ ] 两份 Quantized DLC 都存在、非空，并能被对应版本 QNN 工具读取。
- [ ] AR1、AR128 都已完成 Host/QNN 推理并与各自 Golden Output 对拍；若未做，已明确接受删除源束后无法快速回滚或重编的风险。
- [ ] 已保存两份 DLC 的 SHA-256、文件大小、配置、软件版本和完整日志。
- [ ] 已确认 `input_list` 中的绝对路径在当前机器仍然有效。
- [ ] 已分清符号链接和真实数据目录，没有把“删链接”误认为释放了模型空间。
- [ ] Context Binary 尚未验收时，没有删除两份 Quantized DLC。
- [ ] 已核对 `1_of_1` 与 example3 当前 `6_of_6` 配置差异。
- [ ] 没有删除 `example1/`、`example2/` 源码、原始 Hugging Face 模型或唯一配置记录。

---

## 九、一句话总结

> **example2 当前真正交付到下一阶段的是 AR1、AR128 两份 Quantized DLC；磁盘紧张时，可在逐级验收和备份后依次清理 AR/Split/SHA/普通 DLC 等中间产物，但 Context Binary 尚未生成并通过目标端验证前，不能删除这两份最终 DLC。**
