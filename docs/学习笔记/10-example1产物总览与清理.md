# 10 · example1 产物总览与清理

> **上游原理**：[08 · ONNX 导出与测试向量](./08-ONNX导出与测试向量.md)
>
> **运行总览**：[09 · 工程运行指南](./09-工程运行指南-LLM与ViT双链路.md)
>
> **下游衔接**：[example2 主机编译全景](./example2/00-example2主机编译全景.md)
>
> **本篇范围**：只总结 `example1` 各阶段落盘的文件、下游依赖和清理边界，不重复讲解量化和 ONNX 导出原理。
>
> **一句话本质**：先按“后续是否消费”和“重新生成成本”给产物分类，再决定保留、备份或删除。

---

## 零、先看结论

1. `example2` 开始前必须保留四类数据：`qwen25llm.onnx`、全部 ONNX 外置参数、`qwen25llm.encodings` 和 `qt_0.pkl`。
2. `qwen25llm.onnx` 与外置 `*.weight/*.bias/*.data` 是一个整体；只留小小的 `.onnx` 主文件等于丢掉模型参数。
3. 只为了继续跑当前 `example2` 时，`qwen25llm.pth`、`qwen25llm_torch.encodings` 和 `fp_0.pkl` 都不是必需输入。其中 `.pth` 是 AIMET 导出的无量化模拟算子的 PyTorch 浮点模型存档，不是紧密打包的 INT4 权重；重新加载仍可能依赖兼容的模型结构、自定义模块和 AIMET/PyTorch 环境。
4. `.tmp/` 可在 `example1` 进程完全结束后清理；`prepare/` 是昂贵的可复用中间产物，删除不会破坏已生成的 DLC，但会让以后重跑 `example1` 必须重做 Prepare。
5. 不要在只看到 AR1/AR128 ONNX 时就删除原始 ONNX 束。当前 `change_hardcoding.py` 没有按 ONNX `external_data.location` 复制外置数据文件，新 ONNX 可能缺少实际参数。
6. **ViT 还没跑，不妨碍在 LLM 的 Quantized DLC 通过结构检查和 Host/QNN 数值对拍、完成备份后清理 `example1` 输出目录**。但不要连 `example1/` 源码目录或原始 Hugging Face 模型一起删除。

---

## 一、输出根目录与产物全景

`example1/config.yaml` 当前配置的运行输出根目录是：

```text
/root/autodl-tmp/zgj/Qwen25/outputs/output
```

下文用 `$OUT` 代表这个目录。一次完整运行可以按下面的结构理解：

```text
$OUT/
├── .tmp/                                  # ONNX 导出等阶段的临时文件
├── prepare/                               # Prepare 后可重新加载的模型及中间文件
│   ├── qwen25llm_kvcache_36_layer.py
│   ├── qwen25llm_kvcache_36_layer.json
│   ├── qwen25llm_kvcache_36_layer_io_map.json
│   ├── *.safetensors                    # 完整恢复 prepared model 时通常需要
│   └── 中间 ONNX / External Data / Constant / UUID 目录
├── test_vectors/
│   ├── fp_0.pkl                        # 关闭 Quantizer 的浮点参考数据
│   └── qt_0.pkl                        # 开启 Quantizer 的量化模拟数据
└── onnx/
    ├── qwen25llm.onnx                  # 计算图和外置参数引用
    ├── External Data                   # ONNX external_data.location 引用的全部文件
    │                                           # 当前主要表现为 *.weight/*.bias/*.data
    ├── qwen25llm.encodings             # ONNX/QNN 名称下的量化规则
    ├── qwen25llm.pth                   # AIMET 导出的 PyTorch 浮点模型存档
    └── qwen25llm_torch.encodings       # PyTorch/QuantSim 名称下的量化规则
```

> **写作时的工作区快照**：截至 2026-08-15，仓库本地 `output/prepare/` 只保留了 `.py` 和两个 JSON，`output/onnx/` 为空，也没有 `output/test_vectors/`。这不是一套完整的本地 example1 输出；文档中的完整文件和大小来自代码以及已记录的远程成功运行。以后阅读时应以当时的实际目录扫描为准。

---

## 二、各阶段产物说明

| 产物 | 生成阶段 | 作用 | 后续消费者 | 重建成本 |
|---|---|---|---|---|
| `.tmp/` | `run.sh` 指定的临时目录 | 避免大模型导出写满系统 `/tmp` | 仅当前运行过程 | 低，下次自动重建 |
| `prepare/*.py` | QAIRT `model_preparer` | Prepare 后的静态模型结构和 `forward` | `skip_prepare=true` 重载 | 高 |
| `prepare/*.json` | QAIRT `model_preparer` | 原模块、新模块、参数名和 Tensor 名的映射 | 名称追溯、Encoding 对齐和排错 | 高 |
| `prepare/*.safetensors` | QAIRT `model_preparer` | prepared model 的实际权重 | `load_torch_model_using_safetensors()` | 高 |
| Prepare 中间 ONNX、External Data、`_Constant_*`、UUID 目录 | Prepare 内部导出 | 中间图转换与落盘 | 当前 Prepare 过程 | 中到高 |
| `test_vectors/fp_0.pkl` | `generate_test_vectors()` | 浮点参考输入、输出和部分层输出 | 数值对拍、排查量化误差 | 中，需重跑模型前向 |
| `test_vectors/qt_0.pkl` | `generate_test_vectors()` | QuantSim 量化模拟的输入、输出和部分层输出 | **example2 Split/RAW/Golden Output** | 中，需重跑模型前向 |
| `onnx/qwen25llm.onnx` | `quantsim.export()` | 计算图、Shape、Tensor 名和 External Data 引用 | **example2 AR 改图、Split、MHA2SHA、QNN** | 高 |
| ONNX `external_data.location` 引用的全部文件 | `quantsim.export()` | ONNX 外置参数；当前主要是 `*.weight/*.bias/*.data` | **ONNX 加载以及 example2 各图处理阶段** | 高 |
| `onnx/qwen25llm.encodings` | `quantsim.export()` | 与 ONNX Tensor 名对齐的位宽、scale、offset 等规则 | **Split/MHA2SHA 处理，随后由 `qairt-converter --quantization_overrides` 消费派生的 SHA Encoding** | 高 |
| `onnx/qwen25llm.pth` | `quantsim.export()` | 无量化模拟算子的 PyTorch 浮点模型存档 | 与兼容模型结构、自定义模块和 AIMET/PyTorch 环境配合恢复、排错 | 高，但 example2 不用 |
| `onnx/qwen25llm_torch.encodings` | `quantsim.export()` | 与 PyTorch 模块名对齐的量化规则 | QuantSim 恢复和排错 | 高，但 example2 不用 |

已记录的一次远程运行中，`qwen25llm.pth` 约 12 GB，`qwen25llm_torch.encodings` 约 336 MB。所以它们往往是“只继续做 QNN 部署”时最先考虑的空间回收对象；大小不是格式保证，不同版本和配置可能变化。

Encoding 的直接消费链要记成：

```text
qwen25llm.encodings
  → Split / MHA2SHA 映射
  → sha_output/*.encodings
  → qairt-converter --quantization_overrides
  → 普通 DLC
  → qairt-quantizer 读取“普通 DLC + input_list/RAW”
  → Quantized DLC
```

`qairt-quantizer` 不直接读取原始 `qwen25llm.encodings`。

---

## 三、交给 example2 的最小必要集合

```text
$OUT/
├── onnx/
│   ├── qwen25llm.onnx
│   ├── qwen25llm.encodings
│   └── 该 ONNX 的 external_data.location 引用的全部文件
└── test_vectors/
    └── qt_0.pkl
```

这个集合中：

- `qwen25llm.onnx` 管“怎么计算”；
- External Data 管“参数具体是什么”；
- `qwen25llm.encodings` 管“这些参数和激活怎么量化”；
- `qt_0.pkl` 管“编译后用什么输入和黄金输出做验证”。

当前 `split_onnx_utils` 的主流程只遍历 `qt` 测试向量，所以 `fp_0.pkl` 不是生成当前 Quantized DLC 的硬依赖。

### 3.1 外置权重的特别风险

`example2/G2G/change_hardcoding.py` 使用 `load_external_data=False` 读图，并只在辅助文件复制列表中显式包含 `.encodings/.json/.yaml`，没有根据 `external_data.location` 复制外置数据。因此：

> `assets/models_ar_n/ar1-cl2048/onnx/qwen25llm.onnx` 或 AR128 的 ONNX 存在，只能证明图主文件已保存，不能证明它的外置参数已完整复制。`onnx.load(path, load_external_data=False)` 也只能解析主图，不是权重完整性验收。

只有 `onnx.load(path, load_external_data=True)` 成功，或枚举出所有 `external_data.location` 并逐一确认目标文件存在且非空，才能说 AR ONNX 的 External Data 完整。在两份 AR ONNX 通过这项检查，或后续 Split、MHA2SHA、DLC 结构检查和 Host/QNN 数值对拍全部完成以前，建议继续保留 example1 的原始 ONNX 束。

---

## 四、什么可以删，什么暂时不能删

### 4.1 只继续跑 example2 时，可优先清理

| 产物 | 可删前提 | 删除后失去什么 |
|---|---|---|
| `$OUT/.tmp/` | `example1` 和导出子进程已完全结束 | 只失去当次临时文件，下次会重建 |
| `qwen25llm.pth` | ONNX 束已验收，不打算在 PyTorch/AIMET 中恢复或排错 | 失去约 12 GB 的 PyTorch 浮点模型存档 |
| `qwen25llm_torch.encodings` | 不打算恢复或排查 QuantSim | 失去 Torch 名称下的 Encoding |
| `fp_0.pkl` | 不做 FP 与 QT/ONNX/QNN 数值对拍 | 失去浮点参考基线 |
| 已明确属于失败轮次的旧导出残留 | 已用路径、时间和当前 ONNX 引用核对过 | 失去旧轮次排错现场 |

### 4.2 满足条件后才能清理

| 产物 | 最早可删节点 | 更稳妥的建议 |
|---|---|---|
| Prepare 中间 ONNX、External Data、`_Constant_*`、UUID 目录 | 完整 prepared bundle 已实际用 `skip_prepare=true` 重载并通过前向/PPL 回归 | 只清理能明确识别的中间项，保留 `.py/.json/_io_map.json/.safetensors` |
| 整个 `$OUT/prepare/` | 已不再需要重跑 `example1` 量化或重新导出 | 等 Quantized DLC 通过结构检查和 Host/QNN 数值对拍、完成备份后再删；恢复时需重做高成本 Prepare |
| `qt_0.pkl` | example2 Split 已生成完整 RAW、`input_list` 和 Golden Output | 保留到两份 Quantized DLC 完成 Host/QNN 数值对拍，便于重跑和排查误差 |
| `qwen25llm.onnx` + 全部 External Data + `qwen25llm.encodings` | 下游已有通过 External Data 完整性检查的替代束 | 等 AR1/AR128 两份 Quantized DLC 都通过结构检查和 Host/QNN 数值对拍，并做 checksum 备份后再删 |

### 4.3 不要与输出产物一起误删

| 对象 | 为什么还可能需要 |
|---|---|
| `example1/` 源码目录 | `example3` 的嵌入表/测试向量脚本会导入 `example1.llm_utils` |
| 原始 Hugging Face Qwen2.5-VL 模型 | 独立 ViT 链路和 `example3` 生成 embedding 仍会读取原模型 |
| `example1/config.yaml`、完整日志、环境版本和 checksum | 它们决定产物是如何生成的，体积小但复现价值高 |

---

## 五、按里程碑制定清理策略

| 里程碑 | 必须保留 | 此时可考虑清理 |
|---|---|---|
| example1 刚完成 | ONNX 完整束、`qwen25llm.encodings`、`qt_0.pkl` | `.tmp/`；不做 PyTorch 调试时可备份或删除 `.pth` 和 `_torch.encodings` |
| AR1/AR128 改图完成 | 仍建议保留 example1 核心束 | 只有 AR 新图能以 `load_external_data=True` 成功加载，或所有 `external_data.location` 都存在且非空时，才具备提前删源束的技术条件 |
| Split/MHA2SHA 完成 | 新 ONNX、新 Encoding、RAW/Golden Output | `qt_0.pkl` 已无后续硬依赖，但建议等 DLC 完成后再删 |
| AR1/AR128 Quantized DLC 都通过结构检查、Host/QNN 数值对拍并备份 | 两份 Quantized DLC、配置、日志、Golden Output 和 checksum | 如果不再重编，可清理 example1 的 `prepare/`、`test_vectors/` 和 `onnx/` |
| Context Binary/端侧验证完成 | 真正交付的 DLC/Context Binary 与运行配置 | 其他中间图可按是否需要重编决定归档或删除 |

这里要分清“技术上最早可删”与“工程上建议何时删”：当磁盘非常紧张时，可在下游替代束通过 External Data 完整性检查后提前释放；但这只能证明“文件齐”，不能证明“数值对”。普通情况下，等最终 DLC 完成 Host/QNN 数值对拍再清理，更容易回滚。

---

## 六、删除前的验收和备份

### 6.1 确认 example1 核心输入非空

```bash
OUT=$(python -c "import yaml; print(yaml.safe_load(open('example1/config.yaml'))['environment']['output_dir'])")

test -s "$OUT/onnx/qwen25llm.onnx" || echo "ERROR: ONNX missing"
test -s "$OUT/onnx/qwen25llm.encodings" || echo "ERROR: Encoding missing"
test -s "$OUT/test_vectors/qt_0.pkl" || echo "ERROR: qt_0.pkl missing"

find "$OUT/onnx" -maxdepth 1 -type f \
  \( -name '*.weight' -o -name '*.bias' -o -name '*.data' \) \
  -size +0c | head
```

`find` 有输出只证明“存在外置数据文件”，不代表 ONNX 引用的每一份都齐全。External Data 完整性必须用下面两种方式之一确认：

1. `onnx.load(path, load_external_data=True)` 成功；这种方式会实际加载大权重，要预留足够内存。
2. 用 `load_external_data=False` 读取主图，枚举所有 `external_data.location`，再相对于 ONNX 所在目录逐一确认文件存在且非空。

仅执行 `onnx.load(path, load_external_data=False)` 不能证明外置权重齐全。

### 6.2 确认当前脚本的两份最终 DLC

```bash
cd example2/host_linux

test -s assets/artifacts/ar1-cl2048/1_of_1/compiled_model/ar1-cl2048_1_of_1_quantized.dlc
test -s assets/artifacts/ar128-cl2048/1_of_1/compiled_model/ar128-cl2048_1_of_1_quantized.dlc
```

这里要分两层验收：

1. **结构完整性检查**：DLC 存在且非空；Converter 和 Quantizer 的实际返回码为 `0`；当前 QAIRT/QNN SDK 的 DLC 检查工具能成功读取。
2. **功能/数值验收**：至少运行一次 Host/QNN 推理，将输出与 Split 生成的 Golden Output 按项目容差或 SQNR 规则对拍。

当前脚本没有检查 `proc.returncode`，异常也会在子进程中被捕获，`executor.map()` 的结果没有显式消费，所以日志打印 `done` 不能证明返回码为 `0`，更不能代替数值对拍。

如果只做了结构完整性检查，就只能说“DLC 文件可读”，不能说“模型功能已验收”。此时若因磁盘压力删除源束，需要明确接受以后无法快速回滚或重编的风险。

### 6.3 保留最小复现信息

即使为了省空间删除大体积中间产物，也建议保留：

- `example1/config.yaml` 与 `example2/host_linux/qnn_compile_deploy.py` 当次配置；
- AIMET、PyTorch、Transformers、ONNX、QAIRT/QNN SDK 版本；
- example1/example2 完整日志；
- 最终 DLC 的文件大小和 SHA-256；
- 目标 SoC、DSP/HTP 架构、AR 和 Context Length。

---

## 七、ViT 未运行时能否删除 example1 产物

可以，但要等 LLM 链路自己走到稳定的下游产物：

```text
LLM 链路：example1 output → example2 → AR1/AR128 Quantized DLC
ViT 链路：原始 HF 模型 → vit/qwen2_5_vl/example1 → VEG 产物
```

ViT 链路不读取 LLM `$OUT/onnx/qwen25llm.*` 或 `$OUT/test_vectors/*.pkl`。因此，在两份 LLM Quantized DLC 已通过结构检查和 Host/QNN 数值对拍、完成备份，并且确认不再重编 LLM 后，即使 ViT 还没跑，也可以清理 LLM 的 example1 大体积输出。

但后续 `example3` 的嵌入表/测试向量脚本会导入 `example1.llm_utils` 并加载原始 Hugging Face 模型，所以此处的“删除 example1 产物”只指删除 `$OUT` 下的生成物，不是删除仓库里的 `example1/` 代码或 HF Checkpoint。

---

## 八、清理检查表

- [ ] 已确认要清理的是 `$OUT` 产物目录，不是仓库源码或原始 HF 模型。
- [ ] Converter 和 Quantizer 的实际返回码都为 `0`，AR1/AR128 的 `*_quantized.dlc` 都存在、非空并能被 QNN 工具读取。
- [ ] 已运行 Host/QNN 推理并与 Golden Output 对拍；若没做，已明确接受删除源束后无法快速回滚或重编的风险。
- [ ] 已保存两份 DLC 的 SHA-256、配置、环境版本和完整日志。
- [ ] 已明确以后是否还要重跑 example1、Split、MHA2SHA 或 QNN 编译。
- [ ] 若提前删除 example1 ONNX 束，下游替代 ONNX 已以 `load_external_data=True` 成功加载，或已逐一验证所有 `external_data.location` 存在且非空。
- [ ] 删除 `prepare/` 前，已接受将来可能要重做高成本 Prepare。
- [ ] 删除 `fp_0.pkl` 前，已确认不再做 FP/QT/QNN 数值对拍。
- [ ] 没有仅凭日志中的 `done` 就判定下游成功。

---

## 九、一句话总结

> **example2 开始前把 ONNX、所有 `external_data.location` 引用的外置参数、ONNX Encoding 和 `qt_0.pkl` 当成不可拆的交接集合；只跑 QNN 时可优先清理 `.pth`、Torch Encoding、FP Test Vector 和已结束进程的临时文件，其余大体积产物最好等 AR1/AR128 Quantized DLC 通过结构检查和 Host/QNN 数值对拍、完成 checksum 备份后再删。**
