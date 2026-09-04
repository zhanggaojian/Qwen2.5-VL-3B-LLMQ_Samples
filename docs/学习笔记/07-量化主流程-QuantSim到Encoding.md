# 07 · 量化主流程：QuantSim 到 Encoding

> **流程位置**：承接 [06 · Prepare 模型准备阶段](./06-Prepare模型准备阶段.md)，从 prepared 浮点模型出发，完成量化模拟、Encoding 优化、精度验证和导出。
>
> **本篇范围**：只介绍量化流程。QuantSim、QDQ、Encoding、SeqMSE 等具体原理统一放到 07 系列附录。
>
> **一句话本质**：给 prepared 浮点模型安装 Quantizer，用真实数据确定权重和激活的 Encoding，在 PyTorch 中模拟 W4A16 量化误差并验证精度。

---

## 一、完整流程图

主线代码位于 `example1/llm_quant.py` 约 L437～603：

```text
Prepare 后的浮点模型 prepared_model
                │
                ▼
① deepcopy + LLMForwardPassManager
   保留浮点基线，管理量化模型副本的定长前向
                │
                ▼
② QuantizationSimModel
   根据 dummy input 捕获图，插入 QDQ Quantizer
                │
                ▼
③ 配置量化规则
   默认 W4A16 + HTP profile + 局部混合精度规则
                │
                ▼
④ SeqMSE
   用真实样本优化权重 Encoding
                │
                ▼
⑤ compute_encodings
   用真实校准样本确定激活 Encoding，并计算可覆盖的参数 Encoding
                │
                ▼
⑥ 量化后 PPL
   对比浮点 PPL，检查量化精度损失
                │
                ▼
⑦ 导出
   ONNX + Quantization Encodings + 测试向量
                │
                ▼
后续 QNN 编译与 HTP/NPU 部署
```

---

## 二、每个阶段发生了什么

| 阶段 | 输入 | 核心动作 | 产物 |
|---|---|---|---|
| ① 创建 `sim_fpm` | `prepared_model` | 深拷贝模型，建立定长前向管理器 | 独立的浮点模型副本 |
| ② 创建 QuantSim | 模型副本 + `dummy_input` | 捕获固定图并插入 Quantizer | 假量化模型骨架 |
| ③ 配置规则 | QuantSim 模型 | 应用 HTP、MatMul/KV、Concat 和混合精度规则 | 最终量化配置结构 |
| ④ SeqMSE | 浮点参考 + QuantSim + 真实样本 | 搜索更合适的权重量化范围 | weight Encoding |
| ⑤ `compute_encodings` | QuantSim + 校准集 | 前向统计 activation；直接从参数张量计算或重算允许覆盖的 parameter Encoding | activation 与可覆盖的 parameter Encoding |
| ⑥ PPL 验证 | 量化模拟模型 + 测试集 | 与浮点 PPL 对比 | 精度损失结果 |
| ⑦ 导出 | 已完成 Encoding 的模型 | 固化计算图与量化描述 | [ONNX、Encoding、测试向量](./08-ONNX导出与测试向量.md) |

---

## 三、主流程需要记住的边界

- `LLMForwardPassManager(...)` 只是创建前向管理器，不代表已经跑完一遍真实数据。
- `dummy_input` 用来确定输入结构、shape 和计算图；真实 calibration 数据才用来确定量化范围。
- 当前以 W4A16 为默认方案，但 MatMul、KV Cache、Concat 和混合精度配置可以覆盖局部规则。
- QuantSim 阶段得到的是带 QDQ 的 PyTorch 假量化模型，不是已经打包好的 INT4 端侧模型。
- SeqMSE 优化并冻结受支持层的权重 Encoding；`compute_encodings()` 完成真实数据下的 activation Encoding，并从参数张量计算或重算其他允许覆盖的 parameter Encoding。
- 当前代码在标定后计算并打印 PPL，随后继续导出，没有自动精度门禁；导出的 ONNX 与 Encoding 还需要经过 QNN 编译才能在 HTP/NPU 上真实执行。

模型主线可以简记为：

```text
prepared_model
  → 深拷贝浮点模型
  → QuantSim 假量化模型
  → 完成 Encoding 的假量化模型
  → ONNX + Encoding
  → QNN/HTP 真实量化执行
```

---

## 四、主线代码定位

| 流程 | `example1/llm_quant.py` 位置 |
|---|---:|
| 创建 `sim_fpm` 与 QuantSim | 约 L437～462 |
| MatMul、Concat、混合精度规则 | 约 L464～478 |
| SeqMSE | 约 L480～527 |
| `compute_encodings` | 约 L534～568 |
| 量化后 PPL | 约 L570～577 |
| 测试向量与导出 | 约 L579～603 |

---

## 五、07 系列附录

| 附录 | 详细内容 | 状态 |
|---|---|---|
| [07-附录A · QuantSim 模型骨架与 QDQ](./07-附录A-QuantSim模型骨架与QDQ.md) | FPM 对象关系、QuantSim 参数、QDQ、真实部署是否反量化 | 已建立 |
| [07-附录B · Encoding 量化参数基础](./07-附录B-Encoding量化参数基础.md) | Encoding、scale、offset、对称与非对称量化 | 已建立 |
| [07-附录C · MatMul、Concat 与混合精度](./07-附录C-量化规则配置-MatMul-Concat与混合精度.md) | KV 8bit、Concat Encoding 共享与手工例外规则 | 已建立 |
| [07-附录D · SeqMSE 权重量化优化](./07-附录D-SeqMSE权重量化优化.md) | 两种前向回调、参数含义、逐层候选搜索与执行后清理 | 已建立 |
| [07-附录E · `compute_encodings()` 激活标定](./07-附录E-compute_encodings激活标定.md) | 两种校准回调、Quantizer 观察、切块 KV 路径及当前单块覆盖边界 | 已建立 |
| [07-附录F · 量化方法总览与选型](./07-附录F-量化方法总览与选型.md) | 量化对象、PTQ/QAT、静态/动态、粒度、常见算法与项目选型 | 已建立 |

相关主篇：[05 · LLMForwardPassManager](./05-通用前向处理流程.md)、[06 · Prepare](./06-Prepare模型准备阶段.md)、[04 · PPL](./04-PPL困惑度评估.md)、[08 · ONNX 导出与测试向量](./08-ONNX导出与测试向量.md)。

---

## 六、一句话总结

> **量化主流程就是：复制 prepared 浮点模型 → 创建 QuantSim → 配置量化规则 → SeqMSE 优化权重 → 真实数据标定激活 → PPL 验证 → 导出 ONNX 与 Encoding → 交给 QNN 编译部署。**
