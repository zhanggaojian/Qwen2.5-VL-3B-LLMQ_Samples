# 06 · 附录D · Prepare 面试速答

> **定位**：本篇只保留 Prepare 阶段的面试结论。主线见 [06 · Prepare](./06-Prepare模型准备阶段.md)，dummy input 细节见 [附录A](./06-附录A-Prepare-Dummy-Input输入模具.md)，工具关系见 [附录B](./06-附录B-QAIRT-QNN-AIMET-QuantSim概念与关系.md)，完整转换证据见 [附录C](./06-附录C-QAIRT-model_preparer内部流程.md)。

## 一、30 秒总答

本项目的 Prepare 使用固定 shape 的 dummy input 跑通已经完成端侧适配的浮点 PyTorch 模型，由 QAIRT `model_preparer` 编排，经过 `ONNX → QuIR → QNNIR → Emitter`，最终重建出一个**输入输出扁平、算子显式、shape 固定、浮点计算语义基本等价**的 PyTorch 模型，供后续 AIMET QuantSim 使用。

```text
已适配的浮点 PyTorch 模型 + dummy input
                  │
                  ▼
        ONNX → QuIR → QNNIR
                  │
                  ▼ Emitter
        prepared PyTorch 浮点模型
                  │
                  ▼
          PPL 检查 → QuantSim
```

工具边界要说准确：整体流程由 **QAIRT `model_preparer`** 编排；Torch 导出 ONNX 使用 PyTorch 的导出能力；QuIR、QNNIR 和 Emitter 属于 QAIRT 内部转换链。

---

## 二、高频面试速答

### 1. Prepare 的目标是什么？

把“适合 Transformers 运行”的浮点模型，整理成“适合静态图、QuantSim 和后续端侧导出”的浮点模型。Prepare 本身**不是量化**，也不是生成 HTP 可执行文件。

### 2. dummy input 有什么特点？

- 当前输入：随机 token ID 或随机 embedding，数值没有真实语义；
- shape：使用真实部署规格，当前为 Current=1073、Past=975、Context=2048；
- 原始 2D padding mask：全 1；
- 最终 4D combined mask：不是全 0，而是带有 `0/-100` 的因果结构；
- RoPE：由位置 `0～1072` 确定性计算出的外部 `cos/sin`；
- 36 层 Past KV：全 0，K 为 `[1,2,128,975]`，V 为 `[1,2,975,128]`。

RoPE 计算简版：`head_dim=128` 对应 64 个固定旋转速度；每个位置分别乘这 64 个速度得到角度，再逐项计算 `cos(角度)` 和 `sin(角度)`：

```text
64 个旋转速度 ← head_dim=128、rope_theta=1,000,000
角度表         = 位置 [0...1072] × 旋转速度
cos/sin 表     = cos(角度表)、sin(角度表)
最终 shape     = [1, 1, 1073, 64]
```

> **一句话总结**：RoPE 表就是把每个 token 的位置乘上 64 个不同的固定转速，再对所得角度计算 `cos/sin`，供 Attention 旋转 Q、K；它不是随机数，也不是模型权重。

一句口诀：**输入随机、shape 全真、Mask 因果、RoPE 确定、KV 全零。** dummy 只负责建图，不是量化校准数据。

### 3. ONNX、QuIR、QNNIR 各自解决什么问题？

| 阶段 | 面试理解 |
|---|---|
| ONNX | 把动态 PyTorch 执行过程表达成标准静态计算图 |
| QuIR | QAIRT 内部的统一工作表示，便于做框架无关的图规范化与变换 |
| QNNIR | 进一步对齐 QNN 的算子、Tensor、布局和 I/O 契约 |
| Emitter | 把处理后的图重新生成可运行的 PyTorch 模型 |

QNNIR 仍是中间表示，**不是 DLC、Context Binary 或 HTP 机器码**。

这里是面向当前 QAIRT 2.42 产物的工作性理解；QuIR/QNNIR 的官方 schema 和逐 pass 清单仍应以对应版本 SDK 文档或源码为准。

### 4. 为什么同时需要 QuIR 和 QNNIR，一个 IR 不够吗？

因为职责不同：QuIR 先解决“不同前端如何统一表示”，QNNIR 再解决“如何符合 QNN 后端契约”。两级 IR 把通用图优化与后端适配分开，便于复用、排错和支持不同前后端。

### 5. 输入输出扁平是什么意思？

原模型接口可以嵌套：

```text
inputs = {
  embeds,
  position_ids: (cos, sin),
  past_key_values: ((K0,V0), (K1,V1))
}
```

Prepare 后变成有固定顺序的 Tensor 列表：

```text
forward(embeds, mask, cos, sin, K0, V0, K1, V1)
→ (logits, K0_new, V0_new, K1_new, V1_new)
```

当前 36 层模型是 **76 个输入 Tensor、73 个输出 Tensor**。扁平化改变的是接口组织方式，不是 Tensor 数值语义。

### 6. “算子显式”是什么意思？

原来写在 `forward` 表达式里的运算会变成可识别的模块节点。例如：

```python
x = x + residual
```

会整理成类似：

```python
self.Add = Add()
x = self.Add(x, residual)
```

这样 QuantSim 更容易找到量化插入点，后续 QNN 也更容易做算子映射。**显式化不等于已经量化。**

### 7. Tensor 不就是 shape 和 dtype 吗？

不止。计算图里的 Tensor 是算子之间传递数据的“边”，除 shape、dtype 外，还包含名字、维度语义、layout、生产者、消费者及输入输出端口等信息。算子是节点，Tensor 是带数据契约的连线。

### 8. Permute 是做什么的？

Permute 改变维度顺序，例如将 `[B,S,1,H]` 变成 Conv2d 需要的 `[B,H,1,S]`。它改变逻辑 layout，但不改变元素语义；逻辑换序也不代表此处一定立即发生物理内存复制。

### 9. 为什么说“激活 Tensor 连线映射”，不只说“算子连线”？

因为只说 A 连到 B，无法说明 A 的哪个输出端口接到 B 的哪个输入端口，也无法表达分支和复用。若同一个 Tensor 名同时出现在 `A.output[0]` 与 `B.input[0]`，就表示：

```text
A.output[0] ── Tensor T ──→ B.input[0]
```

这里映射的是**名称和端口关系**，不是保存运行时的激活数值。

### 10. Prepare 产物文件分别有什么用？

| 文件 | 作用 | 记忆方式 |
|---|---|---|
| `.py` | 定义重建后的模块和 `forward` | 怎么算 |
| `.json` | 记录旧模块到新模块/节点的来源关系 | 从哪来 |
| `_io_map.json` | 映射新旧参数名，并记录激活 Tensor 的端口连线 | 怎么对应 |
| `.safetensors` | 保存 prepared model 的完整实际权重 | 参数是多少 |

贯穿例子：旧的 `q_proj_conv` 在新图中可能变成 `Conv2d + Permute`；普通 `.json` 记录这组来源关系，`_io_map.json` 把新 Conv 的 weight/bias 对回旧参数并标出 Conv 与 Permute 之间的 Tensor，`.safetensors` 则保存新参数名下的实际数值。

### 11. prepared 权重和 Prepare 前有什么变化？

没有训练，也没有量化，因此核心权重学到的数值语义基本不变；变化主要是参数名、所属模块、保存对象，个别转换也可能需要 reshape/transpose。当前 `q_proj_conv` 前后逻辑 shape 一致，没有证据说明其权重被转置。

激活 layout 明确会因 Reshape/Permute 改变；权重是否物理重排要用完整 `.safetensors` 逐张量验证。最终 HTP 的物理内存排布由后续编译阶段决定，不能归到 Prepare。

另外，`.safetensors` 是**完整权重文件，不是根据 JSON 生成的差异包**；JSON 只能辅助名称对齐，真正比较仍要读取两边 Tensor。

### 12. 为什么 Prepare 后还要跑一次 PPL？

为了确认图重写没有明显破坏浮点数值语义。应在相同数据和前处理下比较 Prepare 前后的 PPL；结果应非常接近。PPL 是整体门禁，必要时再做 logits 或中间 Tensor 的逐层误差定位。

### 13. Prepare 阶段没有做什么？

它没有产生 INT4/INT8 权重，没有统计 activation encoding，没有做 SeqMSE，也没有生成 DLC、Context Binary 或 HTP 可执行产物；这些属于后续量化与编译阶段。

---

## 三、面试易错句纠正

| 容易说错 | 更准确的说法 |
|---|---|
| dummy input 全是 0 | 当前输入是随机值，只有 Past KV 是全 0 |
| mask 全 0 | 原始 2D mask 全 1；最终 combined mask 是 `0/-100` 因果结构 |
| 全程每一步都是 QAIRT 自己实现 | QAIRT 负责编排；Torch→ONNX 使用 PyTorch 导出能力 |
| QNNIR 就是 QNN 机器码 | QNNIR 仍是中间图表示 |
| Prepare 已经量化模型 | Prepare 输出仍是浮点模型 |
| `_io_map.json` 保存激活值 | 它保存激活 Tensor 名与模块端口的对应关系 |
| `.safetensors` 是权重差异 | 它保存完整 prepared 权重，不是 diff |
| 扁平化改变 Tensor 数值 | 它主要改变输入输出的容器、名字与顺序 |

---

## 四、10 秒记忆

> **Dummy 定模具，ONNX 画静态图，QuIR 统一语言，QNNIR 对齐 QNN，Emitter 重建 Torch，PPL 负责守门，QuantSim 接着量化。**

## 五、简短总结

Prepare 的核心不是“优化权重”，而是把已适配的浮点模型整理成一个**静态、扁平、显式、固定 shape、可被量化工具继续处理**的等价浮点图。
