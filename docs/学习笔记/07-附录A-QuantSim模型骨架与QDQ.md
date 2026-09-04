# 07-附录A · QuantSim 模型骨架与 QDQ

> **所属主篇**：[07 · 量化主流程：QuantSim 到 Encoding](./07-量化主流程-QuantSim到Encoding.md)
>
> **本篇范围**：只展开 `LLMForwardPassManager + QuantizationSimModel(...)` 这一段，解释对象关系、构造参数、QDQ 和真实部署边界。

---

## 一、当前代码

```python
sim_fpm = LLMForwardPassManager(
    cfg=llm_config,
    model=copy.deepcopy(prepared_model),
    tokenizer=tokenizer,
    separate_tuple_input_output=True,
    num_tokens=ARN,
)

with sim_fpm.place_on_device("cuda"):
    quantsim = QuantizationSimModel(
        model=sim_fpm.model,
        quant_scheme=getattr(QuantScheme, _quant['quant_scheme']),
        dummy_input=dummy_input,
        default_output_bw=_quant['default_output_bw'],
        default_param_bw=_quant['default_param_bw'],
        in_place=True,
        config_file=htp_config_file,
    )
```

一句话说明：

> **先深拷贝 prepared 浮点模型并交给 FPM 管理，再让 AIMET 根据 dummy input 和 HTP 配置给这份模型安装 Quantizer，建立后续 SeqMSE 与标定需要的假量化骨架。**

---

## 二、三个对象分别负责什么

```text
prepared_model
    │  Prepare 后的原始浮点模型，保留作浮点参考
    │
    └── deepcopy
          ▼
      sim_fpm.model
          │  真正被 AIMET 原地改造的模型
          ▼
      quantsim.model
             插入 Quantizer 后的假量化模型
```

| 对象 | 作用 |
|---|---|
| `prepared_model` | 保留未插入 Quantizer 的浮点基线 |
| `sim_fpm` | 准备定长输入、Mask、RoPE、扁平 KV，并循环运行前向 |
| `sim_fpm.model` | FPM 管理的模型副本，也是传给 AIMET 的模型 |
| `quantsim` | QuantSim 控制器，提供 `compute_encodings()`、`export()` 等接口 |
| `quantsim.model` | 实际带 Quantizer 的模型 |

因为设置了 `in_place=True`，AIMET 原地改造 `sim_fpm.model`。因此构造后：

```text
sim_fpm.model  ──运行──► 带 Quantizer 的模型
quantsim.model ──指向──► 同一份被改造后的模型
```

### 为什么需要 `deepcopy`

QuantSim 会改变模块结构。如果直接传入 `prepared_model`，浮点参考模型也会被改造，不方便做浮点对照、SeqMSE 参考和问题定位。

```text
prepared_model：保留浮点基线
模型深拷贝：允许 QuantSim 原地修改
```

### 创建 `sim_fpm` 是否已经跑了一遍前向

没有。`LLMForwardPassManager(...)` 这里只是创建管理器并保存模型、tokenizer 和定长配置。

真正执行前向的时机包括：

- QuantSim 使用 `dummy_input` 捕获计算图；
- SeqMSE 运行真实样本；
- `compute_encodings()` 运行校准样本；
- PPL 评估运行测试样本。

### `separate_tuple_input_output=True` 的作用

Prepare 后模型使用扁平 I/O：

```text
inputs_embeds, mask, cos, sin, K0, V0, K1, V1, ...
```

该开关让 FPM 按这套接口组织输入和输出。它是模型接口适配开关，不是量化开关。

---

## 三、`QuantizationSimModel` 每个参数做什么

| 参数 | 当前含义 | 重点 |
|---|---|---|
| `model=sim_fpm.model` | 改造 prepared 模型的深拷贝 | 传的是底层模型，不是整个 FPM |
| `quant_scheme=post_training_tf` | 使用 AIMET 的 PTQ/min-max 风格 Encoding 方案 | 名字中的 TF 不代表调用 TensorFlow |
| `dummy_input` | 用固定接口和 shape 捕获执行图 | 不是量化校准数据 |
| `default_output_bw=16` | 默认激活量化位宽为 16 bit | 后续配置和例外可以覆盖 |
| `default_param_bw=4` | 默认参数/权重量化位宽为 4 bit | 即默认 W4 |
| `in_place=True` | 原地修改传入的深拷贝模型 | 原始 `prepared_model` 不受影响 |
| `config_file=htp_config_file` | 应用 HTP v73 对应的量化规则 | 只是目标硬件 profile，不是 QNN 编译 |

当前默认目标可以记成：

```text
W4A16
W4  = weight 默认 4 bit
A16 = activation 默认 16 bit
```

它只是默认配置。MatMul、KV Cache、Concat 和混合精度规则还可能覆盖局部 Tensor 的位宽或 Encoding 关系。

### `place_on_device("cuda")` 做什么

QuantSim 构造时需要使用 dummy input 执行模型来捕获图，因此模型与输入必须在同一设备上。

```text
进入 with：模型临时放到 CUDA
with 内部：QuantSim 使用 dummy input 捕获图并改造模块
离开 with：模型恢复到上下文管理器记录的设备
```

这里仍然是 CUDA 上的 PyTorch/AIMET 离线处理，不代表已经在 HTP 上运行。

---

## 四、模型结构发生了什么变化

构造前是普通浮点模块：

```text
Conv2d
Add
MatMul
```

构造后，可量化模块会被转换或包装为带 Quantizer 的模块：

```text
QuantizedConv2d
 ├─ input_quantizers
 ├─ output_quantizers
 └─ param_quantizers["weight"]

QuantizedAdd
QuantizedMatMul
```

这些 Quantizer 描述“在哪里模拟量化、使用多少 bit、采用什么 Encoding”。AIMET 还会根据连接图和 HTP 配置决定哪些端口启用、禁用、共享或传播 Encoding。

---

## 五、QDQ 到底是什么

QuantSim 不会马上把整个模型打包成真正的 INT4 文件，而是在浮点计算路径中模拟低比特误差：

```text
浮点 Tensor x
      │
      │ Quantize：缩放、取整、截断到整数范围
      ▼
整数编码 q
      │
      │ Dequantize：映射回该整数代表的浮点网格值
      ▼
近似值 x_hat
```

常用公式：

```text
q     = clamp(round(x / scale) + offset)
x_hat = (q - offset) × scale
```

PyTorch 浮点算子最终接收的是 `x_hat`，因此模型仍能使用普通浮点 kernel，同时携带低比特量化误差。

### Q 和 DQ 是否会造成两次量化误差

通常不会。主要不可逆误差发生在 Q 的取整和截断：

```text
x = 1.27，scale = 0.1

Q ：round(1.27 / 0.1) = 13
DQ：13 × 0.1 = 1.3

总量化误差 = 1.3 - 1.27 = 0.03
```

DQ 只是把整数编码映射为它代表的网格值，不会再次做一次低比特取整。实际浮点计算可能存在很小的 FP 表示误差，但它不是第二次低比特量化误差。

真正会继续增加误差的是新的 Quantize/Requantize：

```text
Q → DQ → 浮点计算 → 再次 Q
                       ▲
                  又发生一次取整
```

因此需要 Encoding 传播、共享 Encoding 和 QDQ 融合，减少没有必要的重新量化边界。

### 真实部署是否需要反量化

取决于后续消费者：

| 后续路径 | 是否需要显式 DQ |
|---|---|
| 下一个算子支持量化输入 | 通常不需要转回浮点，编译器可融合量化链路 |
| 下一个算子只支持浮点 | 需要在边界处 DQ |
| 最终输出交给浮点 CPU/上层逻辑 | 通常需要 DQ |

导出的 QDQ/Encoding 主要向 QNN 描述量化边界和参数。QNN/HTP 编译时可以把连续 QDQ 融合为真正的量化 kernel，不会机械地在每一层都执行一次“整数转浮点”。

---

## 六、dummy input 和校准数据不要混淆

| 对比项 | `dummy_input` | calibration 数据 |
|---|---|---|
| 来源 | 固定 shape 的模拟输入 | `train_dataloader` 真实样本 |
| 主要目的 | 捕获执行图和确认接口 | 观察真实 activation 分布 |
| 是否要求代表真实分布 | 否 | 是 |
| 是否确定最终 activation Encoding | 否 | 主要由它确定 |

一句话：

> **dummy input 定图，真实 calibration 数据定量化范围。**

---

## 七、构造 QuantSim 后完成了什么

已经完成：

- 建立模型连接图和量化边界；
- 转换或包装可量化模块；
- 插入参数、输入和输出 Quantizer；
- 装载默认 W4A16 与 HTP profile；
- 得到可以继续配置、SeqMSE 和标定的假量化模型。

尚未完成：

- 尚未应用完整的 MatMul、Concat 和混合精度例外；
- 尚未通过 SeqMSE 优化权重 Encoding；
- 尚未通过真实数据完成 activation Encoding；
- 尚未计算量化后 PPL；
- 尚未得到最终打包的 INT4 权重或 HTP 可执行产物。

阶段边界可以记成：

> **QuantSim 构造负责安装和配置 Quantizer；SeqMSE 与 `compute_encodings()` 负责确定更具体的量化 Encoding。**

---

## 八、面试速记

### 为什么 QuantSim 需要 dummy input

为了执行固定路径、捕获模型连接关系，并确定 Quantizer 应插入在哪些 Tensor 边界；不是为了学习真实数据分布。

### 为什么使用 `copy.deepcopy(prepared_model)`

为了保留未量化的浮点参考，同时允许 `in_place=True` 原地改造模型副本。

### QuantSim 创建完成是否等于量化完成

不等于。此时只建立了假量化骨架，后面还需要规则配置、SeqMSE、真实数据标定、PPL 验证和导出。

### 一句话总结

> **`QuantizationSimModel(...)` 根据 dummy input 和 HTP 配置，在 prepared 模型副本中插入 QDQ Quantizer，建立浮点环境下的量化误差模拟模型；它只是量化流程的起点，不是最终量化产物。**

---

## 九、源码与关联笔记

- `example1/llm_quant.py`：QuantSim 构造约 L437～462；后续规则约 L464～478。
- `example1/llm_utils/forward_pass_wrapper.py`：`place_on_device()` 约 L261～268。
- [05 · LLMForwardPassManager](./05-通用前向处理流程.md)
- [06 · Prepare 模型准备阶段](./06-Prepare模型准备阶段.md)
- [AIMET QuantizationSimModel 官方说明](https://quic.github.io/aimet-pages/releases/2.26.0/techniques/qat.html)
