# 第七章 量化 —— 从 FP16 到 INT4

> *放弃指数：⭐⭐⭐⭐ 本章需要理解数值表示和误差传播，建议动手实验*

---

## 7.1 PTQ vs QAT：训练后量化 vs 量化感知训练

### 为什么需要量化？

```
┌─────────────────────────────────────────────────────────────┐
│                    量化的收益与代价                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  收益：                                                      │
│  1. 模型体积减小：FP16 → INT8 → INT4                         │
│     70B 模型：140GB → 70GB → 35GB                            │
│                                                             │
│  2. 推理速度提升：                                            │
│     INT8 计算：GPU Tensor Core 加速                          │
│     INT4 计算：更高的吞吐                                     │
│                                                             │
│  3. 显存占用降低：                                            │
│     更大的 batch size                                        │
│     更长的上下文                                             │
│                                                             │
│  4. 成本节省：                                                │
│     单卡服务更大模型                                          │
│     或用更便宜的硬件                                          │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  代价：                                                      │
│  1. 精度损失：模型变"傻"                                      │
│  2. 实现复杂：量化方法多样                                    │
│  3. 兼容性：部分框架/硬件不支持                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 数值表示基础

在讨论量化之前，必须理解数值在计算机中的表示方式：

```
┌─────────────────────────────────────────────────────────────┐
│                    浮点数表示                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FP32 (Single Precision):                                   │
│  [符号位 1bit][指数位 8bits][尾数位 23bits]                   │
│  范围: ±3.4 × 10^38                                         │
│  精度: ~7 位有效数字                                         │
│  应用: 训练默认精度                                          │
│                                                             │
│  FP16 (Half Precision):                                     │
│  [符号位 1bit][指数位 5bits][尾数位 10bits]                   │
│  范围: ±65504                                               │
│  精度: ~3 位有效数字                                         │
│  应用: 训练常用，推理默认                                     │
│                                                             │
│  BF16 (BFloat16):                                           │
│  [符号位 1bit][指数位 8bits][尾数位 7bits]                    │
│  范围: 与 FP32 相同                                          │
│  精度: ~2-3 位有效数字                                       │
│  应用: 训练更稳定，避免溢出                                   │
│                                                             │
│  FP8 (E4M3 / E5M2):                                         │
│  E4M3: [符号位 1bit][指数位 4bits][尾数位 3bits]              │
│  E5M2: [符号位 1bit][指数位 5bits][尾数位 2bits]              │
│  应用: H100+ 推理                                            │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  定点数表示：                                                 │
│                                                             │
│  INT8:                                                     │
│  范围: -128 ~ +127 (有符号)                                  │
│       0 ~ 255 (无符号)                                       │
│                                                             │
│  INT4:                                                     │
│  范围: -8 ~ +7 (有符号)                                      │
│       0 ~ 15 (无符号)                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键概念：精度 vs 范围**

```
FP16 问题：
- 范围有限：max = 65504
- 梯度容易溢出

BF16 解决：
- 范围与 FP32 相同
- 精度降低但训练稳定

量化挑战：
- INT8 范围极小
- 需要仔细处理分布
```

---

### 量化的基本原理

**量化公式：**

```
量化：Q = round((R - zero_point) / scale)
反量化：R = Q × scale + zero_point

其中：
- R: 原始浮点数（Real）
- Q: 量化后的整数（Quantized）
- scale: 缩放因子
- zero_point: 零点偏移
```

**对称 vs 非对称量化：**

```
┌─────────────────────────────────────────────────────────────┐
│                    量化类型对比                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  对称量化（Symmetric）：                                      │
│                                                             │
│  FP16:     -2.0  -1.0   0.0   1.0   2.0                     │
│              ↓    ↓     ↓     ↓     ↓                       │
│  INT8:    -128  -64     0    64   128                       │
│                                                             │
│  公式：Q = round(R / scale)                                  │
│  scale = max(|R_max|, |R_min|) / 127                        │
│  zero_point = 0                                             │
│                                                             │
│  优点：实现简单，计算效率高                                   │
│  缺点：不适用于非对称分布                                     │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  非对称量化（Asymmetric）：                                   │
│                                                             │
│  FP16:      0.0   0.5   1.0   1.5   2.0                     │
│              ↓     ↓     ↓     ↓     ↓                       │
│  INT8:       0    64   128   192   255                       │
│                                                             │
│  公式：Q = round((R - R_min) / scale)                        │
│  scale = (R_max - R_min) / 255                              │
│  zero_point = round(-R_min / scale)                          │
│                                                             │
│  优点：精度更高，适用范围广                                   │
│  缺点：计算稍复杂                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### PTQ：Post-Training Quantization

**训练后量化**：在模型训练完成后进行量化，不需要重新训练。

```
┌─────────────────────────────────────────────────────────────┐
│                      PTQ 流程                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 训练模型（FP32/FP16）                                    │
│     │                                                       │
│     ▼                                                       │
│  2. 收集校准数据（Calibration Data）                         │
│     - 选取代表数据样本（100-500 个）                         │
│     - 前向传播，收集各层激活分布                             │
│     │                                                       │
│     ▼                                                       │
│  3. 确定 scale 和 zero_point                                │
│     - Min-Max：使用最大最小值                                │
│     - Percentile：使用百分位数                               │
│     - KL Divergence：最小化分布差异                          │
│     │                                                       │
│     ▼                                                       │
│  4. 量化权重和激活                                           │
│     │                                                       │
│     ▼                                                       │
│  5. 部署量化模型                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**PTQ 代码示例：**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PTQQuantizer:
    def __init__(self, model_name, calibration_samples=200):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.calibration_samples = calibration_samples
        self.calibration_data = []
    
    def collect_calibration_data(self, dataset):
        """收集校准数据"""
        self.model.eval()
        
        with torch.no_grad():
            for i, sample in enumerate(dataset[:self.calibration_samples]):
                inputs = self.tokenizer(sample, return_tensors="pt")
                outputs = self.model(**inputs)
                
                # 收集各层激活值
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # 注册 hook 收集激活
                        pass
    
    def compute_scale_zero_point(self, tensor, method="minmax"):
        """计算 scale 和 zero_point"""
        if method == "minmax":
            r_min = tensor.min().item()
            r_max = tensor.max().item()
        elif method == "percentile":
            r_min = torch.quantile(tensor.flatten(), 0.01).item()
            r_max = torch.quantile(tensor.flatten(), 0.99).item()
        
        # 对称量化
        scale = max(abs(r_min), abs(r_max)) / 127
        zero_point = 0
        
        return scale, zero_point
    
    def quantize_weight(self, weight, scale, zero_point=0):
        """量化权重"""
        quantized = torch.round(weight / scale)
        quantized = torch.clamp(quantized, -128, 127).to(torch.int8)
        return quantized
    
    def dequantize_weight(self, quantized, scale, zero_point=0):
        """反量化权重"""
        return quantized.to(torch.float32) * scale + zero_point


# 实际使用
quantizer = PTQQuantizer("meta-llama/Llama-2-7b-hf")
quantizer.collect_calibration_data(calibration_dataset)
quantized_model = quantizer.quantize()
```

---

### QAT：Quantization-Aware Training

**量化感知训练**：在训练过程中模拟量化效果，让模型适应量化误差。

```
┌─────────────────────────────────────────────────────────────┐
│                      QAT 流程                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 从预训练模型开始                                         │
│     │                                                       │
│     ▼                                                       │
│  2. 插入量化/反量化节点（Fake Quantization）                  │
│     ┌─────────────────────────────────────────────────────┐ │
│     │                                                     │ │
│     │  FP32 Weight → Quantize → INT8 → Dequantize → FP32   │ │
│     │                              ↑                        │ │
│     │                     直通估计器（STE）                  │ │
│     │                              ↓                        │ │
│     │                     梯度反向传播                       │ │
│     │                                                     │ │
│     └─────────────────────────────────────────────────────┘ │
│     │                                                       │
│     ▼                                                       │
│  3. 继续训练（Fine-tune）                                    │
│     - 量化误差被训练过程吸收                                 │
│     - 前向传播模拟量化效果                                   │
│     - 反向传播使用 STE                                       │
│     │                                                       │
│     ▼                                                       │
│  4. 导出量化模型                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**直通估计器：**

```python
class FakeQuantize(torch.autograd.Function):
    """前向量化，反向直通"""
    
    @staticmethod
    def forward(ctx, x, scale, zero_point):
        # 前向：量化 + 反量化
        quantized = torch.round((x - zero_point) / scale)
        quantized = torch.clamp(quantized, -128, 127)
        dequantized = quantized * scale + zero_point
        
        # 保存用于反向传播
        ctx.scale = scale
        ctx.zero_point = zero_point
        
        return dequantized
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向：直接传递梯度（直通估计器）
        return grad_output, None, None


class QATLinear(torch.nn.Module):
    """QAT 线性层"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.scale = 1.0
        self.zero_point = 0
    
    def forward(self, x):
        # 权重量化（训练时）
        if self.training:
            weight_q = FakeQuantize.apply(self.weight, self.scale, self.zero_point)
        else:
            weight_q = self.weight
        
        return torch.nn.functional.linear(x, weight_q, self.bias)
```

---

### PTQ vs QAT 对比

| 维度 | PTQ | QAT |
|------|-----|-----|
| 训练成本 | 低（无需训练） | 高（需要训练） |
| 精度损失 | 较大 | 较小 |
| 适用场景 | 已有模型快速部署 | 新模型/高精度需求 |
| 实现复杂度 | 简单 | 复杂 |
| 时间成本 | 分钟级 | 小时级 |
| INT8 效果 | 较好 | 好 |
| INT4 效果 | 较差 | 尚可 |

**选择建议：**

```
低精度需求（INT8）：PTQ 足够
高精度需求（INT4）：QAT 更优
时间紧迫：PTQ
追求最佳性能：QAT
```

---

## 7.2 GPTQ、AWQ、GGUF：量化方法百花齐放

### GPTQ：基于 Hessian 的量化

**GPTQ** 是目前最流行的 LLM 量化方法之一，基于 Optimal Brain Quantization（OBQ）。

```
┌─────────────────────────────────────────────────────────────┐
│                    GPTQ 原理                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  核心思想：                                                   │
│  逐层量化权重，最小化量化误差对输出的影响                      │
│                                                             │
│  关键洞察：                                                   │
│  量化权重 w_i 对输出误差的影响取决于 Hessian 矩阵 H            │
│                                                             │
│  算法：                                                      │
│  1. 计算 Hessian 矩阵 H = 2X^T X                             │
│  2. 对每个权重：                                              │
│     a. 找到使量化误差最小的量化值                            │
│     b. 更新未量化权重补偿误差                                │
│     c. 更新 Hessian 逆矩阵                                   │
│                                                             │
│  优势：                                                      │
│  - 理论上最优的逐层量化                                      │
│  - INT4 精度保持良好                                         │
│  - 已被 vLLM、AutoGPTQ 支持                                  │
│                                                             │
│  劣势：                                                      │
│  - 量化时间较长（70B 模型需要几小时）                         │
│  - 需要校准数据                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**GPTQ 实践：**

```python
# 安装
# pip install auto-gptq optimum

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# 配置量化
quantization_config = GPTQConfig(
    bits=4,                    # INT4
    dataset="c4",              # 校准数据集
    group_size=128,            # 分组量化
    desc_act=False,            # 不使用激活顺序
)

# 量化和加载
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# 保存
model.save_pretrained("./llama-2-7b-gptq-int4")
tokenizer.save_pretrained("./llama-2-7b-gptq-int4")


# CLI 方式（推荐）
# pip install auto-gptq

# 量化命令
"""
python -m auto_gptq \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./llama-2-7b-gptq-int4 \
    --bits 4 \
    --group_size 128 \
    --dataset c4 \
    --num_samples 512
"""
```

---

### AWQ：Activation-aware Weight Quantization

**AWQ** 基于一个关键观察：**并非所有权重都同等重要**。

```
┌─────────────────────────────────────────────────────────────┐
│                    AWQ 原理                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  关键洞察：                                                   │
│  1% 的权重对输出影响很大（salient weights）                   │
│  这些权重对应高激活通道                                      │
│                                                             │
│  方法：                                                      │
│  1. 分析激活分布，找出 salient channels                      │
│  2. 对 salient weights 保持高精度或特殊处理                  │
│  3. 对其他权重量化                                          │
│                                                             │
│  数学表达：                                                   │
│  Y = W × X                                                  │
│  ∂Y/∂W = X (激活值)                                         │
│  激活值大 → 权重敏感 → 需要高精度                            │
│                                                             │
│  策略：                                                      │
│  - 不量化 salient 权重，或                                   │
│  - 对 salient channel 整体缩放                              │
│                                                             │
│  优势：                                                      │
│  - INT4 精度优于 GPTQ                                       │
│  - 量化速度快（无需复杂计算）                                │
│  - 无需校准数据的复杂处理                                    │
│                                                             │
│  劣势：                                                      │
│  - 需要激活分布分析                                         │
│  - 对特殊分布的模型效果可能有限                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**AWQ 实践：**

```python
# 安装
# pip install autoawq

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加载模型
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 配置量化
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
}

# 量化
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval"  # 校准数据
)

# 保存
model.save_quantized("./llama-2-7b-awq-int4")
tokenizer.save_pretrained("./llama-2-7b-awq-int4")


# CLI 方式
"""
awq_quantize \
    --model_path meta-llama/Llama-2-7b-hf \
    --output_path ./llama-2-7b-awq-int4 \
    --w_bit 4 \
    --q_group_size 128 \
    --calib_data pileval
"""
```

---

### GGUF：llama.cpp 的量化格式

**GGUF** 是 llama.cpp 使用的量化格式，专注于 CPU/Apple Silicon 推理。

```
┌─────────────────────────────────────────────────────────────┐
│                    GGUF 特点                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  设计目标：                                                   │
│  - 在消费级硬件上运行大模型                                  │
│  - CPU / Apple Silicon 推理优化                             │
│  - 单文件，易于分发                                          │
│                                                             │
│  量化类型：                                                   │
│  - Q4_0: 4-bit，无缩放                                       │
│  - Q4_K_M: 4-bit，K-quant，中等精度                         │
│  - Q5_K_M: 5-bit，K-quant，中等精度                         │
│  - Q6_K: 6-bit，K-quant                                     │
│  - Q8_0: 8-bit，无缩放                                       │
│                                                             │
│  推荐选择：                                                   │
│  - Q4_K_M: 平衡精度和体积（推荐）                            │
│  - Q5_K_M: 高精度需求                                        │
│  - Q8_0: 最高精度                                            │
│                                                             │
│  文件大小对比（7B 模型）：                                    │
│  FP16: ~14 GB                                               │
│  Q8_0: ~7.2 GB                                              │
│  Q5_K_M: ~5.1 GB                                            │
│  Q4_K_M: ~4.1 GB                                            │
│  Q4_0: ~3.8 GB                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**GGUF 实践：**

```bash
# 下载 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 转换模型为 GGUF
python convert.py /path/to/llama-2-7b --outfile llama-2-7b-f16.gguf --outtype f16

# 量化
./quantize llama-2-7b-f16.gguf llama-2-7b-q4_k_m.gguf Q4_K_M

# 运行
./main -m llama-2-7b-q4_k_m.gguf -p "Hello, world!" -n 128

# 或使用 Python 绑定
# pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama(
    model_path="./llama-2-7b-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=8
)

output = llm("Q: Hello, world! A:", max_tokens=128)
print(output)
```

---

### 量化方法对比

```
┌─────────────────────────────────────────────────────────────┐
│                  GPTQ vs AWQ vs GGUF                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              精度对比（INT4）                          │  │
│  │                                                      │  │
│  │        FP16 ████████████████████████████████ 100%   │  │
│  │        AWQ  ████████████████████████████░░░░  95%   │  │
│  │       GPTQ  ███████████████████████████░░░░░  92%   │  │
│  │       GGUF  ██████████████████████████░░░░░░  90%   │  │
│  │              (以 MMLU 为例)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              特性对比                                 │  │
│  │                                                      │  │
│  │        GPTQ        AWQ         GGUF                 │  │
│  │ 精度     高         更高        中                   │  │
│  │ 速度     快         更快        中（CPU）            │  │
│  │ 量化时间 长         短          中                   │  │
│  │ GPU支持  好         好          一般                 │  │
│  │ CPU支持  一般       一般        极好                 │  │
│  │ 易用性   中         高          高                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  选择建议：                                                   │
│  - GPU 推理：AWQ（首选）或 GPTQ                              │
│  - CPU/Mac 推理：GGUF                                        │
│  - 最高精度：AWQ                                             │
│  - 最易使用：GGUF                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 量化实践：vLLM 部署量化模型

```python
# 安装 vLLM
# pip install vllm

from vllm import LLM, SamplingParams

# 加载 AWQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    tensor_parallel_size=1
)

# 加载 GPTQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    tensor_parallel_size=1
)

# 推理
prompts = [
    "The capital of France is",
    "The largest planet in our solar system is"
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")
```

---

## 7.3 FP8：新标准还是过渡方案？

### FP8 格式详解

**FP8** 是 NVIDIA H100 引入的新格式，有两种变体：

```
┌─────────────────────────────────────────────────────────────┐
│                      FP8 格式                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  E4M3 (适合权重/激活)：                                       │
│  ┌───┬────────┬────────┐                                   │
│  │ S │ EEEE   │ MMM    │                                    │
│  │ 1 │ 4 bits │ 3 bits │                                    │
│  └───┴────────┴────────┘                                   │
│                                                             │
│  范围: -448 ~ +448                                          │
│  精度: 更高的精度                                            │
│  应用: 权重、激活值                                          │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  E5M2 (适合梯度)：                                           │
│  ┌───┬─────────┬───────┐                                   │
│  │ S │ EEEEE   │ MM    │                                    │
│  │ 1 │ 5 bits  │ 2 bits│                                    │
│  └───┴─────────┴───────┘                                   │
│                                                             │
│  范围: -57344 ~ +57344                                      │
│  精度: 更大的范围                                            │
│  应用: 梯度                                                 │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  为什么需要两种格式？                                         │
│                                                             │
│  权重/激活：精度更重要 → E4M3                                │
│  梯度：范围更重要 → E5M2                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### FP8 vs INT8

```
┌─────────────────────────────────────────────────────────────┐
│                    FP8 vs INT8 对比                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INT8：                                                      │
│  - 整数表示，均匀量化                                        │
│  - 需要仔细设计 scale 和 zero_point                          │
│  - 对非均匀分布效果差                                        │
│  - 需要额外的缩放因子存储                                   │
│                                                             │
│  FP8：                                                       │
│  - 浮点表示，动态范围                                        │
│  - 自适应精度                                                │
│  - 对任意分布效果更好                                        │
│  - 更符合神经网络参数分布                                    │
│                                                             │
│  性能对比（H100）：                                           │
│                                                             │
│  │                                                          │
│  │   算力                                                   │
│  │    ↑     FP8 ████████████████████████████ 1979 TFLOPS   │
│  │         INT8 ████████████████ 1979 TFLOPS               │
│  │         FP16 ██████████ 989 TFLOPS                      │
│  │    └───────────────────────────────────────→             │
│  │                                                          │
│  带宽优势：                                                  │
│  FP16 → FP8: 带宽需求减半                                   │
│  更大的 batch size，更长的上下文                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### FP8 量化实践

```python
import torch
from torch.float8 import float8_e4m3fn, float8_e5m2

# FP8 类型
fp8_e4m3 = torch.float8_e4m3fn  # E4M3
fp8_e5m2 = torch.float8_e5m2    # E5M2

# 转换
fp16_tensor = torch.randn(1024, 1024, dtype=torch.float16)
fp8_tensor = fp16_tensor.to(fp8_e4m3)

# 反量化
fp16_back = fp8_tensor.to(torch.float16)


# Transformer Engine（NVIDIA 官方库）
# pip install transformer-engine

import transformer_engine as te
from transformer_engine.common.recipe import Format, DelayedScaling

# 配置 FP8
fp8_recipe = DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=Format.HYBRID  # E4M3 for activations, E5M2 for gradients
)

# 使用 FP8 训练
model = te.Linear(4096, 4096)

with te.fp8_autocast(fp8_recipe=fp8_recipe):
    output = model(input)


# vLLM FP8 推理
from vllm import LLM

llm = LLM(
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    quantization="fp8",
    tensor_parallel_size=1
)
```

---

### FP8 的挑战

```
挑战 1：精度损失
- FP8 精度比 FP16/BF16 低
- 某些模型/任务对精度敏感
- 需要 calibration 或 fine-tune

挑战 2：硬件依赖
- 仅 H100+ 支持 FP8
- A100 及更早硬件不支持
- 限制了适用范围

挑战 3：软件生态
- 框架支持有限
- 调试工具不足
- 最佳实践还在探索

挑战 4：动态范围溢出
- E4M3 范围有限（-448 ~ +448）
- 需要缩放因子
- scaling 策略需要设计
```

---

### FP8：新标准还是过渡方案？

```
┌─────────────────────────────────────────────────────────────┐
│                    FP8 前景分析                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  支持 FP8 的观点：                                           │
│                                                             │
│  1. 硬件趋势：                                               │
│     - NVIDIA H100/H200                                                                                      │
│     - AMD MI300                                             │
│     - Intel Gaudi2                                          │
│     等新一代芯片都在支持 FP8                                 │
│                                                             │
│  2. 行业标准：                                               │
│     - OCP（开放计算项目）标准化                              │
│     - MLPerf 接受 FP8 结果                                  │
│     - 主流框架开始支持                                      │
│                                                             │
│  3. 实际收益：                                               │
│     - 2× 存储节省                                           │
│     - 2× 带宽效率                                           │
│     - 训练和推理都受益                                      │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  质疑 FP8 的观点：                                           │
│                                                             │
│  1. INT8 更成熟：                                            │
│     - 量化方法丰富                                           │
│     - 精度损失可控                                          │
│     - 硬件支持广泛                                          │
│                                                             │
│  2. FP8 可能是过渡：                                         │
│     - < 8-bit 量化在研究中                                  │
│     - INT4 实际应用可行                                     │
│     - 未来可能有 FP4                                        │
│                                                             │
│  3. 精度敏感任务：                                           │
│     - INT4 对某些任务的精度损失过大                         │
│     - 下一步可能是 FP4 或混合精度                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**结论：当前阶段建议**

```
有 H100/AMD MI300：
  → FP8 是好选择，训练和推理都受益

A100 或更早硬件：
  → INT8/INT4 量化 + AWQ/GPTQ

跨硬件部署：
  → FP8 作为主格式，INT8/INT4 作为备选

未来展望：
  → FP8 可能成为主流
  → INT4 在边缘/成本敏感场景仍重要
```

---

## 7.4 量化后模型变傻了怎么办？

### 量化精度损失诊断

```
┌─────────────────────────────────────────────────────────────┐
│                   量化精度损失诊断流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 基准测试                                           │
│  ┌────────────────────────────────────────────────────────┐│
│  │  测试 FP16 模型在关键任务上的表现                        ││
│  │  - MMLU（知识能力）                                      ││
│  │  - HumanEval（代码）                                     ││
│  │  - GSM8K（数学推理）                                     ││
│  │  - 业务专属测试集                                        ││
│  │                                                         ││
│  │  记录：accuracy, latency, throughput                    ││
│  └────────────────────────────────────────────────────────┘│
│                         ↓                                   │
│  Step 2: 量化测试                                           │
│  ┌────────────────────────────────────────────────────────┐│
│  │  测试量化模型，对比 FP16 基准                            ││
│  │                                                         ││
│  │  精度保留率 = Quantized_Acc / FP16_Acc                  ││
│  │                                                         ││
│  │  一般期望：                                              ││
│  │  - INT8: > 95% 保留                                     ││
│  │  - INT4: > 90% 保留                                     ││
│  └────────────────────────────────────────────────────────┘│
│                         ↓                                   │
│  Step 3: 层级分析                                           │
│  ┌────────────────────────────────────────────────────────┐│
│  │  找出精度损失最大的层                                    ││
│  │                                                         ││
│  │  方法：                                                  ││
│  │  1. 逐层计算量化误差                                    ││
│  │  2. 激活分布分析                                        ││
│  │  3. 权重敏感度分析                                      ││
│  │                                                         ││
│  │  工具：                                                  ││
│  │  - torch.autograd.gradcheck                            ││
│  │  - 权重扰动测试                                         ││
│  └────────────────────────────────────────────────────────┘│
│                         ↓                                   │
│  Step 4: 针对性优化                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 精度损失的常见原因

```
原因 1：激活分布不均

┌─────────────────────────────────────────────────────────────┐
│                    激活分布问题                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题：某些层的激活值分布极度不均匀                          │
│                                                             │
│  理想分布：                                                  │
│  │                                                          │
│  │     ▄▄▄▄▄▄▄                                             │
│  │    ▄████████▄                                           │
│  │   ▄██████████▄                                          │
│  │  ▄████████████▄                                         │
│  │ ──────────────────→                                     │
│  │ -10  0  10                                              │
│                                                             │
│  问题分布：                                                  │
│  │                                                          │
│  │ ▄                                                        │
│  │ █                                            ▄▄         │
│  │ █                                           ▄██▄        │
│  │ █▄▄                                     ▄▄▄▄█████▄      │
│  │ ────────────────────────────────────────────────→       │
│  │ -100                   0                   10000        │
│                                                             │
│  对称量化会把大部分信息丢失                                  │
│                                                             │
│  解决：使用非对称量化或 per-channel 量化                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```
原因 2：权重异常值

问题：某些权重值远大于其他值（outliers）
这些权重对模型输出影响巨大

解决：
1. SmoothQuant：平滑激活和权重
2. AWQ：识别并保护 salient weights
3. 混合精度：关键层保持 FP16
```

```
原因 3：量化粒度不当

问题：per-tensor 量化精度损失大
per-channel 量化显存开销大

解决：
1. per-group 量化（group_size=128）
2. 根据层重要性选择不同粒度
```

---

### 精度恢复策略

**策略 1：混合精度量化**

```python
# 不同层使用不同精度
def mixed_precision_quantize(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 分析层重要度
            importance = analyze_layer_importance(module)
            
            if importance > 0.9:
                # 关键层：保持 FP16
                continue
            elif importance > 0.7:
                # 重要层：INT8
                quantize_to_int8(module)
            else:
                # 一般层：INT4
                quantize_to_int4(module)
    
    return model
```

**策略 2：量化后微调**

```python
# QAT 微调恢复精度
def qat_finetune(model, calibration_data, epochs=3):
    # 1. 插入量化节点
    model = insert_quant_nodes(model)
    
    # 2. 微调
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for batch in calibration_data:
            outputs = model(batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # 3. 导出量化模型
    return export_quantized_model(model)
```

**策略 3：SmoothQuant**

```python
# SmoothQuant: 平滑激活和权重
def smooth_quant(model, calibration_data):
    """
    核心公式：
    Y = (X / s) @ (W × s)
    
    通过缩放因子 s，将激活的难度转移到权重
    """
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 收集激活
            activations = collect_activations(module, calibration_data)
            
            # 计算缩放因子
            x_max = activations.abs().max()
            w_max = module.weight.abs().max()
            
            # 平衡因子
            smooth_scale = (x_max.pow(0.5) / w_max.pow(0.5)).clamp(min=1e-5)
            
            # 缩放
            module.weight.data = module.weight.data * smooth_scale
            # 相应缩放上一层的输出（需要处理层间依赖）
    
    return model
```

**策略 4：权重裁剪**

```python
# 裁剪异常权重值
def clip_weights(model, percentile=99.9):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 计算裁剪阈值
            threshold = torch.quantile(param.abs(), percentile / 100)
            
            # 裁剪
            param.data = torch.clamp(param.data, -threshold, threshold)
    
    return model
```

---

### 精度损失排查清单

```
┌─────────────────────────────────────────────────────────────┐
│                   精度损失排查清单                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  □ 检查量化方法是否适合模型类型                              │
│    - Transformer: AWQ/GPTQ                                  │
│    - CNN: 标准量化                                          │
│                                                             │
│  □ 检查校准数据是否代表真实分布                              │
│    - 数据量是否足够（100-500 samples）                      │
│    - 数据是否多样化                                          │
│                                                             │
│  □ 检查量化粒度                                              │
│    - per-tensor → per-group → per-channel                   │
│    - group_size 是否合适（128 推荐）                         │
│                                                             │
│  □ 检查异常值                                                │
│    - 激活分布是否有 outlier                                 │
│    - 权重分布是否有 outlier                                 │
│                                                             │
│  □ 检查关键层                                                │
│    - Embedding 层是否量化                                   │
│    - Output 头是否量化                                       │
│    - LayerNorm 是否保持 FP32                                │
│                                                             │
│  □ 检查推理框架支持                                          │
│    - vLLM / SGLang 是否支持该量化格式                       │
│    - 是否有特殊配置需求                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 实战：精度恢复完整流程

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QuantizationRecovery:
    """量化精度恢复工具"""
    
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.fp16_results = None
        self.quantized_results = None
    
    def benchmark(self, tasks=["mmlu", "gsm8k"]):
        """基准测试"""
        results = {}
        for task in tasks:
            results[task] = evaluate_on_task(self.model, task)
        return results
    
    def diagnose_layers(self):
        """层级诊断"""
        layer_sensitivity = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 模拟量化并测量误差
                original_weight = module.weight.data.clone()
                
                # 量化
                quantized = self.quantize_weight(original_weight)
                dequantized = self.dequantize_weight(quantized)
                
                # 计算误差
                error = (original_weight - dequantized).abs().mean()
                layer_sensitivity[name] = error.item()
        
        # 排序输出最敏感的层
        return sorted(layer_sensitivity.items(), key=lambda x: x[1], reverse=True)
    
    def apply_smooth_quant(self):
        """应用 SmoothQuant"""
        # 实现见上文
        pass
    
    def mixed_precision_quantize(self, sensitive_layers):
        """混合精度量化"""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in sensitive_layers:
                    # 保持 FP16
                    continue
                else:
                    # 量化为 INT4
                    self.quantize_layer(module, bits=4)
    
    def qat_finetune(self, calibration_data, epochs=3):
        """QAT 微调"""
        # 实现见上文
        pass
    
    def full_recovery_pipeline(self):
        """完整恢复流程"""
        
        # 1. FP16 基准
        print("Step 1: Benchmarking FP16...")
        self.fp16_results = self.benchmark()
        
        # 2. 初步量化
        print("Step 2: Initial quantization...")
        self.quantize_model(bits=4)
        self.quantized_results = self.benchmark()
        
        # 3. 分析精度损失
        accuracy_drop = self.fp16_results['mmlu'] - self.quantized_results['mmlu']
        print(f"Accuracy drop: {accuracy_drop:.2f}%")
        
        if accuracy_drop > 5:
            # 4. 诊断
            print("Step 3: Diagnosing...")
            sensitive_layers = self.diagnose_layers()[:10]
            
            # 5. 应用恢复策略
            print("Step 4: Applying recovery strategies...")
            
            # 5a. SmoothQuant
            self.apply_smooth_quant()
            
            # 5b. 混合精度
            self.mixed_precision_quantize([l[0] for l in sensitive_layers[:3]])
            
            # 5c. QAT 微调（可选）
            # self.qat_finetune(calibration_data)
        
        # 6. 最终测试
        print("Step 5: Final benchmark...")
        final_results = self.benchmark()
        
        return {
            "fp16": self.fp16_results,
            "initial_quant": self.quantized_results,
            "recovered": final_results
        }
```

---

## 本章小结

1. **PTQ vs QAT**：PTQ 快速但精度损失大，QAT 精度高但需要训练。根据实际需求选择。

2. **主流量化方法**：GPTQ 基于 Hessian 理论最优，AWQ 激活感知精度更佳，GGUF 适合 CPU 推理。

3. **FP8 是新标准**：NVIDIA H100 支持，训练推理都受益。但硬件依赖强，可能是过渡方案。

4. **精度恢复是系统工程**：诊断原因、分析层级、选择合适策略（混合精度、SmoothQuant、QAT）。

下一章，我们将进入 CUDA 编程的世界，看看如何通过手写 Kernel 极致优化性能。

---

*放弃指数：⭐⭐⭐⭐ 量化涉及数值计算和误差分析，建议动手实验，对比不同方法的效果。*

---

*（未完待续...）*