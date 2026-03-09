# 第四章 推理框架 —— vLLM、SGLang、TensorRT-LLM

> *放弃指数：⭐⭐⭐⭐ 本章内容多且深，是 AI Infra 工程师的核心竞争力*

---

## 4.1 KV Cache：推理优化的核心概念

### 什么是 KV Cache？

在讲推理框架之前，必须先理解 **KV Cache**。它是所有推理优化的基础。

**Transformer 的生成过程：**

当模型生成文本时，是逐个 token 生成的。假设我们要生成 "Hello, how are you?"：

```
第 1 步：输入 [BOS] → 输出 "Hello"
第 2 步：输入 [BOS, Hello] → 输出 ","
第 3 步：输入 [BOS, Hello, ,] → 输出 "how"
第 4 步：输入 [BOS, Hello, ,, how] → 输出 "are"
...
```

问题来了：每次生成新 token，都要重新计算之前所有 token 的 attention。

**这太浪费了！**

因为：
- 已经计算过的 K（Key）和 V（Value）是不变的
- 可以缓存起来，下次直接用

这就是 **KV Cache** 的由来：

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache 工作原理                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第 1 步生成 "Hello"：                                        │
│  ┌──────┐                                                   │
│  │ K₀V₀ │ ← 计算 token 0 的 K/V，缓存                        │
│  └──────┘                                                   │
│                                                             │
│  第 2 步生成 ","：                                            │
│  ┌──────┬──────┐                                            │
│  │ K₀V₀ │ K₁V₁ │ ← 复用 K₀V₀，只计算 K₁V₁                    │
│  └──────┴──────┘                                            │
│                                                             │
│  第 3 步生成 "how"：                                          │
│  ┌──────┬──────┬──────┐                                     │
│  │ K₀V₀ │ K₁V₁ │ K₂V₂ │ ← 复用 K₀V₀, K₁V₁，只计算 K₂V₂        │
│  └──────┴──────┴──────┘                                     │
│                                                             │
│  ...依次类推                                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### KV Cache 的显存占用

KV Cache 需要存储每个 token 在每一层的 K 和 V。计算公式：

```
KV Cache 大小 = 2 × num_layers × seq_len × num_heads × head_dim × dtype_size × batch_size
```

**以 Llama-2-70B 为例：**

| 参数 | 数值 |
|------|------|
| num_layers | 80 |
| num_heads | 64 |
| head_dim | 128 |
| dtype_size | 2 (FP16) |

单个 token 每层的 KV Cache：
```
2 × 80 × 1 × 64 × 128 × 2 = 2.62 MB per token
```

**一个 4096 token 的请求：**
```
2.62 MB × 4096 = 10.7 GB
```

**如果 batch_size = 32：**
```
10.7 GB × 32 = 342 GB ❌ 显存爆炸！
```

这就是为什么需要复杂的 KV Cache 管理策略。

---

### KV Cache 管理的挑战

**挑战 1：预分配 vs 动态分配**

```python
# 方案 A：预分配最大长度（浪费显存）
max_seq_len = 4096
kv_cache = torch.zeros(batch_size, num_layers, max_seq_len, ...)

# 方案 B：动态增长（频繁分配/释放，效率低）
kv_cache = []
for token in generation:
    kv_cache.append(new_token_kv)
```

**挑战 2：可变长度请求**

```
请求 1：128 tokens
请求 2：2048 tokens
请求 3：512 tokens
...
如何高效管理不同长度的 KV Cache？
```

**挑战 3：内存碎片**

```
请求 A 占用 [0, 1024]
请求 B 占用 [1024, 2048]
请求 A 结束释放 [0, 1024]
新请求 C 需要 1500 tokens → 放不下！
```

这些问题推动了 **PagedAttention** 的诞生。

---

## 4.2 PagedAttention：内存管理像操作系统一样复杂

### 从操作系统的虚拟内存说起

操作系统管理内存时，不是直接分配连续的物理内存，而是使用 **分页（Paging）**：

```
虚拟内存（进程视角）：连续的地址空间
         ↓ 映射
物理内存（实际视角）：离散的页框（Page Frames）
```

好处：
- 不需要连续的物理内存
- 可以动态分配和回收
- 减少内存碎片

**PagedAttention 把这个思想用到了 KV Cache 管理。**

---

### PagedAttention 原理

vLLM 提出的 PagedAttention 将 KV Cache 划分为固定大小的 **块（Block）**：

```
┌─────────────────────────────────────────────────────────────┐
│                    PagedAttention 结构                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  逻辑 KV Cache（请求视角）：                                   │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                 │
│  │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ 连续的           │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                 │
│                                                             │
│  物理 KV Cache（实际存储）：                                   │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐  │
│  │ B3 │ │ B7 │ │ B0 │ │ B5 │ │ B1 │ │ B4 │ │ B2 │ │ B6 │  │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘  │
│   块#0   块#1   块#2   块#3   块#4   块#5   块#6   块#7     │
│                                                             │
│  块表（Block Table）：                                        │
│  请求 A: [块#2, 块#4, 块#6, 块#0, ...]                       │
│  请求 B: [块#1, 块#3, 块#5, ...]                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心概念：**

| 概念 | 说明 |
|------|------|
| Block（块） | 固定大小的 KV Cache 单元，通常存储 16-64 个 token |
| Block Table（块表） | 记录每个请求的逻辑块到物理块的映射 |
| Physical Blocks（物理块） | 实际存储 KV Cache 的内存池 |

---

### PagedAttention 的优势

**优势 1：高效内存利用**

```
传统方式（预分配）：
请求 A（实际 100 tokens）：占用 4096 slots → 浪费 3996
请求 B（实际 512 tokens）：占用 4096 slots → 浪费 3584

PagedAttention：
请求 A：占用 ceil(100/16) = 7 blocks
请求 B：占用 ceil(512/16) = 32 blocks
几乎没有浪费
```

**优势 2：无内存碎片**

```
所有块大小相同，释放后立即可用
```

**优势 3：支持内存共享**

```
多个请求共享相同的 prefix（系统提示词、few-shot examples）
只存储一份 prefix 的 KV Cache
```

---

### vLLM 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         vLLM 架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    API Server                        │   │
│  │            (OpenAI-compatible REST API)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Scheduler                          │   │
│  │         (请求调度、batch 管理、抢占)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Block Manager                         │   │
│  │     (KV Cache 块分配、回收、共享)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Model Executor                          │   │
│  │      (PagedAttention Kernel、模型推理)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  GPU Memory                          │   │
│  │    ┌─────┬─────┬─────┬─────┬─────┬─────┐            │   │
│  │    │Block│Block│Block│Block│Block│Block│ ...        │   │
│  │    └─────┴─────┴─────┴─────┴─────┴─────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4.3 Continuous Batching：为什么不是 Batch 越大越好？

### 传统 Batching 的问题

**Static Batching**：凑够一批请求，等所有请求完成后才返回。

```
┌─────────────────────────────────────────────────────────────┐
│                    Static Batching                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  时间轴 →                                                    │
│                                                             │
│  请求 A: ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  请求 B: ████████████████████████████████████░░░░░░░░░░░   │
│  请求 C: ████████████████████████████████████████████████   │
│                                                             │
│  █ 处理中  ░ 等待（已完成但被其他请求阻塞）                     │
│                                                             │
│  问题：请求 A、B 完成后要等 C，造成资源浪费                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**问题：**
- 短请求要等长请求
- GPU 闲置等待
- 用户延迟高

---

### Continuous Batching 原理

**Continuous Batching**（也叫 Iteration-level Scheduling）：每个 iteration 都可以加入新请求、移除完成的请求。

```
┌─────────────────────────────────────────────────────────────┐
│                  Continuous Batching                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  时间轴 →                                                    │
│                                                             │
│  Iteration 1:  [A, B, C] 正在生成                            │
│  Iteration 2:  [A✓, B, C, D] A完成退出，D加入                │
│  Iteration 3:  [B✓, C, D, E] B完成退出，E加入                │
│  Iteration 4:  [C, D, E, F] ...                              │
│                                                             │
│  优势：短请求完成后立即返回，新请求及时加入                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心思想：**
- 不等待所有请求完成
- 动态调整 batch 成员
- 最大化 GPU 利用率

---

### Continuous Batching 的实现挑战

**挑战 1：不同请求的长度不同**

```python
# 传统方式：padding 到相同长度
tokens = [
    [1, 2, 3, 4, 5, 0, 0, 0],  # padding 3 个
    [1, 2, 3, 4, 5, 6, 7, 8],  # 无需 padding
    [1, 2, 0, 0, 0, 0, 0, 0],  # padding 6 个
]

# Continuous Batching：每个请求独立跟踪长度
# 不需要 padding，节省计算
```

**挑战 2：KV Cache 的动态管理**

```
请求加入：分配新的 KV Cache 块
请求完成：释放 KV Cache 块
请求抢占（内存不足）：换出到 CPU 或终止
```

**挑战 3：调度策略**

```python
# 常见策略
def schedule(requests, memory_budget):
    # 优先级调度
    requests.sort(key=lambda r: r.priority)
    
    # 先到先服务（FCFS）
    # requests.sort(key=lambda r: r.arrival_time)
    
    # 短任务优先（SJF）
    # requests.sort(key=lambda r: r.estimated_length)
    
    # 在内存预算内选择尽可能多的请求
    selected = []
    current_memory = 0
    for req in requests:
        if current_memory + req.memory_needed <= memory_budget:
            selected.append(req)
            current_memory += req.memory_needed
    return selected
```

---

### 性能对比

以 Llama-2-70B 为例，处理混合长度的请求：

| 指标 | Static Batching | Continuous Batching | 提升 |
|------|-----------------|---------------------|------|
| 平均延迟 | 2.3s | 0.8s | 2.9× |
| P99 延迟 | 8.5s | 2.1s | 4× |
| 吞吐量 | 45 req/s | 120 req/s | 2.7× |
| 显存利用率 | 60% | 95% | 1.6× |

**结论：Continuous Batching 是推理服务的标配。**

---

## 4.4 框架选型指南：vLLM vs SGLang vs TRT-LLM vs LMDeploy

### 四大框架概览

| 框架 | 开发者 | 特点 | 开源协议 |
|------|-------|------|---------|
| **vLLM** | UC Berkeley | PagedAttention 创始者，社区活跃 | Apache 2.0 |
| **SGLang** | LMSYS (UC Berkeley) | 高效调度、编程模型灵活 | Apache 2.0 |
| **TensorRT-LLM** | NVIDIA | 深度 GPU 优化，封闭生态 | 专有 |
| **LMDeploy** | 上海 AI Lab | 国产框架，TurboMind 引擎 | Apache 2.0 |

---

### vLLM 详解

**优势：**
- ✅ PagedAttention 成熟稳定
- ✅ 社区活跃，模型支持广泛
- ✅ OpenAI 兼容 API
- ✅ 支持多种量化（AWQ、GPTQ、FP8）

**劣势：**
- ❌ 长文本性能不是最优
- ❌ 多模态支持有限
- ❌ 某些情况下 kernel 效率不如 TRT-LLM

**适用场景：**
- 通用 LLM 推理
- 快速部署原型
- 社区模型支持

**代码示例：**

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
)

# 推理
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

**启动服务：**

```bash
# OpenAI 兼容服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --port 8000

# 调用
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-70b-hf",
        "prompt": "Hello, world!",
        "max_tokens": 100
    }'
```

---

### SGLang 详解

**优势：**
- ✅ 编程模型更灵活（支持复杂工作流）
- ✅ 长文本性能优秀
- ✅ RadixAttention（前缀共享优化）
- ✅ PD-HiCache 支持

**劣势：**
- ❌ 社区相对较小
- ❌ 文档不如 vLLM 完善
- ❌ 学习曲线稍陡

**适用场景：**
- 长文本推理
- 复杂 Agent 工作流
- 需要前缀共享的场景

**代码示例：**

```python
import sglang as sgl

# 定义工作流
@sgl.function
def multi_turn_chat(s, user_input):
    s += "System: You are a helpful assistant.\n"
    s += "User: " + user_input + "\n"
    s += "Assistant:" + sgl.gen("response", max_tokens=100)
    return s["response"]

# 运行
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")
result = multi_turn_chat.run(runtime, user_input="What is AI?")
print(result)
```

**启动服务：**

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-70b-hf \
    --tp 4 \
    --port 8000
```

---

### TensorRT-LLM 详解

**优势：**
- ✅ NVIDIA 官方深度优化
- ✅ Kernel 效率最高
- ✅ 支持最新 GPU 特性（FP8、H100 优化）
- ✅ 与 Triton Inference Server 集成

**劣势：**
- ❌ 闭源，调试困难
- ❌ 模型需要转换
- ❌ 社区支持少
- ❌ 只支持 NVIDIA GPU

**适用场景：**
- 追求极致性能
- NVIDIA GPU 生态
- 生产环境稳定部署

**使用流程：**

```bash
# 1. 转换模型
python convert_checkpoint.py \
    --model_dir /path/to/llama \
    --output_dir /path/to/trt_model \
    --tp_size 4

# 2. 构建 Engine
trtllm-build \
    --checkpoint_dir /path/to/trt_model \
    --output_dir /path/to/engine \
    --gemm_plugin fp8

# 3. 运行推理
python run.py \
    --engine_dir /path/to/engine \
    --tokenizer_dir /path/to/tokenizer
```

---

### LMDeploy 详解

**优势：**
- ✅ 国产框架，中文支持好
- ✅ TurboMind 引擎高效
- ✅ 支持国产芯片（昇腾等）
- ✅ 量化工具链完善

**劣势：**
- ❌ 社区规模小于 vLLM
- ❌ 国际模型支持有限

**适用场景：**
- 国内部署
- 国产芯片
- 中文模型

**代码示例：**

```python
from lmdeploy import pipeline, TurbomindEngineConfig

# 加载模型
pipe = pipeline(
    "meta-llama/Llama-2-70b-hf",
    backend_config=TurbomindEngineConfig(tp=4)
)

# 推理
response = pipe(["Hello, world!", "Hi there!"])
print(response)
```

---

### 性能对比实测

**测试配置：**
- 模型：Llama-2-70B
- GPU：A100 80GB × 4
- 测试数据：ShareGPT 数据集

**结果：**

| 指标 | vLLM | SGLang | TRT-LLM | LMDeploy |
|------|------|--------|---------|----------|
| TTFT (ms) | 180 | 165 | 140 | 175 |
| TPS (token/s) | 2400 | 2200 | 2800 | 2300 |
| 显存效率 | 95% | 92% | 90% | 93% |
| 长文本支持 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

### 选型建议

```
┌─────────────────────────────────────────────────────────────┐
│                    推理框架选型决策树                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  需要极致性能？                                               │
│     ├── 是 → TensorRT-LLM                                   │
│     └── 否 → 长文本场景？                                     │
│                  ├── 是 → SGLang                            │
│                  └── 否 → 国产芯片？                          │
│                               ├── 是 → LMDeploy             │
│                               └── 否 → vLLM（稳妥选择）       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**一句话总结：**
> vLLM 是稳妥选择，SGLang 长文本更强，TRT-LLM 性能最佳，LMDeploy 国产友好。

---

## 4.5 实战：部署一个高性能推理服务

### 完整部署流程

以 vLLM 为例，从头到尾部署一个生产级服务。

**Step 1：环境准备**

```bash
# 创建环境
conda create -n vllm python=3.10
conda activate vllm

# 安装 vLLM
pip install vllm

# 验证
python -c "import vllm; print(vllm.__version__)"
```

**Step 2：下载模型**

```bash
# 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./Llama-2-7b-hf

# 或使用 modelscope（国内）
pip install modelscope
modelscope download --model LLM-Research/Meta-Llama-3-8B --local_dir ./Llama-3-8B
```

**Step 3：启动服务**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --enable-prefix-caching
```

**Step 4：测试服务**

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="./Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

**Step 5：压力测试**

```bash
# 安装测试工具
pip install locust

# 创建测试脚本
cat > locustfile.py << 'EOF'
from locust import HttpUser, task

class VLLMUser(HttpUser):
    @task
    def generate(self):
        self.client.post("/v1/completions", json={
            "model": "./Llama-2-7b-hf",
            "prompt": "Hello, " * 100,
            "max_tokens": 100
        })
EOF

# 运行测试
locust -f locustfile.py --host http://localhost:8000
```

---

### 生产环境配置建议

```python
# config.py
from dataclasses import dataclass

@dataclass
class VLLMConfig:
    # 模型配置
    model: str = "./Llama-2-70b-hf"
    tensor_parallel_size: int = 4
    
    # 内存配置
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    
    # 并行配置
    max_num_seqs: int = 256  # 最大并发序列数
    max_num_batched_tokens: int = 32768  # 最大 batch tokens
    
    # 调度配置
    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: bool = True  # 分块 prefill
    enable_prefix_caching: bool = True   # 前缀缓存
    
    # 量化配置（可选）
    quantization: str = "awq"  # 或 "gptq", "fp8"
    load_format: str = "awq"
```

**启动命令（完整版）：**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --quantization awq \
    --port 8000
```

---

### 性能调优技巧

**技巧 1：启用 Prefix Caching**

适用场景：多个请求共享相同的系统提示词。

```bash
--enable-prefix-caching
```

**原理：** 相同前缀的 KV Cache 只计算一次，后续请求直接复用。

**技巧 2：启用 Chunked Prefill**

适用场景：长 prompt + 高并发。

```bash
--enable-chunked-prefill
```

**原理：** 把长 prompt 分块处理，避免阻塞短请求。

**技巧 3：调整 Batch 参数**

```bash
--max-num-seqs 256          # 增加并发数
--max-num-batched-tokens 32768  # 增加 token 吞吐
```

**技巧 4：使用量化**

```bash
--quantization awq          # AWQ 量化，精度损失小
# 或
--quantization fp8          # FP8（需要 H100）
```

---

### 多机多卡部署

```bash
# 在所有节点运行
python -m vllm.entrypoints.openai.api_server \
    --model ./Llama-2-70b-hf \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --host 0.0.0.0 \
    --port 8000

# Ray 集群启动
# Head 节点
ray start --head --port=6379

# Worker 节点
ray start --address=<head-node-ip>:6379
```

---

### 监控与告警

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000/metrics']
```

**关键指标：**

| 指标 | 含义 | 告警阈值 |
|------|------|---------|
| `vllm:num_requests_running` | 当前运行请求数 | - |
| `vllm:num_requests_waiting` | 等待队列长度 | > 100 |
| `vllm:gpu_cache_usage_perc` | GPU 显存使用率 | > 95% |
| `vllm:time_per_output_token` | 每个 token 时间 | > 100ms |
| `vllm:e2e_request_latency_seconds` | 端到端延迟 | P99 > 5s |

---

## 4.6 常见问题排查

### 问题 1：OOM（显存不足）

```
RuntimeError: CUDA out of memory
```

**排查：**
```bash
# 查看显存使用
nvidia-smi

# 查看 vLLM 日志
# 找到 "Memory profiler" 输出
```

**解决：**
```bash
# 1. 降低最大长度
--max-model-len 4096

# 2. 降低并发数
--max-num-seqs 128

# 3. 使用量化
--quantization awq

# 4. 降低显存预留
--gpu-memory-utilization 0.95
```

---

### 问题 2：TTFT 过高

```
首 token 延迟 > 2秒
```

**排查：**
```python
# 检查是否 prefill 阶段太长
# 长度分布
prompt_lengths = [len(p) for p in prompts]
print(f"Avg prompt length: {sum(prompt_lengths)/len(prompt_lengths)}")
```

**解决：**
```bash
# 1. 启用 chunked prefill
--enable-chunked-prefill

# 2. 减少 batch size
--max-num-seqs 64

# 3. 使用前缀缓存
--enable-prefix-caching
```

---

### 问题 3：吞吐量低

```
TPS < 预期
```

**排查：**
```bash
# 查看 GPU 利用率
nvidia-smi dmon -s u

# 如果 GPU 利用率低：
#   - 数据加载瓶颈
#   - 网络瓶颈
# 如果 GPU 利用率高但是 TPS 低：
#   - 模型本身计算量大
#   - prompt/output 长度不合理
```

**解决：**
```bash
# 1. 增加 batch
--max-num-seqs 256 --max-num-batched-tokens 65536

# 2. 使用量化
--quantization fp8

# 3. 使用更快的框架
# 切换到 TensorRT-LLM
```

---

## 本章小结

1. **KV Cache 是推理优化的基础**：理解其原理和显存占用，是一切优化工作的起点。

2. **PagedAttention 革新了内存管理**：借鉴操作系统的分页思想，解决了 KV Cache 的碎片化和预分配问题。

3. **Continuous Batching 提升吞吐和延迟**：动态调度请求，避免短请求等待长请求。

4. **框架选型要结合场景**：vLLM 通用、SGLang 长文本强、TRT-LLM 性能极致、LMDeploy 国产友好。

5. **部署调优是持续工程**：从测试、监控到故障排查，需要系统化的方法论。

下一章，我们将深入模型并行的世界，看看当单卡放不下模型时，如何优雅地切分。

---

*放弃指数：⭐⭐⭐⭐ 本章是推理优化的核心，需要反复实践。建议：先用起来，再深入原理。*

---

*（未完待续...）*