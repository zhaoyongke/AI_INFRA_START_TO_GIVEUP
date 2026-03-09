# 附录

---

## 附录 A：GPU 性能对比表

### NVIDIA 数据中心 GPU

| GPU | 显存 | 带宽 | FP16 TFLOPS | INT8 TOPS | TDP | NVLink | 发布年份 |
|-----|------|------|-------------|-----------|-----|--------|---------|
| A100 40GB | 40GB HBM2e | 1.6 TB/s | 312 | 624 | 400W | 600 GB/s | 2020 |
| A100 80GB | 80GB HBM2e | 2.0 TB/s | 312 | 624 | 400W | 600 GB/s | 2020 |
| A800 80GB | 80GB HBM2e | 2.0 TB/s | 312 | 624 | 400W | 400 GB/s | 2022 |
| H100 SXM | 80GB HBM3 | 3.35 TB/s | 989 | 3958 | 700W | 900 GB/s | 2022 |
| H100 PCIe | 80GB HBM3 | 2.0 TB/s | 756 | 3025 | 350W | - | 2022 |
| H200 | 141GB HBM3e | 4.8 TB/s | 989 | 3958 | 700W | 900 GB/s | 2024 |
| B200 | 192GB HBM3e | 8 TB/s | 2250 | 9000 | 1000W | 1.8 TB/s | 2024 |
| B100 | 192GB HBM3e | 8 TB/s | 1750 | 7000 | 700W | 1.8 TB/s | 2024 |

### NVIDIA 消费级 GPU

| GPU | 显存 | 带宽 | FP16 TFLOPS | TDP | 价格（约） |
|-----|------|------|-------------|-----|-----------|
| RTX 4090 | 24GB GDDR6X | 1.0 TB/s | 330 | 450W | ¥1.5万 |
| RTX 4080 Super | 16GB GDDR6X | 736 GB/s | 264 | 320W | ¥9000 |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 142 | 350W | ¥8000 |
| RTX 3080 | 10GB GDDR6X | 760 GB/s | 136 | 320W | ¥5000 |

### 国产 GPU

| GPU | 显存 | 带宽 | FP16 TFLOPS | 备注 |
|-----|------|------|-------------|------|
| 华为昇腾 910B | 64GB HBM | ~1.2 TB/s | 310 | 主流国产算力 |
| 壁仞 BR100 | 64GB HBM2e | ~1.0 TB/s | 230 | 国产通用 GPU |
| 海光 DCU Z100 | 32GB HBM2 | ~1.0 TB/s | 120 | 类 ROCm 生态 |
| 寒武纪 MLU370 | 48GB | - | 192 | neuware SDK |
| 摩尔线程 MTT S4000 | 48GB | - | 48 (FP32) | MUSA 生态 |
| 燧原 云燧T21 | 32GB | - | 160 | 自研软件栈 |

### 显存计算公式

```
模型权重显存 (GB) = 参数量(B) × 字节数
- FP32: 参数量 × 4
- FP16/BF16: 参数量 × 2
- INT8: 参数量 × 1
- INT4: 参数量 × 0.5

KV Cache显存 (GB) = 2 × L × H × D × S × B × bytes
- L: 层数
- H: 注意力头数
- D: 头维度
- S: 序列长度
- B: Batch Size
- bytes: 数据类型字节数

示例 (LLaMA-2-70B, BF16):
- 权重: 70B × 2 = 140GB
- KV Cache (L=80, H=64, D=128, S=4096, B=1): 
  2 × 80 × 64 × 128 × 4096 × 2 / 1e9 ≈ 10.7GB
```

---

## 附录 B：常用推理框架配置模板

### vLLM 配置模板

```yaml
# vllm-config.yaml
model: "./models/llama-2-70b"
tokenizer: "./models/llama-2-70b"

# 并行配置
tensor_parallel_size: 4
pipeline_parallel_size: 1

# 内存配置
gpu_memory_utilization: 0.9
max_model_len: 8192
block_size: 16

# 批处理配置
max_num_seqs: 256
max_num_batched_tokens: 32768

# 优化配置
enable_chunked_prefill: true
enable_prefix_caching: true

# 量化配置（可选）
quantization: awq
load_format: awq

# 服务配置
host: 0.0.0.0
port: 8000
```

**启动命令：**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/llama-2-70b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --port 8000
```

---

### SGLang 配置模板

```yaml
# sglang-config.yaml
model_path: "./models/llama-2-70b"

# 并行配置
tp: 4

# 内存配置
mem_fraction_static: 0.9
context_length: 8192

# 其他配置
disable_cuda_graph: false
disable_radix_cache: false
enable_p2p_check: true

# 服务配置
host: 0.0.0.0
port: 8000
```

**启动命令：**

```bash
python -m sglang.launch_server \
    --model-path ./models/llama-2-70b \
    --tp 4 \
    --mem-fraction-static 0.9 \
    --port 8000
```

---

### TensorRT-LLM 配置模板

```bash
# 1. 转换模型
python convert_checkpoint.py \
    --model_dir ./models/llama-2-70b \
    --output_dir ./trt_model \
    --tp_size 4 \
    --pp_size 1

# 2. 构建 Engine
trtllm-build \
    --checkpoint_dir ./trt_model \
    --output_dir ./trt_engine \
    --gemm_plugin fp16 \
    --max_batch_size 32 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --max_beam_width 1

# 3. 运行推理
python run.py \
    --engine_dir ./trt_engine \
    --tokenizer_dir ./models/llama-2-70b \
    --max_output_len 1024
```

---

### DeepSpeed 配置模板

```json
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1
        }
    },
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    
    "gradient_checkpointing": {
        "enabled": true
    },
    
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true
    }
}
```

---

## 附录 C：显存占用估算公式

### 通用公式

```
总显存 = 权重 + 激活值 + KV Cache + 优化器状态 + 其他

1. 模型权重
   显存 = 参数量 × 字节数
   - FP32: 4 bytes/参数
   - FP16/BF16: 2 bytes/参数
   - INT8: 1 byte/参数
   - INT4: 0.5 bytes/参数

2. 激活值
   显存 ≈ batch_size × seq_len × hidden_size × num_layers × bytes
   粗略估计：权重显存 × batch_size / 32

3. KV Cache
   显存 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × bytes
   简化：显存 ≈ 2 × L × D × S × B × bytes
   其中 D = hidden_size

4. 优化器状态
   - Adam: 2× 权重（一阶和二阶动量）
   - SGD: 0（无额外状态）
```

---

### 常见模型显存估算

| 模型 | 参数量 | FP16 权重 | FP32 权重 | KV Cache (2K) | 推荐 GPU |
|------|--------|----------|----------|---------------|----------|
| LLaMA-7B | 7B | 14 GB | 28 GB | ~2 GB | RTX 4090 |
| LLaMA-13B | 13B | 26 GB | 52 GB | ~3 GB | A100 40GB |
| LLaMA-70B | 70B | 140 GB | 280 GB | ~11 GB | A100 80GB × 2 |
| GPT-3 175B | 175B | 350 GB | 700 GB | ~25 GB | A100 80GB × 8 |
| GLM-5 | 744B | 1488 GB | - | ~? | H100 80GB × 20+ |

---

### 计算示例代码

```python
def estimate_memory(
    num_params_billion: float,
    precision: str = "fp16",
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 32,
):
    """估算显存占用"""
    
    # 字节数
    bytes_map = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    bytes_per_param = bytes_map[precision]
    
    # 权重显存
    weight_memory_gb = num_params_billion * bytes_per_param
    
    # 激活值（粗略）
    activation_memory_gb = batch_size * seq_len * hidden_size * num_layers * bytes_per_param / 1e9
    
    # KV Cache
    kv_cache_gb = 2 * num_layers * hidden_size * seq_len * batch_size * bytes_per_param / 1e9
    
    # 总计（推理）
    inference_memory_gb = weight_memory_gb + kv_cache_gb
    
    # 训练额外需要
    # 优化器状态（Adam）
    optimizer_memory_gb = num_params_billion * 8  # 2 states × 4 bytes
    
    # 梯度
    gradient_memory_gb = num_params_billion * bytes_per_param
    
    # 训练总计
    training_memory_gb = (weight_memory_gb + activation_memory_gb + 
                         kv_cache_gb + optimizer_memory_gb + gradient_memory_gb)
    
    return {
        "weight_gb": weight_memory_gb,
        "activation_gb": activation_memory_gb,
        "kv_cache_gb": kv_cache_gb,
        "inference_total_gb": inference_memory_gb,
        "training_total_gb": training_memory_gb,
    }

# 示例
result = estimate_memory(
    num_params_billion=70,
    precision="fp16",
    batch_size=1,
    seq_len=4096,
    hidden_size=8192,
    num_layers=80
)

print(f"LLaMA-70B 显存估算:")
print(f"  权重: {result['weight_gb']:.1f} GB")
print(f"  KV Cache: {result['kv_cache_gb']:.1f} GB")
print(f"  推理总计: {result['inference_total_gb']:.1f} GB")
print(f"  训练总计: {result['training_total_gb']:.1f} GB")
```

---

## 附录 D：常见错误速查表

### CUDA 相关错误

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `CUDA out of memory` | 显存不足 | 减小 batch_size、使用量化、清空 cache |
| `CUDA error: invalid device ordinal` | GPU ID 无效 | 检查 CUDA_VISIBLE_DEVICES |
| `CUDA error: device-side assert triggered` | 越界访问 | 检查索引范围 |
| `CUDA error: misaligned address` | 内存对齐问题 | 检查数据类型和访问模式 |
| `illegal memory access` | 非法内存访问 | 检查指针和边界 |

### NCCL 相关错误

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `NCCL WARN Call to NCCL timed out` | 通信超时 | 增加 NCCL_TIMEOUT |
| `NCCL WARN Connection refused` | 网络不通 | 检查防火墙和网络配置 |
| `NCCL WARN GPU Direct P2P disabled` | P2P 不可用 | 检查 NVLink 或用 NCCL_P2P_DISABLE=1 |
| `NCCL WARN Cuda failure` | CUDA 错误 | 检查 GPU 状态和代码 |
| `NCCL WARN unhandled system error` | 系统错误 | 检查内存和网络 |

### PyTorch 相关错误

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `RuntimeError: Expected all tensors on the same device` | 张量设备不一致 | 统一到同一设备 |
| `ValueError: too many dimensions` | 张量维度错误 | 检查 shape |
| `KeyError: "xxx"` | 键不存在 | 检查 checkpoint 的 key |
| `RuntimeError: CUDA error: an illegal memory access` | 非法内存访问 | 检查索引和对齐 |
| `AssertionError: Torch not compiled with CUDA enabled` | 无 CUDA 支持 | 安装 CUDA 版本 |

### 分布式相关错误

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `Connection reset by peer` | 连接断开 | 检查网络和防火墙 |
| `Address already in use` | 端口被占用 | 换端口或关闭占用进程 |
| `Master node unreachable` | 主节点不可达 | 检查主节点状态和网络 |
| `Timeout waiting for master` | 等待超时 | 增加超时时间 |

---

## 附录 E：推荐阅读与资源

### 官方文档

| 资源 | 链接 | 说明 |
|------|------|------|
| PyTorch 文档 | https://pytorch.org/docs/ | 核心框架文档 |
| vLLM 文档 | https://vllm.readthedocs.io/ | 高性能推理框架 |
| SGLang 文档 | https://github.com/sgl-project/sglang | 灵活推理框架 |
| HuggingFace 文档 | https://huggingface.co/docs | 模型和数据集 |
| NVIDIA CUDA 文档 | https://docs.nvidia.com/cuda/ | CUDA 编程指南 |
| NCCL 文档 | https://docs.nvidia.com/deeplearning/nccl/ | 分布式通信 |
| Flash Attention | https://github.com/Dao-AILab/flash-attention | 注意力优化 |

---

### 论文推荐

**架构类：**
- Attention Is All You Need (Transformer)
- LLaMA: Open and Efficient Foundation Language Models
- Mistral 7B
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

**优化类：**
- Flash Attention (v1, v2, v3)
- PagedAttention (vLLM)
- GPTQ: Accurate Post-Training Quantization
- AWQ: Activation-aware Weight Quantization

**并行类：**
- Megatron-LM: Training Multi-Billion Parameter Language Models
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- DeepSpeed: Extreme-scale model training

---

### 开源项目

| 项目 | GitHub | 说明 |
|------|--------|------|
| vLLM | https://github.com/vllm-project/vllm | 高性能推理 |
| SGLang | https://github.com/sgl-project/sglang | 灵活推理框架 |
| llama.cpp | https://github.com/ggerganov/llama.cpp | 边缘推理 |
| DeepSpeed | https://github.com/microsoft/DeepSpeed | 分布式训练 |
| Megatron-LM | https://github.com/NVIDIA/Megatron-LM | 大模型训练 |
| Flash Attention | https://github.com/Dao-AILab/flash-attention | 注意力优化 |
| AutoGPTQ | https://github.com/AutoGPTQ/AutoGPTQ | GPTQ 量化 |
| AutoAWQ | https://github.com/casper-hansen/AutoAWQ | AWQ 量化 |
| Triton | https://github.com/openai/triton | GPU 编程 |

---

### 博客和教程

- [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Lilian Weng - Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [HuggingFace - Transformers 进阶教程](https://huggingface.co/docs/transformers/zh/main/zh)
- [NVIDIA - Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)
- [Flash Attention 官方博客](https://tridao.me/publications/flash2/flash2.pdf)

---

### 社区

- [HuggingFace 社区](https://huggingface.co/)
- [PyTorch 论坛](https://discuss.pytorch.org/)
- [NVIDIA 开发者论坛](https://forums.developer.nvidia.com/)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [知乎 - 机器学习话题](https://www.zhihu.com/topic/19559450)

---

### 持续学习资源

- [arXiv - cs.CL](https://arxiv.org/list/cs.CL/recent) - NLP 最新论文
- [arXiv - cs.LG](https://arxiv.org/list/cs.LG/recent) - 机器学习最新论文
- [Papers With Code](https://paperswithcode.com/) - 论文 + 代码
- [HuggingFace Daily Papers](https://huggingface.co/papers) - 每日论文精选

---

*感谢阅读本书！欢迎反馈和建议。*

---

*全书完*