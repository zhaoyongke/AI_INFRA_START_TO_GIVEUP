# 第六章 PD 分离与推理优化

> *放弃指数：⭐⭐⭐⭐⭐ 本章是推理优化的前沿，需要深入理解计算/访存特性*

---

## 6.1 Prefill vs Decode：计算密集 vs 内存密集

### 两阶段的本质差异

大模型生成文本的过程分为两个阶段：**Prefill（预填充）** 和 **Decode（解码）**。

```
┌─────────────────────────────────────────────────────────────┐
│                  Prefill vs Decode 对比                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户请求："请用 200 字介绍北京"                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Prefill 阶段                                            ││
│  │                                                         ││
│  │ 输入："请用 200 字介绍北京" (8 tokens)                    ││
│  │ 输出：首个 token "北"                                    ││
│  │                                                         ││
│  │ 特点：                                                   ││
│  │ - 并行处理所有输入 tokens                                 ││
│  │ - 计算密集型                                             ││
│  │ - GPU 利用率高                                           ││
│  │ - 生成首 token 延迟（TTFT）                               ││
│  └─────────────────────────────────────────────────────────┘│
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Decode 阶段                                              ││
│  │                                                         ││
│  │ 输入：当前已生成的 tokens                                 ││
│  │ 输出：下一个 token                                        ││
│  │                                                         ││
│  │ 特点：                                                   ││
│  │ - 串行生成，每次 1 个 token                               ││
│  │ - 内存密集型                                             ││
│  │ - GPU 利用率低                                           ││
│  │ - 每个 token 生成速度（TPS）                              ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  完整生成：北(Decode) → 京(Decode) → 是(Decode) → ...        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 为什么会有本质差异？

**Prefill 阶段：**

```python
# Prefill: 处理 prompt 中所有 tokens
# 假设 prompt_len = 1024

# 计算量（Attention）
FLOPs_prefill = 2 × num_layers × num_heads × head_dim × prompt_len²

# 对于 1024 tokens:
# Attention 计算量 ∝ prompt_len² = 1,048,576

# 特点：
# 1. 每个 token 可以并行处理
# 2. 矩阵乘法规模大 → GPU 利用率高
# 3. 算术强度高 → 计算密集
```

**Decode 阶段：**

```python
# Decode: 每次生成 1 个 token

# 计算量（Attention）
FLOPs_decode = 2 × num_layers × num_heads × head_dim × generated_len

# 每生成 1 个 token:
# 只需要处理当前 token 与之前所有 tokens 的 attention

# 特点：
# 1. 每个 token 独立计算，无法并行
# 2. 矩阵乘法规模小 → GPU 利用率低
# 3. 需要读取所有 KV Cache → 显存带宽瓶颈
```

---

### 算术强度分析

**算术强度 = 计算量 / 数据访问量**

```
┌─────────────────────────────────────────────────────────────┐
│                    算术强度对比                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prefill 阶段（以 1024 prompt 为例）：                        │
│                                                             │
│  计算量 ≈ 2 × L × H × D × 1024²                              │
│  数据量 ≈ 2 × L × H × D × 1024  (模型权重)                   │
│                                                             │
│  算术强度 ≈ 2 × 1024 = 2048 ops/byte                        │
│                                                             │
│  结论：算术强度高 → 计算密集 → GPU 计算能力是瓶颈             │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Decode 阶段（生成第 N 个 token）：                           │
│                                                             │
│  计算量 ≈ 2 × L × H × D × N                                  │
│  数据量 ≈ 2 × L × H × D × N (KV Cache) + 权重               │
│                                                             │
│  算术强度 ≈ O(1) ops/byte                                   │
│                                                             │
│  结论：算术强度低 → 内存密集 → 显存带宽是瓶颈                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 实测数据对比

以 Llama-2-70B 在 A100 80GB 上为例：

| 指标 | Prefill (1024 tokens) | Decode (per token) |
|------|----------------------|-------------------|
| 计算量 | ~140 TFLOPs | ~0.14 TFLOPs |
| 数据量 | ~280 GB (权重) | ~280 GB (权重) + KV Cache |
| 耗时 | ~100ms | ~30ms |
| GPU 利用率 | ~85% | ~15% |
| 瓶颈 | GPU 计算 | 显存带宽 |

**关键发现：**
> Decode 阶段的 GPU 利用率远低于 Prefill 阶段。这说明存在优化空间。

---

### 阶段特性总结

| 特性 | Prefill | Decode |
|------|---------|--------|
| 计算模式 | 并行处理 prompt 中的所有 tokens | 串行生成，每次 1 token |
| 算术强度 | 高（计算密集） | 低（内存密集） |
| GPU 利用率 | 高 | 低 |
| 主要瓶颈 | GPU 计算能力 | 显存带宽 |
| 优化方向 | 更强的 GPU、Flash Attention | 更高的带宽、KV Cache 优化 |
| 关键指标 | TTFT（首 Token 延迟） | TPS（每秒生成 token 数） |

**这一差异是 PD 分离的理论基础。**

---

## 6.2 PD Disaggregation：预填充和解码分家

### 核心思想

既然 Prefill 和 Decode 有不同的计算特性，为什么不把它们放在不同的硬件上优化？

```
┌─────────────────────────────────────────────────────────────┐
│                    PD 分离架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传统架构（混合部署）：                                        │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │                     单一 GPU 集群                        ││
│  │                                                        ││
│  │  Request 1: Prefill ──→ Decode ──→ Decode ──→ ...     ││
│  │  Request 2:      Prefill ──→ Decode ──→ ...            ││
│  │  Request 3:           Prefill ──→ ...                  ││
│  │                                                        ││
│  │  问题：Prefill 和 Decode 竞争资源，效率不优              ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  PD 分离架构：                                                │
│                                                             │
│  ┌────────────────┐              ┌────────────────┐        │
│  │  Prefill 节点   │              │  Decode 节点   │        │
│  │                │              │                │        │
│  │ · 高算力 GPU    │   KV Cache   │ · 高带宽 GPU   │        │
│  │   (H100)       │ ─────────→  │   (H100)       │        │
│  │ · 大 batch     │   传输       │ · 长连接       │        │
│  │ · 批处理优化   │              │ · 高带宽优化   │        │
│  └────────────────┘              └────────────────┘        │
│                                                             │
│  优势：资源专精，整体效率提升                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### PD 分离的收益分析

**收益 1：资源利用率提升**

```
传统架构：
- GPU 需要同时满足 Prefill 和 Decode 的需求
- Prefill 期间 Decode 请求排队
- Decode 期间 GPU 计算能力闲置

PD 分离：
- Prefill 节点：专注于批量 Prefill，GPU 利用率 > 90%
- Decode 节点：专注于 Decode，带宽利用率最大化
```

**收益 2：延迟优化**

```
传统架构 TTFT：
- 如果有 Decode 请求在处理，Prefill 需要等待
- P99 TTFT 可能很高

PD 分离 TTFT：
- Prefill 节点专用，无需等待
- P99 TTFT 显著降低
```

**收益 3：硬件选型优化**

```
Prefill 节点：
- 需要：高 FLOPS、大显存
- 选择：H100（高算力）

Decode 节点：
- 需要：高带宽、足够显存
- 选择：H100（高带宽）或 A100（成本优化）
```

---

### PD 分离架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   PD 分离完整架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                     ┌─────────────┐                         │
│                     │   客户端    │                          │
│                     └─────────────┘                         │
│                           ↓                                 │
│                     ┌─────────────┐                         │
│                     │  负载均衡器  │                          │
│                     │  (Router)   │                          │
│                     └─────────────┘                         │
│                      ↙         ↘                            │
│            ┌─────────────┐  ┌─────────────┐                 │
│            │ Prefill Pool│  │ Decode Pool │                 │
│            │             │  │             │                 │
│            │ ┌─────────┐ │  │ ┌─────────┐ │                 │
│            │ │Prefill 1│ │  │ │Decode 1 │ │                 │
│            │ └─────────┘ │  │ └─────────┘ │                 │
│            │ ┌─────────┐ │  │ ┌─────────┐ │                 │
│            │ │Prefill 2│ │  │ │Decode 2 │ │                 │
│            │ └─────────┘ │  │ └─────────┘ │                 │
│            │    ...      │  │    ...      │                 │
│            └─────────────┘  └─────────────┘                 │
│                    ↓               ↑                        │
│            ┌─────────────────────────────────┐              │
│            │         KV Cache 传输            │              │
│            │   (RDMA / 高速网络 / 共享存储)   │              │
│            └─────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### KV Cache 传输

PD 分离的关键是 KV Cache 的高效传输：

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache 传输方案                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方案 1：RDMA 直接传输                                       │
│  ┌────────────────┐                    ┌────────────────┐  │
│  │ Prefill GPU    │  RDMA Write        │ Decode GPU     │  │
│  │ KV Cache Buffer│ ─────────────────→ │ KV Cache Buffer│  │
│  └────────────────┘                    └────────────────┘  │
│                                                             │
│  延迟：< 1ms（同机房）                                        │
│  带宽：100+ GB/s (InfiniBand)                               │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  方案 2：共享存储                                            │
│  ┌────────────────┐      ┌────────┐      ┌────────────────┐│
│  │ Prefill GPU    │ ───→ │ Redis/ │ ───→ │ Decode GPU     ││
│  │                │      │ SSD    │      │                ││
│  └────────────────┘      └────────┘      └────────────────┘│
│                                                             │
│  延迟：~10ms                                                 │
│  适用：跨数据中心                                             │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  方案 3：Mooncake 传输层                                     │
│  - 优化的 KV Cache 序列化                                    │
│  - 多种传输协议支持                                          │
│  - 自动拓扑感知                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### PD 分离的实现（SGLang 风格）

```python
import sglang as sgl
from sglang import Router

# 定义 PD 分离架构
class PDCluster:
    def __init__(self, prefill_nodes, decode_nodes):
        self.prefill_pool = prefill_nodes  # Prefill 节点池
        self.decode_pool = decode_nodes    # Decode 节点池
        self.kv_cache_store = KVCacheStore()  # KV Cache 存储
    
    def process_request(self, request):
        # 1. 在 Prefill 节点执行 Prefill
        prefill_node = self.select_prefill_node()
        kv_cache = prefill_node.prefill(request.prompt)
        
        # 2. 传输 KV Cache 到 Decode 节点
        decode_node = self.select_decode_node()
        self.transfer_kv_cache(kv_cache, decode_node)
        
        # 3. 在 Decode 节点执行 Decode
        output = decode_node.decode(
            kv_cache=kv_cache,
            max_tokens=request.max_tokens
        )
        
        return output
    
    def select_prefill_node(self):
        # 负载均衡选择 Prefill 节点
        return min(self.prefill_pool, key=lambda n: n.load)
    
    def select_decode_node(self):
        # 负载均衡选择 Decode 节点
        return min(self.decode_pool, key=lambda n: n.load)


# 单个节点的实现
class PrefillNode:
    def __init__(self, model_path, gpu_ids):
        self.engine = sgl.Engine(model_path, tp=len(gpu_ids))
    
    def prefill(self, prompt):
        # 执行 Prefill，返回 KV Cache
        output = self.engine.prefill(prompt)
        return output.kv_cache


class DecodeNode:
    def __init__(self, model_path, gpu_ids):
        self.engine = sgl.Engine(model_path, tp=len(gpu_ids))
    
    def decode(self, kv_cache, max_tokens):
        # 接收 KV Cache，执行 Decode
        output = self.engine.decode(
            kv_cache=kv_cache,
            max_tokens=max_tokens
        )
        return output


# 启动 PD 集群
cluster = PDCluster(
    prefill_nodes=[
        PrefillNode("meta-llama/Llama-2-70b", [0, 1, 2, 3])
    ],
    decode_nodes=[
        DecodeNode("meta-llama/Llama-2-70b", [4, 5, 6, 7])
    ]
)

# 处理请求
result = cluster.process_request(request)
```

---

### PD 分离的挑战

**挑战 1：KV Cache 传输延迟**

```
问题：KV Cache 可能很大（数十 GB），传输时间长

解决：
1. RDMA 高速网络
2. KV Cache 压缩
3. 流水线传输（边 Prefill 边传输）
```

**挑战 2：负载均衡**

```
问题：Prefill 和 Decode 负载不均衡

解决：
1. 动态调度
2. Chunked Prefill（分块 Prefill）
3. 抢占式调度
```

**挑战 3：系统复杂度**

```
问题：架构复杂，运维成本高

解决：
1. 自动化部署
2. 统一监控
3. 成熟的框架
```

---

## 6.3 HiCache / KVCache Offloading：把 KV Cache 搬到 CPU/SSD

### 问题背景

当上下文长度从 4K 增长到 128K+，KV Cache 成为显存大户：

```
Llama-2-70B, 128K context:
KV Cache ≈ 320 GB ❌ 单卡放不下
```

即使 PD 分离，Decode 节点也可能面临显存压力。

---

### 多级存储方案

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU 多级存储架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: GPU HBM (最快，最小，最贵)                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 容量: 80 GB (A100) / 141 GB (H100)                      ││
│  │ 带宽: 2 TB/s (A100) / 3.35 TB/s (H100)                 ││
│  │ 延迟: ~100 ns                                            ││
│  │ 存储: 当前活跃的 KV Cache                                 ││
│  └─────────────────────────────────────────────────────────┘│
│                         ↓                                   │
│  Level 2: CPU DRAM (中速，中大，中贵)                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 容量: 512 GB - 2 TB                                      ││
│  │ 带宽: ~100 GB/s                                          ││
│  │ 延迟: ~100 ns                                            ││
│  │ 存储: 待使用的 KV Cache / 换出的 KV Cache                 ││
│  └─────────────────────────────────────────────────────────┘│
│                         ↓                                   │
│  Level 3: SSD (慢速，最大，便宜)                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 容量: 4 TB - 100 TB                                      ││
│  │ 带宽: ~5 GB/s (NVMe)                                     ││
│  │ 延迟: ~10 μs                                             ││
│  │ 存储: 冷 KV Cache / 长期存储                             ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### HiCache：层级缓存

**HiCache (Hierarchical Cache)** 是 SGLang 提出的多层级 KV Cache 管理方案：

```
┌─────────────────────────────────────────────────────────────┐
│                     HiCache 工作流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  请求到达：                                                  │
│                                                             │
│  1. 检查 GPU HBM                                             │
│     ├── 命中 → 直接使用                                      │
│     └── 未命中 → 检查 CPU DRAM                               │
│                    ├── 命中 → 加载到 GPU HBM                 │
│                    └── 未命中 → 检查 SSD                     │
│                                   ├── 命中 → 加载到 DRAM→GPU │
│                                   └── 未命中 → 重新计算       │
│                                                             │
│  显存满时：                                                  │
│                                                             │
│  1. 根据策略选择淘汰的 KV Cache                              │
│  2. 写入下一级存储                                          │
│  3. 释放 GPU 显存                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Offloading 策略

**策略 1：异步卸载**

```python
class KVCacheManager:
    def __init__(self, gpu_capacity, cpu_capacity):
        self.gpu_cache = {}  # GPU HBM
        self.cpu_cache = {}  # CPU DRAM
        self.gpu_capacity = gpu_capacity
        self.cpu_capacity = cpu_capacity
    
    def get(self, request_id):
        # 1. 检查 GPU
        if request_id in self.gpu_cache:
            return self.gpu_cache[request_id]
        
        # 2. 检查 CPU（需要异步加载）
        if request_id in self.cpu_cache:
            kv_cache = self.cpu_cache[request_id]
            # 异步加载到 GPU
            future = self.async_load_to_gpu(kv_cache)
            return future
    
    def put(self, request_id, kv_cache):
        # 如果 GPU 满了，先卸载
        if self.gpu_memory_used + kv_cache.size > self.gpu_capacity:
            self.evict_lru()
        
        self.gpu_cache[request_id] = kv_cache
    
    def evict_lru(self):
        # 找到最久未使用的 KV Cache
        lru_id = self.find_lru_request()
        kv_cache = self.gpu_cache.pop(lru_id)
        
        # 异步卸载到 CPU
        self.async_offload_to_cpu(lru_id, kv_cache)
```

---

### Offloading 的性能开销

```
┌─────────────────────────────────────────────────────────────┐
│                   Offloading 性能分析                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景：Llama-2-70B, 128K context                            │
│                                                             │
│  不使用 Offloading：                                          │
│  - 需要 GPU: ~400 GB (4 × H100)                             │
│  - Decode 延迟: ~30ms/token                                 │
│                                                             │
│  使用 CPU Offloading：                                        │
│  - 需要 GPU: ~100 GB (1 × H100)                             │
│  - 额外开销: ~5-10ms (PCIe 传输)                             │
│  - Decode 延迟: ~35-40ms/token                              │
│                                                             │
│  结论：                                                      │
│  - 显存节省 75%                                              │
│  - 性能损失 17-33%                                           │
│  - 适合成本敏感场景                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Offloading 优化技术

**优化 1：预取**

```python
# 预测下一步需要的 KV Cache，提前加载
def prefetch_next_kv_cache(current_request):
    # 根据访问模式预测
    next_request = predict_next_request(current_request)
    if next_request in cpu_cache:
        async_load_to_gpu(next_request)
```

**优化 2：流水线卸载**

```python
# 在生成过程中异步卸载
def decode_with_offload(kv_cache):
    for token in range(max_tokens):
        # 生成 token
        output_token = generate_token(kv_cache)
        
        # 异步卸载旧的 KV Cache
        if token % offload_interval == 0:
            async_offload_older_layers(kv_cache)
        
        yield output_token
```

**优化 3：压缩**

```python
# KV Cache 压缩
def compress_kv_cache(kv_cache):
    # 方法 1：量化
    kv_cache_int8 = quantize_to_int8(kv_cache)
    
    # 方法 2：稀疏化
    kv_cache_sparse = sparsify(kv_cache)
    
    return kv_cache_compressed  # 压缩比 2-4x
```

---

### Offloading 实现（简化版）

```python
import torch
import torch.distributed as dist
from typing import Dict, Optional
import asyncio

class KVCacheOffloader:
    """KV Cache 多级存储管理器"""
    
    def __init__(self, gpu_layers: int, cpu_layers: int, ssd_path: str):
        self.gpu_layers = gpu_layers  # GPU 存储的层数
        self.cpu_layers = cpu_layers  # CPU 存储的层数
        self.ssd_path = ssd_path      # SSD 存储路径
        
        # 存储结构
        self.gpu_cache: Dict[str, torch.Tensor] = {}
        self.cpu_cache: Dict[str, torch.Tensor] = {}
        self.ssd_cache: Dict[str, str] = {}
        
        # 统计信息
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.ssd_hits = 0
        self.misses = 0
    
    async def get(self, request_id: str, layer_idx: int) -> Optional[torch.Tensor]:
        """获取 KV Cache"""
        
        # GPU 缓存
        cache_key = f"{request_id}_{layer_idx}"
        if cache_key in self.gpu_cache:
            self.gpu_hits += 1
            return self.gpu_cache[cache_key]
        
        # CPU 缓存 -> 异步加载到 GPU
        if cache_key in self.cpu_cache:
            self.cpu_hits += 1
            kv_cache = self.cpu_cache[cache_key]
            await self.load_to_gpu(cache_key, kv_cache)
            return self.gpu_cache[cache_key]
        
        # SSD 缓存 -> 异步加载到 CPU -> GPU
        if cache_key in self.ssd_cache:
            self.ssd_hits += 1
            file_path = self.ssd_cache[cache_key]
            kv_cache = await self.load_from_ssd(file_path)
            await self.load_to_gpu(cache_key, kv_cache)
            return self.gpu_cache[cache_key]
        
        self.misses += 1
        return None
    
    async def put(self, request_id: str, layer_idx: int, kv_cache: torch.Tensor):
        """存储 KV Cache"""
        
        cache_key = f"{request_id}_{layer_idx}"
        
        # 如果 GPU 满了，先淘汰
        if len(self.gpu_cache) >= self.gpu_layers:
            await self.evict_lru()
        
        self.gpu_cache[cache_key] = kv_cache
    
    async def load_to_gpu(self, key: str, kv_cache: torch.Tensor):
        """加载到 GPU"""
        if len(self.gpu_cache) >= self.gpu_layers:
            await self.evict_lru()
        
        self.gpu_cache[key] = kv_cache.cuda()
    
    async def evict_lru(self):
        """淘汰最久未使用的缓存"""
        # 找 LRU key
        lru_key = min(self.gpu_cache.keys())  # 简化实现
        
        # 写入 CPU
        kv_cache = self.gpu_cache.pop(lru_key)
        self.cpu_cache[lru_key] = kv_cache.cpu()
        
        # 如果 CPU 也满了，写入 SSD
        if len(self.cpu_cache) >= self.cpu_layers:
            await self.offload_to_ssd()
    
    async def offload_to_ssd(self):
        """卸载到 SSD"""
        lru_key = min(self.cpu_cache.keys())
        kv_cache = self.cpu_cache.pop(lru_key)
        
        # 保存到文件
        file_path = f"{self.ssd_path}/{lru_key}.pt"
        torch.save(kv_cache, file_path)
        self.ssd_cache[lru_key] = file_path
    
    async def load_from_ssd(self, file_path: str) -> torch.Tensor:
        """从 SSD 加载"""
        return torch.load(file_path)
    
    def stats(self) -> Dict:
        """返回统计信息"""
        total = self.gpu_hits + self.cpu_hits + self.ssd_hits + self.misses
        return {
            "gpu_hits": self.gpu_hits,
            "cpu_hits": self.cpu_hits,
            "ssd_hits": self.ssd_hits,
            "misses": self.misses,
            "hit_rate": (total - self.misses) / total if total > 0 else 0
        }
```

---

## 6.4 Speculative Decoding：推测解码

### 核心思想

Decode 阶段低效的原因是每次只生成 1 个 token，GPU 利用率低。

**Speculative Decoding** 的想法是：用一个小模型快速"猜"多个 token，然后用大模型验证。

```
┌─────────────────────────────────────────────────────────────┐
│                  Speculative Decoding 原理                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传统 Decode：                                               │
│                                                             │
│  大模型: T1 → T2 → T3 → T4 → T5                             │
│         30ms  30ms  30ms  30ms  30ms                        │
│         总计: 150ms                                         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Speculative Decoding:                                       │
│                                                             │
│  Step 1: 小模型快速猜测                                       │
│  小模型: T1' → T2' → T3' → T4' → T5' (猜测 5 个)            │
│          2ms   2ms   2ms   2ms   2ms                        │
│          总计: 10ms                                          │
│                                                             │
│  Step 2: 大模型验证                                          │
│  大模型: 验证 T1' T2' T3' T4' T5'                            │
│          30ms (并行验证)                                     │
│          假设 T4' 被拒绝，保留 T1' T2' T3'                    │
│                                                             │
│  Step 3: 大模型生成正确 token                                 │
│  大模型: T4 (5ms)                                            │
│                                                             │
│  总计: 10ms + 30ms + 5ms = 45ms                             │
│  加速: 150ms / 45ms = 3.3x                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Speculative Decoding 数学原理

**为什么可以一次验证多个 token？**

关键洞察：Transformer 从第 i 个位置开始的计算，可以并行产生第 i, i+1, i+2, ... 位置的 logits。

```
传统方式（串行）：
位置 0: 输入 [BOS] → 输出 logit[0] → sample T1
位置 1: 输入 [BOS, T1] → 输出 logit[1] → sample T2
...

Speculative 方式（并行验证）：
输入 [BOS, T1', T2', T3', T4', T5']
输出 [logit[0], logit[1], logit[2], logit[3], logit[4], logit[5]]

一次 forward 得到所有位置的 logit，可以并行验证！
```

---

### 接受/拒绝机制

```python
def speculative_decode_step(draft_model, target_model, prompt, num_spec_tokens=5):
    """单步推测解码"""
    
    # 1. Draft model 快速生成多个 token
    draft_tokens = []
    current_input = prompt
    for _ in range(num_spec_tokens):
        draft_token = draft_model.generate(current_input, max_tokens=1)
        draft_tokens.append(draft_token)
        current_input = torch.cat([current_input, draft_token], dim=-1)
    
    # 2. Target model 并行验证
    # 输入: prompt + draft_tokens
    verify_input = torch.cat([prompt] + draft_tokens, dim=-1)
    target_logits = target_model.forward(verify_input)  # 一次 forward
    
    # 3. 逐个验证并接受/拒绝
    accepted_tokens = []
    for i, draft_token in enumerate(draft_tokens):
        # 计算接受概率
        draft_prob = softmax(draft_model.get_logits(prompt, draft_tokens[:i]))
        target_prob = softmax(target_logits[i])
        
        p_draft = draft_prob[draft_token]
        p_target = target_prob[draft_token]
        
        # 接受条件
        accept_prob = min(1, p_target / p_draft) if p_draft > 0 else 1
        
        if random() < accept_prob:
            accepted_tokens.append(draft_token)
        else:
            # 拒绝后，从 target 分布采样
            corrected_token = sample_from_dist(target_prob)
            accepted_tokens.append(corrected_token)
            break
    
    # 4. 如果全部接受，额外采样一个 token
    if len(accepted_tokens) == num_spec_tokens:
        bonus_token = sample_from_dist(target_logits[-1])
        accepted_tokens.append(bonus_token)
    
    return accepted_tokens
```

---

### Speculative Decoding 性能分析

**加速比计算：**

```
假设：
- 大模型 Decode 时间: T_large = 30ms
- 小模型 Decode 时间: T_small = 2ms
- 推测 tokens 数: k = 5
- 接受率: α (每个 token 被接受的概率)

加速比 ≈ (平均接受的 tokens 数) × (时间节省)

具体公式：
E[接受的 tokens 数] = 1 + α + α² + ... + α^{k} + α^{k+1}
                   = (1 - α^{k+2}) / (1 - α)  (几何级数)

时间开销：
- 小模型推测: k × T_small
- 大模型验证: T_large
- 总时间: k × T_small + T_large

加速比 = E[接受的 tokens] × T_large / (k × T_small + T_large)
```

**示例计算：**

```
α = 0.7, k = 5, T_large = 30ms, T_small = 2ms

E[接受的 tokens] = (1 - 0.7^7) / (1 - 0.7) ≈ 3.3

加速比 = 3.3 × 30 / (5 × 2 + 30) = 99 / 40 = 2.48x
```

---

### Draft Model 选择

**方案 1：独立小模型**

```python
# 使用一个独立的小模型作为 draft
draft_model = Llama-2-7B
target_model = Llama-2-70B

# 优势：速度快
# 劣势：需要额外显存，模型不匹配可能导致接受率低
```

**方案 2：Self-Speculative（自推测）**

```python
# 用大模型的部分层作为 draft
draft_model = Llama-2-70B 的前 8 层
target_model = Llama-2-70B 完整模型

# 优势：无需额外模型，接受率高
# 劣势：draft 稍慢
```

**方案 3：Medusa Heads**

```python
# 在大模型上加多个解码头
class MedusaModel(nn.Module):
    def __init__(self, base_model, num_heads=4):
        self.base_model = base_model
        self.medusa_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size) 
            for _ in range(num_heads)
        ])
    
    def forward(self, input_ids):
        hidden_states = self.base_model(input_ids)
        
        # 主输出
        main_logits = self.base_model.lm_head(hidden_states)
        
        # Medusa heads 预测后续 tokens
        medusa_logits = [head(hidden_states) for head in self.medusa_heads]
        
        return main_logits, medusa_logits

# 优势：无需额外模型，高度集成
# 劣势：需要训练 medusa heads
```

---

### Speculative Decoding 实战

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    def __init__(self, draft_model_name, target_model_name, device="cuda"):
        # 加载模型
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name).to(device)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.device = device
    
    def generate(self, prompt, max_tokens=100, num_spec_tokens=4):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        generated = input_ids.clone()
        total_accepted = 0
        total_speculated = 0
        
        while generated.shape[1] < input_ids.shape[1] + max_tokens:
            # 1. Draft model 推测
            draft_tokens = self._draft_generate(generated, num_spec_tokens)
            total_speculated += num_spec_tokens
            
            # 2. Target model 验证
            accepted, new_tokens = self._verify(generated, draft_tokens)
            total_accepted += len(new_tokens)
            
            # 3. 更新
            generated = torch.cat([generated, new_tokens.unsqueeze(0)], dim=1)
            
            # 4. 检查 EOS
            if new_tokens[-1] == self.tokenizer.eos_token_id:
                break
        
        acceptance_rate = total_accepted / total_speculated
        print(f"Acceptance rate: {acceptance_rate:.2%}")
        print(f"Avg tokens per step: {total_accepted / (max_tokens / num_spec_tokens):.2f}")
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def _draft_generate(self, input_ids, num_tokens):
        """Draft model 快速生成"""
        draft_tokens = []
        current = input_ids
        
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = self.draft_model(current)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                draft_tokens.append(next_token.item())
                current = torch.cat([current, next_token.unsqueeze(0)], dim=1)
        
        return draft_tokens
    
    def _verify(self, input_ids, draft_tokens):
        """Target model 验证"""
        # 构建验证输入
        draft_tensor = torch.tensor(draft_tokens, device=self.device).unsqueeze(0)
        verify_input = torch.cat([input_ids, draft_tensor], dim=1)
        
        # Target model forward
        with torch.no_grad():
            outputs = self.target_model(verify_input)
            logits = outputs.logits
        
        # 验证每个 draft token
        accepted = []
        for i, draft_token in enumerate(draft_tokens):
            target_logits = logits[0, input_ids.shape[1] + i - 1, :]
            
            # 简化：直接比较 argmax
            target_token = torch.argmax(target_logits).item()
            
            if target_token == draft_token:
                accepted.append(draft_token)
            else:
                accepted.append(target_token)
                break
        
        # 如果全部接受，生成一个 bonus token
        if len(accepted) == len(draft_tokens):
            bonus_logits = logits[0, -1, :]
            bonus_token = torch.argmax(bonus_logits).item()
            accepted.append(bonus_token)
        
        return len(accepted), torch.tensor(accepted, device=self.device)


# 使用示例
decoder = SpeculativeDecoder(
    draft_model_name="meta-llama/Llama-2-7b-hf",
    target_model_name="meta-llama/Llama-2-70b-hf"
)

output = decoder.generate("The capital of France is", max_tokens=50)
print(output)
```

---

### Speculative Decoding 适用场景

| 场景 | 适用性 | 说明 |
|------|--------|------|
| 低延迟推理 | ⭐⭐⭐⭐⭐ | 显著降低 TPOT |
| 高吞吐推理 | ⭐⭐⭐ | 加速有限 |
| 长文本生成 | ⭐⭐⭐⭐ | 更明显的加速效果 |
| 模型本身很快 | ⭐⭐ | 额外开销可能抵消收益 |
| 批量推理 | ⭐⭐ | 实现复杂，收益不稳定 |

---

## 本章小结

1. **Prefill vs Decode 本质差异**：Prefill 计算密集、GPU 利用率高；Decode 内存密集、GPU 利用率低。这一差异是所有优化的理论基础。

2. **PD 分离架构**：将 Prefill 和 Decode 分离到不同节点，资源专精化，显著提升整体效率。关键技术是 KV Cache 的高效传输。

3. **KV Cache Offloading**：多级存储方案（GPU→CPU→SSD），显著节省显存，会有性能损失。适合长上下文和成本敏感场景。

4. **Speculative Decoding**：用小模型猜测 + 大模型验证，一次验证多个 token，显著加速 Decode 阶段。关键在于选择合适的 draft model 和保持高的接受率。

下一章，我们将进入量化的世界，看看如何通过降低精度来压缩模型、加速推理。

---

*放弃指数：⭐⭐⭐⭐⭐ 本章是推理优化的前沿，涉及大量架构设计。理解原理后，建议阅读 vLLM/SGLang 源码深入学习。*

---

*（未完待续...）*