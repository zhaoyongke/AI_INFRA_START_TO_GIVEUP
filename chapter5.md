# 第五章 模型并行 —— 单卡塞不下了

> *放弃指数：⭐⭐⭐⭐⭐ 本章是硬核中的硬核，建议配合实践理解*

---

## 5.1 Tensor Parallelism (TP)：把模型切开

### 为什么需要 TP？

当你有一个 70B 参数的模型，FP16 存储，需要显存：
```
70B × 2 bytes = 140GB
```

而 A100 80GB 放不下。怎么办？

**答案：把模型切成多份，放在多张卡上。**

---

### TP 原理：矩阵分块乘法

Tensor Parallelism 的核心思想是：**把矩阵乘法分块并行计算**。

假设我们要计算矩阵乘法 `Y = X @ W`：

```
┌─────────────────────────────────────────────────────────────┐
│              Tensor Parallelism 原理                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始计算：Y = X @ W                                         │
│                                                             │
│  X: [batch, hidden]                                         │
│  W: [hidden, hidden]                                        │
│  Y: [batch, hidden]                                         │
│                                                             │
│  ────────────────────────────────────────────────────────  │
│                                                             │
│  TP 切分（按列切分）：                                         │
│                                                             │
│  W = [W₁ | W₂]  (将 W 按列切成两半)                          │
│                                                             │
│  Y₁ = X @ W₁  → GPU 0 计算                                   │
│  Y₂ = X @ W₂  → GPU 1 计算                                   │
│                                                             │
│  Y = [Y₁ | Y₂]  (拼接结果)                                   │
│                                                             │
│  ┌───────────────┐   ┌───────────────┐                      │
│  │    GPU 0      │   │    GPU 1      │                      │
│  │  X @ W₁ → Y₁  │   │  X @ W₂ → Y₂  │                      │
│  └───────────────┘   └───────────────┘                      │
│           ↘                ↙                                │
│            ┌─────────────────┐                              │
│            │ Y = concat(Y₁,Y₂)│                              │
│            └─────────────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Transformer 中的 TP

Transformer 的核心结构是 **Self-Attention** 和 **MLP**，我们来分别看怎么切。

#### Self-Attention 的 TP 切分

```
┌─────────────────────────────────────────────────────────────┐
│                  Self-Attention TP 切分                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始结构：                                                   │
│  Q = X @ Wq     K = X @ Wk     V = X @ Wv                   │
│  Attention(Q, K, V)                                          │
│  Output = Attention_result @ Wo                             │
│                                                             │
│  ────────────────────────────────────────────────────────  │
│                                                             │
│  TP 切分后：                                                  │
│                                                             │
│  GPU 0:                           GPU 1:                     │
│  Q₀ = X @ Wq₀                    Q₁ = X @ Wq₁               │
│  K₀ = X @ Wk₀                    K₁ = X @ Wk₁               │
│  V₀ = X @ Wv₀                    V₁ = X @ Wv₁               │
│  Attn₀ = Attention(Q₀,K₀,V₀)     Attn₁ = Attention(Q₁,K₁,V₁)│
│  Y₀ = Attn₀ @ Wo₀                Y₁ = Attn₁ @ Wo₁           │
│                                                             │
│  最终输出：Y = Y₀ + Y₁ (AllReduce 求和)                       │
│                                                             │
│  关键点：                                                     │
│  - QKV 权重按列切分（Column Parallel）                        │
│  - Output 权重按行切分（Row Parallel）                        │
│  - 每个 attention head 完整在一个 GPU 上                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### MLP 的 TP 切分

```
┌─────────────────────────────────────────────────────────────┐
│                      MLP TP 切分                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始结构：                                                   │
│  Hidden → Gate_proj → SiLU                                  │
│         → Up_proj   → *                                     │
│         → Down_proj                                         │
│                                                             │
│  TP 切分后（TP=2）：                                          │
│                                                             │
│  GPU 0:                           GPU 1:                     │
│  Gate₀ = X @ Gate_proj₀          Gate₁ = X @ Gate_proj₁     │
│  Up₀ = X @ Up_proj₀              Up₁ = X @ Up_proj₁         │
│  Hidden₀ = Gate₀ * SiLU(Up₀)     Hidden₁ = Gate₁ * SiLU(Up₁)│
│  Out₀ = Hidden₀ @ Down_proj₀     Out₁ = Hidden₁ @ Down_proj₁│
│                                                             │
│  最终输出：Out = Out₀ + Out₁ (AllReduce)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### TP 的通信开销

TP 需要同步的点：

```python
# TP = 2 示例

# 通信点 1：Forward 开始时，广播输入 X（或每个 GPU 自己计算）
# 通常是 Broadcast 或每个 GPU 从同一输入读取

# 通信点 2：Attention 输出后，AllReduce
output = all_reduce(output_shards)  # SUM

# 通信点 3：MLP 输出后，AllReduce
output = all_reduce(output_shards)  # SUM

# 通信点 4：Backward 时，梯度 AllReduce
grad = all_reduce(grad_shards)
```

**每一层的通信量：**
```
通信量 ≈ 2 × batch_size × seq_len × hidden_size × 4 bytes (FP32)
```

**关键结论：TP 适合高带宽互联（NVLink）。**

---

### TP 的实现（PyTorch）

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """按列切分的线性层"""
    
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        assert out_features % world_size == 0
        
        # 每个 GPU 只存储 1/world_size 的权重
        self.weight = nn.Parameter(
            torch.empty(out_features // world_size, in_features)
        )
        self.bias = nn.Parameter(
            torch.empty(out_features // world_size)
        )
    
    def forward(self, x):
        # 每个 GPU 计算自己的部分
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        return output  # [batch, out_features // world_size]


class RowParallelLinear(nn.Module):
    """按行切分的线性层"""
    
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        assert in_features % world_size == 0
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features // world_size)
        )
    
    def forward(self, x):
        # 每个 GPU 计算自己的部分
        output = torch.nn.functional.linear(x, self.weight)
        
        # AllReduce 求和
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


class TPLinearBlock(nn.Module):
    """完整的 TP 线性块：Column → ReLU → Row"""
    
    def __init__(self, hidden_size, world_size):
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden_size, 4 * hidden_size, world_size)
        self.fc2 = RowParallelLinear(4 * hidden_size, hidden_size, world_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
```

---

### TP 的局限性

| 优点 | 缺点 |
|------|------|
| 无需修改模型逻辑 | 通信开销大（每层都通信） |
| 延迟低（单次迭代） | 需要 NVLink 高带宽 |
| 实现相对简单 | TP 规模受限（通常 ≤ 8） |
| 每个 GPU 有完整激活 | 显存节省不如 PP |

**经验法则：**
- NVLink 互联：TP ≤ 8
- PCIe 互联：TP ≤ 2（性能急剧下降）

---

## 5.2 Pipeline Parallelism (PP)：把层排好队

### PP 原理：流水线思想

TP 是把每一层横向切开，PP 是把模型纵向切开——不同层放在不同 GPU 上。

```
┌─────────────────────────────────────────────────────────────┐
│                  Pipeline Parallelism 示意图                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GPU 0: Layer 0-7                                           │
│  GPU 1: Layer 8-15                                          │
│  GPU 2: Layer 16-23                                         │
│  GPU 3: Layer 24-31                                         │
│                                                             │
│  数据流动：                                                   │
│                                                             │
│  Batch 1: GPU 0 → GPU 1 → GPU 2 → GPU 3                     │
│  Batch 2:        GPU 0 → GPU 1 → GPU 2 → GPU 3              │
│  Batch 3:               GPU 0 → GPU 1 → GPU 2 → GPU 3       │
│                                                             │
│  问题：气泡（Bubble）                                         │
│  ┌────────────────────────────────────────────────────────┐│
│  │ 时间 →                                                 ││
│  │ GPU 0: ████████░░░░░░░░████████░░░░░░░░████████        ││
│  │ GPU 1: ░░░░████████░░░░░░░░████████░░░░░░░░████████    ││
│  │ GPU 2: ░░░░░░░░████████░░░░░░░░████████░░░░░░░░████████││
│  │ GPU 3: ░░░░░░░░░░░░████████░░░░░░░░████████░░░░░░░░████││
│  │        █ 处理中  ░ 空闲（气泡）                          ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### PP 的气泡问题

朴素 PP 的问题是 GPU 空闲时间多。解决方法是 **Micro-batching**：

```
┌─────────────────────────────────────────────────────────────┐
│                   Micro-batching 优化                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  将一个 batch 分成多个 micro-batch：                          │
│                                                             │
│  原始：batch_size = 32                                       │
│  拆分：4 个 micro-batch，每个 8 samples                       │
│                                                             │
│  流水线填充后：                                               │
│  ┌────────────────────────────────────────────────────────┐│
│  │ 时间 →                                                 ││
│  │ GPU 0: ████████████████████████████████████████████    ││
│  │ GPU 1: ░░░░████████████████████████████████████████    ││
│  │ GPU 2: ░░░░░░░░████████████████████████████████████    ││
│  │ GPU 3: ░░░░░░░░░░░░████████████████████████████████    ││
│  │                                                         ││
│  │ 气泡比例 ≈ (PP_size - 1) / (PP_size + num_micro_batches - 1) │
│  │                                                        ││
│  │ 例：PP=4, micro_batches=32，气泡比例 ≈ 9%               ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### PP 的调度策略

**1F1B (One Forward One Backward)** 是最常用的调度策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    1F1B 调度示意                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段 1：Warm-up（填充流水线）                                 │
│  阶段 2：稳态（1 forward + 1 backward）                       │
│  阶段 3：Cool-down（排空流水线）                               │
│                                                             │
│  GPU 0: F0→F1→F2→F3 → F4B0 → F5B1 → F6B2 → F7B3 → B4→B5→B6→B7│
│  GPU 1:   F0→F1→F2 → F3B0 → F4B1 → F5B2 → F6B3 → B4→B5→B6→B7 │
│  GPU 2:     F0→F1 → F2B0 → F3B1 → F4B2 → F5B3 → B4→B5→B6→B7  │
│  GPU 3:       F0 → F1B0 → F2B1 → F3B2 → F4B3 → B4→B5→B6→B7   │
│                                                             │
│  F = Forward, B = Backward, 数字 = micro-batch 编号          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### PP 的通信开销

PP 每次只在相邻 GPU 之间传递激活值：

```python
# PP 的通信点

# Forward: 传递激活值
activation = layer_forward(activation)
send_to_next_gpu(activation)  # P2P 通信

# Backward: 传递梯度
grad = layer_backward(grad)
send_to_prev_gpu(grad)  # P2P 通信

# 通信量
activation_size = batch_size × seq_len × hidden_size × num_layers_per_stage
```

**PP vs TP 通信对比：**

| 方式 | 通信频率 | 通信量/次 | 依赖带宽 |
|------|---------|----------|---------|
| TP | 每层 | medium | 高（NVLink） |
| PP | 每层传递一次 | 大 | 中（可用 PCIe） |

---

### PP 的实现（PyTorch + DeepSpeed）

```python
import torch
import torch.nn as nn
from deepspeed import pipe

# 定义模型层
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Pipeline 并行包装
class PipelineModel(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, num_stages):
        super().__init__()
        self.num_stages = num_stages
        layers_per_stage = num_layers // num_stages
        
        # 根据 rank 决定当前 stage 有哪些层
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads) 
            for _ in range(layers_per_stage)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# DeepSpeed Pipeline 配置
ds_config = {
    "pipeline": {
        "enabled": True,
        "parallel_size": 4,  # PP = 4
        "micro_batches": 32,
        "activation_checkpoint_interval": 1
    }
}
```

---

### PP 的优缺点

| 优点 | 缺点 |
|------|------|
| 通信量小（仅需 P2P） | 有气泡，效率无法 100% |
| 可扩展性好（可用 PCIe） | 延迟高（需等整个管道） |
| 显存节省大 | 实现复杂 |
| 与 TP 可组合 | 需要调整 batch 策略 |

---

## 5.3 Expert Parallelism (EP)：MoE 的艺术

### MoE 架构基础

Mixture of Experts (MoE) 是一种稀疏激活的架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE 架构示意                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  标准Transformer：                                           │
│  Input → Attention → MLP (所有参数参与) → Output             │
│                                                             │
│  MoE Transformer：                                           │
│  Input → Attention → MoE Layer → Output                     │
│                          ↓                                  │
│                  ┌───────────────────┐                      │
│                  │     Router        │                      │
│                  │   (决定选哪些专家)  │                      │
│                  └───────────────────┘                      │
│                     ↙   ↓   ↘                               │
│              ┌──────┬──────┬──────┬──────┐                  │
│              │专家1 │专家2 │专家3 │专家4 │ ...               │
│              │      │      │      │      │                  │
│              └──────┴──────┴──────┴──────┘                  │
│                     ↘   ↓   ↙                               │
│                  ┌───────────────────┐                      │
│                  │   Combine 输出    │                      │
│                  └───────────────────┘                      │
│                                                             │
│  特点：每个 token 只激活 top-k 个专家，稀疏计算              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### EP 原理：专家分布在不同 GPU

**Expert Parallelism** 将不同的专家放在不同的 GPU 上：

```
┌─────────────────────────────────────────────────────────────┐
│                    Expert Parallelism                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  假设 8 个专家，EP = 4：                                       │
│                                                             │
│  GPU 0: 专家 0, 专家 1                                        │
│  GPU 1: 专家 2, 专家 3                                        │
│  GPU 2: 专家 4, 专家 5                                        │
│  GPU 3: 专家 6, 专家 7                                        │
│                                                             │
│  工作流程：                                                   │
│                                                             │
│  1. Router 决定每个 token 应该去哪些专家                       │
│  2. All-to-All 通信：把 token 发送到对应专家所在的 GPU         │
│  3. 专家计算                                                  │
│  4. All-to-All 通信：把结果返回原 GPU                         │
│  5. Combine 输出                                              │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │                                                         ││
│  │  Token 0 → 专家 2 (GPU 1) ──┐                          ││
│  │  Token 1 → 专家 0 (GPU 0)   │                          ││
│  │  Token 2 → 专家 5 (GPU 2)   │                          ││
│  │  Token 3 → 专家 7 (GPU 3) ──┤                          ││
│  │                            ↓                          ││
│  │                    All-to-All 通信                      ││
│  │                            ↓                          ││
│  │                    专家计算 → All-to-All 返回           ││
│  │                            ↓                          ││
│  │                    Combine 输出                         ││
│  │                                                         ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### EP 的通信开销

EP 的核心通信是 **All-to-All**：

```python
# EP 通信示意
def expert_parallel_forward(tokens, router_logits):
    # 1. Router 计算
    expert_indices = router(router_logits)  # 每个 token 选哪些专家
    
    # 2. All-to-All 发送 token 到对应专家所在的 GPU
    tokens_for_experts = all_to_all(tokens, expert_indices)
    
    # 3. 专家计算
    expert_outputs = expert_layers(tokens_for_experts)
    
    # 4. All-to-All 返回结果
    outputs = all_to_all(expert_outputs, reverse=True)
    
    # 5. Combine
    final_output = combine(outputs, expert_weights)
    
    return final_output
```

**All-to-All 通信量：**
```
通信量 = 2 × batch_size × seq_len × hidden_size × k (top-k experts)
```

---

### EP 与 TP/PP 的组合

```
┌─────────────────────────────────────────────────────────────┐
│                  EP + TP 组合示意                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  总共 64 个专家，TP=2，EP=4                                   │
│                                                             │
│  物理布局：                                                   │
│                                                             │
│       EP 维度（专家分布）                                      │
│       ─────────────────────────────────────────→            │
│  │   ┌─────────┬─────────┬─────────┬─────────┐              │
│  │   │ GPU 0   │ GPU 1   │ GPU 2   │ GPU 3   │              │
│  T   │ 专家    │ 专家    │ 专家    │ 专家    │              │
│  P   │ 0-7     │ 8-15    │ 16-23   │ 24-31   │              │
│  维  │ (前半)  │ (前半)  │ (前半)  │ (前半)  │              │
│  度  ├─────────┼─────────┼─────────┼─────────┤              │
│  │   │ GPU 4   │ GPU 5   │ GPU 6   │ GPU 7   │              │
│  │   │ 专家    │ 专家    │ 专家    │ 专家    │              │
│  ↓   │ 0-7     │ 8-15    │ 16-23   │ 24-31   │              │
│      │ (后半)  │ (后半)  │ (后半)  │ (后半)  │              │
│      └─────────┴─────────┴─────────┴─────────┘              │
│                                                             │
│  - EP 把不同专家分布到不同 GPU                                │
│  - TP 把单个专家切分到多个 GPU                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### EP 的实现（Megablocks 风格）

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k, ep_size):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.experts_per_rank = num_experts // ep_size
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # 专家网络（每个 rank 只有一部分专家）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(self.experts_per_rank)
        ])
    
    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        
        # 1. Router 计算
        router_logits = self.router(x_flat)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = router_probs.topk(self.top_k, dim=-1)
        
        # 2. 确定 token 应该发送到哪个 rank
        target_ranks = topk_indices // self.experts_per_rank
        
        # 3. All-to-All 发送
        # （实际实现更复杂，需要处理 token 重排序）
        x_dispatched = self._all_to_all_dispatch(x_flat, target_ranks)
        
        # 4. 专家计算
        expert_outputs = self._compute_experts(x_dispatched, topk_indices)
        
        # 5. All-to-All 返回
        outputs = self._all_to_all_combine(expert_outputs, target_ranks)
        
        # 6. Combine
        final_output = (outputs * topk_probs.unsqueeze(-1)).sum(dim=1)
        
        return final_output.view(batch_size, seq_len, hidden)
    
    def _all_to_all_dispatch(self, x, target_ranks):
        # All-to-All 通信实现
        # ...
        pass
    
    def _compute_experts(self, x, expert_indices):
        # 专家计算
        outputs = []
        for i, expert in enumerate(self.experts):
            mask = (expert_indices % self.experts_per_rank) == i
            if mask.any():
                outputs.append(expert(x[mask]))
        return torch.cat(outputs)
    
    def _all_to_all_combine(self, x, target_ranks):
        # All-to-All 通信实现
        # ...
        pass
```

---

### EP 的挑战

**挑战 1：负载不均衡**

```
问题：某些专家总是被选中，某些专家很少被用到

解决：
1. 负载均衡损失
2. 专家容量限制
3. Token 溢出处理
```

**挑战 2：All-to-All 通信开销**

```
问题：All-to-All 是集合通信，效率依赖于网络拓扑

解决：
1. 使用 NVLink / NVSwitch
2. 合并小通信
3. 异步通信
```

---

## 5.4 Context Parallelism (CP)：长文本的新战场

### 长文本的显存挑战

当 seq_len 从 4K 增长到 128K，显存会发生什么？

```
KV Cache 显存（以 70B 模型为例）：

seq_len = 4K:
  KV Cache ≈ 10GB

seq_len = 128K:
  KV Cache ≈ 320GB  ❌ 单卡放不下
```

**Context Parallelism (CP)** 专门解决这个问题。

---

### CP 原理：切分序列

CP 将长序列切分成多段，每个 GPU 处理一段：

```
┌─────────────────────────────────────────────────────────────┐
│                   Context Parallelism                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始序列：长度 128K                                          │
│  ┌────────────────────────────────────────────────────────┐│
│  │ S₀   S₁   S₂   S₃   ...   S₁₂₇  (128K tokens)          ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  CP=4 切分后：                                               │
│  ┌──────────────┐                                           │
│  │ GPU 0: S₀-S₃₁ (32K)                                      │
│  │ GPU 1: S₃₂-S₆₃ (32K)                                     │
│  │ GPU 2: S₆₄-S₉₅ (32K)                                     │
│  │ GPU 3: S₉₆-S₁₂₇ (32K)                                    │
│  └──────────────┘                                           │
│                                                             │
│  关键问题：Self-Attention 需要整个序列！                       │
│                                                             │
│  解决方案：                                                  │
│  1. 每个 GPU 计算当前 segment 的 Q                           │
│  2. 通过 All-Gather 获取其他 segment 的 K, V                 │
│  3. 计算 Attention                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### CP 的 Attention 计算

```
┌─────────────────────────────────────────────────────────────┐
│                  CP Attention 计算流程                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 本地 QKV 计算                                        │
│  GPU 0: Q₀ = Wq × X₀, K₀ = Wk × X₀, V₀ = Wv × X₀            │
│  GPU 1: Q₁ = Wq × X₁, K₁ = Wk × X₁, V₁ = Wv × X₁            │
│  ...                                                        │
│                                                             │
│  Step 2: All-Gather K, V                                     │
│  GPU 0: 获取 [K₀, K₁, K₂, K₃], [V₀, V₁, V₂, V₃]              │
│  GPU 1: 获取 [K₀, K₁, K₂, K₃], [V₀, V₁, V₂, V₃]              │
│  ...                                                        │
│                                                             │
│  Step 3: 计算本地 Attention                                   │
│  GPU 0: Attn₀ = Q₀ × K^T × V                                 │
│  GPU 1: Attn₁ = Q₁ × K^T × V                                 │
│  ...                                                        │
│                                                             │
│  注意：K, V 需要 All-Gather，但不需要广播回                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### CP 与 Ring Attention

**Ring Attention** 是 CP 的一种高效实现：

```
Ring Attention 优化通信：

传统 All-Gather：
每个 GPU 一次性获取所有 K, V → 显存不够！

Ring 方式：
每个 GPU 只保留一个 segment 的 K, V
逐轮传递，计算完成后丢弃

Round 1:
GPU 0: 有 K₀,V₀ → 传给 GPU 1，收到 K₃,V₃
GPU 1: 有 K₁,V₁ → 传给 GPU 2，收到 K₀,V₀
GPU 2: 有 K₂,V₂ → 传给 GPU 3，收到 K₁,V₁
GPU 3: 有 K₃,V₃ → 传给 GPU 0，收到 K₂,V₂

Round 2, 3, ... 依此类推

优势：
- 显存友好（只需一个 segment 的 K,V）
- 通信计算重叠
```

---

### CP 的通信量

```python
# CP 通信量计算

# All-Gather K, V
# 每个 segment 的 K, V 大小
kv_size_per_segment = seq_len // cp_size × hidden_size × 2  # K + V

# All-Gather 总通信量
allgather_size = kv_size_per_segment × (cp_size - 1)

# Ring Attention 逐轮通信
# 每轮通信量
ring_size_per_round = kv_size_per_segment
# 总轮数
num_rounds = cp_size - 1
```

---

### CP 的实现（简化版）

```python
import torch
import torch.distributed as dist

class ContextParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, cp_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.cp_size = cp_size
        self.head_dim = hidden_size // num_heads
        
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        # 每个 GPU 只有 seq_len / cp_size 的数据
        
        batch_size, seq_len, _ = x.shape
        
        # 1. 本地 QKV 计算
        q = self.wq(x)  # [batch, seq_len/cp_size, hidden_size]
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 2. All-Gather K, V
        k_all = self._all_gather_kv(k)
        v_all = self._all_gather_kv(v)
        
        # 3. 计算 Attention
        attn_output = self._attention(q, k_all, v_all)
        
        # 4. Output projection
        output = self.wo(attn_output)
        
        return output
    
    def _all_gather_kv(self, tensor):
        # All-Gather 实现
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.cp_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=1)
    
    def _attention(self, q, k, v):
        # 标准 scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
```

---

## 5.5 3D 并行：算力不够凑数来凑

### 3D 并行组合

当模型足够大时，单一并行策略不够用，需要 **组合拳**：

```
┌─────────────────────────────────────────────────────────────┐
│                      3D 并行示意                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模型：Llama-2-70B                                           │
│  GPU：64 张 A100 80GB                                        │
│                                                             │
│  并行策略：TP=4, PP=4, DP=4                                  │
│                                                             │
│  总 GPU = TP × PP × DP = 4 × 4 × 4 = 64                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    DP 维度（数据并行）                     ││
│  │  ┌─────────────────────────────────────────────────────┐││
│  │  │ DP 0            DP 1            DP 2            DP 3 │││
│  │  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │││
│  │  │ │ PP 维度     │ │ PP 维度     │ │ PP 维度     │     │││
│  │  │ │ Stage 0-3   │ │ Stage 0-3   │ │ Stage 0-3   │     │││
│  │  │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │     │││
│  │  │ │ │TP 维度  │ │ │ │TP 维度  │ │ │ │TP 维度  │ │     │││
│  │  │ │ │GPU 0-3  │ │ │ │GPU 16-19│ │ │ │GPU 32-35│ │     │││
│  │  │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │     │││
│  │  └─────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  显存占用分析：                                               │
│  - TP=4: 模型权重 × 1/4                                      │
│  - PP=4: 权重 × 1/4, 激活 × 1/4                              │
│  - 总计: 权重 × 1/16                                         │
│                                                             │
│  通信开销：                                                   │
│  - TP: 高带宽需求（NVLink）                                   │
│  - PP: 中带宽需求                                            │
│  - DP: 低频 But 大量                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3D 并行的物理拓扑

```
┌─────────────────────────────────────────────────────────────┐
│                 64 GPU 物理布局示例                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  机柜 1 (节点 0-1)          机柜 2 (节点 2-3)                 │
│  ┌─────────────────┐        ┌─────────────────┐             │
│  │ 节点 0          │        │ 节点 2           │             │
│  │ GPU 0-7         │        │ GPU 16-23        │             │
│  │ (TP=4, PP=2)    │──┬─────│ (TP=4, PP=2)     │             │
│  ├─────────────────┤  │     ├─────────────────┤             │
│  │ 节点 1          │  │     │ 节点 3           │             │
│  │ GPU 8-15        │  │     │ GPU 24-31        │             │
│  │ (TP=4, PP=2)    │  │     │ (TP=4, PP=2)     │             │
│  └─────────────────┘  │     └─────────────────┘             │
│                       │                                     │
│  机柜 3 (节点 4-5)     │     机柜 4 (节点 6-7)                 │
│  ┌─────────────────┐  │     ┌─────────────────┐             │
│  │ 节点 4          │  │     │ 节点 6           │             │
│  │ GPU 32-39       │  ├─────│ GPU 48-55        │             │
│  ├─────────────────┤  │     ├─────────────────┤             │
│  │ 节点 5          │  │     │ 节点 7           │             │
│  │ GPU 40-47       │  │     │ GPU 56-63        │             │
│  └─────────────────┘  │     └─────────────────┘             │
│                       │                                     │
│  ──────── 互联 ───────┘                                     │
│  NVLink: 同节点内 TP 通信                                    │
│  InfiniBand: 跨节点 PP/DP 通信                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 并行策略选择指南

```
┌─────────────────────────────────────────────────────────────┐
│                   并行策略选择决策树                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模型能放入单卡？                                             │
│     ├── 是 → 使用数据并行 (DP)                               │
│     └── 否 → 模型大小？                                       │
│              ├── 单卡 < 模型 < 单节点                        │
│              │    └── TP + DP                               │
│              │                                              │
│              ├── 单节点 < 模型 < 多节点                      │
│              │    └── TP + PP + DP                          │
│              │                                              │
│              └── MoE 模型？                                   │
│                   ├── 是 → EP + TP + DP/PP                  │
│                   └── 否 → 评估带宽                          │
│                             ├── 高带宽 → 更大 TP             │
│                             └── 低带宽 → 更大 PP             │
│                                                             │
│  需要长上下文？                                               │
│     └── 是 → 加入 CP                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 典型配置案例

**案例 1：Llama-2-70B 训练**

```yaml
# 配置
GPU: 64 × A100 80GB
并行策略: TP=4, PP=4, DP=4

显存分析:
  - 模型权重: 140GB / 16 = 8.75GB per GPU
  - 激活值: ~10GB per GPU
  - KV Cache: 取决于 batch_size
  - 总计: ~20-25GB per GPU ✓

通信拓扑:
  - TP 通信: 节点内 NVLink (600 GB/s)
  - PP 通信: 跨节点 IB (200 GB/s)
  - DP 通信: 梯度 AllReduce (跨节点)
```

**案例 2：Mixtral-8×7B (MoE) 推理**

```yaml
# 配置
GPU: 8 × A100 80GB
并行策略: EP=8 或 TP=8

方案对比:
  - EP=8: 每卡 1 个专家，显存省，通信多
  - TP=8: 模型切分，通信少，显存多

选择:
  - 单节点 NVLink → TP=8 (更快)
  - 多节点或长文本 → EP + TP
```

**案例 3：128K 长上下文推理**

```yaml
# 配置
模型: Llama-3-70B
GPU: 8 × H100 80GB
并行策略: TP=4, CP=2

显存分析:
  - KV Cache (128K): ~320GB
  - CP=2 后: ~160GB per GPU group
  - TP=4 后: ~40GB per GPU ✓
```

---

## 本章小结

1. **Tensor Parallelism (TP)**：把模型权重按列/行切分，适合高带宽互联，低延迟但规模受限。

2. **Pipeline Parallelism (PP)**：把模型层切分到不同 GPU，有气泡但扩展性好，可配合 micro-batching 优化。

3. **Expert Parallelism (EP)**：MoE 模型的专属方案，专家分布在不同 GPU，All-to-All 通信是关键。

4. **Context Parallelism (CP)**：长文本的解决方案，序列切分后 All-Gather K/V，或用 Ring Attention 优化。

5. **3D 并行**：组合策略，TP×PP×DP，应对超大模型；根据网络拓扑和显存预算选择最优组合。

下一章，我们将进入 PD 分离的世界，看看预填充和解码分离如何提升推理效率。

---

*放弃指数：⭐⭐⭐⭐⭐ 模型并行是硬核中的硬核，需要结合实践才能真正理解。建议：从 TP 开始，逐步深入。*

---

*（未完待续...）*