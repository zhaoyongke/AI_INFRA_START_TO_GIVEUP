# 第八章 CUDA 编程与 Kernel 优化

> *放弃指数：⭐⭐⭐⭐⭐ 本章是硬核中的硬核，需要 C++/CUDA 基础和大量实践*

---

## 8.1 CUDA 入门：Thread、Block、Grid 的世界观

### GPU 并行模型

GPU 的核心设计理念是 **大规模并行**。理解 CUDA 的执行模型是优化的第一步。

```
┌─────────────────────────────────────────────────────────────┐
│                    CUDA 执行模型                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Grid（网格）                                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   一个 Kernel 启动                       ││
│  │                                                         ││
│  │  Block(0,0)    Block(1,0)    Block(2,0)    Block(3,0)  ││
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ ││
│  │  │ Threads │   │ Threads │   │ Threads │   │ Threads │ ││
│  │  │  0-255  │   │  0-255  │   │  0-255  │   │  0-255  │ ││
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘ ││
│  │  Block(0,1)    Block(1,1)    Block(2,1)    Block(3,1)  ││
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ ││
│  │  │ Threads │   │ Threads │   │ Threads │   │ Threads │ ││
│  │  │  0-255  │   │  0-255  │   │  0-255  │   │  0-255  │ ││
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘ ││
│  │                                                         ││
│  │  Grid = 所有 Block 的集合                                ││
│  │  Block = 一组 Thread（最多 1024 个线程）                  ││
│  │                                                         ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  关键概念：                                                   │
│  - Thread：最小执行单元（一个 CUDA Core）                    │
│  - Block：一组协作线程（共享内存）                           │
│  - Grid：所有 Block 的集合（一次 Kernel 启动）               │
│  - SM (Streaming Multiprocessor)：执行 Block 的硬件单元      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 硬件映射

CUDA 的软件抽象如何映射到硬件：

```
┌─────────────────────────────────────────────────────────────┐
│                   软件到硬件的映射                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  软件抽象          硬件执行                                  │
│  ────────────────────────────────────────                   │
│  Grid         →    整个 GPU                                  │
│  Block        →    一个 SM（Streaming Multiprocessor）       │
│  Warp         →    32 个同步执行的线程                       │
│  Thread       →    一个 CUDA Core                            │
│                                                             │
│  ────────────────────────────────────────────────────────── │
│                                                             │
│  Warp 是关键概念：                                            │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Warp (32 threads)                   │  │
│  │                                                       │  │
│  │  T0  T1  T2  T3  ... T31                              │  │
│  │  ───────────────────────                              │  │
│  │           SIMD 执行                                   │  │
│  │  所有线程同时执行相同指令                               │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Warp Divergence（分支分歧）：                               │
│  - if (threadIdx.x < 16) { A } else { B }                  │
│  - 前 16 线程执行 A，后 16 线程执行 B                        │
│  - 实际执行：先执行 A（后 16 线程等待），再执行 B（前 16 等待）│
│  - 性能折半！                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### CUDA 内存层次

```
┌─────────────────────────────────────────────────────────────┐
│                 CUDA 内存层次结构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Global Memory (HBM)                     │   │
│  │           大容量（80GB），高延迟（~400 cycles）        │   │
│  │              所有线程可访问                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↑                                 │
│         ┌─────────────────┼─────────────────┐               │
│         ↓                 ↓                 ↓               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Block 0     │  │ Block 1     │  │ Block N     │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │
│  │ │Shared   │ │  │ │Shared   │ │  │ │Shared   │ │         │
│  │ │Memory   │ │  │ │Memory   │ │  │ │Memory   │ │         │
│  │ │(SRAM)   │ │  │ │(SRAM)   │ │  │ │(SRAM)   │ │         │
│  │ │~100KB   │ │  │ │~100KB   │ │  │ │~100KB   │ │         │
│  │ │低延迟   │ │  │ │低延迟   │ │  │ │低延迟   │ │         │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │
│  │ │Registers│ │  │ │Registers│ │  │ │Registers│ │         │
│  │ │最快     │ │  │ │最快     │ │  │ │最快     │ │         │
│  │ │私有     │ │  │ │私有     │ │  │ │私有     │ │         │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  访问速度：Register >> Shared Memory >> Global Memory        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 第一个 CUDA Kernel

```cpp
// vector_add.cu

#include <cuda_runtime.h>
#include <cstdio>

// CUDA Kernel 定义
__global__ void vector_add(float *a, float *b, float *c, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 主机代码
int main() {
    int n = 1 << 20;  // 1M 元素
    size_t size = n * sizeof(float);
    
    // 分配主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 启动 Kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    
    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }
    
    // 释放内存
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

**编译和运行：**

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

---

### Kernel 关键概念

**1. Thread 索引计算**

```cpp
// 1D Grid, 1D Block
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D Grid, 2D Block
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;

// 3D Grid, 3D Block
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * height * width + y * width + x;
```

**2. 共享内存使用**

```cpp
__global__ void matrix_vector_mul(float *A, float *x, float *y, int n) {
    // 共享内存缓存
    __shared__ float shared_x[256];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // 协作加载 x 到共享内存
    if (tid < n) {
        shared_x[tid] = x[tid];
    }
    __syncthreads();  // 同步等待所有线程加载数据
    
    // 计算
    float sum = 0.0f;
    for (int col = 0; col < n; col++) {
        sum += A[row * n + col] * shared_x[col];
    }
    
    y[row] = sum;
}
```

**3. 常用 CUDA 内置变量**

```cpp
// 线程索引
threadIdx.x, threadIdx.y, threadIdx.z  // block 内的线程索引

// Block 索引
blockIdx.x, blockIdx.y, blockIdx.z     // grid 内的 block 索引

// Block 维度
blockDim.x, blockDim.y, blockDim.z     // block 的大小

// Grid 维度
gridDim.x, gridDim.y, gridDim.z        // grid 的大小
```

---

### 性能优化基础

```
┌─────────────────────────────────────────────────────────────┐
│                   CUDA 性能优化关键指标                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Occupancy（占用率）                                      │
│     - 一个 SM 上活跃 warp 数 / 最大 warp 数                  │
│     - 目标：接近 100%                                       │
│                                                             │
│  2. Memory Bandwidth（内存带宽）                             │
│     - 实际带宽 / 理论带宽                                    │
│     - 目标：> 80%                                           │
│                                                             │
│  3. Compute Throughput（计算吞吐）                           │
│     - 实际 FLOPS / 峰值 FLOPS                                │
│     - 目标：> 50%（计算密集型）                              │
│                                                             │
│  4. Warp Efficiency（Warp 效率）                             │
│     - 非 idle 线程比例                                       │
│     - 避免 warp divergence                                  │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  优化优先级：                                                 │
│                                                             │
│  1. 最大化并行度                                            │
│     - 足够的 block 和 thread                                │
│     - 避免 SM 空闲                                          │
│                                                             │
│  2. 优化内存访问                                            │
│     - Coalesced memory access（合并访问）                   │
│     - 使用共享内存缓存                                      │
│                                                             │
│  3. 减少分支分歧                                            │
│     - 避免 warp divergence                                 │
│                                                             │
│  4. 隐藏延迟                                                │
│     - 更多独立指令                                          │
│     - Memory-compute overlap                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8.2 为什么手写 Kernel 比调用 cuBLAS 快？

### 库函数的局限性

cuBLAS、cuDNN 等库已经高度优化，为什么还需要手写 Kernel？

```
┌─────────────────────────────────────────────────────────────┐
│                 库函数 vs 手写 Kernel                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  库函数优势：                                                 │
│  ✓ 高度优化，经过广泛测试                                    │
│  ✓ 通用性强，适用于大多数场景                                │
│  ✓ 稳定可靠，无需维护                                        │
│                                                             │
│  库函数局限：                                                 │
│  ✗ 无法针对特定问题优化                                      │
│  ✗ 多个 kernel 调用有额外开销                                │
│  ✗ 中间结果需要写回显存                                      │
│  ✗ 无法利用特定问题结构                                      │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  示例：Transformer Attention                                │
│                                                             │
│  使用 cuBLAS（标准实现）：                                    │
│  ┌────────────────────────────────────────────────────────┐│
│  │  Q = X @ Wq      → cuBLAS GEMM (kernel 1)              ││
│  │  K = X @ Wk      → cuBLAS GEMM (kernel 2)              ││
│  │  V = X @ Wv      → cuBLAS GEMM (kernel 3)              ││
│  │  Scores = Q @ K^T → cuBLAS GEMM (kernel 4)             ││
│  │  Write to global memory → 延迟                         ││
│  │  Attn = softmax(Scores) → 自定义 kernel (kernel 5)     ││
│  │  Output = Attn @ V → cuBLAS GEMM (kernel 6)            ││
│  │                                                        ││
│  │  总计：6 次 kernel 启动 + 5 次显存读写                   ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  手写 Fused Kernel：                                         │
│  ┌────────────────────────────────────────────────────────┐│
│  │  Flash Attention (单 kernel)：                          ││
│  │  1. 分块加载 Q, K, V 到 shared memory                  ││
│  │  2. 片计算 attention scores                            ││
│  │  3. 在线 softmax（避免存储完整 attention 矩阵）         ││
│  │  4. 片加权求和                                          ││
│  │                                                        ││
│  │  总计：1 次 kernel 启动 + 最小显存访问                   ││
│  │  结果：2-4x 加速                                        ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Kernel Fusion 的威力

**示例：Fused LayerNorm + Linear**

```cpp
// 标准：LayerNorm + Linear 分开实现
// LayerNorm 实现
__global__ void layernorm_kernel(float* input, float* output, 
                                   float* gamma, float* beta,
                                   int batch, int hidden) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // 计算 mean
    for (int i = tid; i < hidden; i += blockDim.x) {
        sum += input[b * hidden + i];
    }
    sum = block_reduce_sum(sum);
    float mean = sum / hidden;
    
    __syncthreads();
    
    // 计算 variance
    for (int i = tid; i < hidden; i += blockDim.x) {
        float diff = input[b * hidden + i] - mean;
        sum_sq += diff * diff;
    }
    sum_sq = block_reduce_sum(sum_sq);
    float var = sum_sq / hidden;
    float inv_std = rsqrtf(var + 1e-5f);
    
    __syncthreads();
    
    // LayerNorm 输出
    for (int i = tid; i < hidden; i += blockDim.x) {
        int idx = b * hidden + i;
        output[idx] = (input[idx] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// 然后 Linear 需要：
// 1. 读 LayerNorm 输出
// 2. 矩阵乘法
// 3. 写 Linear 输出

// Fused Kernel：LayerNorm + Linear 合并
__global__ void fused_layernorm_linear_kernel(
    float* input, float* output,
    float* gamma, float* beta, float* weight,
    int batch, int hidden_in, int hidden_out
) {
    __shared__ float shared_ln[1024];  // 存 LayerNorm 结果
    
    // Step 1: LayerNorm（同上）
    // ... layer norm 计算 ...
    
    // 与此同时存入共享内存，无需写全局内存
    if (tid < hidden_in) {
        shared_ln[tid] = normalized_value;
    }
    
    __syncthreads();
    
    // Step 2: Linear（直接从_shared_读取）
    for (int j = tid; j < hidden_out; j += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_in; k++) {
            sum += shared_ln[k] * weight[j * hidden_in + k];
        }
        output[blockIdx.x * hidden_out + j] = sum;
    }
}
```

**Fused Kernel 的收益：**

```
场景：Transformer FFN（LayerNorm + Linear × 2 + GELU + Linear）

未优化：5 次 kernel 启动 + 4 次 global memory 读写

Fused 后：1 次 kernel 启动 + 1 次 global memory 读写

性能提升：
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   │ Latency (ms)                                          │
│   │                                                       │
│   │  ██                                    未优化: 12ms   │
│   │  ████                                  ████████████   │
│   │  ██████                                                      │
│   │  ████████                                    Fused: 3ms   │
│   │  ██████████                                  ████         │
│   │  ████████████                                ████         │
│   │  ───────────────────────────────                        │
│   │        加速 4x                                          │
│   │                                                        │
│   └────────────────────────────────────────────────────────┘
```

---

### 什么时候手写 Kernel？

```
┌─────────────────────────────────────────────────────────────┐
│               手写 Kernel 决策指南                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  应该手写 Kernel 的场景：                                     │
│                                                             │
│  1. 算子融合收益明显                                         │
│     - 多个简单算子                                         │
│     - 中间结果大（memory-bound）                            │
│     - 典型：Flash Attention, Fused LayerNorm               │
│                                                             │
│  2. 标准库不支持的操作                                       │
│     - 自定义采样算法                                        │
│     - 特殊激活函数                                          │
│     - 量化/反量化融合                                       │
│                                                             │
│  3. 特定硬件优化                                             │
│     - 利用 Tensor Core                                     │
│     - 针对架构优化（A100 vs H100）                          │
│                                                             │
│  4. 极致性能需求                                             │
│     - 推理服务延迟要求                                       │
│     - 训练吞吐瓶颈                                          │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  应该使用库函数的场景：                                       │
│                                                             │
│  1. 标准操作（GEMM, 卷积）                                   │
│     - cuBLAS, cuDNN 已极致优化                              │
│                                                             │
│  2. 开发效率优先                                             │
│     - 快速验证想法                                          │
│     - 性能不是瓶颈                                          │
│                                                             │
│  3. 维护成本考量                                             │
│     - 手写 kernel 难维护                                   │
│     - 硬件升级可能需要重写                                  │
│                                                             │
│  4. 跨平台兼容                                               │
│     - 需要支持多种 GPU                                      │
│     - 自定义 kernel 移植成本高                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8.3 Flash Attention：如何拯救显存带宽

### Attention 的显存瓶颈

标准 Attention 实现：

```
┌─────────────────────────────────────────────────────────────┐
│                  标准 Attention 分析                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  计算流程：                                                   │
│                                                             │
│  Q [B, H, N, D]  K [B, H, N, D]  V [B, H, N, D]            │
│       │               │               │                     │
│       │       K^T [B, H, D, N]        │                     │
│       │               │               │                     │
│       └───────┬───────┘               │                     │
│               ↓                       │                     │
│         S = Q @ K^T                   │                     │
│      [B, H, N, N]  ← O(N²) 显存！     │                     │
│               │                       │                     │
│               ↓ softmax               │                     │
│         P = softmax(S)                │                     │
│      [B, H, N, N]  ← O(N²) 显存！     │                     │
│               │                       │                     │
│               └───────────┬───────────┘                     │
│                           ↓                                 │
│                     O = P @ V                               │
│                    [B, H, N, D]                             │
│                                                             │
│  显存占用：O(N²)                                             │
│  - N=4096: 16M floats × batch × heads ≈ 数 GB              │
│  - N=16384: 256M floats → 显存爆炸                          │
│                                                             │
│  带宽消耗：                                                   │
│  - 写 S: B × H × N² × 4 bytes                               │
│  - 读 S (softmax): B × H × N² × 4 bytes                     │
│  - 写 P: B × H × N² × 4 bytes                               │
│  - 读 P: B × H × N² × 4 bytes                               │
│  总计：4 × B × H × N² × 4 bytes                             │
│                                                             │
│  N=4096, B=1, H=32: 4 × 1 × 32 × 16M × 4 ≈ 8 GB 带宽        │
│  这只是 Attention 一层！                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Flash Attention 核心思想

**核心观察：**
- S 和 P 是中间结果，最终不需要存储
- 可以分块计算，每次只处理一小块

**Flash Attention 策略：**

```
┌─────────────────────────────────────────────────────────────┐
│                  Flash Attention 原理                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  分块计算（Tiling）：                                         │
│                                                             │
│  Q, K, V 切分成小块（tiles）：                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Q: [Q₀, Q₁, Q₂, ...]                                │  │
│  │  K: [K₀, K₁, K₂, ...]                                │  │
│  │  V: [V₀, V₁, V₂, ...]                                │  │
│  │                                                      │  │
│  │  每个 tile 大小：如 128 或 256 个 tokens              │  │
│  │  可以放入 SRAM（Shared Memory）                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  分块 Attention 计算：                                        │
│                                                             │
│  for each Q_tile i:                                         │
│    load Qᵢ to SRAM                                          │
│    for each K,V_tile j:                                     │
│      load Kⱼ, Vⱼ to SRAM                                    │
│      compute Sᵢⱼ = Qᵢ @ Kⱼ^T                                │
│      compute Pᵢⱼ = softmax(Sᵢⱼ)                             │
│      accumulate Oᵢ += Pᵢⱼ @ Vⱼ                              │
│    write Oᵢ to HBM                                          │
│                                                             │
│  显存占用：O(B × H × N × D × tile_size) << O(N²)            │
│                                                             │
│  核心挑战：如何分块做 softmax？                               │
│  答案：Online Softmax（在线 softmax）                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Online Softmax

普通 softmax 需要先知道所有值才能计算，但 Flash Attention 需要增量计算。

```
┌─────────────────────────────────────────────────────────────┐
│                    Online Softmax 推导                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  标准 Softmax：                                               │
│  softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)                         │
│                                                             │
│  问题：需要知道所有 x 才能计算                               │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  增量更新（Online Softmax）：                                 │
│                                                             │
│  假设已计算了前 k 个元素的统计量：                            │
│  - mₖ = max(x₀, x₁, ..., xₖ₋₁)    （当前最大值）            │
│  - lₖ = Σᵢ₌₀ᵏ⁻¹ exp(xᵢ - mₖ)      （归一化因子）            │
│                                                             │
│  当加入新元素 xₖ：                                           │
│  - mₖ₊₁ = max(mₖ, xₖ)                                       │
│  - lₖ₊₁ = lₖ × exp(mₖ - mₖ₊₁) + exp(xₖ - mₖ₊₁)             │
│                                                             │
│  也需要更新之前的归一化结果（因为 m 可能变化）                │
│                                                             │
│  输出：                                                      │
│  - Oₖ₊₁ = Oₖ × (lₖ × exp(mₖ - mₖ₊₁) / lₖ₊₁)                │
│         + Vₖ × exp(xₖ - mₖ₊₁) / lₖ₊₁                        │
│                                                             │
│  核心：维护两个标量 m 和 l，就可以增量更新                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Flash Attention 简化实现

```cpp
// flash_attention.cu - 简化版实现

template<int TILE_SIZE>
__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* O,
    int B, int H, int N, int D
) {
    // Shared memory for tiles
    __shared__ float Q_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float K_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float V_tile[TILE_SIZE][HEAD_DIM];
    
    // 每个线程块处理一个 query tile 和所有 key/value tiles
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int q_tile = blockIdx.x;
    
    // 当前 tile 的 query 起始位置
    int q_start = q_tile * TILE_SIZE;
    
    // 加载 Q tile 到共享内存
    for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            int q_idx = batch * H * N * D + head * N * D + (q_start + i) * D + j;
            Q_tile[i][j] = Q[q_idx];
        }
    }
    __syncthreads();
    
    // 每个 query 需要维护的统计量
    float row_max[TILE_SIZE];      // m[i]
    float row_sum[TILE_SIZE];      // l[i]
    float output[TILE_SIZE][HEAD_DIM];  // O[i]
    
    // 初始化
    for (int i = 0; i < TILE_SIZE; i++) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
        for (int j = 0; j < HEAD_DIM; j++) {
            output[i][j] = 0.0f;
        }
    }
    
    // 遍历所有 K, V tiles
    for (int kv_tile = 0; kv_tile < N / TILE_SIZE; kv_tile++) {
        int kv_start = kv_tile * TILE_SIZE;
        
        // 加载 K, V tile 到共享内存
        for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
            for (int j = threadIdx.x; j < D; j += blockDim.x) {
                int k_idx = batch * H * N * D + head * N * D + (kv_start + i) * D + j;
                K_tile[i][j] = K[k_idx];
                V_tile[i][j] = V[k_idx];
            }
        }
        __syncthreads();
        
        // 计算当前 tile 的 attention scores
        // S = Q_tile @ K_tile^T
        float S[TILE_SIZE][TILE_SIZE];
        
        for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x) {
                float score = 0.0f;
                for (int k = 0; k < D; k++) {
                    score += Q_tile[i][k] * K_tile[j][k];
                }
                S[i][j] = score / sqrtf((float)D);  // scale
            }
        }
        __syncthreads();
        
        // 在线 softmax 更新
        for (int i = 0; i < TILE_SIZE; i++) {
            // 计算当前 tile 的最大值
            float tile_max = -INFINITY;
            for (int j = 0; j < TILE_SIZE; j++) {
                tile_max = fmaxf(tile_max, S[i][j]);
            }
            
            // 更新全局最大值
            float new_max = fmaxf(row_max[i], tile_max);
            
            // 计算重新缩放因子
            float exp_old = expf(row_max[i] - new_max);
            float exp_new = expf(tile_max - new_max);
            
            // 更新归一化因子
            row_sum[i] = row_sum[i] * exp_old;
            for (int j = 0; j < TILE_SIZE; j++) {
                row_sum[i] += expf(S[i][j] - new_max);
            }
            
            // 更新输出
            for (int j = 0; j < HEAD_DIM; j++) {
                output[i][j] *= exp_old;
            }
            
            // 加上新 tile 的贡献
            for (int k = 0; k < TILE_SIZE; k++) {
                float prob = expf(S[i][k] - new_max);
                for (int j = 0; j < HEAD_DIM; j++) {
                    output[i][j] += prob * V_tile[k][j];
                }
            }
            
            row_max[i] = new_max;
        }
        __syncthreads();
    }
    
    // 归一化并写回全局内存
    for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < HEAD_DIM; j += blockDim.x) {
            int o_idx = batch * H * N * D + head * N * D + (q_start + i) * D + j;
            O[o_idx] = output[i][j] / row_sum[i];
        }
    }
}
```

---

### Flash Attention 性能对比

```
┌─────────────────────────────────────────────────────────────┐
│               Flash Attention 性能提升                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  测试配置：GPT-3 (175B), seq_len=2048, A100 80GB            │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              显存占用对比                              │  │
│  │                                                      │  │
│  │  标准 Attention:                                     │  │
│  │  ████████████████████████████████████████ 48GB      │  │
│  │                                                      │  │
│  │  Flash Attention:                                    │  │
│  │  ████ 8GB                                            │  │
│  │                                                      │  │
│  │  节省：6x 显存                                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              速度对比                                  │  │
│  │                                                      │  │
│  │  标准 Attention:                                     │  │
│  │  ████████████████████████ 2.5s                       │  │
│  │                                                      │  │
│  │  Flash Attention:                                    │  │
│  │  ████████████ 1.2s                                   │  │
│  │                                                      │  │
│  │  加速：2.1x                                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  关键收益：                                                   │
│  1. 显存从 O(N²) 降到 O(N)                                  │
│  2. 更少的 HBM 访问 → 更快的执行                             │
│  3. 支持更长的上下文                                         │
│  4. 训练更快，推理更高效                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8.4 Triton：写 Kernel 不用学 CUDA？

### Triton 简介

**Triton** 是 OpenAI 开发的 GPU 编程语言，目标是让写 GPU kernel 更简单。

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton vs CUDA                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CUDA：                                                      │
│  - 底层 C++ 扩展                                             │
│  - 需要管理 Thread、Block、Shared Memory                     │
│  - 手动优化内存访问                                          │
│  - 学习曲线陡峭                                              │
│                                                             │
│  Triton：                                                    │
│  - Python 语法                                               │
│  - 自动向量化（类似 NumPy）                                   │
│  - 编译器自动优化                                            │
│  - 学习曲线平缓                                              │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  代码对比（Vector Add）：                                     │
│                                                             │
│  CUDA:                                                      │
│  __global__ void add(float* a, float* b, float* c, int n) { │
│      int i = blockIdx.x * blockDim.x + threadIdx.x;         │
│      if (i < n) c[i] = a[i] + b[i];                         │
│  }                                                          │
│                                                             │
│  Triton:                                                    │
│  @triton.jit                                                │
│  def add_kernel(a, b, c, n, BLOCK: tl.constexpr):           │
│      pid = tl.program_id(0)                                 │
│      offs = pid * BLOCK + tl.arange(0, BLOCK)               │
│      mask = offs < n                                        │
│      a_val = tl.load(a + offs, mask=mask)                   │
│      b_val = tl.load(b + offs, mask=mask)                   │
│      tl.store(c + offs, a_val + b_val, mask=mask)           │
│                                                             │
│  看起来更高级，但隐藏了复杂性                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Triton 编程模型

Triton 把 GPU 编程抽象为 "Program" 概念：

```
┌─────────────────────────────────────────────────────────────┐
│                  Triton 执行模型                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CUDA Model:                                                │
│  Grid → Blocks → Threads → SIMD                            │
│                                                             │
│  Triton Model:                                               │
│  Grid → Programs → SIMD Lanes                               │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                                                      │  │
│  │  Program 0     Program 1     Program 2     Program 3 │  │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐│  │
│  │  │ [0..127]│   │[128..255]│  │[256..383]│  │[384..511]││  │
│  │  │ 向量操作 │   │ 向量操作 │  │ 向量操作 │  │ 向量操作 ││  │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘│  │
│  │                                                      │  │
│  │  每个 Program 处理一个 block 的数据                   │  │
│  │  编译器自动映射到 CUDA threads/warps                  │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  核心抽象：                                                   │
│  - tl.program_id(): 当前 program 的索引                     │
│  - tl.arange(): 生成向量索引                                 │
│  - tl.load() / tl.store(): 向量化内存访问                   │
│  - 所有操作自动向量化                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Triton 实现 Flash Attention

```python
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 获取当前 program 的位置
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 计算 batch 和 head 索引
    off_z = off_hz // H
    off_h = off_hz % H
    
    # 计算 Q 的起始位置
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )
    
    # K, V 的 block pointer
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(D_HEAD, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(D_HEAD, BLOCK_N),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, D_HEAD),
        order=(1, 0),
    )
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # 加载 Q block
    q = tl.load(Q_block_ptr)
    
    # 遍历 K, V blocks
    lo = 0
    hi = (N_CTX + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    
    for start_n in range(lo, hi, BLOCK_N):
        # 加载 K, V
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        # 计算 QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= 1.0 / (D_HEAD ** 0.5)
        
        # 在线 softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # 更新 m, l, acc
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v) * beta[:, None]
        
        l_i = l_i * alpha + l_ij * beta
        m_i = m_new
        
        # 移动 K, V block pointer
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # 归一化
    acc = acc / l_i[:, None]
    
    # 存储结果
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(tl.float16))


# Python wrapper
def flash_attention(q, k, v):
    # q, k, v: [batch, heads, seq_len, head_dim]
    batch, heads, seq_len, head_dim = q.shape
    
    o = torch.empty_like(q)
    
    grid = lambda META: (
        (seq_len + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        batch * heads,
    )
    
    flash_attention_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, heads, seq_len, head_dim,
        BLOCK_M=128, BLOCK_N=64,
    )
    
    return o
```

---

### Triton 的优势与局限

```
┌─────────────────────────────────────────────────────────────┐
│                   Triton 优势                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 开发效率高                                               │
│     - Python 语法                                           │
│     - 自动向量化                                            │
│     - 编程模型简单                                          │
│                                                             │
│  2. 性能接近 CUDA                                           │
│     - 编译器优化                                            │
│     - 自动内存合并                                          │
│     - 支持 Tensor Core                                      │
│                                                             │
│  3. 可移植性好                                               │
│     - 跨 GPU 架构                                           │
│     - 编译器适配                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
│                   Triton 局限                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 灵活性受限                                               │
│     - 不如 CUDA 底层控制精细                                │
│     - 某些优化难以实现                                      │
│                                                             │
│  2. 调试困难                                                 │
│     - 编译错误信息晦涩                                      │
│     - 缺乏成熟的调试工具                                    │
│                                                             │
│  3. 生态不成熟                                               │
│     - 文档有限                                              │
│     - 社区规模小                                            │
│                                                             │
│  4. 性能可能不如手动优化 CUDA                                │
│     - 对专家级 CUDA 程序员                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8.5 当你的 Kernel 被 NVCC 编译器优化掉了

### NVCC 编译器行为

NVCC（NVIDIA CUDA Compiler）会进行各种优化，有时会把你的代码"优化掉"。

```
┌─────────────────────────────────────────────────────────────┐
│                 NVCC 常见优化行为                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 死代码消除（Dead Code Elimination）                      │
│                                                             │
│  问题代码：                                                   │
│  __global__ void kernel(float* data) {                      │
│      float temp = data[0];  // 读取                         │
│      temp = temp * 2;       // 计算，但结果未存储            │
│      // temp 从未使用                                       │
│  }                                                          │
│                                                             │
│  结果：整个 kernel 可能被优化为空操作                        │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  2. 常量折叠（Constant Folding）                             │
│                                                             │
│  问题代码：                                                   │
│  const int SIZE = 1024;                                     │
│  float array[SIZE];                                         │
│  for (int i = 0; i < SIZE * 2; i++) {  // SIZE * 2 编译时计算│
│      array[i] = 0;                                          │
│  }                                                          │
│                                                             │
│  如果编译器发现某些分支永远不会执行，可能直接删除             │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  3. 循环展开（Loop Unrolling）                               │
│                                                             │
│  for (int i = 0; i < 4; i++) {                              │
│      data[i] = i;                                           │
│  }                                                          │
│                                                             │
│  可能被展开为：                                              │
│  data[0] = 0;                                               │
│  data[1] = 1;                                               │
│  data[2] = 2;                                               │
│  data[3] = 3;                                               │
│                                                             │
│  这通常是好事，但可能改变内存访问模式                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 为什么代码被优化掉了？

```
常见原因：

1. 结果未存储或使用
   - 编译器认为计算无意义
   - 常见于测试代码

2. 写后读被消除
   __global__ void kernel(float* data) {
       data[0] = 1.0f;  // 写
       float x = data[0];  // 读同一位置
       // 编译器可能直接用常量 1.0f 替代读取
   }

3. 内存屏障缺失
   - 没有 __syncthreads()
   - 编译器可能重排内存操作

4. 违反内存模型假设
   - 两次写入同一地址（无中间读取）
   - 可能被合并或覆盖
```

---

### 如何防止被优化掉

```cpp
// 方法 1：使用 volatile
__global__ void kernel(volatile float* data) {
    data[0] = data[0] + 1;  // 强制从内存读取
}

// 方法 2：使用编译器屏障
__global__ void kernel(float* data) {
    data[0] = 1.0f;
    __syncthreads();  // 内存屏障
    float x = data[0];  // 确保读取
}

// 方法 3：输出结果
__global__ void kernel(float* data, float* output) {
    float temp = data[0];
    temp = temp * 2;
    output[0] = temp;  // 结果被使用，不会被优化掉
}

// 方法 4：使用 __restrict__ 和 volatile 组合
__global__ void kernel(
    float* __restrict__ output,
    const float* __restrict__ input
) {
    // __restrict__ 告诉编译器指针不重叠
    // 编译器可以更激进优化，但保持语义
}
```

---

### CUDA 性能分析工具

```bash
# 1. nvprof（旧版）
nvprof ./my_cuda_program

# 输出示例：
# ==PROF== Profiling "kernel_name": Array size 1024
#            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
#  GPU activities:   85.43%  12.345ms         1  12.345ms  12.345ms  12.345ms  kernel_name
#      API calls:    67.89%  56.789ms         1  56.789ms  56.789ms  56.789ms  cudaMalloc

# 2. Nsight Compute（推荐）
ncu ./my_cuda_program

# 详细分析：
# - Kernel 性能指标
# - 内存吞吐
# - Occupancy
# - Warp 效率

# 3. Nsight Systems（系统级）
nsys profile ./my_cuda_program

# 输出：
# - CPU/GPU 时间线
# - Kernel 启动延迟
# - 内存拷贝时间

# 4. CUDA-MEMCHECK（内存检查）
cuda-memcheck ./my_cuda_program

# 检查：
# - 越界访问
# - 竞争条件
# - 内存泄漏
```

---

### 常见性能问题与解决

```
┌─────────────────────────────────────────────────────────────┐
│                 CUDA 性能问题诊断                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题 1：Occupancy 低                                        │
│                                                             │
│  症状：GPU 利用率低                                          │
│  诊断：ncu --metrics sm__warps_active.avg.pct_of_peak       │
│                                                             │
│  原因：                                                      │
│  - Block 数量不足                                           │
│  - 每个 Block 线程数不当                                    │
│  - 寄存器/显存占用过高                                      │
│                                                             │
│  解决：                                                      │
│  - 增加 Grid 大小                                           │
│  - 调整 Block 大小（128/256/512）                          │
│  - 减少寄存器使用（--maxrregcount）                        │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题 2：内存带宽未打满                                      │
│                                                             │
│  症状：计算快但整体慢                                        │
│  诊断：ncu --metrics dram__throughput.avg.pct_of_peak_sustained│
│                                                             │
│  原因：                                                      │
│  - 内存访问不合并                                           │
│  - 缺乏内存访问局部性                                       │
│                                                             │
│  解决：                                                      │
│  - 重排内存访问模式                                         │
│  - 使用共享内存缓存                                         │
│  - 检查合并访问                                             │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题 3：Warp Divergence                                    │
│                                                             │
│  症状：分支语句性能差                                        │
│  诊断：ncu --metrics smsp__sass_branch_targets.divergent.sum│
│                                                             │
│  原因：                                                     │
│  - warp 内线程走不同分支                                   │
│                                                             │
│  解决：                                                     │
│  - 重排数据显示分支分歧                                    │
│  - 使用 warp 级原语（__ballot_sync）                       │
│  - 避免 threadIdx 决定的条件分支                           │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题 4：Shared Memory Bank Conflict                       │
│                                                             │
│  症状：共享内存访问慢                                        │
│  诊断：ncu --metrics l1tex__data_bank_conflicts_pipe_bank.sum       │
│                                                             │
│  原因：                                               │
│  - 多线程访问同一 bank                                    │
│                                                             │
│  解决：                                                │
│  - Padding（填充）                                       │
│  - 改变访问模式                                         │
│  - 使用交叉存取                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 本章小结

1. **CUDA 执行模型**：理解 Grid→Block→Warp→Thread 的层次结构是优化的基础。

2. **Kernel Fusion 收益大**：减少 kernel 启动开销和显存访问是关键优化方向。

3. **Flash Attention 是经典案例**：通过分块计算和在线 softmax，将显存从 O(N²) 降到 O(N)。

4. **Triton 降低门槛**：Python 语法、自动向量化，适合快速开发和实验。

5. **调试和性能分析**：使用 Nsight Compute/Systems 定位瓶颈，理解 NVCC 优化行为。

下一章，我们将进入大模型架构演进的世界，看看 Transformer 如何进化到现代大模型。

---

*放弃指数：⭐⭐⭐⭐⭐ CUDA 编程需要大量实践，建议从简单 kernel 开始，逐步深入。*

---

*（未完待续...）*