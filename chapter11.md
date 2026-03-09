# 第十一章 经典踩坑案例集

> *放弃指数：⭐⭐⭐ 本章是实战经验总结，建议收藏备用*

---

## 11.1 显存泄漏排查：tcmalloc vs jemalloc

### 问题现象

```
┌─────────────────────────────────────────────────────────────┐
│                   显存泄漏问题现象                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景：vLLM 服务运行一段时间后 OOM                           │
│                                                             │
│  时间线：                                                    │
│  ─────────────────────────────────────────────────────────  │
│  T=0h:    启动服务，显存 50GB/80GB                          │
│  T=6h:    显存 58GB/80GB                                    │
│  T=12h:   显存 65GB/80GB                                    │
│  T=18h:   显存 72GB/80GB                                    │
│  T=24h:   显存 77GB/80GB                                    │
│  T=26h:   OOM！服务崩溃                                      │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  典型报错：                                                   │
│  RuntimeError: CUDA out of memory. Tried to allocate 2.00 GB│
│                                                             │
│  疑问：                                                      │
│  - 模型权重固定，KV Cache 有上限                             │
│  - 为什么显存还在不断增长？                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 排查步骤

```
┌─────────────────────────────────────────────────────────────┐
│                   显存泄漏排查流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 确认是 GPU 显存还是 CPU 内存                        │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  $ nvidia-smi                                               │
│  +-------------------------------------------------------+  │
│  | GPU  Memory-Usage |                                   |  │
│  |   0  77GB / 80GB  |  ← GPU 显存确实在增长              │  │
│  +-------------------------------------------------------+  │
│                                                             │
│  $ ps aux | grep python                                     │
│  user  12345  ... 45G  ← 进程内存也在增长                    │
│                                                             │
│  Step 2: 使用 PyTorch 内存分析工具                          │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  import torch                                               │
│                                                             │
│  # 打印内存快照                                             │
│  torch.cuda.memory_summary()                                │
│                                                             │
│  # 内存分配历史                                             │
│  snapshot = torch.cuda.memory._record_memory_history()      │
│  # ... 一段时间后 ...                                       │
│  torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")    │
│                                                             │
│  # 用可视化工具分析                                         │
│  # python -m torch.cuda.memory                             │
│                                                             │
│  Step 3: 分析内存碎片                                       │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  $ python -c "                                              │
│  import torch                                               │
│  print('Allocated:', torch.cuda.memory_allocated() / 1e9, 'GB')│
│  print('Reserved:', torch.cuda.memory_reserved() / 1e9, 'GB')│
│  print('碎片率:', (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / torch.cuda.memory_reserved() * 100, '%')│
│  "                                                          │
│                                                             │
│  常见输出：                                                   │
│  Allocated: 45.2 GB                                         │
│  Reserved: 77.1 GB                                          │
│  碎片率: 41.4%  ← 高碎片率！                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 内存分配器的影响

```
┌─────────────────────────────────────────────────────────────┐
│                内存分配器对比                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Linux 默认内存分配器：glibc malloc                          │
│  问题：多线程场景下性能差，容易碎片化                         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  tcmalloc (Google):                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 优点：                                                │  │
│  │ - 多线程性能好                                        │  │
│  │ - 碎片管理优秀                                        │  │
│  │ - 适合 TensorFlow/PyTorch                            │  │
│  │                                                       │  │
│  │ 安装：                                                │  │
│  │ $ apt-get install libtcmalloc-minimal4               │  │
│  │ 或                                                    │  │
│  │ $ yum install gperftools-libs                        │  │
│  │                                                       │  │
│  │ 使用：                                                │  │
│  │ $ LD_PRELOAD=/usr/lib/libtcmalloc.so python train.py │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  jemalloc (Facebook):                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 优点：                                                │  │
│  │ - 内存占用更低                                        │  │
│  │ - 碎片控制更好                                        │  │
│  │ - 适合长时间运行的服务                                │  │
│  │                                                       │  │
│  │ 安装：                                                │  │
│  │ $ apt-get install libjemalloc2                       │  │
│  │                                                       │  │
│  │ 使用：                                                │  │
│  │ $ LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \│
│  │   python serve.py                                     │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  性能对比（8小时推理服务）：                                   │
│                                                             │
│  分配器     启动内存   8小时后内存   碎片率                   │
│  ─────────────────────────────────────────────              │
│  glibc      50 GB      77 GB        41%                     │
│  tcmalloc   48 GB      55 GB        15%                     │
│  jemalloc   47 GB      51 GB        8%                      │
│                                                             │
│  结论：长时间运行的服务推荐 jemalloc                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 常见显存泄漏原因

```python
# 原因 1：未 detach 的计算图
def bad_inference(model, input):
    output = model(input)
    loss = compute_loss(output)  # loss 保留了计算图
    return loss  # 返回了带梯度的 tensor → 内存泄漏

# 修复
def good_inference(model, input):
    with torch.no_grad():
        output = model(input)
    return output  # 无计算图

# ─────────────────────────────────────────────────────────────

# 原因 2：缓存未清理
class ModelCache:
    def __init__(self):
        self.cache = {}  # 缓存不断增长
    
    def get(self, key):
        if key not in self.cache:
            self.cache[key] = expensive_compute(key)
        return self.cache[key]
    
    # 问题：cache 无限增长

# 修复：使用 LRU 缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached(key):
    return expensive_compute(key)

# ─────────────────────────────────────────────────────────────

# 原因 3：CUDA Stream 未同步
import torch.cuda

stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    # 异步操作
    result = model(input)

# 问题：stream 资源未释放
# PyTorch 会保留 stream 直到所有操作完成

# ─────────────────────────────────────────────────────────────

# 原因 4：Tensor 在 CPU 和 GPU 之间频繁移动
for batch in data_loader:
    batch = batch.cuda()  # 每次分配新的 GPU 内存
    output = model(batch)
    result = output.cpu()  # 移回 CPU
    # GPU 内存没有立即释放

# 修复：使用 pin_memory + async
data_loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # 锁页内存
    num_workers=4
)

for batch in data_loader:
    batch = batch.cuda(non_blocking=True)
    output = model(batch)
    result = output.cpu(non_blocking=True)
```

---

### 排查工具脚本

```bash
#!/bin/bash
# memory_debug.sh - 显存内存调试脚本

echo "=== GPU 显存监控 ==="
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

echo ""
echo "=== 进程内存监控 ==="
# 获取 Python 进程 PID
PID=$(pgrep -f "python.*serve")

if [ -n "$PID" ]; then
    echo "监控进程: $PID"
    watch -n 1 "ps -p $PID -o pid,vsz,rss,pmem,comm"
fi

echo ""
echo "=== 内存映射分析 ==="
if [ -n "$PID" ]; then
    pmap -x $PID | tail -20
fi

echo ""
echo "=== CUDA 内存碎片分析 ==="
python3 << 'EOF'
import torch
import gc

print("CUDA 内存状态:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"  碎片率:    {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / torch.cuda.memory_reserved() * 100:.1f}%")

print("\n尝试清理碎片...")
gc.collect()
torch.cuda.empty_cache()

print("\n清理后状态:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
EOF
```

---

## 11.2 NCCL 通信失败：网络配置的玄学

### 问题现象

```
┌─────────────────────────────────────────────────────────────┐
│                   NCCL 错误现象                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  常见错误消息：                                               │
│                                                             │
│  错误 1: Timeout                                            │
│  ─────────────────────────────────────────────────────────  │
│  NCCL WARN : Call to NCCL timed out...                      │
│  RuntimeError: NCCL error in: /path/to/file.cc:123         │
│                                                             │
│  错误 2: Connection refused                                  │
│  ─────────────────────────────────────────────────────────  │
│  NCCL WARN : NET/Socket : Connection refused                │
│                                                             │
│  错误 3: Unhandled system error                              │
│  ─────────────────────────────────────────────────────────  │
│  NCCL WARN : Cuda failure 'invalid device ordinal'          │
│                                                             │
│  错误 4: P2P failed                                          │
│  ─────────────────────────────────────────────────────────  │
│  NCCL WARN : GPU Direct P2P is disabled                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### NCCL 调试方法

```bash
# 开启 NCCL 调试输出
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 输出到文件
export NCCL_DEBUG_FILE=/tmp/nccl_debug.log

# 然后运行训练脚本
python train.py

# 分析输出
cat /tmp/nccl_debug.log | grep -i "error\|warn\|fail"
```

---

### 常见问题与解决方案

```
┌─────────────────────────────────────────────────────────────┐
│                 NCCL 问题排查指南                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题 1: 网络接口选择错误                                     │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  症状：NCCL bind to wrong interface                         │
│                                                             │
│  排查：                                                      │
│  $ ip addr show                                             │
│  # 通常有 eth0, docker0, lo 等                              │
│                                                             │
│  解决：明确指定网络接口                                      │
│  export NCCL_SOCKET_IFNAME=eth0  # 或 ib0 (InfiniBand)     │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题 2: P2P 通信失败                                        │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  症状：GPU Direct P2P disabled                               │
│                                                             │
│  排查：                                                      │
│  # 检查 P2P 状态                                            │
│  $ nvidia-smi nvlink --status                               │
│  $ nvidia-smi --query-gpu=pci.bus_id --format=csv          │
│                                                             │
│  解决：                                                      │
│  # 方案 1: 启用 P2P                                         │
│  export NCCL_P2PLevel=SYS  # 或 PHB/PGA/NVL                 │
│                                                             │
│  # 方案 2: 使用 NCCL 而非 P2P                               │
│  export NCCL_P2P_LEVEL=PIX                                  │
│                                                             │
│  # 方案 3: 禁用 P2P（降性能）                                │
│  export NCCL_P2P_DISABLE=1                                  │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题 3: 超时问题                                            │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  症状：NCCL call timed out                                   │
│                                                             │
│  解决：增加超时时间                                          │
│  export NCCL_TIMEOUT=1800  # 30 minutes                     │
│  export NCCL_BLOCKING_WAIT=1                                │
│                                                             │
│  如果仍然超时，检查：                                        │
│  1. 防火墙是否阻止了通信端口                                 │
│  2. 集群网络是否正常                                         │
│  3. 某个节点是否卡住                                         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题 4: IB（InfiniBand）问题                                │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  症状：IB related errors                                     │
│                                                             │
│  排查：                                                      │
│  $ ibv_devinfo  # 检查 IB 设备                              │
│  $ ibstat       # IB 状态                                   │
│                                                             │
│  解决：                                                      │
│  # 启用 IB                                                  │
│  export NCCL_IB_DISABLE=0                                   │
│  export NCCL_IB_HCA=mlx5_0  # 指定 IB 设备                  │
│                                                             │
│  # 或禁用 IB 使用以太网                                      │
│  export NCCL_IB_DISABLE=1                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### NCCL 配置最佳实践

```bash
#!/bin/bash
# nccl_config.sh - NCCL 配置模板

# ========== 基础配置 ==========
# 调试级别
export NCCL_DEBUG=WARN

# ========== 网络配置 ==========
# 指定网络接口（重要！）
export NCCL_SOCKET_IFNAME=eth0

# IB 配置
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0

# ========== P2P 配置 ==========
# GPU 直连通信
export NCCL_P2P_LEVEL=NVL  # NVLink 连接
# 或
# export NCCL_P2P_LEVEL=PIX  # PCIe 同 root complex
# export NCCL_P2P_LEVEL=SYS  # 跨域通信

# ========== 性能调优 ==========
# 算法选择
export NCCL_ALGO=Ring  # 或 Tree / Pat

# 协议选择
export NCCL_PROTO=Simple  # 或 LL (Low Latency)

# ========== 容错配置 ==========
# 超时设置
export NCCL_TIMEOUT=600  # 10 minutes

# 阻塞等待
export NCCL_BLOCKING_WAIT=1

# ========== 特定场景配置 ==========

# 多机训练
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# 大模型训练
export NCCL_MAX_NRINGS=4
export NCCL_MIN_NCHANNELS=4

echo "NCCL 配置完成"
env | grep NCCL
```

---

### NCCL 性能测试

```bash
# nccl_test.sh - NCCL 性能测试脚本

# 下载 NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/usr/local/cuda

# 测试 all-reduce 性能
mpirun -np 8 -hostfile hosts \
  ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 1

# 测试 all-gather 性能
mpirun -np 8 -hostfile hosts \
  ./build/all_gather_perf -b 8 -e 256M -f 2 -g 1

# 测试 broadcast 性能
mpirun -np 8 -hostfile hosts \
  ./build/broadcast_perf -b 8 -e 256M -f 2 -g 1

# 输出解读：
# - busbw: 有效带宽（GB/s）
# - algbw: 算法带宽（GB/s）
# - time: 延迟（us）
# 
# 理想情况：
# - NVLink: > 50 GB/s
# - PCIe: > 10 GB/s
# - Ethernet: 取决于网卡（25Gbps ≈ 3 GB/s）
```

---

## 11.3 热点问题：为什么这个 GPU 总是 100%？

### 问题现象

```
┌─────────────────────────────────────────────────────────────┐
│                   GPU 热点问题现象                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  $ nvidia-smi dmon -s u                                     │
│                                                             │
│  gpu   sm   mem   enc   dec   Jasper                        │
│  Idx     %     %     %     %       %                        │
│    0   98    45     0     0       0  ← 热点！               │
│    1   35    42     0     0       0                          │
│    2   33    40     0     0       0                          │
│    3   34    41     0     0       0                          │
│    4   32    39     0     0       0                          │
│    5   35    43     0     0       0                          │
│    6   33    41     0     0       0                          │
│    7   36    40     0     0       0                          │
│                                                             │
│  问题：GPU 0 几乎 100%，其他 GPU 闲置                        │
│                                                             │
│  影响：                                                      │
│  - 整体吞吐下降                                              │
│  - GPU 0 容易过热                                           │
│  - 训练/推理速度慢                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 热点原因排查

```
┌─────────────────────────────────────────────────────────────┐
│                   GPU 热点原因分析                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原因 1: 数据加载瓶颈                                        │
│  ─────────────────────────────────────────────────────────  │
│  问题：数据加载在 CPU 上串行处理                             │
│        GPU 0 负责分发数据到其他 GPU                         │
│                                                             │
│  排查：                                                      │
│  $ nvidia-smi dmon -s u -c 10  # 监控 10 次                │
│  如果 GPU-Util 波动大（忽高忽低），很可能是数据加载问题       │
│                                                             │
│  解决：                                                      │
│  - 增加 DataLoader workers                                  │
│  - 使用 pin_memory                                          │
│  - 预取数据                                                 │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  原因 2: 模型并行负载不均                                    │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题：Tensor Parallel 或 Pipeline Parallel 分配不均        │
│        某些层计算量更大，负责的 GPU 负载更高                 │
│                                                             │
│  排查：                                                      │
│  # 使用 Nsight Systems 分析每层的计算时间                   │
│  nsys profile -o report python train.py                     │
│                                                             │
│  解决：                                                      │
│  - 重新设计并行策略                                         │
│  - 调整层分配                                               │
│  - 使用更均衡的模型架构                                     │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  原因 3: 参数服务器模式                                      │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题：中心化的梯度聚合节点成为瓶颈                          │
│        GPU 0 作为参数服务器                                  │
│                                                             │
│  解决：                                                      │
│  - 使用 Ring AllReduce（如 Horovod）                        │
│  - 避免中心化聚合                                           │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  原因 4: 显存碎片                                            │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  问题：某些 GPU 显存碎片严重                                 │
│        导致有效 Batch Size 不同                             │
│                                                             │
│  排查：                                                      │
│  python -c "                                                │
│  import torch                                               │
│  for i in range(torch.cuda.device_count()):                │
│      torch.cuda.set_device(i)                               │
│      allocated = torch.cuda.memory_allocated() / 1e9        │
│      reserved = torch.cuda.memory_reserved() / 1e9          │
│      print(f'GPU {i}: {allocated:.1f}GB / {reserved:.1f}GB')│
│  "                                                          │
│                                                             │
│  解决：                                                      │
│  - 定期清空 cache                                           │
│  - 使用更合理的 Batch Size                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 热点问题解决方案

```python
# 解决方案 1: 优化数据加载
# ─────────────────────────────────────────────────────────────

from torch.utils.data import DataLoader

# 优化前
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,  # 单进程
)

# 优化后
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,          # 多进程加载
    pin_memory=True,        # 锁页内存
    prefetch_factor=4,       # 预取因子
    persistent_workers=True  # 保持 worker 存活
)

# ─────────────────────────────────────────────────────────────

# 解决方案 2: 使用分布式采样器
# ─────────────────────────────────────────────────────────────

from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True  # 避免最后一个 batch 大小不均
)

loader = DataLoader(
    dataset,
    batch_size=per_gpu_batch_size,
    sampler=sampler,
    num_workers=8,
    pin_memory=True
)

# ─────────────────────────────────────────────────────────────

# 解决方案 3: 监控和告警
# ─────────────────────────────────────────────────────────────

import subprocess
import time

def monitor_gpu_balance(threshold=30):
    """监控 GPU 负载均衡"""
    while True:
        # 获取各 GPU 利用率
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        
        utils = [float(x.strip()) for x in result.stdout.strip().split('\n')]
        
        if utils:
            avg_util = sum(utils) / len(utils)
            max_util = max(utils)
            
            # 检查是否均衡
            if max_util - avg_util > threshold:
                print(f"[WARN] GPU 不均衡! 最大利用率: {max_util:.1f}%, 平均: {avg_util:.1f}%")
                print(f"       GPU 利用率: {utils}")
                # 发送告警...
        
        time.sleep(10)

# 启动监控线程
import threading
monitor_thread = threading.Thread(target=monitor_gpu_balance, daemon=True)
monitor_thread.start()
```

---

## 11.4 CPU 瓶颈：预处理比推理还慢

### 问题现象

```
┌─────────────────────────────────────────────────────────────┐
│                   CPU 瓶颈问题现象                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  典型场景：                                                  │
│  ┌────────────────────────────────────────────────────────┐│
│  │                                                        ││
│  │  Timeline:                                              ││
│  │  ─────────────────────────────────────────────────────  ││
│  │  预处理 (CPU): ████████████████████  200ms              ││
│  │  推理 (GPU):   ████████               80ms              ││
│  │  ─────────────────────────────────────────────────────  ││
│  │  总时间: 280ms                                          ││
│  │                                                        ││
│  │  问题：                                                  ││
│  │  - 预处理时间 > 推理时间 × 2                            ││
│  │  - GPU 大部分时间在等待 CPU                             ││
│  │  - GPU 利用率低（< 50%）                                ││
│  │                                                        ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  常见原因：                                                   │
│  1. Tokenization 在 CPU 上串行处理                          │
│  2. 图像解码/预处理复杂                                      │
│  3. 数据增强操作耗时                                        │
│  4. Python 解释器开销                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### CPU 瓶颈排查

```python
import time
import functools

# 性能分析装饰器
def timeit(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"[{name or func.__name__}] 耗时: {(end-start)*1000:.2f}ms")
            return result
        return wrapper
    return decorator

# 分析各阶段时间
@timeit("Tokenization")
def tokenize(texts, tokenizer):
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

@timeit("数据传输")
def to_device(data, device):
    return {k: v.to(device) for k, v in data.items()}

@timeit("推理")
def inference(model, inputs):
    with torch.no_grad():
        return model(**inputs)

# 完整流程分析
def analyze_pipeline():
    texts = ["这是一段测试文本" * 100] * 32  # 32个样本
    
    # Step 1: Tokenization
    inputs = tokenize(texts, tokenizer)
    
    # Step 2: 数据传输
    inputs = to_device(inputs, device)
    
    # Step 3: 推理
    outputs = inference(model, inputs)
    
    # Step 4: 后处理
    results = postprocess(outputs)

# 输出示例：
# [Tokenization] 耗时: 150ms
# [数据传输] 耗时: 5ms
# [推理] 耗时: 80ms
# [后处理] 耗时: 10ms
# → Tokenization 是瓶颈！
```

---

### CPU 优化方案

```
┌─────────────────────────────────────────────────────────────┐
│                   CPU 瓶颈优化方案                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方案 1: 并行预处理                                          │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  from torch.utils.data import DataLoader                   │
│                                                             │
│  loader = DataLoader(                                      │
│      dataset,                                              │
│      batch_size=32,                                        │
│      num_workers=8,        # 使用 8 个进程并行加载          │
│      pin_memory=True,                                      │
│      prefetch_factor=4,                                    │
│  )                                                         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  方案 2: 批量 Tokenization                                  │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  # 优化前：逐个处理                                         │
│  for text in texts:                                        │
│      tokens = tokenizer(text)  # 每次调用有开销             │
│                                                             │
│  # 优化后：批量处理                                         │
│  tokens = tokenizer(texts, return_tensors="pt",            │
│                     padding=True, truncation=True)         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  方案 3: 预处理缓存                                          │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  import hashlib                                            │
│  from functools import lru_cache                           │
│                                                             │
│  @lru_cache(maxsize=10000)                                 │
│  def cached_tokenize(text: str):                           │
│      return tokenizer(text, return_tensors="pt")           │
│                                                             │
│  # 或者离线预处理保存                                        │
│  def precompute_and_save(dataset, save_dir):               │
│      for idx, item in enumerate(dataset):                  │
│          tokens = tokenizer(item["text"])                  │
│          torch.save(tokens, f"{save_dir}/{idx}.pt")        │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  方案 4: 使用高效库                                          │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  # tokenizers 库比 transformers 快 10x                      │
│  from tokenizers import Tokenizer                          │
│  tokenizer = Tokenizer.from_file("tokenizer.json")         │
│                                                             │
│  # 或使用 blingfire (极快的分词器)                          │
│  import blingfire                                          │
│  tokens = blingfire.text_to_ids(text, tokenizer_path)      │
│                                                             │
│  # 图像处理：kornia (GPU 加速)                              │
│  import kornia.augmentation as K                           │
│  transform = K.AugmentationSequential(...)                 │
│  images = transform(images.cuda())  # GPU 上处理           │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  方案 5: 异步流水线                                          │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  import threading                                          │
│  import queue                                              │
│                                                             │
│  class AsyncDataLoader:                                    │
│      def __init__(self, loader, queue_size=10):            │
│          self.loader = loader                              │
│          self.queue = queue.Queue(maxsize=queue_size)      │
│          self._stop = False                                │
│          self.thread = threading.Thread(target=self._load) │
│          self.thread.start()                               │
│                                                             │
│      def _load(self):                                      │
│          for batch in self.loader:                         │
│              if self._stop:                                │
│                  break                                     │
│              self.queue.put(batch)                         │
│          self.queue.put(None)  # 结束标志                   │
│                                                             │
│      def __iter__(self):                                   │
│          while True:                                       │
│              batch = self.queue.get()                      │
│              if batch is None:                             │
│                  break                                     │
│              yield batch                                   │
│                                                             │
│      def stop(self):                                       │
│          self._stop = True                                 │
│          self.thread.join()                                │
│                                                             │
│  # 使用                                                    │
│  async_loader = AsyncDataLoader(train_loader)              │
│  for batch in async_loader:                                │
│      # batch 已在后台准备好                                 │
│      train_step(batch)                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 性能对比

```
┌─────────────────────────────────────────────────────────────┐
│                 CPU 优化效果对比                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景：32 个文本样本的推理服务                                │
│                                                             │
│  优化前：                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Tokenization: 150ms                                  │  │
│  │ 数据传输:      5ms                                   │  │
│  │ 推理:         80ms                                   │  │
│  │ ─────────────────────────────────────────────────    │  │
│  │ 总计:        235ms                                   │  │
│  │ GPU 利用率:   34%                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  优化后 (批量 + 并行 + GPU预处理)：                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Tokenization:  20ms (批量)                           │  │
│  │ 数据传输:       5ms                                  │  │
│  │ 推理:          80ms                                  │  │
│  │ ─────────────────────────────────────────────────    │  │
│  │ 总计:        105ms                                   │  │
│  │ GPU 利用率:   76%                                    │  │
│  │                                                      │  │
│  │ 加速: 2.2x                                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  关键优化点：                                                 │
│  1. 批量处理替代逐个处理                                     │
│  2. 多 worker 并行加载                                       │
│  3. GPU 上做预处理                                           │
│  4. 缓存重复计算                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 本章小结

1. **显存泄漏很隐蔽**：可能是分配器问题、计算图未释放、缓存增长等多种原因，需要系统化排查。

2. **NCCL 问题复杂**：网络、P2P、超时等问题需要逐层排查，开启 Debug 日志是第一步。

3. **GPU 热点问题常见**：数据加载瓶颈、负载不均、参数服务器模式都是可能原因，监控和优化缺一不可。

4. **CPU 瓶颈容易被忽视**：预处理比推理慢很常见，批量处理、并行化、GPU加速是主要优化方向。

下一章，我们将讨论成本优化，这是老板最喜欢听的内容。

---

*放弃指数：⭐⭐⭐ 这些案例都是实战经验，建议收藏并定期回顾。*

---

*（未完待续...）*