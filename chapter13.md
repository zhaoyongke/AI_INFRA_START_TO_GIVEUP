# 第十三章 故障排查指南

> *放弃指数：⭐⭐⭐ 本章是排错宝典，建议配合实际排查使用*

---

## 13.1 OOM：显存不够的一百种原因

### OOM 分类

```
┌─────────────────────────────────────────────────────────────┐
│                   OOM 类型分类                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  类型 1: 模型权重放不下                                      │
│  ─────────────────────────────────────────────────────────  │
│  症状：模型加载时 OOM                                        │
│  原因：模型参数量 > 显存容量                                 │
│                                                             │
│  类型 2: 训练时 OOM                                          │
│  ─────────────────────────────────────────────────────────  │
│  症状：前向传播或反向传播时 OOM                              │
│  原因：激活值、梯度、优化器状态占用过多                      │
│                                                             │
│  类型 3: 推理时 OOM                                          │
│  ─────────────────────────────────────────────────────────  │
│  症状：生成过程中 OOM                                        │
│  原因：KV Cache 增长、请求过长、Batch Size 不当              │
│                                                             │
│  类型 4: 显存碎片 OOM                                        │
│  ─────────────────────────────────────────────────────────  │
│  症状：理论够用但实际 OOM                                    │
│  原因：显存碎片化严重                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### OOM 排查流程

```python
def diagnose_oom():
    """OOM 诊断函数"""
    import torch
    import gc
    
    print("=== GPU 显存诊断 ===\n")
    
    # 1. 基本信息
    print("1. GPU 基本信息:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
    
    # 2. 当前使用情况
    print("\n2. 当前显存使用:")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0
        
        print(f"   GPU {i}:")
        print(f"     - 已分配: {allocated:.2f} GB")
        print(f"     - 已预留: {reserved:.2f} GB")
        print(f"     - 碎片率: {fragmentation:.1f}%")
    
    # 3. 内存分布
    print("\n3. 内存占用分析:")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    print(f"   Tensor: {obj.shape}, {obj.dtype}, {obj.device}")
        except:
            pass
    
    # 4. 清理建议
    print("\n4. 清理建议:")
    
    # 清理垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    
    print("   - 已执行 gc.collect()")
    print("   - 已执行 torch.cuda.empty_cache()")
    
    # 清理后状态
    print("\n5. 清理后显存:")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU {i}: {allocated:.2f} GB / {reserved:.2f} GB")

# 执行诊断
diagnose_oom()
```

---

### OOM 解决方案速查表

```
┌─────────────────────────────────────────────────────────────┐
│                   OOM 解决方案速查表                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题                          解决方案                      │
│  ─────────────────────────────────────────────────────────  │
│  模型放不下                    - 使用模型并行 (TP/PP)        │
│                                - 使用量化 (INT8/INT4)        │
│                                - 使用 FSDP/ZeRO             │
│                                                             │
│  训练 OOM                      - 减小 batch_size             │
│                                - 启用 gradient checkpointing │
│                                - 使用 mixed precision       │
│                                - 启用 offloading            │
│                                - 减少 accumulation steps    │
│                                                             │
│  推理 OOM                      - 限制 max_tokens            │
│                                - 减小 batch_size            │
│                                - 启用 prefix caching        │
│                                - 使用连续批处理              │
│                                                             │
│  长序列 OOM                    - 启用 flash attention        │
│                                - 使用 KV Cache 优化          │
│                                - 分段处理                    │
│                                                             │
│  显存碎片                      - 定期 empty_cache            │
│                                - 使用 jemalloc               │
│                                - 重启服务                    │
│                                                             │
│  临时峰值                      - 增加显存（换 GPU）          │
│                                - 降低峰值计算                │
│                                - 分批处理                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 13.2 NCCL Timeout：分布式训练最头疼的问题

### NCCL Timeout 排查步骤

```bash
#!/bin/bash
# nccl_troubleshoot.sh - NCCL 问题排查脚本

echo "=== NCCL 问题排查 ==="

echo ""
echo "1. 检查 NCCL 版本"
python -c "import torch; print(f'NCCL version: {torch.cuda.nccl.version()}')"

echo ""
echo "2. 检查 GPU 可见性"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

echo ""
echo "3. 检查网络接口"
ip addr show | grep -E "^[0-9]|inet "

echo ""
echo "4. 测试 P2P 连接"
python << 'EOF'
import torch
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPU 数量: {num_gpus}")
    
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"  GPU {i} -> GPU {j}: {'✓' if can_access else '✗'}")
EOF

echo ""
echo "5. 检查 NCCL 环境变量"
env | grep NCCL

echo ""
echo "6. 快速测试 NCCL"
python << 'EOF'
import torch
import torch.distributed as dist
import os

os.environ['NCCL_DEBUG'] = 'WARN'

try:
    dist.init_process_group(backend='nccl')
    print(f"NCCL 初始化成功")
    print(f"  Rank: {dist.get_rank()}")
    print(f"  World size: {dist.get_world_size()}")
    
    # 测试 all-reduce
    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"  All-reduce 测试: {tensor.item()}")
    
    dist.destroy_process_group()
    print("NCCL 测试通过 ✓")
except Exception as e:
    print(f"NCCL 测试失败: {e}")
EOF
```

---

### NCCL Timeout 常见原因

```
┌─────────────────────────────────────────────────────────────┐
│                NCCL Timeout 常见原因                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原因 1: 网络配置错误                                        │
│  ─────────────────────────────────────────────────────────  │
│  检查：NCCL_SOCKET_IFNAME 是否正确                          │
│  解决：export NCCL_SOCKET_IFNAME=eth0                       │
│                                                             │
│  原因 2: 防火墙阻止                                          │
│  ─────────────────────────────────────────────────────────  │
│  检查：iptables/firewall 状态                                │
│  解决：开放 NCCL 通信端口（通常 29500 等）                   │
│                                                             │
│  原因 3: GPU P2P 不可用                                      │
│  ─────────────────────────────────────────────────────────  │
│  检查：nvidia-smi topo -m                                   │
│  解决：export NCCL_P2P_LEVEL=SYS 或 NCCL_P2P_DISABLE=1      │
│                                                             │
│  原因 4: 某个节点卡住                                        │
│  ─────────────────────────────────────────────────────────  │
│  检查：每个节点的 CPU/内存/GPU 状态                          │
│  解决：重启卡住的节点                                       │
│                                                             │
│  原因 5: DHPC/IB 配置问题                                    │
│  ─────────────────────────────────────────────────────────  │
│  检查：ibstat, ibv_devinfo                                  │
│  解决：检查 IB 驱动和配置                                   │
│                                                             │
│  原因 6: NCCL 版本不兼容                                     │
│  ─────────────────────────────────────────────────────────  │
│  检查：所有节点的 NCCL 版本                                  │
│  解决：统一 NCCL 版本                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 13.3 权重加载失败：格式不统一的痛苦

### 常见权重格式

```
┌─────────────────────────────────────────────────────────────┐
│                   常见权重格式                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  格式          后缀        特点                  常用框架   │
│  ─────────────────────────────────────────────────────────  │
│  PyTorch      .pt/.pth    标准 PyTorch 格式      PyTorch   │
│  Safetensors  .safetensors 安全快速              HF/Diffusers│
│  GGUF         .gguf       llama.cpp 格式         llama.cpp │
│  GGML         .ggml       旧版 llama.cpp         llama.cpp │
│  ONNX         .onnx       跨平台                 ONNX      │
│  TensorRT     .engine     NVIDIA 优化            TensorRT  │
│  AWQ          .pt         AWQ 量化格式           AutoAWQ   │
│  GPTQ         .pt         GPTQ 量化格式          AutoGPTQ  │
│                                                             │
│  格式转换工具：                                               │
│  - transformers: convert_slow_tokenizer, save_pretrained   │
│  - safetensors: safe_open, save_file                       │
│  - llama.cpp: convert.py                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 权重加载排查脚本

```python
def debug_weight_loading(model_path: str):
    """权重加载调试"""
    import torch
    from pathlib import Path
    import json
    
    print(f"=== 权重加载调试: {model_path} ===\n")
    
    path = Path(model_path)
    
    # 1. 检查路径
    print("1. 路径检查:")
    print(f"   路径存在: {path.exists()}")
    print(f"   是目录: {path.is_dir()}")
    
    if path.is_dir():
        files = list(path.glob("*"))
        print(f"   文件数: {len(files)}")
        for f in files[:10]:  # 显示前 10 个
            print(f"     - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    
    # 2. 检查配置
    print("\n2. 配置检查:")
    config_path = path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"   模型类型: {config.get('model_type', 'unknown')}")
        print(f"   隐藏维度: {config.get('hidden_size', 'unknown')}")
        print(f"   层数: {config.get('num_hidden_layers', 'unknown')}")
    else:
        print("   未找到 config.json")
    
    # 3. 检查权重文件
    print("\n3. 权重文件检查:")
    
    # PyTorch 格式
    pt_files = list(path.glob("*.bin")) + list(path.glob("*.pt")) + list(path.glob("*.pth"))
    print(f"   PyTorch 文件: {len(pt_files)}")
    
    # Safetensors 格式
    safetensor_files = list(path.glob("*.safetensors"))
    print(f"   Safetensors 文件: {len(safetensor_files)}")
    
    # 4. 尝试加载
    print("\n4. 加载测试:")
    
    # 尝试 safetensors
    if safetensor_files:
        try:
            from safetensors import safe_open
            with safe_open(safetensor_files[0], framework="pt") as f:
                keys = f.keys()
                print(f"   Safetensors 加载成功, {len(list(keys))} 个张量")
        except Exception as e:
            print(f"   Safetensors 加载失败: {e}")
    
    # 尝试 PyTorch
    if pt_files:
        try:
            state_dict = torch.load(pt_files[0], map_location="cpu")
            print(f"   PyTorch 加载成功, {len(state_dict)} 个张量")
            
            # 检查张量形状
            for i, (k, v) in enumerate(state_dict.items()):
                if i < 3:
                    print(f"     - {k}: {v.shape}")
        except Exception as e:
            print(f"   PyTorch 加载失败: {e}")

# 使用
debug_weight_loading("/path/to/model")
```

---

## 13.4 性能不达预期：从 Profiling 到瓶颈定位

### 性能 Profiling 工具

```python
# 使用 PyTorch Profiler

import torch
import torch.profiler as profiler

def profile_model(model, inputs):
    """模型性能分析"""
    
    # 预热
    for _ in range(10):
        _ = model(inputs)
    
    # Profile
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as p:
        for _ in range(10):
            model(inputs)
            p.step()
    
    # 打印摘要
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 使用
model = ... # 你的模型
inputs = ... # 输入数据
profile_model(model, inputs)
```

---

### 性能瓶颈定位清单

```
┌─────────────────────────────────────────────────────────────┐
│                  性能瓶颈定位清单                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 检查 GPU 利用率                                          │
│     $ nvidia-smi dmon -s u                                  │
│     - < 50%: 可能是 CPU/内存/IO 瓶颈                        │
│     - 50-80%: 正常范围                                      │
│     - > 90%: 计算密集                                       │
│                                                             │
│  2. 检查显存带宽                                             │
│     $ nvidia-smi dmon -s m                                  │
│     - 查看显存读写速率                                       │
│     - 对比理论带宽 (A100: 2TB/s)                             │
│                                                             │
│  3. 检查 PCIe 带宽                                           │
│     $ bandwidthTest                                         │
│     - 影响 CPU-GPU 数据传输                                 │
│                                                             │
│  4. 检查 CPU 占用                                            │
│     $ htop                                                  │
│     - 是否 CPU 瓶颈                                         │
│     - 单线程还是多线程                                       │
│                                                             │
│  5. 检查内存使用                                             │
│     $ free -h                                               │
│     - 是否有 swap                                           │
│     - 是否内存不足                                          │
│                                                             │
│  6. 检查磁盘 IO                                              │
│     $ iostat -x 1                                           │
│     - 数据加载是否成为瓶颈                                   │
│                                                             │
│  7. 检查网络延迟                                             │
│     $ ping, iperf                                           │
│     - 分布式训练的网络延迟                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 本章小结

1. **OOM 有多种类型**：模型放不下、训练/推理过程中的内存增长、显存碎片，需要针对性解决。

2. **NCCL Timeout 排查要有系统化思路**：网络接口、P2P 连接、超时设置、版本兼容逐项检查。

3. **权重格式不统一很常见**：PyTorch、Safetensors、GGUF、量化格式各有特点，转换时要注意兼容性。

4. **性能 Profiling 是定位瓶颈的关键**：使用 PyTorch Profiler、Nsight Systems 等工具，数据驱动地优化。

下一章，我们将展望 AI Infra 的未来趋势。

---

*放弃指数：⭐⭐⭐ 故障排查需要经验和直觉，多积累案例。*

---

*（未完待续...）*