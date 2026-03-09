# 第十二章 成本优化 —— 老板最喜欢听的章节

> *放弃指数：⭐⭐ 本章是省钱攻略，建议和老板一起看*

---

## 12.1 Spot Instance：用便宜显卡的代价

### Spot Instance 原理

```
┌─────────────────────────────────────────────────────────────┐
│                   Spot Instance 机制                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  云厂商的闲置计算资源：                                        │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  按需价格 (On-Demand):  $3.00/小时/GPU (A100)         │ │
│  │  Spot 价格:            $0.90/小时/GPU                 │ │
│  │                                                       │ │
│  │  节省：70%                                            │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  代价：                                                      │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  云厂商可以随时回收：                                   │ │
│  │  - 给出 2 分钟通知                                    │ │
│  │  - 实例被强制终止                                     │ │
│  │  - 未保存的数据丢失                                   │ │
│  │                                                       │ │
│  │  回收概率：                                            │ │
│  │  - 低需求时段：< 5%                                    │ │
│  │  - 高需求时段：> 20%                                   │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  适用场景：                                                   │
│  ✓ 批处理任务（可中断）                                      │
│  ✓ 分布式训练（支持 checkpoint）                             │
│  ✓ 开发测试环境                                              │
│  ✓ 容错性高的推理服务                                        │
│                                                             │
│  不适用：                                                     │
│  ✗ 长时间不可中断的任务                                      │
│  ✗ 实时推理服务（无备援）                                    │
│  ✗ 数据敏感任务                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Spot Instance 使用实践

```bash
# AWS Spot Instance 配置示例

# 1. 查看当前 Spot 价格
aws ec2 describe-spot-price-history \
  --instance-types p4d.24xlarge \
  --start-time $(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --query 'SpotPriceHistory[*].{AZ:AvailabilityZone,Price:SpotPrice}'

# 2. 创建 Spot Fleet
cat > spot-fleet-config.json << 'EOF'
{
  "IamFleetRole": "arn:aws:iam::123456789:role/aws-ec2-spot-fleet-tagging-role",
  "AllocationStrategy": "capacityOptimized",
  "TargetCapacity": 10,
  "SpotMaintenanceStrategies": {
    "CapacityRebalance": {
      "ReplacementStrategy": "launch"
    }
  },
  "LaunchTemplateConfigs": [{
    "LaunchTemplateSpecification": {
      "LaunchTemplateId": "lt-1234567890abcdef0",
      "Version": "$Latest"
    },
    "Overrides": [
      {"InstanceType": "p4d.24xlarge", "AvailabilityZone": "us-east-1a"},
      {"InstanceType": "p4d.24xlarge", "AvailabilityZone": "us-east-1b"},
      {"InstanceType": "p4d.24xlarge", "AvailabilityZone": "us-east-1c"}
    ]
  }]
}
EOF

aws ec2 request-spot-fleet --spot-fleet-request-config file://spot-fleet-config.json
```

---

### 处理 Spot Instance 中断

```python
import os
import signal
import time
import threading
from typing import Optional

class SpotInterruptionHandler:
    """Spot Instance 中断处理器"""
    
    def __init__(self, checkpoint_dir: str = "/tmp/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.interruption_received = False
        self.save_callback: Optional[callable] = None
        
        # 监听中断信号
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        # 启动元数据监控线程
        self._start_metadata_monitor()
    
    def _handle_signal(self, signum, frame):
        """处理中断信号"""
        print(f"\n[WARN] 收到中断信号 {signum}，准备保存状态...")
        self.interruption_received = True
        
        if self.save_callback:
            self.save_callback()
        
        # 优雅退出
        os._exit(0)
    
    def _start_metadata_monitor(self):
        """监控 AWS 元数据服务（检测即将中断）"""
        def monitor():
            while True:
                try:
                    # 检查是否收到终止通知
                    import urllib.request
                    req = urllib.request.Request(
                        "http://169.254.169.254/latest/meta-data/spot/instance-action",
                        headers={"X-aws-ec2-metadata-token": self._get_metadata_token()}
                    )
                    response = urllib.request.urlopen(req, timeout=1)
                    action = response.read().decode()
                    
                    if action in ["terminate", "stop"]:
                        print(f"[WARN] Spot 即将 {action}，保存状态...")
                        self.interruption_received = True
                        if self.save_callback:
                            self.save_callback()
                        break
                except:
                    pass
                
                time.sleep(5)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _get_metadata_token(self) -> str:
        """获取 AWS 元数据 token"""
        import urllib.request
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            method="PUT"
        )
        response = urllib.request.urlopen(req, timeout=1)
        return response.read().decode()
    
    def register_save_callback(self, callback: callable):
        """注册状态保存回调"""
        self.save_callback = callback
    
    def should_stop(self) -> bool:
        """检查是否应该停止"""
        return self.interruption_received


# 使用示例
handler = SpotInterruptionHandler(checkpoint_dir="/mnt/checkpoints")

# 注册保存函数
def save_checkpoint():
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': global_step,
    }, "/mnt/checkpoints/latest.pt")

handler.register_save_callback(save_checkpoint)

# 训练循环中使用
for epoch in range(num_epochs):
    for batch in train_loader:
        if handler.should_stop():
            save_checkpoint()
            break
        
        # 正常训练
        train_step(batch)
```

---

### Spot 成本优化策略

```
┌─────────────────────────────────────────────────────────────┐
│                 Spot Instance 成本优化策略                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  策略 1: 多可用区分散                                        │
│  ─────────────────────────────────────────────────────────  │
│  - 不同 AZ 的 Spot 价格和可用性不同                          │
│  - 分散到多个 AZ 降低同时被回收的概率                        │
│                                                             │
│  策略 2: 实例类型多样化                                      │
│  ─────────────────────────────────────────────────────────  │
│  - 同时请求 p4d.24xlarge 和 p4de.24xlarge                   │
│  - 某种类型被回收时，其他类型还能继续                         │
│                                                             │
│  策略 3: 使用 Spot Fleet / Auto Scaling                     │
│  ─────────────────────────────────────────────────────────  │
│  - 自动补充被回收的实例                                      │
│  - 保持目标容量                                             │
│                                                             │
│  策略 4: Checkpoint 频繁保存                                 │
│  ─────────────────────────────────────────────────────────  │
│  - 每 N 步保存一次                                          │
│  - 最大化利用中断前的计算成果                                │
│                                                             │
│  策略 5: 混合使用 Spot 和 On-Demand                          │
│  ─────────────────────────────────────────────────────────  │
│  - 关键节点使用 On-Demand                                   │
│  - Worker 节点使用 Spot                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 12.2 推理批处理：如何在延迟和吞吐之间平衡

### 延迟 vs 吞吐权衡

```
┌─────────────────────────────────────────────────────────────┐
│                  延迟 vs 吞吐 权衡                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  延迟敏感场景：                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  用户交互、聊天机器人、实时生成                         │ │
│  │                                                       │ │
│  │  优化目标：                                            │ │
│  │  - TTFT < 500ms                                       │ │
│  │  - TPOT < 50ms                                        │ │
│  │  - P99 延迟 < 2s                                      │ │
│  │                                                       │ │
│  │  代价：低 Batch Size，吞吐量低                         │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  吞吐敏感场景：                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  批量处理、离线生成、数据处理                           │ │
│  │                                                       │ │
│  │  优化目标：                                            │ │
│  │  - 最大化 tokens/second                               │ │
│  │  - GPU 利用率 > 80%                                   │ │
│  │                                                       │ │
│  │  代价：高 Batch Size，延迟高                           │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  Batch Size vs 性能关系图                             │  │
│  │                                                       │  │
│  │  吞吐 │                    ┌─────────────            │  │
│  │       │                   ╱                          │  │
│  │  (tokens│                 ╱                           │  │
│  │   /sec) │                ╱                            │  │
│  │         │───────────────╱                             │  │
│  │         └──────────────────────────────→ Batch Size   │  │
│  │                                                       │  │
│  │  延迟  │                                       ┌──    │  │
│  │         │                                     ╱       │  │
│  │   (ms) │                                    ╱        │  │
│  │         │───────────────────╱─────────────╱          │  │
│  │         └──────────────────────────────────→ Batch   │  │
│  │                        Size                           │  │
│  │                                                       │  │
│  │  结论：                                                │  │
│  │  - 小 Batch: 低延迟，低吞吐                            │  │
│  │  - 大 Batch: 高延迟，高吞吐                            │  │
│  │  - 存在最优平衡点                                      │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 批处理优化代码

```python
import time
import threading
from queue import Queue, Empty
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class Request:
    """推理请求"""
    request_id: str
    prompt: str
    max_tokens: int = 100
    arrival_time: float = 0.0

@dataclass
class Response:
    """推理响应"""
    request_id: str
    text: str
    latency_ms: float

class DynamicBatchScheduler:
    """动态批处理调度器"""
    
    def __init__(
        self,
        model,
        max_batch_size: int = 32,
        max_wait_ms: int = 50,  # 最大等待时间
        max_batch_tokens: int = 8192,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.max_batch_tokens = max_batch_tokens
        
        self.request_queue = Queue()
        self.response_queue = Queue()
        
        self._running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
    
    def submit(self, request: Request) str:
        """提交请求"""
        request.arrival_time = time.time()
        self.request_queue.put(request)
        return request.request_id
    
    def get_response(self, request_id: str, timeout: float = 30.0) -> Optional[Response]:
        """获取响应"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.request_id == request_id:
                    return response
                else:
                    # 放回队列
                    self.response_queue.put(response)
            except Empty:
                pass
        return None
    
    def _process_loop(self):
        """处理循环"""
        while self._running:
            batch = self._assemble_batch()
            
            if batch:
                # 批量推理
                prompts = [r.prompt for r in batch]
                outputs = self.model.generate(prompts, max_new_tokens=100)
                
                # 分发响应
                for request, output in zip(batch, outputs):
                    latency = (time.time() - request.arrival_time) * 1000
                    response = Response(
                        request_id=request.request_id,
                        text=output,
                        latency_ms=latency
                    )
                    self.response_queue.put(response)
    
    def _assemble_batch(self) -> List[Request]:
        """组装批次"""
        batch = []
        total_tokens = 0
        start_time = time.time()
        
        while len(batch) < self.max_batch_size:
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 超时或达到 token 限制
            if elapsed_ms >= self.max_wait_ms:
                break
            
            if total_tokens >= self.max_batch_tokens:
                break
            
            try:
                # 带超时的获取
                remaining_ms = max(1, self.max_wait_ms - elapsed_ms)
                request = self.request_queue.get(timeout=remaining_ms / 1000)
                
                batch.append(request)
                # 估算 token 数
                total_tokens += len(request.prompt.split()) + request.max_tokens
                
            except Empty:
                break
        
        return batch


# 使用示例
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

scheduler = DynamicBatchScheduler(
    model=model,
    max_batch_size=32,
    max_wait_ms=50,
    max_batch_tokens=4096
)

# 提交请求
request_id = scheduler.submit(Request(
    request_id="req-1",
    prompt="Hello, world!",
    max_tokens=100
))

# 获取响应
response = scheduler.get_response(request_id, timeout=10)
print(f"Response: {response.text}")
print(f"Latency: {response.latency_ms:.1f}ms")
```

---

## 12.3 模型缓存与预热：减少冷启动时间

### 冷启动问题

```
┌─────────────────────────────────────────────────────────────┐
│                   冷启动时间分析                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模型启动流程（70B 模型）：                                   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  1. 模型下载/加载                                     │  │
│  │     ████████████████████  45s                        │  │
│  │                                                       │  │
│  │  2. 权重加载到 GPU                                    │  │
│  │     ████████████████████████████  60s                │  │
│  │                                                       │  │
│  │  3. 预编译 Kernel                                     │  │
│  │     ████████  15s                                     │  │
│  │                                                       │  │
│  │  4. 首次推理预热                                      │  │
│  │     ██████  10s                                       │  │
│  │                                                       │  │
│  │  ─────────────────────────────────────────────────    │  │
│  │  总计: ~130s (2 分钟+)                                 │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  优化目标：                                                   │
│  - 减少首请求延迟                                           │
│  - 快速扩容响应                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 模型缓存策略

```python
import os
import json
import hashlib
from pathlib import Path
from typing import Optional
import torch

class ModelCache:
    """模型权重缓存管理器"""
    
    def __init__(self, cache_dir: str = "/mnt/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """加载缓存清单"""
        if self.cache_manifest_path.exists():
            return json.load(open(self.cache_manifest_path))
        return {}
    
    def _save_manifest(self):
        """保存缓存清单"""
        json.dump(self.manifest, open(self.cache_manifest_path, 'w'))
    
    def get_cache_key(self, model_name: str, quantization: Optional[str] = None) -> str:
        """生成缓存 key"""
        key_str = f"{model_name}_{quantization or 'fp16'}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def is_cached(self, model_name: str, quantization: Optional[str] = None) -> bool:
        """检查是否已缓存"""
        cache_key = self.get_cache_key(model_name, quantization)
        return cache_key in self.manifest
    
    def get_cache_path(self, model_name: str, quantization: Optional[str] = None) -> Path:
        """获取缓存路径"""
        cache_key = self.get_cache_key(model_name, quantization)
        return self.cache_dir / f"{cache_key}.pt"
    
    def save_weights(
        self,
        model_name: str,
        weights: Dict,
        quantization: Optional[str] = None
    ):
        """保存权重到缓存"""
        cache_key = self.get_cache_key(model_name, quantization)
        cache_path = self.get_cache_path(model_name, quantization)
        
        torch.save(weights, cache_path)
        
        self.manifest[cache_key] = {
            "model_name": model_name,
            "quantization": quantization,
            "path": str(cache_path),
            "size_mb": cache_path.stat().st_size / (1024 * 1024),
            "timestamp": time.time()
        }
        self._save_manifest()
    
    def load_weights(
        self,
        model_name: str,
        quantization: Optional[str] = None
    ) -> Optional[Dict]:
        """从缓存加载权重"""
        if not self.is_cached(model_name, quantization):
            return None
        
        cache_path = self.get_cache_path(model_name, quantization)
        return torch.load(cache_path)


# 使用示例
cache = ModelCache("/mnt/cache")

# 检查缓存
if cache.is_cached("llama-2-70b", "awq"):
    print("从缓存加载...")
    weights = cache.load_weights("llama-2-70b", "awq")
else:
    print("下载并缓存...")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
    cache.save_weights("llama-2-70b", model.state_dict(), "awq")
```

---

### 模型预热

```python
import time
import torch

class ModelWarmup:
    """模型预热器"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def warmup(
        self,
        batch_sizes: List[int] = [1, 4, 16],
        prompt_lengths: List[int] = [128, 512, 1024],
        output_lengths: List[int] = [32, 64],
    ):
        """
        预热模型
        
        目的：
        1. 预编译 CUDA Kernel
        2. 初始化 KV Cache
        3. 预执行 Attention 计算
        """
        print("开始模型预热...")
        start_time = time.time()
        
        for batch_size in batch_sizes:
            for prompt_len in prompt_lengths:
                for output_len in output_lengths:
                    self._warmup_step(batch_size, prompt_len, output_len)
        
        elapsed = time.time() - start_time
        print(f"预热完成，耗时: {elapsed:.1f}s")
    
    def _warmup_step(
        self,
        batch_size: int,
        prompt_length: int,
        output_length: int
    ):
        """单步预热"""
        # 生成假输入
        dummy_text = "test " * (prompt_length // 5)
        inputs = self.tokenizer(
            [dummy_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_length
        ).to(self.model.device)
        
        # 执行推理
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=output_length,
                do_sample=False
            )
        
        # 清理
        del inputs
        torch.cuda.empty_cache()
    
    def profile_memory(self):
        """显存分析"""
        torch.cuda.reset_peak_memory_stats()
        
        # 执行一次推理
        inputs = self.tokenizer("Hello", return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=100)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"峰值显存: {peak_memory:.2f} GB")
        
        return peak_memory


# 集成到服务启动
def start_inference_service():
    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
    
    # 2. 预热
    warmuper = ModelWarmup(model.cuda(), tokenizer)
    warmuper.warmup(
        batch_sizes=[1, 8],
        prompt_lengths=[512, 2048],
        output_lengths=[64, 128]
    )
    
    # 3. 分析显存
    warmuper.profile_memory()
    
    # 4. 服务就绪
    print("服务就绪！")
    return model, tokenizer
```

---

## 12.4 成本监控：你的模型每个 token 花多少钱？

### 成本计算公式

```
┌─────────────────────────────────────────────────────────────┐
│                   推理成本计算                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  基础公式：                                                   │
│                                                             │
│  单次推理成本 = GPU 时间 × GPU 小时价格                      │
│                                                             │
│  Token 成本 = 单次推理成本 / 输出 token 数                  │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  详细计算：                                                   │
│                                                             │
│  1. GPU 时间计算                                            │
│     T_total = T_prefill + T_decode                          │
│     T_prefill = input_tokens / prefill_speed                │
│     T_decode = output_tokens / decode_speed                 │
│                                                             │
│  2. GPU 小时价格（示例）                                     │
│     - A100 (云厂商): $2.5 - $3.5 / 小时                    │
│     - H100 (云厂商): $4.0 - $6.0 / 小时                    │
│     - 自建 A100: ¥电费 + ¥折旧                             │
│                                                             │
│  3. 实际例子                                                 │
│     模型: Llama-2-70B                                       │
│     GPU: A100 80GB × 4                                      │
│     GPU 价格: $3.0 / 小时 / 卡                              │
│     总价: $12 / 小时                                        │
│                                                             │
│     输入: 1000 tokens                                       │
│     输出: 200 tokens                                        │
│     Prefill 速度: 50000 tokens/s                           │
│     Decode 速度: 100 tokens/s (per card, TP=4)              │
│                                                             │
│     T_prefill = 1000 / 50000 = 0.02s                       │
│     T_decode = 200 / 100 = 2s                              │
│     T_total = 2.02s                                        │
│                                                             │
│     成本 = 2.02s × $12 / 3600s = $0.0067                   │
│     Token 成本 = $0.0067 / 200 = $0.000033/token           │
│                   = 0.0033 cents/token                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 成本监控系统

```python
import time
import json
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

@dataclass
class InferenceRecord:
    """推理记录"""
    timestamp: float
    request_id: str
    user_id: str
    model: str
    
    # Token 数量
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # 时间
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    
    # GPU 信息
    gpu_type: str
    gpu_count: int
    gpu_hours: float  # GPU 时间（小时）
    
    # 成本
    cost_usd: float
    cost_per_1k_tokens: float


class CostMonitor:
    """成本监控器"""
    
    def __init__(
        self,
        gpu_hourly_price: float = 3.0,  # 每张 GPU 每小时价格
        gpu_type: str = "A100",
        gpu_count: int = 1,
    ):
        self.gpu_hourly_price = gpu_hourly_price
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        
        self.records: List[InferenceRecord] = []
        self.lock = threading.Lock()
        
        # 统计
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_gpu_hours": 0.0,
        }
    
    def record(
        self,
        request_id: str,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        prefill_time_ms: float,
        decode_time_ms: float,
    ) -> InferenceRecord:
        """记录一次推理"""
        
        total_tokens = input_tokens + output_tokens
        total_time_ms = prefill_time_ms + decode_time_ms
        
        # 计算 GPU 时间（小时）
        total_time_hours = total_time_ms / 1000 / 3600
        gpu_hours = total_time_hours * self.gpu_count
        
        # 计算成本
        cost_usd = gpu_hours * self.gpu_hourly_price
        cost_per_1k_tokens = (cost_usd / total_tokens * 1000) if total_tokens > 0 else 0
        
        record = InferenceRecord(
            timestamp=time.time(),
            request_id=request_id,
            user_id=user_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
            total_time_ms=total_time_ms,
            gpu_type=self.gpu_type,
            gpu_count=self.gpu_count,
            gpu_hours=gpu_hours,
            cost_usd=cost_usd,
            cost_per_1k_tokens=cost_per_1k_tokens,
        )
        
        with self.lock:
            self.records.append(record)
            self.stats["total_requests"] += 1
            self.stats["total_tokens"] += total_tokens
            self.stats["total_cost"] += cost_usd
            self.stats["total_gpu_hours"] += gpu_hours
        
        return record
    
    def get_stats(self, hours: int = 24) -> Dict:
        """获取统计信息"""
        cutoff = time.time() - hours * 3600
        
        with self.lock:
            recent_records = [r for r in self.records if r.timestamp > cutoff]
            
            if not recent_records:
                return {"error": "No records in period"}
            
            total_cost = sum(r.cost_usd for r in recent_records)
            total_tokens = sum(r.total_tokens for r in recent_records)
            total_gpu_hours = sum(r.gpu_hours for r in recent_records)
            
            return {
                "period_hours": hours,
                "total_requests": len(recent_records),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 4),
                "total_gpu_hours": round(total_gpu_hours, 6),
                "avg_tokens_per_request": round(total_tokens / len(recent_records), 1),
                "avg_cost_per_request": round(total_cost / len(recent_records), 6),
                "avg_cost_per_1k_tokens": round(total_cost / total_tokens * 1000, 6) if total_tokens > 0 else 0,
                "gpu_type": self.gpu_type,
                "gpu_count": self.gpu_count,
                "gpu_hourly_price": self.gpu_hourly_price,
            }
    
    def export_report(self, filepath: str):
        """导出报告"""
        with self.lock:
            data = {
                "generated_at": datetime.now().isoformat(),
                "stats": self.get_stats(24),
                "records": [asdict(r) for r in self.records[-1000:]]  # 最近 1000 条
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)


# 使用示例
monitor = CostMonitor(
    gpu_hourly_price=12.0,  # 4 × A100
    gpu_type="A100",
    gpu_count=4
)

# 记录推理
record = monitor.record(
    request_id="req-123",
    user_id="user-456",
    model="llama-2-70b",
    input_tokens=500,
    output_tokens=200,
    prefill_time_ms=20,
    decode_time_ms=2000
)

print(f"本次成本: ${record.cost_usd:.6f}")
print(f"Token 成本: ${record.cost_per_1k_tokens:.6f} / 1k tokens")

# 获取统计
stats = monitor.get_stats(24)
print(f"\n24 小时统计:")
print(f"  总请求数: {stats['total_requests']}")
print(f"  总 Token: {stats['total_tokens']:,}")
print(f"  总成本: ${stats['total_cost_usd']:.4f}")
print(f"  平均每请求: ${stats['avg_cost_per_request']:.6f}")
print(f"  平均每 1k token: ${stats['avg_cost_per_1k_tokens']:.6f}")
```

---

### 成本优化建议

```
┌─────────────────────────────────────────────────────────────┐
│                   成本优化建议清单                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 模型选择                                                │
│     □ 使用量化模型（INT8/INT4）                             │
│     □ 选择合适尺寸的模型（不要过度配置）                     │
│     □ 小模型 + Speculative Decoding                         │
│                                                             │
│  2. 资源优化                                                │
│     □ 使用 Spot Instance                                   │
│     □ 合理配置 autoscaling                                  │
│     □ 避免资源闲置                                          │
│                                                             │
│  3. 推理优化                                                │
│     □ 提高 Batch Size（提升吞吐）                           │
│     □ 启用 KV Cache 复用                                   │
│     □ 使用 vLLM/SGLang 等高效框架                           │
│                                                             │
│  4. 缓存策略                                                │
│     □ 缓存常见请求结果                                      │
│     □ 使用 Prefix Caching                                  │
│     □ 预热减少冷启动                                        │
│                                                             │
│  5. 产品策略                                                │
│     □ 限制 max_tokens                                      │
│     □ 削峰填谷（延迟非紧急请求）                            │
│     □ 路由到不同成本的模型                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 本章小结

1. **Spot Instance 节省成本高达 70%**，但需要处理中断风险，适合可恢复的任务。

2. **批处理是吞吐优化的关键**，在延迟和吞吐之间找到平衡点。

3. **模型预热和缓存**能显著减少冷启动时间，提升用户体验。

4. **成本监控必须做**，了解每个请求、每个 token 的成本，才能做出正确决策。

---

*放弃指数：⭐⭐ 成本优化是个系统工程，需要持续监控和调整。*

---

*（未完待续...）*