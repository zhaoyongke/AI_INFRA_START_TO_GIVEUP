# 第十章 分布式系统设计

> *放弃指数：⭐⭐⭐⭐ 本章需要分布式系统基础，建议结合实践理解*

---

## 10.1 服务发现与负载均衡：不只是 Nginx

### AI 推理服务的负载均衡挑战

传统 Web 服务和 AI 推理服务有本质区别：

```
┌─────────────────────────────────────────────────────────────┐
│              传统 Web vs AI 推理 负载均衡对比                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传统 Web 服务：                                              │
│  ┌────────────────────────────────────────────────────────┐│
│  │                                                        ││
│  │  请求特点：                                              ││
│  │  - 请求时间：毫秒级                                      ││
│  │  - 资源消耗：CPU 为主                                    ││
│  │  - 请求差异：相对均匀                                    ││
│  │                                                        ││
│  │  负载均衡策略：                                          ││
│  │  - Round Robin（轮询）                                  ││
│  │  - Least Connections（最小连接数）                      ││
│  │  - IP Hash                                             ││
│  │                                                        ││
│  │  效果：均衡，可预测                                      ││
│  │                                                        ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  AI 推理服务：                                                │
│  ┌────────────────────────────────────────────────────────┐│
│  │                                                        ││
│  │  请求特点：                                              ││
│  │  - 请求时间：秒级甚至分钟级                              ││
│  │  - 资源消耗：GPU 显存 + 计算                             ││
│  │  - 请求差异：极大（短文本 vs 长文本）                     ││
│  │  - 批处理：需要考虑 batch 组装                           ││
│  │                                                        ││
│  │  负载均衡挑战：                                          ││
│  │  ❌ Round Robin 可能不公平                              ││
│  │     （短任务节点 vs 长任务节点）                          ││
│  │  ❌ Least Connections 忽略请求复杂度                    ││
│  │  ❌ 单纯 CPU/内存指标不足以反映负载                      ││
│  │                                                        ││
│  │  需要考虑：                                              ││
│  │  ✓ GPU 显存使用率                                       ││
│  │  ✓ KV Cache 占用                                        ││
│  │  ✓ 队列长度和等待时间                                   ││
│  │  ✓ 请求预估复杂度                                       ││
│  │                                                        ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 服务发现机制

```
┌─────────────────────────────────────────────────────────────┐
│                    服务发现架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  架构图：                                                     │
│                                                             │
│                    ┌─────────────────┐                      │
│                    │    客户端        │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│                             ▼                               │
│                    ┌─────────────────┐                      │
│                    │  API Gateway    │                      │
│                    │  (Kong/Nginx)   │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│                             ▼                               │
│                    ┌─────────────────┐                      │
│                    │ Service Registry│                      │
│                    │ (Consul/Etcd/   │                      │
│                    │  Kubernetes DNS)│                      │
│                    └────────┬────────┘                      │
│                             │                               │
│              ┌──────────────┼──────────────┐                │
│              ▼              ▼              ▼                │
│        ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│        │Inference │  │Inference │  │Inference │            │
│        │ Server 1 │  │ Server 2 │  │ Server N │            │
│        │          │  │          │  │          │            │
│        │ GPU: A100│  │ GPU: H100│  │ GPU: A100│            │
│        │ Load: 60%│  │ Load: 45%│  │ Load: 80%│            │
│        └──────────┘  └──────────┘  └──────────┘            │
│                                                             │
│  服务注册信息：                                               │
│  {                                                          │
│    "service_id": "inference-server-1",                      │
│    "address": "10.0.1.101",                                 │
│    "port": 8000,                                            │
│    "tags": ["gpu-a100", "model-llama2-70b"],                │
│    "metadata": {                                            │
│      "gpu_utilization": 0.6,                                │
│      "gpu_memory_used": 48,                                 │
│      "gpu_memory_total": 80,                                │
│      "queue_length": 12,                                    │
│      "avg_latency_ms": 45,                                  │
│      "models": ["llama2-70b", "llama2-13b"]                 │
│    }                                                        │
│  }                                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 智能负载均衡算法

```python
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ServerMetrics:
    """服务器指标"""
    server_id: str
    address: str
    port: int
    
    # GPU 指标
    gpu_utilization: float = 0.0      # 0-1
    gpu_memory_used: float = 0.0      # GB
    gpu_memory_total: float = 80.0    # GB
    
    # 请求指标
    queue_length: int = 0
    active_requests: int = 0
    avg_latency_ms: float = 0.0
    
    # 健康状态
    healthy: bool = True
    last_heartbeat: float = 0.0
    
    # 模型信息
    models: List[str] = None
    
    # 能力
    max_batch_size: int = 32
    max_sequence_length: int = 8192


class AIInferenceLoadBalancer:
    """AI 推理服务负载均衡器"""
    
    def __init__(self):
        self.servers: Dict[str, ServerMetrics] = {}
        self.request_history: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def register_server(self, metrics: ServerMetrics):
        """注册服务器"""
        with self.lock:
            self.servers[metrics.server_id] = metrics
    
    def update_metrics(self, server_id: str, **kwargs):
        """更新服务器指标"""
        with self.lock:
            if server_id in self.servers:
                server = self.servers[server_id]
                for key, value in kwargs.items():
                    if hasattr(server, key):
                        setattr(server, key, value)
                server.last_heartbeat = time.time()
    
    def select_server(
        self,
        model: str,
        estimated_tokens: int = 512,
        priority: str = "latency"  # latency / throughput / cost
    ) -> Optional[ServerMetrics]:
        """
        选择最佳服务器
        
        Args:
            model: 需要的模型
            estimated_tokens: 预估 token 数量
            priority: 优先级 (latency/throughput/cost)
        
        Returns:
            选中的服务器，如果没有合适的返回 None
        """
        with self.lock:
            # 1. 筛选健康且支持该模型的服务器
            candidates = [
                s for s in self.servers.values()
                if s.healthy 
                and model in (s.models or [])
                and self._check_capacity(s, estimated_tokens)
            ]
            
            if not candidates:
                return None
            
            # 2. 根据优先级排序
            if priority == "latency":
                # 优先选择延迟低的
                candidates.sort(key=lambda s: self._score_for_latency(s))
            elif priority == "throughput":
                # 优先选择吞吐高的
                candidates.sort(key=lambda s: self._score_for_throughput(s), reverse=True)
            elif priority == "cost":
                # 优先选择成本低的（如 GPU 利用率低的）
                candidates.sort(key=lambda s: self._score_for_cost(s), reverse=True)
            else:
                # 默认：综合评分
                candidates.sort(key=lambda s: self._composite_score(s), reverse=True)
            
            return candidates[0]
    
    def _check_capacity(self, server: ServerMetrics, estimated_tokens: int) -> bool:
        """检查服务器是否有足够容量"""
        # 检查显存
        memory_available = server.gpu_memory_total - server.gpu_memory_used
        if memory_available < 10:  # 预留 10GB
            return False
        
        # 检查队列
        if server.queue_length > 100:  # 队列过长
            return False
        
        # 检查 token 限制
        if estimated_tokens > server.max_sequence_length:
            return False
        
        return True
    
    def _score_for_latency(self, server: ServerMetrics) -> float:
        """延迟评分（越低越好）"""
        # 综合考虑当前队列和平均延迟
        queue_factor = server.queue_length * 50  # 每个排队请求增加 50ms 估算
        return server.avg_latency_ms + queue_factor
    
    def _score_for_throughput(self, server: ServerMetrics) -> float:
        """吞吐评分（越高越好）"""
        # GPU 利用率适中时吞吐最高
        optimal_util = 0.8
        utilization_score = 1 - abs(server.gpu_utilization - optimal_util)
        
        # 考虑剩余容量
        capacity_score = (server.gpu_memory_total - server.gpu_memory_used) / server.gpu_memory_total
        
        return utilization_score * 0.6 + capacity_score * 0.4
    
    def _score_for_cost(self, server: ServerMetrics) -> float:
        """成本评分（越高越节省）"""
        # GPU 利用率低 = 单位成本高
        # 但同时也意味着更快响应
        return 1 - server.gpu_utilization
    
    def _composite_score(self, server: ServerMetrics) -> float:
        """综合评分"""
        latency_score = 1 / (1 + self._score_for_latency(server) / 100)
        throughput_score = self._score_for_throughput(server)
        
        # 加权平均
        return latency_score * 0.5 + throughput_score * 0.5


# 使用示例
lb = AIInferenceLoadBalancer()

# 注册服务器
lb.register_server(ServerMetrics(
    server_id="server-1",
    address="10.0.1.101",
    port=8000,
    gpu_utilization=0.6,
    gpu_memory_used=48,
    queue_length=5,
    models=["llama2-70b", "llama2-13b"]
))

# 选择服务器
server = lb.select_server(
    model="llama2-70b",
    estimated_tokens=2048,
    priority="latency"
)

if server:
    print(f"Selected server: {server.server_id} at {server.address}:{server.port}")
```

---

### 生产环境实践

**Kubernetes + Istio 方案：**

```yaml
# inference-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-service
  annotations:
    # 自定义负载均衡
    traffic.sidecar.istio.io/loadBalancer: "LEAST_REQUEST"
spec:
  selector:
    app: inference
  ports:
  - port: 8000
    targetPort: 8000

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
      annotations:
        # Prometheus 监控
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      containers:
      - name: inference
        image: inference-server:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "128Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "64Gi"
        env:
        - name: MODEL_NAME
          value: "llama2-70b"
        - name: MAX_BATCH_SIZE
          value: "32"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## 10.2 推理集群调度：GPU 算力怎么分？

### 调度系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    推理集群调度架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                     用户请求                              ││
│  └────────────────────────┬────────────────────────────────┘│
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   请求队列 (Queue)                        ││
│  │                                                         ││
│  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐             ││
│  │   │Req 1│ │Req 2│ │Req 3│ │Req 4│ │  ...│             ││
│  │   │P=高 │ │P=中 │ │P=低 │ │P=高 │ │     │             ││
│  │   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘             ││
│  │                                                         ││
│  │   P = Priority                                          ││
│  └────────────────────────┬────────────────────────────────┘│
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   调度器 (Scheduler)                      ││
│  │                                                         ││
│  │   职责：                                                 ││
│  │   1. 请求优先级排序                                      ││
│  │   2. 资源匹配                                            ││
│  │   3. Batch 组装                                          ││
│  │   4. 路由决策                                            ││
│  │                                                         ││
│  │   ┌─────────────────────────────────────────────────┐   ││
│  │   │              Batch 组装器                        │   ││
│  │   │                                                 │   ││
│  │   │  将多个请求合并成一个 batch：                     │   ││
│  │   │  - 考虑 token 长度                               │   ││
│  │   │  - 考虑模型兼容性                                │   ││
│  │   │  - 考虑优先级                                    │   ││
│  │   │                  ┌─────────────────────────┐    │   ││
│  │   │  Req1 Req2 Req3 │ Batch: [Req1, Req2, Req3]│    │   ││
│  │   │  ↓    ↓    ↓    │ Tokens: 512+256+1024     │    │   ││
│  │   │  └──────────┬───┘ Priority: 高（取最高）    │    │   ││
│  │   │             └─────────────────────────────┘    │   ││
│  │   └─────────────────────────────────────────────────┘   ││
│  │                                                         ││
│  └────────────────────────┬────────────────────────────────┘│
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   执行器池 (Executor Pool)                ││
│  │                                                         ││
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐             ││
│  │   │Executor 1│  │Executor 2│  │Executor N│             ││
│  │   │ GPU: 0   │  │ GPU: 1   │  │ GPU: N   │             ││
│  │   │ Batch: 32│  │ Batch: 16│  │ Batch: 32│             ││
│  │   │ Load: 80%│  │ Load: 45%│  │ Load: 60%│             ││
│  │   └──────────┘  └──────────┘  └──────────┘             ││
│  │                                                         ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 请求调度器实现

```python
import heapq
import time
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

class RequestPriority(Enum):
    HIGH = 1      # 付费用户、实时交互
    MEDIUM = 2    # 普通用户
    LOW = 3       # 批处理后台任务


@dataclass(order=True)
class InferenceRequest:
    """推理请求"""
    request_id: str
    prompt: str
    model: str
    max_tokens: int = 512
    priority: RequestPriority = RequestPriority.MEDIUM
    
    # 预估信息
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 512
    
    # 时间信息
    arrival_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None  # 截止时间（可选）
    
    # 用户信息
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # 回调
    callback: Optional[callable] = None
    
    def __lt__(self, other):
        # 优先级排序：优先级高 + 到达早
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.arrival_time < other.arrival_time


class RequestScheduler:
    """请求调度器"""
    
    def __init__(self, max_batch_size: int = 32, max_wait_time_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        
        # 多级优先级队列
        self.queues: Dict[RequestPriority, List[InferenceRequest]] = {
            RequestPriority.HIGH: [],
            RequestPriority.MEDIUM: [],
            RequestPriority.LOW: [],
        }
        
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_wait_time_ms": 0,
        }
    
    def submit(self, request: InferenceRequest) -> str:
        """提交请求"""
        with self.lock:
            heapq.heappush(self.queues[request.priority], request)
            self.stats["total_requests"] += 1
            return request.request_id
    
    def get_batch(self, model: str) -> List[InferenceRequest]:
        """
        获取一批待处理的请求
        
        策略：
        1. 优先从高优先级队列取
        2. 考虑最大等待时间
        3. 合并相似请求
        """
        batch = []
        current_time = time.time()
        earliest_arrival = float('inf')
        
        with self.lock:
            # 按优先级遍历队列
            for priority in [RequestPriority.HIGH, RequestPriority.MEDIUM, RequestPriority.LOW]:
                queue = self.queues[priority]
                
                while queue and len(batch) < self.max_batch_size:
                    request = heapq.heappop(queue)
                    
                    # 检查模型兼容性
                    if request.model != model:
                        # 放回队列
                        heapq.heappush(queue, request)
                        continue
                    
                    batch.append(request)
                    earliest_arrival = min(earliest_arrival, request.arrival_time)
                    
                    # 检查是否达到最大等待时间
                    wait_time_ms = (current_time - earliest_arrival) * 1000
                    if wait_time_ms >= self.max_wait_time_ms:
                        break
                
                # 如果已经等待很久，不再从低优先级队列取
                if earliest_arrival < float('inf'):
                    wait_time_ms = (current_time - earliest_arrival) * 1000
                    if wait_time_ms >= self.max_wait_time_ms:
                        break
        
        if batch:
            self.stats["total_batches"] += 1
            avg_wait = sum(
                (current_time - r.arrival_time) * 1000 for r in batch
            ) / len(batch)
            self.stats["avg_wait_time_ms"] = (
                self.stats["avg_wait_time_ms"] * 0.9 + avg_wait * 0.1
            )
        
        return batch
    
    def get_queue_stats(self) -> Dict:
        """获取队列统计"""
        with self.lock:
            return {
                "high_priority": len(self.queues[RequestPriority.HIGH]),
                "medium_priority": len(self.queues[RequestPriority.MEDIUM]),
                "low_priority": len(self.queues[RequestPriority.LOW]),
                **self.stats
            }


class BatchAssembler:
    """Batch 组装器"""
    
    def __init__(self, max_batch_tokens: int = 8192):
        self.max_batch_tokens = max_batch_tokens
    
    def assemble(
        self,
        requests: List[InferenceRequest],
        gpu_memory_available: float
    ) -> List[List[InferenceRequest]]:
        """
        将请求组装成多个 batch
        
        考虑因素：
        1. Token 长度限制
        2. GPU 显存限制
        3. 延迟要求
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        # 按 token 长度排序（短请求优先）
        sorted_requests = sorted(
            requests,
            key=lambda r: r.estimated_input_tokens + r.estimated_output_tokens
        )
        
        for request in sorted_requests:
            request_tokens = (
                request.estimated_input_tokens + request.estimated_output_tokens
            )
            
            # 检查是否可以加入当前 batch
            if (
                len(current_batch) < self.max_batch_tokens // 128  # 近似限制
                and current_tokens + request_tokens <= self.max_batch_tokens
            ):
                current_batch.append(request)
                current_tokens += request_tokens
            else:
                # 开始新 batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [request]
                current_tokens = request_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches


class ClusterScheduler:
    """集群调度器"""
    
    def __init__(self, executors: List[Dict]):
        self.executors = executors  # 执行器列表
        
        self.request_scheduler = RequestScheduler()
        self.batch_assembler = BatchAssembler()
        
        self.executor_status: Dict[str, Dict] = {}
    
    def update_executor_status(self, executor_id: str, status: Dict):
        """更新执行器状态"""
        self.executor_status[executor_id] = status
    
    def schedule(self) -> List[Dict]:
        """调度决策"""
        decisions = []
        
        # 对每个模型进行调度
        models = set(e["model"] for e in self.executors)
        
        for model in models:
            # 获取该模型的请求 batch
            batch = self.request_scheduler.get_batch(model)
            
            if not batch:
                continue
            
            # 选择最佳执行器
            candidates = [
                e for e in self.executors
                if e["model"] == model and e.get("healthy", True)
            ]
            
            if not candidates:
                # 没有可用执行器，请求放回队列
                continue
            
            # 选择负载最低的
            selected = min(
                candidates,
                key=lambda e: self.executor_status.get(e["id"], {}).get("load", 0)
            )
            
            decisions.append({
                "executor_id": selected["id"],
                "batch": batch,
                "model": model,
            })
        
        return decisions
```

---

## 10.3 多租户隔离：我的请求被别人的长文本堵住了

### 多租户问题描述

```
┌─────────────────────────────────────────────────────────────┐
│                   多租户问题示意                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题场景：                                                   │
│                                                             │
│  时间轴 →                                                    │
│                                                             │
│  租户 A（短请求）：                                           │
│  [Prompt: 50 tokens] → [等待...] → [Response: 100 tokens]   │
│                         ↑被阻塞                              │
│                                                             │
│  租户 B（长请求）：                                           │
│  [Prompt: 8000 tokens] → [====长文本生成=====]               │
│                          ↑占用大量资源                        │
│                                                             │
│  问题：                                                      │
│  1. Head-of-Line Blocking：短请求被长请求阻塞                 │
│  2. 资源垄断：长请求占用大量 KV Cache                         │
│  3. 公平性：不同租户的服务质量差异大                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 隔离策略

```
┌─────────────────────────────────────────────────────────────┐
│                   多租户隔离策略                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  策略 1：资源配额 (Resource Quota)                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  租户 A 配额：                                         │ │
│  │  - 最大并发：10                                        │ │
│  │  - 最大 tokens/分钟：100,000                          │ │
│  │  - 最大 batch_size：16                                │ │
│  │                                                       │ │
│  │  租户 B 配额：                                         │ │
│  │  - 最大并发：5                                         │ │
│  │  - 最大 tokens/分钟：50,000                           │ │
│  │  - 最大 batch_size：8                                 │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  策略 2：优先级队列 (Priority Queue)                          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  优先级分类：                                          │ │
│  │  - P0: 实时交互（延迟敏感）                            │ │
│  │  - P1: 普通请求                                       │ │
│  │  - P2: 批处理后台任务                                  │ │
│  │                                                       │ │
│  │  调度规则：                                            │ │
│  │  P0 请求优先处理，P2 请求在资源空闲时处理              │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  策略 3：时间片轮转 (Time Slicing)                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  定期抢占总执行器                                      │ │
│  │  │                                                    │ │
│  │  │  T=0: 租户 A 使用                                  │ │
│  │  │  T=100ms: 切换到租户 B                             │ │
│  │  │  T=200ms: 切换到租户 C                             │ │
│  │  │  T=300ms: 切换回租户 A                             │ │
│  │                                                       │ │
│  │  问题：GPU 任务不好中断                                │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  策略 4：专用资源池 (Dedicated Pool)                          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │ │
│  │  │Pool A (VIP)  │ │Pool B (标准) │ │Pool C (批量) │   │ │
│  │  │ GPU: 2       │ │ GPU: 4       │ │ GPU: 2       │   │ │
│  │  │ 延迟 < 100ms │ │ 延迟 < 1s    │ │ 延迟不限     │   │ │
│  │  │ 高优先级     │ │ 中优先级     │ │ 低优先级     │   │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘   │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  策略 5：Preemption (抢占)                                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                                       │ │
│  │  当高优先级请求到达时：                                 │ │
│  │  1. 检查是否有低优先级任务在执行                        │ │
│  │  2. 暂停低优先级任务（保存状态）                        │ │
│  │  3. 执行高优先级任务                                   │ │
│  │  4. 恢复低优先级任务                                   │ │
│  │                                                       │ │
│  │  注意：需要支持任务暂停和恢复                          │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 多租户隔离实现

```python
import threading
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TenantConfig:
    """租户配置"""
    tenant_id: str
    
    # 资源配额
    max_concurrent_requests: int = 10
    max_tokens_per_minute: int = 100000
    max_tokens_per_request: int = 8192
    
    # 优先级
    priority_class: str = "standard"  # premium / standard / bulk
    
    # QPS 限制
    max_requests_per_second: float = 10.0
    
    # 成本控制
    max_cost_per_day: float = 100.0


class TenantResourceMonitor:
    """租户资源监控"""
    
    def __init__(self):
        self.usage: Dict[str, Dict] = defaultdict(lambda: {
            "current_requests": 0,
            "tokens_this_minute": 0,
            "requests_this_second": 0,
            "cost_today": 0.0,
            "last_minute_reset": time.time(),
            "last_second_reset": time.time(),
            "last_day_reset": time.time(),
        })
        self.lock = threading.RLock()
    
    def check_quota(self, tenant: TenantConfig, tokens: int) -> bool:
        """检查是否有足够的配额"""
        with self.lock:
            usage = self.usage[tenant.tenant_id]
            current_time = time.time()
            
            # 重置计数器
            if current_time - usage["last_minute_reset"] > 60:
                usage["tokens_this_minute"] = 0
                usage["last_minute_reset"] = current_time
            
            if current_time - usage["last_second_reset"] > 1:
                usage["requests_this_second"] = 0
                usage["last_second_reset"] = current_time
            
            if current_time - usage["last_day_reset"] > 86400:
                usage["cost_today"] = 0.0
                usage["last_day_reset"] = current_time
            
            # 检查各项限制
            if usage["current_requests"] >= tenant.max_concurrent_requests:
                return False
            
            if usage["tokens_this_minute"] + tokens > tenant.max_tokens_per_minute:
                return False
            
            if usage["requests_this_second"] >= tenant.max_requests_per_second:
                return False
            
            if usage["cost_today"] >= tenant.max_cost_per_day:
                return False
            
            return True
    
    def acquire(self, tenant_id: str, tokens: int, cost: float = 0.0):
        """占用资源"""
        with self.lock:
            usage = self.usage[tenant_id]
            usage["current_requests"] += 1
            usage["tokens_this_minute"] += tokens
            usage["requests_this_second"] += 1
            usage["cost_today"] += cost
    
    def release(self, tenant_id: str):
        """释放资源"""
        with self.lock:
            self.usage[tenant_id]["current_requests"] -= 1


class MultiTenantScheduler:
    """多租户调度器"""
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self.monitor = TenantResourceMonitor()
        
        # 按优先级分类的队列
        self.queues = {
            "premium": [],    # 高优先级
            "standard": [],   # 标准优先级
            "bulk": [],       # 批处理低优先级
        }
        
        self.lock = threading.RLock()
    
    def register_tenant(self, config: TenantConfig):
        """注册租户"""
        self.tenants[config.tenant_id] = config
    
    def submit_request(
        self,
        tenant_id: str,
        request: InferenceRequest
    ) -> bool:
        """提交请求"""
        if tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[tenant_id]
        
        # 检查配额
        tokens = (
            request.estimated_input_tokens + request.estimated_output_tokens
        )
        
        if not self.monitor.check_quota(tenant, tokens):
            return False
        
        # 占用资源
        self.monitor.acquire(tenant_id, tokens)
        
        # 放入对应优先级队列
        with self.lock:
            priority_class = tenant.priority_class
            self.queues[priority_class].append(request)
        
        return True
    
    def get_next_batch(
        self,
        batch_size: int = 8
    ) -> List[InferenceRequest]:
        """获取下一批请求"""
        batch = []
        
        with self.lock:
            # 按优先级获取
            for priority in ["premium", "standard", "bulk"]:
                queue = self.queues[priority]
                
                while queue and len(batch) < batch_size:
                    request = queue.pop(0)
                    batch.append(request)
        
        return batch
    
    def complete_request(self, tenant_id: str):
        """请求完成"""
        self.monitor.release(tenant_id)
```

---

## 10.4 高可用设计：当显卡烧了怎么办？

### 高可用架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    高可用架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────────┐                      │
│                    │     用户        │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│                             ▼                               │
│                    ┌─────────────────┐                      │
│                    │  DNS 负载均衡   │                      │
│                    │ (Route 53/云DNS)│                      │
│                    └────────┬────────┘                      │
│                             │                               │
│              ┌──────────────┼──────────────┐                │
│              ▼              ▼              ▼                │
│        ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│        │ Region A │  │ Region B │  │ Region C │            │
│        │ (主区域)  │  │ (备区域)  │  │ (备区域)  │            │
│        └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│             │              │              │                 │
│             ▼              ▼              ▼                 │
│        ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│        │  API GW  │  │  API GW  │  │  API GW  │            │
│        └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│             │              │              │                 │
│             ▼              ▼              ▼                 │
│        ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│        │ Inf Pool │  │ Inf Pool │  │ Inf Pool │            │
│        │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │            │
│        │ │ GPU 0 │ │  │ │ GPU 0 │ │  │ │ GPU 0 │ │            │
│        │ │ GPU 1 │ │  │ │ GPU 1 │ │  │ │ GPU 1 │ │            │
│        │ │ GPU 2 │ │  │ │ GPU 2 │ │  │ │ GPU 2 │ │            │
│        │ │ GPU 3 │ │  │ │ GPU 3 │ │  │ │ GPU 3 │ │            │
│        │ └──────┘ │  │ └──────┘ │  │ └──────┘ │            │
│        └──────────┘  └──────────┘  └──────────┘            │
│                                                             │
│  故障转移流程：                                               │
│  1. 健康检查发现 Region A 故障                               │
│  2. DNS 更新，流量切到 Region B                              │
│  3. Region B 扩容                                           │
│  4. Region A 恢复后逐步切回                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 健康检查机制

```python
import time
import threading
import requests
from typing import List, Dict, Callable
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # 部分功能异常
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    server_id: str
    status: HealthStatus
    details: Dict
    timestamp: float
    latency_ms: float


class HealthChecker:
    """健康检查器"""
    
    def __init__(
        self,
        check_interval_seconds: int = 10,
        unhealthy_threshold: int = 3,
        healthy_threshold: int = 2,
    ):
        self.check_interval = check_interval_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold
        
        self.servers: Dict[str, Dict] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.consecutive_failures: Dict[str, int] = defaultdict(int)
        self.consecutive_successes: Dict[str, int] = defaultdict(int)
        
        self.on_status_change: Optional[Callable] = None
        
        self._running = False
    
    def register_server(
        self,
        server_id: str,
        endpoint: str,
        check_paths: List[str] = None
    ):
        """注册服务器"""
        self.servers[server_id] = {
            "endpoint": endpoint,
            "check_paths": check_paths or ["/health", "/ready"],
        }
        self.health_status[server_id] = HealthStatus.HEALTHY
    
    def check_health(self, server_id: str) -> HealthCheckResult:
        """执行健康检查"""
        server = self.servers[server_id]
        start_time = time.time()
        
        details = {
            "checks": {},
            "errors": [],
        }
        
        overall_healthy = True
        
        for path in server["check_paths"]:
            try:
                url = f"{server['endpoint']}{path}"
                response = requests.get(url, timeout=5)
                
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    details["checks"][path] = {
                        "status": "ok",
                        "latency_ms": latency_ms,
                    }
                else:
                    details["checks"][path] = {
                        "status": "error",
                        "code": response.status_code,
                    }
                    overall_healthy = False
                    details["errors"].append(f"{path}: HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                details["checks"][path] = {"status": "timeout"}
                overall_healthy = False
                details["errors"].append(f"{path}: timeout")
            except Exception as e:
                details["checks"][path] = {"status": "error", "message": str(e)}
                overall_healthy = False
                details["errors"].append(f"{path}: {str(e)}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 更新状态计数
        if overall_healthy:
            self.consecutive_successes[server_id] += 1
            self.consecutive_failures[server_id] = 0
            
            if self.consecutive_successes[server_id] >= self.healthy_threshold:
                new_status = HealthStatus.HEALTHY
        else:
            self.consecutive_failures[server_id] += 1
            self.consecutive_successes[server_id] = 0
            
            if self.consecutive_failures[server_id] >= self.unhealthy_threshold:
                new_status = HealthStatus.UNHEALTHY
            else:
                new_status = HealthStatus.DEGRADED
        
        # 检查状态变化
        old_status = self.health_status.get(server_id)
        if new_status != old_status:
            self.health_status[server_id] = new_status
            if self.on_status_change:
                self.on_status_change(server_id, old_status, new_status, details)
        
        return HealthCheckResult(
            server_id=server_id,
            status=new_status,
            details=details,
            timestamp=time.time(),
            latency_ms=latency_ms,
        )
    
    def start(self):
        """启动健康检查线程"""
        self._running = True
        
        def check_loop():
            while self._running:
                for server_id in list(self.servers.keys()):
                    self.check_health(server_id)
                time.sleep(self.check_interval)
        
        thread = threading.Thread(target=check_loop, daemon=True)
        thread.start()
    
    def stop(self):
        """停止健康检查"""
        self._running = False


class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self, health_checker: HealthChecker):
        self.health_checker = health_checker
        self.active_servers: set = set()
        self.standby_servers: set = set()
        
        # 设置健康检查回调
        self.health_checker.on_status_change = self._on_status_change
    
    def _on_status_change(
        self,
        server_id: str,
        old_status: HealthStatus,
        new_status: HealthStatus,
        details: Dict
    ):
        """健康状态变化回调"""
        print(f"[{time.strftime('%H:%M:%S')}] Server {server_id}: {old_status.value} → {new_status.value}")
        
        if new_status == HealthStatus.UNHEALTHY:
            # 从活跃列表移除
            if server_id in self.active_servers:
                self.active_servers.remove(server_id)
                print(f"  → Removed from active pool")
                
                # 尝试启用备用服务器
                self._activate_standby()
        
        elif new_status == HealthStatus.HEALTHY:
            # 恢复活跃
            if server_id not in self.active_servers:
                self.active_servers.add(server_id)
                print(f"  → Added back to active pool")
    
    def _activate_standby(self):
        """激活备用服务器"""
        if self.standby_servers:
            standby = self.standby_servers.pop()
            self.active_servers.add(standby)
            print(f"  → Activated standby server: {standby}")
    
    def get_active_servers(self) -> List[str]:
        """获取活跃服务器列表"""
        return list(self.active_servers)


# 使用示例
health_checker = HealthChecker(
    check_interval_seconds=10,
    unhealthy_threshold=3,
    healthy_threshold=2,
)

# 注册服务器
health_checker.register_server("server-1", "http://10.0.1.101:8000")
health_checker.register_server("server-2", "http://10.0.1.102:8000")
health_checker.register_server("server-3", "http://10.0.1.103:8000")

# 创建故障转移管理器
failover = FailoverManager(health_checker)
failover.active_servers = {"server-1", "server-2"}
failover.standby_servers = {"server-3"}

# 启动健康检查
health_checker.start()
```

---

### 故障恢复策略

```
┌─────────────────────────────────────────────────────────────┐
│                    故障恢复策略                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景 1：单 GPU 故障                                         │
│                                                             │
│  检测：健康检查失败                                          │
│  动作：                                                      │
│  1. 标记 GPU 为 unhealthy                                   │
│  2. 停止向其发送新请求                                       │
│  3. 将该 GPU 上的请求迁移到其他 GPU                          │
│  4. 通知运维                                                 │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  场景 2：整节点故障                                          │
│                                                             │
│  检测：节点心跳丢失                                          │
│  动作：                                                      │
│  1. 更新服务注册，移除节点                                   │
│  2. 流量切到其他节点                                         │
│  3. 自动扩容（如果配置了自动扩展）                           │
│  4. 告警通知                                                 │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  场景 3：区域故障                                            │
│                                                             │
│  检测：区域内多个节点同时故障                                │
│  动作：                                                      │
│  1. DNS 切换到备用区域                                       │
│  2. 备用区域扩容                                             │
│  3. 通知相关人员                                             │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  场景 4：流量突增                                            │
│                                                             │
│  检测：队列长度 / 延迟超过阈值                               │
│  动作：                                                      │
│  1. 自动水平扩展                                             │
│  2. 限流保护                                                 │
│  3. 降级策略（如减少 max_tokens）                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 自动扩缩容实现

```python
import time
import threading
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ScalingPolicy:
    """扩缩容策略"""
    # 扩容条件
    scale_up_threshold: float = 0.8    # GPU 利用率阈值
    scale_up_cooldown: int = 60        # 扩容冷却时间（秒）
    
    # 缩容条件
    scale_down_threshold: float = 0.3  # GPU 利用率阈值
    scale_down_cooldown: int = 300     # 缩容冷却时间（秒）
    
    # 限制
    min_replicas: int = 1
    max_replicas: int = 10


class AutoScaler:
    """自动扩缩容"""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.current_replicas = 1
        
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        
        self.metrics_history: List[Dict] = []
        self._running = False
    
    def evaluate(self, current_metrics: Dict) -> str:
        """
        评估是否需要扩缩容
        
        Args:
            current_metrics: 包含 gpu_utilization, queue_length, avg_latency 等
        
        Returns:
            "scale_up" / "scale_down" / "no_action"
        """
        current_time = time.time()
        
        # 记录历史
        self.metrics_history.append({
            "timestamp": current_time,
            **current_metrics
        })
        
        # 只保留最近 5 分钟的数据
        cutoff = current_time - 300
        self.metrics_history = [
            m for m in self.metrics_history
            if m["timestamp"] > cutoff
        ]
        
        # 计算平均值
        if not self.metrics_history:
            return "no_action"
        
        avg_utilization = sum(
            m["gpu_utilization"] for m in self.metrics_history
        ) / len(self.metrics_history)
        
        avg_queue = sum(
            m.get("queue_length", 0) for m in self.metrics_history
        ) / len(self.metrics_history)
        
        # 扩容判断
        if (
            (avg_utilization > self.policy.scale_up_threshold or avg_queue > 50)
            and current_time - self.last_scale_up_time > self.policy.scale_up_cooldown
            and self.current_replicas < self.policy.max_replicas
        ):
            self.last_scale_up_time = current_time
            return "scale_up"
        
        # 缩容判断
        if (
            avg_utilization < self.policy.scale_down_threshold
            and avg_queue < 5
            and current_time - self.last_scale_down_time > self.policy.scale_down_cooldown
            and self.current_replicas > self.policy.min_replicas
        ):
            self.last_scale_down_time = current_time
            return "scale_down"
        
        return "no_action"
    
    def scale_up(self):
        """执行扩容"""
        if self.current_replicas < self.policy.max_replicas:
            self.current_replicas += 1
            # 实际实现需要调用 Kubernetes API 或云服务 API
    
    def scale_down(self):
        """执行缩容"""
        if self.current_replicas > self.policy.min_replicas:
            self.current_replicas -= 1


# Kubernetes 实际扩缩容示例
"""
import kubernetes.client

def k8s_scale_deployment(namespace: str, deployment: str, replicas: int):
    api = kubernetes.client.AppsV1Api()
    
    body = {
        "spec": {
            "replicas": replicas
        }
    }
    
    api.patch_namespaced_deployment_scale(
        name=deployment,
        namespace=namespace,
        body=body
    )
"""
```

---

### 灾备演练

```
┌─────────────────────────────────────────────────────────────┐
│                    灾备演练清单                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  定期演练（建议每月一次）：                                    │
│                                                             │
│  1. 单点故障演练                                             │
│     □ 模拟单个 GPU 故障                                      │
│     □ 验证请求迁移                                           │
│     □ 确认告警发出                                           │
│     □ 检查服务恢复时间                                       │
│                                                             │
│  2. 多点故障演练                                             │
│     □ 模拟多 GPU 同时故障                                    │
│     □ 验证自动扩容                                           │
│     □ 检查限流保护                                           │
│                                                             │
│  3. 区域切换演练                                             │
│     □ 执行区域故障转移                                       │
│     □ 验证 DNS 切换                                          │
│     □ 检查数据一致性                                         │
│                                                             │
│  4. 流量突增演练                                             │
│     □ 模拟流量峰值                                           │
│     □ 验证自动扩容                                           │
│     □ 确认服务降级策略                                       │
│                                                             │
│  演练后复盘：                                                 │
│  - 记录发现的问题                                            │
│  - 更新应急预案                                              │
│  - 改进监控告警                                              │
│  - 修复暴露的 bug                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 本章小结

1. **AI 推理负载均衡不同于传统 Web 服务**：需要考虑 GPU 显存、KV Cache、请求复杂度等多个维度，传统 Round Robin 不再适用。

2. **智能调度是关键**：结合请求优先级、资源配额、Batch 组装等因素，实现高效、公平的资源分配。

3. **多租户隔离需要综合策略**：资源配额、优先级队列、专用资源池等多管齐下，确保服务质量公平。

4. **高可用是持续工程**：健康检查、故障转移、自动扩缩容、灾备演练，形成完整的可靠性保障体系。

---

*放弃指数：⭐⭐⭐⭐ 分布式系统设计涉及众多知识点，建议结合实际项目经验理解。*

---

*（全书完）*