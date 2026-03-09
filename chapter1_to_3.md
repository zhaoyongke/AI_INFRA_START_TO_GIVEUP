# AI Infra 从入门到放弃

> *"从'我要改变世界'到'我想改变行业'的完整心路历程"*

---

## 目录

- 第一章 AI Infra 是什么？我为什么在这里？
- 第二章 硬件基础 —— 显卡比你想象的贵
- 第三章 训练框架入门

---

# 第一章 AI Infra 是什么？我为什么在这里？

> *放弃指数：⭐ 本章是入门概念，但你可能会被后面章节劝退*

---

## 1.1 从训练一个模型到服务一百万用户

### 一个真实的故事

小明是一名机器学习工程师。某天，老板兴奋地找到他：

> "小明！我们训练了一个超厉害的对话模型，效果非常好，明天上线！"

小明看着老板闪亮的眼睛，点了点头。然后他打开电脑，开始思考几个问题：

1. **这个模型多大？**
   "72B 参数，FP16 存储。"
   小明默默算了一下：72B × 2 bytes = 144GB 显存。单张 A100 80GB 扛不住，需要切分。

2. **预期并发是多少？**
   "上线第一天预计 10 万用户同时在线。"
   小明脑补了一下：如果每个请求平均 500 tokens，10 万并发意味着...50 million tokens 同时在处理。这需要多少张卡？

3. **延迟要求呢？**
   "当然是越快越好，用户等不及。"
   小明查了一下论文，首 token 延迟要在 500ms 以内，平均每个 token 生成要控制在 50ms 内。这意味着什么？

4. **成本预算？**
   老板沉默了一秒："这个...你们技术团队商量一下。"

小明下班后发了条朋友圈：
> *训练一个模型只需要一台机器、一个人、一段时间。服务一个模型需要一整个团队、一整个机房、还有一整个人生。*

这就是 **AI Infra** 要解决的问题。

---

### 什么是 AI Infra？

**AI Infra（AI Infrastructure）**，中文叫"AI 基础设施"，是指支撑 AI 模型从训练到部署全生命周期的技术体系。

用一个不太恰当但很形象的比喻：

```
AI 算法工程师  ≈  建筑设计师
                    ↓
AI Infra 工程师 ≈  建筑施工队
```

设计师画出宏伟蓝图，施工队负责把它变成现实。但现实往往比蓝图复杂得多：

| 模型训练 | 模型服务 |
|---------|---------|
| 单次运行，跑完算数 | 7×24 小时在线，不能挂 |
| 批量处理，吞吐优先 | 实时响应，延迟优先 |
| 显存不够可以存 checkpoint | OOM 就意味着服务不可用 |
| 失败了重新跑一次 | 失败了用户就流失了 |
| 算力成本在预算内 | 每秒钟都在烧钱 |

这就是为什么大模型时代，**AI Infra 工程师** 成了最紧缺的岗位之一。

---

### AI Infra 的核心问题

当你踏上这条路，你会发现每天要处理的问题不外乎这几类：

**1. 显存管理**

显存是 AI Infra 工程师的心头痛。模型权重要占显存，KV Cache 要占显存，激活值要占显存。你永远觉得显存不够用。

> *真实场景：优化后显存占用降低 30%，然后产品说"我们要把上下文从 4K 扩展到 128K"，然后显存又炸了。*

**2. 计算效率**

GPU 很贵，不能让它闲着。但你的计算效率为什么只有理论峰值的 40%？

可能是：
- Kernel 没有优化，显存带宽没打满
- 通信开销太大，GPU 等数据的时间比计算时间还长
- 批处理策略不合理，GPU 利用率忽高忽低

**3. 系统吞吐与延迟**

这是一个永恒的权衡。吞吐量高时，延迟往往也高；延迟低时，吞吐量往往也低。

> *真实场景：老板说"要高并发"，产品说"要低延迟"，运营说"要低成本"。你只能回复："选两个。"*

**4. 可靠性与可观测性**

模型服务不能莫名其妙挂掉。但深入看：

- 为什么偶尔会出现 NaN？
- 为什么某个请求耗时是平均值的 10 倍？
- 为什么 GPU 利用率从 90% 突然降到 0%？

这些问题都需要 Infra 工程师来回答。

---

## 1.2 AI Infra 工程师的一天

### 上午 9:00：看监控

你到公司的第一个动作不是打开 IDE，而是打开 Grafana。

```
昨夜告警记录：
- 02:15 GPU-03 显存使用率 95% 持续 10 分钟
- 03:42 推理服务 P99 延迟飙升至 2.3 秒
- 05:17 某个请求触发了 NCCL 超时
```

你开始排查。不是硬件故障，不是网络问题，而是一个用户发送了 120K tokens 的超长请求，导致 KV Cache 爆炸。

你在群里回复：
> *已处理，添加了 max_prompt_length 限制*

心里想的是：
> *为什么用户不能正常点聊天？*

---

### 上午 10:30：跑 Benchmark

产品经理问："vLLM 和 SGLang 到底用哪个？"

你打开 Jupyter Notebook，开始跑对比测试：

```
测试配置：
- 模型：Llama-3-8B
- GPU：A100 80GB × 4
- 并发：50
- Prompt 长度：512 tokens
- 输出长度：128 tokens

结果：
| 指标 | vLLM | SGLang |
|------|------|--------|
| TTFT (ms) | 156 | 148 |
| TPS | 1240 | 1187 |
| GPU 显存 (GB) | 28.3 | 26.7 |
| P99 延迟 (ms) | 423 | 389 |
```

你写了一份报告，结论是：
> *两个框架各有优势，建议根据具体场景选择。*
>- vLLM：吞吐量略高，社区更活跃
>- SGLang：显存更省，长文本更友好

产品经理满意地走了。你心里清楚测试还有很多变量没控制，但能交差就行。

---

### 下午 2:00：Profiling

发现了一个奇怪的现象：某个模型的推理速度比预期慢了 30%。

你打开 Nsight Systems，开始 trace：

```
发现的问题：
1. 某个 Embedding 层没有正确计算，每次都在做 CPU→GPU 拷贝
2. Attention Kernel 的 occupancy 只有 45%，远低于预期的 80%
3. 前处理阶段（Tokenization）成了瓶颈，CPU 单线程处理
```

修复后，推理速度提升 35%。你在 commit message 里写：
> *优化推理性能，详见内部文档*

其实你知道这只是一个临时方案，真正的优化需要重写 Kernel。但那是下周的事了。

---

### 下午 4:00：会议

本周的 Infra 周会，议程：

1. **故障复盘**（30 分钟）
   - 上周服务抖动的原因定位
   - NCCL 版本升级的讨论

2. **技术方案评审**（40 分钟）
   - 新模型上线的资源评估
   - PD 分离方案的可行性分析

3. **成本优化讨论**（20 分钟）
   - 能不能用 Spot Instance？
   - 推理集群缩容计划

你在会议中贡献了几个专业观点：
> *"这个需要做个 POC"*
> *"我 isolated 一下看看"*
> *"从原理上分析"*

会后你还要写一份会议纪要。

---

### 晚上 7:00：看论文

工作没做完，但你想看看最新的论文。毕竟这个领域变化太快了。

今天的技术动态：
- Flash Attention 3 发布了，H100 专用优化
- SGLang 新版支持了 PD-HiCache
- 有人提出了新的 KV Cache 压缩方法，压缩比 10×，精度损失 < 1%

你打开论文，看了一会儿，然后意识到自己的数学基础不够，默默收藏了这篇博客：
> *《Flash Attention 详解：从原理到实现》*

你看完博客，感觉自己懂了，然后想起还有个 bug 没修，打开 IDE...

---

### 深夜 11:00：值班

服务突然告警，你登录机器排查：

```bash
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM4...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   32C    P0   114W / 400W  |  78234MiB / 81920MiB |     98%      Default |
|                               |                      |                      |
+-------------------------------+----------------------+----------------------+
```

显存快满了。你检查日志，发现是某个用户在疯狂发超长请求。

你只好临时调整配置：
```bash
# 紧急处理
curl -X POST localhost:8000/admin/config \
  -d '{"max_sequence_length": 32768}'
```

然后写了一个 Jira ticket：
> *异常请求自动熔断机制*

其实这个问题上个月就该做了。但一直没排上。

---

### 凌晨 1:00：回家

地铁已经停运，你打车回家。车上刷了刷朋友圈，看到大佬们讨论 CUDA 编程技巧，默默点了个赞。

躺在床上，你问自己：
> *当初为什么入了这一行？*

想了想，大概是因为：
- 想参与 AI 这个时代浪潮
- 享受解决复杂系统问题的成就感
- 看着自己优化的服务支撑百万用户，感觉很有意义

然后再想想明天的任务：
- NCCL 超时的问题还没定位
- 新模型上线的资源申请还没批
- 那个 CUDA Kernel 还没写完

算了，明天再说。晚安。

---

## 1.3 你需要掌握的技能树

AI Infra 是一个跨学科的领域，需要掌握的知识面很广。但好消息是：**你不用全点满**。

### 核心技能矩阵

```
                    ┌─────────────────────────────────────┐
                    │          AI Infra 技能树            │
                    └─────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
    ┌─────────┐               ┌─────────────┐             ┌───────────┐
    │  硬件   │               │   软件/框架  │             │   系统   │
    └─────────┘               └─────────────┘             └───────────┘
         │                           │                           │
    · GPU 架构                  · PyTorch                   · Linux 运维
    · CUDA 编程                 · 推理框架                  · 分布式系统
    · 显存管理                  · 量化方法                  · 容器化（K8s）
    · 网络通信                  · 并行策略                  · 可观测性
```

---

### 必修技能（必须掌握）

| 技能 | 重要程度 | 学习成本 | 说明 |
|------|---------|---------|------|
| Python 编程 | ⭐⭐⭐⭐⭐ | 低 | 吃 Python 饭的，必须精通 |
| Linux 基础 | ⭐⭐⭐⭐⭐ | 中 | 每天和服务器打交道 |
| GPU 架构基础 | ⭐⭐⭐⭐⭐ | 中 | 知道 Tensor Core 和 CUDA Core 的区别 |
| PyTorch 基础 | ⭐⭐⭐⭐ | 中 | 训练框架，理解计算图 |
| 推理框架使用 | ⭐⭐⭐⭐ | 中 | vLLM / SGLang / TRT-LLM 至少会一个 |
| 显存管理 | ⭐⭐⭐⭐⭐ | 中 | 永远的核心问题 |
| 并行策略基础 | ⭐⭐⭐⭐ | 高 | TP/PP/EP 至少理解原理 |

---

### 进阶技能（掌握更好）

| 技能 | 重要程度 | 学习成本 | 说明 |
|------|---------|---------|------|
| CUDA 编程 | ⭐⭐⭐ | 高 | 手写 Kernel 是高薪技能 |
| NCCL 通信 | ⭐⭐⭐ | 高 | 分布式训练必备 |
| 模型量化 | ⭐⭐⭐ | 中 | 成本优化的关键手段 |
| 性能 Profiling | ⭐⭐⭐⭐ | 中 | nsight / pytorch profiler |
| 分布式系统设计 | ⭐⭐⭐ | 高 | 大规模集群必备 |
| Kubernetes | ⭐⭐⭐ | 中 | 云原生部署标准 |

---

### 学习路径建议

根据你的背景，学习路径会有所不同：

**如果你是算法工程师转 Infra：**

```
已有：Python、PyTorch、模型原理
需要补：GPU 架构、CUDA 基础、并行策略、系统运维
建议：先从推理框架入手，理解显存管理，再深入学习并行计算
```

**如果你是后端工程师转 Infra：**

```
已有：Linux、分布式系统、容器化、性能优化思维
需要补：GPU 架构、深度学习框架、模型原理
建议：先从部署和性能优化入手，理解 GPU 计算模式，再深入学习模型细节
```

**如果你是应届生：**

```
建议：打好基础，从开源项目学起
路径：Python → PyTorch → GPU 基础 → 推理框架 → 并行计算 → CUDA
```

---

### 一个现实的学习比例

根据一线工程师的经验，实际工作中知识运用的比例大概是：

```
30% —— 排查问题和 debugging
25% —— 看文档、读源码、查资料
20% —— 写代码、测试、调优
15% —— 会议、沟通、写文档
10% —— 学新东西、看论文
```

所以，**解决问题的能力** 比 **掌握知识多少** 更重要。

---

### 避免去学的陷阱

有些东西看起来很重要，但入门阶段可以暂时跳过：

**可以以后再学的：**
- 复杂的 CUDA 优化技巧（先会用框架就够了）
- 所有并行策略的实现细节（先理解原理）
- 各种量化的数学推导（先用起来再说）
- 每个新框架的 API（先精通一个）

**一定要避免的：**
- 只看论文不实践
- 追逐热点不深耕
- 忽视基础知识
- 不看日志不看报错

---

## 本章小结

1. **AI Infra 是连接模型与用户的桥梁**，核心任务是解决模型训练后的部署、优化、服务化问题。

2. **AI Infra 工程师的日常工作**包括：监控告警、性能优化、系统设计、故障排查，以及对新技术的持续学习。

3. **技能树很大**，但不用全点满。核心是 Python、Linux、GPU 基础、并行策略，然后根据需要逐步深入。

4. **最重要的是**：保持学习的热情，接受问题的复杂性，承认自己的有限性。

下一章，我们将深入硬件基础，看看那些比你这辈子工资还贵的显卡到底长什么样。

---

*放弃指数：⭐ 本章概念性的，最难的还在后面。*

---

# 第二章 硬件基础 —— 显卡比你想象的贵

> *放弃指数：⭐⭐ 本章有大量概念，但都是职业发展基础*

---

## 2.1 GPU 架构入门：CUDA Core、Tensor Core、显存带宽

### 为什么是 GPU 而不是 CPU？

先看一个最直观的对比：

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 数十个 | 数千个 |
| 单核性能 | 高 | 低 |
| 并行能力 | 弱 | 极强 |
| 内存带宽 | ~100 GB/s | ~3000 GB/s |
| 价格 | 几千元 | 几万~几十万元 |
| 擅长任务 | 复杂逻辑、串行计算 | 大规模并行计算 |

用一个生活化的比喻：
> CPU 像一个顶级教授，能解决复杂的数学证明，但一次只能处理一个学生的问题。
>
> GPU 像一千个高中数学老师，虽然每人只能做简单计算，但可以同时处理一千个学生的作业。

**AI 计算恰好需要大量简单的矩阵乘法** —— 这正是 GPU 的主场。

---

### GPU 的核心组件

打开 NVIDIA 的架构白皮书，你会被各种术语淹没。但本质上，GPU 就三个核心组件：

```
┌─────────────────────────────────────────────────────────┐
│                         GPU                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Streaming Multiprocessor (SM)        │   │
│  │  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │ CUDA Cores  │  │ Tensor Cores│  ... (×N)      │   │
│  │  │ (通用计算)   │  │ (矩阵加速)  │                │   │
│  │  └─────────────┘  └─────────────┘                │   │
│  │                                                   │   │
│  │  ┌─────────────────────────────────────────┐     │   │
│  │  │         Shared Memory / L1 Cache         │     │   │
│  │  └─────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              HBM (High Bandwidth Memory)          │   │
│  │                   显存，80GB                       │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

#### 1. CUDA Core：最基础的计算单元

CUDA Core 是 GPU 最基本的计算单元，可以执行浮点运算、整数运算等通用计算。

**关键数字：**
- A100：每个 SM 有 64 个 CUDA Core，共 108 个 SM，约 6912 个 CUDA Core
- H100：每个 SM 有 128 个 CUDA Core，共 132 个 SM，约 16896 个 CUDA Core

**一个 CUDA Core 每个周期可以完成：**
- 1 次 FP32 加法或乘法
- 或 2 次 FP16 运算

> *面试题：为什么 GPU 的理论峰值 FLOPS = CUDA Cores × 频率 × 每周期运算数？*

---

#### 2. Tensor Core：AI 计算的秘密武器

Tensor Core 是 NVIDIA 为深度学习专门设计的计算单元，能在一个周期内完成 **矩阵乘法 + 累加** 运算。

**一个 Tensor Core 一个周期可以完成：**

| 精度 | A100 | H100 |
|------|------|------|
| FP16/BF16 | 256 FMA | 256 FMA |
| FP8 | - | 512 FMA |
| INT8 | 256 FMA | 512 FMA |

**这意味什么？**

以 BF16 为例，一个 Tensor Core 一次可以计算一个 4×4 矩阵乘法：

```
A (4×4 BF16) × B (4×4 BF16) + C (4×4 FP32) = D (4×4 FP32)
```

如果用 CUDA Core 逐元素计算，需要 64 次乘法 + 48 次加法 = 112 次运算。
而 Tensor Core **一条指令搞定**。

这就是为什么 Tensor Core 是 AI 加速的核心。

---

#### 3. 显存带宽：容易被忽略的瓶颈

很多人只关注 FLOPS（计算能力），但实际训练/推理中，**显存带宽往往是更大的瓶颈**。

**为什么？**

假设你在做矩阵乘法 `C = A × B`：
- 计算量：2mnk 次 FLOPs（乘法+加法各一次）
- 数据读取：矩阵 A + 矩阵 B 的数据量
- 数据写入：矩阵 C 的数据量

如果计算速度（FLOPS）很快，但数据从显存读取很慢，GPU 就会"等数据"。

**算术强度（Arithmetic Intensity）**：

```
算术强度 = 计算量 / 数据传输量
```

- 算术强度高 → 计算密集型，充分利用 GPU
- 算术强度低 → 内存密集型，带宽成瓶颈

**大模型推理的特点**：Decode 阶段是典型的内存密集型。

每次生成一个 token，需要：
1. 读取所有模型权重
2. 读取当前 token 的 KV Cache
3. 做很少的计算（只有 1 个 token）

这就是为什么 Decode 阶段的速度取决于：**显存带宽 / 模型大小**

---

### GPU 规格：你应该关心的数字

看到一张 GPU，你应该立刻关注这些参数：

| 参数 | 含义 | 对 AI 的影响 |
|------|------|-------------|
| FLOPS | 计算能力 | 训练速度、Prefill 速度 |
| 显存容量 | 能放多大模型 | 单卡最大模型尺寸 |
| 显存带宽 | 数据传输速度 | Decode 速度、长文本性能 |
| TDP | 功耗 | 电费、散热要求 |
| 互联带宽 | 卡间通信速度 | 多卡并行效率 |

---

### 主流 GPU 对比

| GPU | 显存 | 带宽 | FP16 TFLOPS | 价格（约） |
|-----|------|------|-------------|-----------|
| RTX 4090 | 24GB | 1TB/s | 330 | ¥1.5万 |
| A100 40GB | 40GB | 1.6TB/s | 312 | ¥8万 |
| A100 80GB | 80GB | 2TB/s | 312 | ¥12万 |
| H100 80GB | 80GB | 3.35TB/s | 989 | ¥25万 |

**一个残酷的事实：**
> H100 的价格 ≈ 一辆中档汽车
>
> 一个 8 卡 H100 服务器的价格 ≈ 一辆豪车

---

## 2.2 为什么 H100 比 A100 贵十万？

### 从 A100 到 H100：架构升级

A100 基于 **Ampere 架构**（2020年发布）
H100 基于 **Hopper 架构**（2022年发布）

NVIDIA 官方的说法：H100 比 A100 快 **6 倍**（训练）、**9 倍**（推理）。

**真实的性能提升来自哪里？**

| 特性 | A100 | H100 | 提升幅度 |
|------|------|------|---------|
| FP16/BF16 Tensor Core | 第 3 代 | 第 4 代 | 2× |
| FP8 支持 | ❌ | ✅ | - |
| 显存带宽 | 2 TB/s | 3.35 TB/s | 1.7× |
| NVLink 速度 | 600 GB/s | 900 GB/s | 1.5× |
| 稀疏计算 | ❌ | ✅ | 最高 2× |

---

### FP8：新的游戏规则

H100 最大的升级是 **FP8 支持**。

**FP8 是什么？**

传统的 FP16 用 16 位表示一个浮点数：
```
[符号位 1位][指数位 5位][尾数位 10位]
```

FP8 用 8 位表示：
```
E4M3 格式：[符号位 1位][指数位 4位][尾数位 3位]
E5M2 格式：[符号位 1位][指数位 5位][尾数位 2位]
```

**两种格式的区别：**
- **E4M3**：精度更高，适合权重
- **E5M2**：范围更大，适合梯度/激活

**好处：**
1. 显存占用减半（FP16 → FP8）
2. 带宽压力减半
3. Tensor Core 吞吐翻倍

**代价：**
- 动态范围和精度降低
- 需要仔细调参避免溢出/下溢

---

### 实测：H100 vs A100

以 GPT-3 175B 训练为例（NVIDIA 官方数据）：

| 指标 | A100 | H100 | 提升 |
|------|------|------|------|
| 训练 TFLOPS | 140 | 850 | 6× |
| 推理 Latency | 29ms | 7ms | 4× |
| 推理 Throughput | 1× | 30× | 30× |

**为什么推理提升这么大？**

因为推理是内存密集型，H100 的 3.35 TB/s 带宽加上 FP8 支持让 Decode 速度大幅提升。

---

### 所以，H100 贵在哪里？

1. **技术溢价**：FP8、Transformer Engine、HBM3
2. **供需关系**：AI 爆发，一卡难求
3. **CUDA 生态护城河**：你也没别的选择
4. **利润率**：NVIDIA 毛利率 70%+

> 一个冷知识：H100 的制造成本估计只有几千美元。

---

## 2.3 国产芯片生态：机遇与坑

### 为什么需要国产芯片？

1. **供应链安全**：美国出口管制
2. **成本考量**：H100 太贵且难买
3. **政策支持**：国产替代是大趋势

### 主流国产 GPU 一览

| 厂商 | 代表产品 | 算力 | 显存 | 软件生态 |
|------|---------|------|------|---------|
| 华为昇腾 | Ascend 910B | 310 TFLOPS (FP16) | 64GB HBM | CANN（类 CUDA） |
| 壁仞 | BR100 | 230 TFLOPS (FP16) | 64GB HBM2e | 对标 CUDA |
| 寒武纪 | MLU370 | 192 TFLOPS (FP16) | 48GB | Neuware |
| 海光 | DCU Z100 | 120 TFLOPS (FP16) | 32GB HBM2 | 类 ROCm |
| 摩尔线程 | MTT S4000 | 48 TFLOPS (FP32) | 48GB | MUSA（类 CUDA） |
| 燧原 | 云燧T21 | 160 TFLOPS (FP16) | 32GB | 自研栈 |

---

### 国产芯片的坑

**坑 1：软件生态不成熟**

```python
# 你想跑的代码
import torch
model = AutoModel.from_pretrained("Qwen/Qwen-72B")

# 国产芯片需要
import torch_npu  # 昇腾
# 或 import torch_mlu  # 寒武纪
# 或 import torch_brcode  # 壁仞

# 然后...
model = model.npu()  # 改代码
# 某些算子不支持
# 某些精度有问题
# 报错信息看不懂
```

**坑 2：文档和社区薄弱**

```
NVIDIA: 遇到问题 → Google → Stack Overflow → 解决
国产芯片: 遇到问题 → Google → 没结果 → 翻文档 → 文档不全 → 找厂家 FAE → 等一周
```

**坑 3：算子和框架适配慢**

大模型技术迭代很快，新型注意力机制、新的并行策略层出不穷。

NVIDIA 的 CUDA 生态有社区贡献，国产芯片往往是：
> "你们什么时候支持 Flash Attention 3？"
> "已经排期了，预计下个版本..."

**坑 4：集群互联弱**

NVIDIA 的 NVLink + NVSwitch 提供了 900 GB/s 的卡间互联。

国产芯片的互联方案：
- 壁仞：400 GB/s
- 昇腾：392 GB/s
- 其他：PCIe 4.0，64 GB/s

多卡训练效率差距明显。

---

### 一线工程师的建议

**可以用国产芯片的场景：**
- 单卡推理（不用多卡互联）
- 模型已经适配过的（如 GLM 系列）
- 对精度要求不苛刻
- 有厂家技术支持

**暂时别用国产芯片的场景：**
- 新模型首发训练
- 大规模分布式训练（>64 卡）
- 追求极致性能优化
- 时间紧迫的生产任务

**一个实操建议：**
> 先用 NVIDIA 完成 POC 和调优，确认可行后再移植到国产芯片。

---

## 2.4 一个 GPU 机柜的年电费比我年薪还高

### 算一笔经济账

假设你有一个标准的 8 卡 H100 服务器：

**硬件成本：**
- 8×H100 80GB：¥200万
- 服务器主机：¥50万
- 网络设备：¥10万
- **总计：约 ¥260万**

**运营成本：**
- 单卡 TDP：700W
- 8 卡功耗：5.6kW
- 服务器整机（含 CPU、风扇等）：≈10kW
- PUE（含空调制冷）：1.5×
- 实际功耗：**15kW**

**年电费：**
```
15kW × 24h × 365天 × ¥1/度 = ¥131,400/年
```

**三年总成本：**
```
硬件 260万 + 电费 39万 + 机房租金 + 运维 = 约 350万
```

---

### 电费优化策略

**策略 1：削峰填谷**

很多地区有峰谷电价，夜间电价便宜 30%-50%。

如果业务允许，可以：
- 白天：在线服务
- 夜间：离线训练/批处理

**策略 2：液冷节能**

传统风冷 PUE 约 1.5，液冷可以降到 1.1。

一个 100 机柜的数据中心，液冷每年可以节省 **数百万电费**。

**策略 3：Spot/竞价实例**

云厂商的 Spot 实例价格可以是按需实例的 20%-50%。

但要注意：
- 可能随时被回收
- 适合可以断点续训的任务

---

### Infra 工程师应有的成本意识

当老板问你："为什么不多加几张卡？"

你应该能回答：

| 场景 | 回答模板 |
|------|---------|
| 峰值流量大 | "高峰期加 4 张卡可以缩短 P99 延迟 40%，需要 ¥40万/年" |
| 模型要扩容 | "72B 模型需要 4 张 80GB 显存，单卡放不下，必须扩" |
| 用户增长快 | "按当前增长率，Q3 需要再上一个机柜" |

> 一个优秀的 Infra 工程师，不仅是技术专家，也是半个财务。

---

## 本章小结

1. **GPU 的三大核心组件**：CUDA Core（通用计算）、Tensor Core（矩阵加速）、HBM（显存）。理解三者关系是优化 AI 工作负载的基础。

2. **H100 的溢价来自**：FP8 支持、更高带宽、Transformer Engine 等技术升级，以及 NVIDIA 的垄断定价。

3. **国产芯片现状**：硬件参数追赶中，软件生态是短板，适合有适配能力的团队选择性采用。

4. **成本意识**：电费、硬件折旧、运维成本是 AI Infra 的隐形挑战，需要纳入技术决策考量。

下一章，我们将进入训练框架的世界，看看 PyTorch 怎么把多卡跑起来——以及为什么你的多卡训练比单卡还慢。

---

*放弃指数：⭐⭐ 本章概念较多，但后面章节会更深入。准备好迎接代码了吗？*

---

# 第三章 训练框架入门

> *放弃指数：⭐⭐⭐ 本章开始有代码了，光看不练是学不会的*

---

## 3.1 PyTorch 分布式训练：DDP、FSDP、DeepSpeed

### 从单卡到多卡：为什么这么难？

你可能会想：
> "不就是多几张卡一起跑吗？有什么难的？"

先看一个简单的问题：

假设你有一个 7B 参数的模型（FP32 存储），单卡 A100 80GB 能放下吗？

```
7B × 4 bytes (FP32) = 28GB ✓ 放得下
```

如果加入优化器状态呢？Adam 优化器需要存储：
- 模型权重：28GB
- 梯度：28GB
- 一阶动量：28GB
- 二阶动量：28GB
- **总计：112GB** ❌ 放不下

这就是为什么需要分布式训练。但引入多卡后，问题才刚刚开始。

---

### 分布式训练的三种策略

```
┌─────────────────────────────────────────────────────────────┐
│                    分布式训练策略                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ 数据并行 DP │   │ 模型并行 MP │   │ 流水线并行 PP│       │
│  │             │   │             │   │             │       │
│  │ 每卡完整模型│   │ 模型切分    │   │ 层切分      │       │
│  │ 不同数据    │   │ 相同数据    │   │ 数据流水    │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
│                                                             │
│  复杂度：低 ────────────────────────────────────→ 高        │
│  通信量：高 ────────────────────────────────────→ 低        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

本章重点讲**数据并行**，这是最常用的方式。

---

### DDP：DistributedDataParallel

**最基础的数据并行方案**，PyTorch 原生支持。

**原理：**

```
                    ┌─────────────────────────────────────┐
                    │             梯度同步                 │
                    │        AllReduce (Ring)             │
                    └─────────────────────────────────────┘
                              ▲    ▲    ▲    ▲
                              │    │    │    │
              ┌───────────────┴────┴────┴────┴───────────────┐
              │                                              │
       ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐     │
       │   GPU 0     │  │   GPU 1     │  │   GPU N     │     │
       │             │  │             │  │             │     │
       │ 完整模型副本│  │ 完整模型副本│  │ 完整模型副本│     │
       │ Batch 1/4   │  │ Batch 2/4   │  │ Batch 4/4   │     │
       └─────────────┘  └─────────────┘  └─────────────┘     │
                                                            │
                 每张卡：完整模型 + 不同数据切片                   │
                                                            │
       └──────────────────────────────────────────────────────┘
```

**代码示例：**

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def train(rank, world_size):
    # 1. 初始化进程组
    dist.init_process_group(
        backend='nccl',  # GPU 用 NCCL
        init_method='tcp://localhost:29500',
        world_size=world_size,
        rank=rank
    )
    
    # 2. 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 3. 创建模型并移到 GPU
    model = MyModel().to(device)
    
    # 4. 用 DDP 包装模型
    model = DDP(model, device_ids=[rank])
    
    # 5. 数据加载器（每个进程不同的数据）
    train_loader = get_data_loader(rank, world_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 6. 训练循环
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            loss = compute_loss(output, batch.label)
            loss.backward()  # DDP 自动同步梯度
            optimizer.step()
    
    # 7. 清理
    dist.destroy_process_group()

# 启动训练
world_size = torch.cuda.device_count()
mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

**DDP 的局限性：**

1. **每张卡都要放完整模型** → 模型大于一张卡显存就跑不了
2. **优化器状态重复** → 显存利用率低
3. **梯度同步开销** → 卡越多，同步时间占比越高

---

### FSDP：Fully Sharded Data Parallel

**DDP 进化版**，把模型、梯度、优化器状态都切分开。

**核心思想（来自 DeepSpeed ZeRO）：**

| 策略 | 切分内容 | 显存节省 |
|------|---------|---------|
| ZeRO-1 | 优化器状态 | 4× |
| ZeRO-2 | + 梯度 | 8× |
| ZeRO-3 | + 模型权重 | N×（N = GPU 数量） |

**原理图：**

```
传统 DDP (每卡完整):
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     GPU 0       │  │     GPU 1       │  │     GPU 2       │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ 完整模型    │ │  │ │ 完整模型    │ │  │ │ 完整模型    │ │
│ │ 完整梯度    │ │  │ │ 完整梯度    │ │  │ │ 完整梯度    │ │
│ │ 完整优化器  │ │  │ │ 完整优化器  │ │  │ │ 完整优化器  │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘  └─────────────────┘

FSDP ZeRO-3 (每卡仅 1/N):
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     GPU 0       │  │     GPU 1       │  │     GPU 2       │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ 模型切片 1  │ │  │ │ 模型切片 2  │ │  │ │ 模型切片 3  │ │
│ │ 梯度切片 1  │ │  │ │ 梯度切片 2  │ │  │ │ 梯度切片 3  │ │
│ │ 优化器切片1 │ │  │ │ 优化器切片2 │ │  │ │ 优化器切片3 │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘  └─────────────────┘
     前向计算时动态 all-gather 需要的权重
```

**代码示例：**

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

def train(rank, world_size):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)
    
    model = MyLargeModel()
    
    # FSDP 配置
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        device_id=torch.cuda.current_device(),
        # 混合精度
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
    )
    
    # ... 训练代码同 DDP

# 更简单的启动方式
torchrun --nproc_per_node=8 train.py
```

**FSDP 的优势：**
- 可以训练超出单卡显存的模型
- 显存效率高
- PyTorch 原生支持，无需额外依赖

**FSDP 的代价：**
- 通信量增加
- 需要精心配置 sharding 策略
- 调试更困难

---

### DeepSpeed：微软的分布式训练神器

**DeepSpeed** 是微软开源的训练优化库，提供了更多高级功能。

**核心特性：**

| 特性 | 说明 |
|------|------|
| ZeRO | 梯度/优化器/权重分片 |
| Offload | 把优化器/梯度卸载到 CPU |
| Mixed Precision | FP16/BF16/FP8 混合精度 |
| Gradient Checkpointing | 重计算节省显存 |
| Pipeline Parallelism | 流水线并行 |
| MoE 支持 | 专家模型训练 |

**配置文件：**

```json
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95]
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
        }
    },
    "gradient_checkpointing": {
        "enabled": true
    }
}
```

**代码示例：**

```python
import deepspeed
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("model_path")

# DeepSpeed 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

# 训练
for batch in train_loader:
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()
```

**DeepSpeed vs FSDP 选择：**

| 场景 | 推荐 |
|------|------|
| 快速上手、PyTorch 原生 | FSDP |
| 需要更多显存优化 | DeepSpeed + Offload |
| Production 环境 | DeepSpeed（更稳定） |
| 大模型开源项目 | 两者都支持（如 Megatron-LM） |

---

## 3.2 为什么我的多卡训练比单卡还慢？

### 一个真实案例

小明兴冲冲地向老板报告：
> "我们申请到了 8 张 A100，训练速度一定能提升 8 倍！"

一周后，小明发现：
> "8 卡训练速度居然比单卡还慢 20%..."

问题出在哪里？

---

### 分布式训练的性能杀手

**杀手 1：负载不均衡**

```python
# 错误示范：数据量不是卡数的整数倍
world_size = 8
total_samples = 1001  # 不能被 8 整除

# GPU 0-6: 125 samples each
# GPU 7: 126 samples → 慢一点
# 其他 GPU 等待 GPU 7 → 整体被拖慢
```

**解决方案：**

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True  # 丢弃不整除的部分
)
```

---

**杀手 2：NCCL 通信瓶颈**

```
单卡：计算时间 = 100ms
8卡：计算时间 = 12.5ms + 通信时间 = ?
```

如果通信时间是 30ms，那么：
```
8 卡效率 = 12.5 / (12.5 + 30) = 29%
```

**诊断方法：**

```python
# 在训练代码中添加
import torch.distributed as dist

# 开启 NCCL 调试
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

# 看 NCCL 日志，找到通信时间
```

**优化手段：**

```bash
# 1. 使用 NVLink（如果可用）
nvidia-smi nvlink --status

# 2. 选择正确的网络拓扑
os.environ['NCCL_IB_DISABLE'] = '0'  # 启用 InfiniBand
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 指定网卡

# 3. 梯度压缩（牺牲精度换速度）
# 在 DeepSpeed 中启用梯度压缩
```

---

**杀手 3：Gradient Accumulation 设置不当**

```python
# 目标：等效 batch size = 128
# 显卡数量 = 8

# 错误设置
batch_size_per_gpu = 128  # 每卡 128
gradient_accumulation_steps = 1  # 不累积
# 实际 batch size = 128 × 8 = 1024 ❌

# 正确设置
batch_size_per_gpu = 16  # 每卡 16
gradient_accumulation_steps = 1  
# 实际 batch size = 16 × 8 = 128 ✓

# 或更大 batch，更少步数
batch_size_per_gpu = 4
gradient_accumulation_steps = 4
# 实际 batch size = 4 × 4 × 8 = 128 ✓
```

---

**杀手 4：数据加载瓶颈**

```python
# GPU 利用率忽高忽低
# 可能是数据加载太慢

# 诊断
nvidia-smi dmon -s u -d 1
# GPU-Util 闪烁：数据加载瓶颈
# GPU-Util 平稳偏低：计算/通信瓶颈

# 解决
train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=8,        # 多进程加载
    pin_memory=True,      # 锁页内存
    prefetch_factor=2,    # 预取
    persistent_workers=True
)
```

---

### 多卡加速效率公式

```
实际加速比 = 单卡时间 / 多卡时间

理想加速比 = 卡数量
实际加速比 = 卡数量 × 并行效率

并行效率 = 计算时间 / (计算时间 + 通信时间 + 等待时间)
```

**经验值：**
- 2 卡：并行效率 90%+
- 4 卡：并行效率 80%+
- 8 卡：并行效率 70%+
- 64 卡：并行效率 50%左右

如果 8 卡效率低于 60%，一定有问题需要排查。

---

## 3.3 显存优化三板斧

当显存不够时，你有三把斧头可以用。

---

### 斧头 1：Gradient Checkpointing（梯度检查点）

**原理：**

```
正常前向传播：保存所有激活值 → 占用大量显存
Gradient Checkpointing：只保存部分激活值，需要时重新计算

┌────────────────────────────────────────────────────────┐
│                  正常模式                                │
│  Layer1 → Layer2 → Layer3 → Layer4 → Layer5 → Layer6   │
│   ↓        ↓        ↓        ↓        ↓        ↓      │
│  保存     保存     保存     保存     保存     保存     │
│                                                        │
│                  Checkpointing 模式                      │
│  Layer1 → Layer2 → Layer3 → Layer4 → Layer5 → Layer6   │
│   ↓              ↓               ↓              ↓     │
│  保存          保存             保存           保存    │
│                               ↑                        │
│                     反向传播时重新计算                    │
└────────────────────────────────────────────────────────┘
```

**代码：**

```python
# PyTorch 原生方式
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([
            TransformerBlock() for _ in range(24)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            # 正常模式
            # x = layer(x)
            
            # Checkpointing 模式
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# HuggingFace 方式
model.gradient_checkpointing_enable()
```

**效果：**
- 显存减少 30%-70%
- 训练速度降低 20%-40%（因为要重计算）

**适用场景：**
- 模型很大，显存放不下
- 愿意用时间换空间

---

### 斧头 2：Mixed Precision（混合精度训练）

**原理：**

```
FP32: 32位浮点数，精度最高，显存占用最大
FP16: 16位浮点数，显存减半，但精度降低
BF16: 16位浮点数，范围更大，适合大模型

混合精度 = 前向传播用 FP16 + 优化器用 FP32 主副本
```

**代码：**

```python
# PyTorch 原生
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.float16):  # 或 torch.bfloat16
        output = model(batch)
        loss = compute_loss(output)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# HuggingFace 方式
from transformers import TrainingArguments

args = TrainingArguments(
    fp16=True,  # 或 bf16=True（如果 GPU 支持）
    ...
)
```

**显存节省：**
- 激活值减半
- 模型权重减半（可选）
- 通常节省 30%-50%

**注意事项：**
- FP16 可能溢出，需要 Loss Scaling
- BF16 不需要 Loss Scaling，推荐优先使用

---

### 斧头 3：Offloading（卸载到 CPU）

**原理：**

```python
当 GPU 显存不够时，把暂时不用的数据放到 CPU 内存
```

**DeepSpeed 配置：**

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**效果：**
- 可以训练超出 GPU 显存的模型
- 速度降低明显（CPU-GPU 数据传输）

**适用场景：**
- 模型太大，GPU 怎么优化都放不下
- 可以接受训练速度变慢

---

### 三板斧组合效果

| 组合方式 | 显存节省 | 速度影响 | 适用场景 |
|---------|---------|---------|---------|
| 仅 Mixed Precision | 30%-50% | 几乎不变 | 首选方案 |
| MP + Gradient Checkpointing | 50%-80% | 降 20%-40% | 大模型标准配置 |
| MP + GC + Offloading | 能放下超大模型 | 降 50%+ | 极端情况 |

---

## 3.4 从 0 开始训练一个大模型（然后放弃）

### 创建一个"能跑"的训练脚本

**Step 1：准备配置**

```python
# config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 模型配置
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 32000
    max_seq_len: int = 4096
    
    # 训练配置
    batch_size: int = 128
    learning_rate: float = 1e-4
    num_epochs: int = 1
    
    # 分布式配置
    world_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # 优化配置
    use_fsdp: bool = True
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
```

**Step 2：定义模型**

```python
# model.py
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

def create_model(config: TrainConfig):
    llama_config = LlamaConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_seq_len,
    )
    
    model = LlamaForCausalLM(llama_config)
    
    # 启用 gradient checkpointing
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e9  # 单位：B

# 示例：7B 模型
# 参数量 ≈ 7 × 10^9
```

**Step 3：分布式初始化**

```python
# distributed.py
import os
import torch
import torch.distributed as dist

def setup_distributed():
    # 从环境变量获取信息
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    
    # 设置设备
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    dist.destroy_process_group()
```

**Step 4：训练主循环**

```python
# train.py
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

def train(config: TrainConfig):
    # 初始化
    rank, world_size, local_rank = setup_distributed()
    
    # 创建模型
    model = create_model(config).cuda()
    
    # FSDP 包装
    if config.use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )
    
    # 数据加载
    train_dataset = create_dataset()  # 你的数据集
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size // world_size // config.gradient_accumulation_steps,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 混合精度
    scaler = GradScaler() if config.use_amp else None
    
    # 训练循环
    model.train()
    for epoch in range(config.num_epochs):
        sampler.set_epoch(epoch)  # 确保 shuffle 正确
        
        for step, batch in enumerate(train_loader):
            batch = batch.cuda()
            
            with autocast(enabled=config.use_amp):
                outputs = model(batch)
                loss = outputs.loss
            
            # 梯度累积
            loss = loss / config.gradient_accumulation_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                if rank == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    # 保存模型（只在 rank 0 做）
    if rank == 0:
        torch.save(model.state_dict(), 'model_final.pt')
    
    cleanup_distributed()

if __name__ == '__main__':
    config = TrainConfig()
    train(config)
```

**Step 5：启动脚本**

```bash
#!/bin/bash
# run.sh

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    train.py

# 或者多机多卡
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=4 \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29500 \
#     train.py
```

---

### 然后你会遇到这些问题

**问题 1：OOM**

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

解决方案：
1. 减小 batch_size
2. 启用 gradient_checkpointing
3. 使用 FSDP + ZeRO-3
4. 启用 offloading

---

**问题 2：NCCL 超时**

```
RuntimeError: NCCL error in: /path/to/file
```

解决方案：
```bash
# 增加超时时间
export NCCL_TIMEOUT=3600  # 1 hour

# 检查网络
export NCCL_DEBUG=INFO
```

---

**问题 3：Loss 不收敛**

```
Step 100: Loss = 10.5
Step 200: Loss = 10.5
Step 300: Loss = 10.5
...
```

可能原因：
1. 学习率太大或太小
2. 数据有问题
3. 模型初始化有问题
4. 混合精度溢出

---

**问题 4：训练一天后 NaN**

```
Step 10000: Loss = 2.3
Step 10001: Loss = nan
```

排查步骤：
```python
# 1. 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN in gradient: {name}")

# 2. 检查 Loss Scaling
if scaler:
    print(f"Loss scale: {scaler.get_scale()}")

# 3. 检查权重
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in weight: {name}")
```

---

### 为什么说"然后放弃"？

因为从头训练一个大模型，你还需要：

1. **高质量数据**：几 TB 到几十 TB
2. **充足的算力**：几百到几千张 GPU
3. **大量时间**：几周到几个月
4. **调参经验**：学习率、批大小、正则化

一个真实的 7B 模型训练成本：

| 项目 | 数值 |
|------|------|
| 训练数据 | 2T tokens |
| GPU 时间 | 2000 A100 小时 |
| 电费 | ¥十几万 |
| 人力 | 3-6 个月 |

**所以大多数公司的选择**：
- 微调现有模型
- 用开源模型服务
- 把训练外包给云厂商

---

## 本章小结

1. **分布式训练三大框架**：DDP 最简单但模型不能超显存，FSDP 原生支持显存分片，DeepSpeed 功能最全面。

2. **多卡比单卡慢的原因**：负载不均衡、通信瓶颈、数据加载瓶颈，需要用 Profiling 工具定位。

3. **显存优化三板斧**：Gradient Checkpointing（时间换空间）、Mixed Precision（精度换空间）、Offloading（CPU 换显存）。

4. **从头训练大模型很难**：需要数据、算力、时间、经验，大多数场景下微调或使用开源模型更实际。

下一章，我们将进入推理优化的世界，看看如何让你的模型跑得更快、更省显存。

---

*放弃指数：⭐⭐⭐ 光看不练是学不会的，光练不动脑也是学不会的。但本章内容是后续章节的基础，值得投入时间。*

---

*（未完待续...）*