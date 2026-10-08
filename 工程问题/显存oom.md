# OOM 的正确归因顺序（工程方法论）

面试里"线上 OOM 你怎么排查"，标准答案顺序是：

1. 区分 `torch.cuda.OutOfMemoryError` 报的 `allocated` vs `reserved`：若 reserved 远大于 allocated → **碎片问题**（开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，或换 paging 内存池）；
2. 区分是 **prefill 时炸** 还是 **decode 跑一会儿炸**：prefill 炸 → 激活峰值/长 prompt，上 chunked prefill；decode 炸 → KV 池不够，上驱逐/卸载/量化；
3. 看是不是**并发峰值**而非稳态：请求调度没有 admission control（没有等待队列水位控制）；
4. 隐性占用：CUDA context（几百 MB）、NCCL buffer（`NCCL_BUFFSIZE`，多卡时每 rank 几百 MB）、CUDA Graph 捕获时的静态 pool、多模态 encoder 临时显存、logits 缓冲。

下面按 **碎片 → KV 池不足 → 权重过大 → 激活峰值 → 隐性占用** 逐个展开。

# 碎片问题

在显存排查中我们通常通过 nvidia-smi info / npu-smi info 去查看实时的显存占用：

![img](https://i-blog.csdnimg.cn/blog_migrate/029bb2587e852c1db1f7d4794adea954.jpeg)

注意：**nvidia-smi 看到的是进程持有的 reserved 显存，不是模型真实用量**。要看 PyTorch allocator 内部账本，用如下代码：

```python
import torch

# PyTorch 为张量实际分配的 GPU 内存量
allocated = torch.cuda.memory_allocated() / 1024**3  # GB
# PyTorch 从 CUDA 预留的内存量（含未使用部分）
reserved = torch.cuda.memory_reserved() / 1024**3  # GB
# 上次重置以来的峰值分配量
peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
# 重置峰值计数器
torch.cuda.reset_peak_memory_stats()

# 排查碎片用这两件套
torch.cuda.memory_summary()   # 文本账本：segment/block 分布、碎片统计
torch.cuda.memory_snapshot()  # 结构化快照，导入 PyTorch memory_viz 可视化
```

**碎片怎么产生的**：caching allocator 以 segment 为单位向 driver 申请显存，再切成小 block 分给张量。变长张量（动态 shape、变长 batch、不 padding 的 embedding）反复 alloc/free 之后，segment 内留下大量不连续的空洞——总空闲量够，但没有一块连续空间满足新分配，于是 `reserved >> allocated` 且分配失败，这就是"假 OOM"。

**治理手段**：

| 手段 | 说明 |
| :--- | :--- |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | segment 可向物理相邻处扩展，大幅减少假 OOM，PyTorch 2.x 推荐，代价极小 |
| `garbage_collection_threshold:0.6~0.8` | reserved 占总显存超过阈值时主动回收空闲 block，把回收压力前置 |
| 定长化 | 变长输入 padding 到固定 block 尺寸，从源头减少变长分配 |
| 框架自管池 | vLLM/SGLang 启动时一次性分配整块 KV 池，运行期 KV 零碎片——**PagedAttention 本质就是把"虚拟内存分页"引入显存管理，解决 KV 的外部碎片**（vLLM 论文实测静态显存管理浪费 60%~80%）；碎片主要残留在权重加载阶段与激活/临时缓冲 |

# kv cache 不足

## 现象与定位

- 启动时看池子大小：vLLM 日志 `# GPU blocks: xxx`（block 默认 16 token）与 `Maximum concurrency for xx tokens per sequence: xx`，不达标就调 `--gpu-memory-utilization` 或者从权重侧省显存；
- 运行中看水位：`gpu_cache_usage` 长期贴近 1、prefix cache hit rate 低、请求被 preempt（`preemption` 计数上涨；swap/recompute 风暴表现为 TTFT/ITL 毛刺、`#running` 锯齿状归零）；
- **decode 跑一段时间才炸 → 基本锁定 KV 池不足**（对应归因顺序第 2 条的 decode 分支）。

## 治理手段（按 ROI 排序）

**1. 架构级：GQA / MLA（选型时决定了 80% 的 KV 命运）**

KV 头数 32 (MHA) → 8 (GQA) → latent 576 (MLA)，单 token KV 依次降 4 倍再 4 倍；MLA 还能矩阵吸收（W_UK 吸进 W_Q、W_UV 吸进 W_O），decode 时连解压开销都省掉。存量模型没得选，但新业务选模型时这是显存的第一变量。

**2. KV 量化（改动最小、见效最快）**

vLLM/SGLang 均支持 `--kv-cache-dtype fp8`，池子直接 ×2，精度损失通常可忽略；INT4 KV 可再 ×4，但长上下文下误差累积明显，需评测把关。注意：KV 量化省的是池子容量，带宽收益要靠对应 FP8 kernel。

**3. Prefix Caching / 多级缓存**

radix tree（SGLang）/ block hash（vLLM）复用公共前缀，多轮对话、RAG、few-shot 场景命中率极高；再往上把冷 prefix 卸到 CPU/NVMe（vLLM CPU offload connector / LMCache），GPU 只留热数据——用存储换显存，KV 生命周期管理和 LRU 类似。

**4. 驱逐 + 准入控制（稳态不炸的关键）**

- 驱逐：LRU 逐出冷 prefix block，watermark 控制回收水位；
- 准入：`--max-num-seqs`（并发上限）、`--max-num-batched-tokens`（单步 prefill 预算）、`--max-model-len`（拒绝超长请求），本质是给 KV 池上水位线，**宁可排队，不许溢出**；
- 实在不够时的请求级 preempt：swap（KV 换出到 CPU）优先于 recompute（重算烧算力）。

**5. 分布式切分**

- TP 下每卡只存本 rank 的 KV head，池子 ×TP（代价是通信）；
- PD 分离：prefill/decode 各自独立规划显存配比，decode 机器可以堆成纯 KV 仓；
- 1M 上下文这种单请求 KV 超过单机显存的，只能走多机 KV 池（如 Mooncake 的 KVCache-centric 分层架构）。

# 权重占用太多显存

权重常驻且刚性，治理就三板斧：**压（量化）、切（并行）、挪（offload）**。

**1. 权重量化（生产主线）**

| 方案 | 权重显存 | 说明 |
| :--- | :--- | :--- |
| BF16 | 2 B/param | 基准 |
| W8A8（FP8 / INT8+SmoothQuant） | 1 B/param | Hopper+ 原生 FP8，精度几乎无损，服务端首选 |
| W4A16（GPTQ / AWQ，per-group） | 0.5 B/param | weight-only，算子仍跑 FP16，单卡/消费级神器，精度需过 ppl + 业务评测 |
| GGUF（llama.cpp） | 0.5~8 B/param 可选 | 生态最好，CPU/GPU 混跑 |

坑：先 load BF16 再在线量化的路径，加载峰值 = 权重 ×1.5~2，小卡直接炸——要直接加载预量化 checkpoint。

**2. 并行切分**

- TP：权重、KV、logits 全部 ÷N，通信量最大，适合卡内/机内 NVLink；
- PP：权重 ÷N 且通信小，但有流水气泡，适合超大模型跨机；
- EP（MoE 专属）：expert 均摊到各卡。DeepSeek-V3 / K2 这种几百个 expert 的模型，EP 是显存 + 负载均衡的双优解。

**3. Offload（单卡跑大模型的现实解）**

- llama.cpp mmap：权重留在磁盘/内存按层 page-in，依赖 OS 调度，本质是拿带宽换显存；
- MoE 热点路由：expert 访问高度不均（PowerInfer / KTransformers 的思路），热 expert 常驻 GPU、冷 expert 留 CPU 并按路由预测 prefetch，DeepSeek 单机可跑；
- 预期管理：offload 吞吐必掉，只适合"能跑 > 好跑"的场景。

**4. 诚实手段：换小模型 / 蒸馏**

多数业务 70B → 32B/14B 蒸馏后指标不掉，显存直接减半再减半——工程上最被低估的显存优化。

# 激活与瞬时峰值（prefill 炸）

对应归因顺序第 2 条的 prefill 分支：OOM 栈落在 attention/MLP 的 forward 上，长 prompt 一来就炸。

- **chunked prefill**：把长 prompt 切成 `max_num_batched_tokens` 大小的块分步计算，激活峰值从 O(prompt_len) 降到 O(chunk)。vLLM/SGLang 均已默认开启，压峰值就是调小该参数（代价是 prefill 变慢、TTFT 上升）；
- **logits 峰值**：大词表（128K+ vocab）× 大 batch 的 FP32 logits 可达 GB 级，用 fused linear + cross-entropy 算子避免物化完整 logits；
- chunked prefill 与 decode 混跑时，激活峰值还要叠上 decode batch 的 logits，注意两者配比。

# 隐性占用

- **CUDA context**：每进程 300~800 MB，多进程 per-rank 部署时每 rank 一份；
- **NCCL buffer**：多卡通信每 rank 几百 MB，`NCCL_BUFFSIZE` 可调小（代价是通信吞吐）；
- **CUDA Graph pool**：捕获后静态池常驻，capture 的 batch size 组合越多占得越大（vLLM 的 cudagraph capture sizes）；
- **杂项**：多模态 encoder 临时张量、embedding/词表缓冲；vLLM 启动 profiling 的一次性 forward 也吃峰值（它就是用这次 forward 测激活来定 KV 池大小，`gpu_memory_utilization` 留太满会在此处炸）。

诊断口径：`nvidia-smi 进程显存 - torch reserved = 框架外占用`，先算清这笔账再谈优化。

# 一页速查（面试口径）

| 症状 | 第一反应 |
| :--- | :--- |
| reserved >> allocated，假 OOM | `expandable_segments:True`、定长化 |
| prefill 炸（长 prompt 触发） | 调小 `max_num_batched_tokens`（chunked prefill）、fused CE |
| decode 跑一会儿炸 | KV 池不足：KV FP8 量化 / prefix caching / 准入控制 / 加卡 |
| 权重放不下 | W8A8 / W4A16 量化，TP/PP/EP 切分，offload |
| 并发毛刺型 OOM | admission control：`max-num-seqs` + 队列水位 |
| 显存对不上账 | allocated/reserved/peak 三件套 + nvidia-smi 差值，查 context/NCCL/graph |
