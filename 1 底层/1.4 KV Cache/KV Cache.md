# KV Cache

KV Cache 是 llm serving中相当关键的一部分

> [!important] \
> 需要理解：\
> 什么是kv cache \
> kv cache 优化
> 然后如何管理kv cache 

# 什么是KV Cache & 瓶颈分析

作答思路: 定义，为什么存在

## KV Cache是什么

定义：KV Cache 是指大模型推理时，把 Transformer 每一层 Attention 中已经计算好的 Key 和 Value 向量缓存下来。

为什么：

大模型计算过程中会计算一个注意力机制，通过输入依次计算query key value 三个矩阵

然而，decode过程是一个自回归过程，每次需要对历史 token 做 Attention。在这个过程中，多次的计算都属于重复。

有了 KV Cache，Prefill 阶段先算好并存入显存，Decode 阶段只计算新 token 的 Q/K/V，并拼接使用缓存，从而把 Decode 的注意力计算从 O(n²) 降到 O(n)，显著提升推理效率。代价是会占用一定显存，实际系统中通常会配合 PagedAttention 或 GQA 来优化。


## KV Cache 显存大小计算

b 为batch size，**<u> seq_len 为序列总长度（包括用户提供的提示词（prompt）以及模型生成的补全部分（completion））</u>**，n_layer 为解码器块/注意力层数，n_heads 为每个注意力层的注意力头数，d_head 为注意力层的隐藏维度，p_a 为精度对应比特数目。kv各一个所以需要乘以2。多头注意力（MHA）模型使用 KV 缓存技术，每个 token 的内存消耗量（以字节为单位）为：
$$
kv\_cache\_memory = 2 × b × seq\_len × (n_{layer} × n_{heads} × d_{head}× (p_a / 8))
$$
其中，

| 精度  | p_a  | 用途               |
| ----- | ---- | ------------------ |
| INT 8 | 8    | 权重量化，kv cache |
| INT 4 | 4    | 超低精度量化       |
| FP 8  | 8    | 混合精度训练       |
| FP 16 | 16   | 推理               |
| FP 32 | 32   | 训练高精度推理     |
| BF 16 | 16   | 训练，推理         |
|       |      |                    |

> %%r:回顾mha，mla几种attention计算方式%%

==**kv cache 节约计算量计算模拟**==

```
假设这一次的输入文本长度为n=1000
# 在不使用kv cache的情况下
第一次 计算 key value 1000*2 key一次 vakue一次
。。。。
需要计算1000次  
所需计算量为key value 1000*3

# 在使用kv cache的情况下
第一次计算 需要计算key value 1000*2
之后都是1次
因此
所需计算量1000*2

```

从以上数据可以看出，**KV 缓存内存消耗可能会变得非常大，甚至超过了加载大型序列模型权重所需的内存量。**

如{@fig:HBM_use}所示，是GPU的显存使用情况,模型权重和kv缓存是两个最大的开销：
![HBM_use](images/HBM_use.png){#fig:HBM_use}

%%r:所以，降低 kv cache 显存占用有以下几个解决方案：
1. 降低kv cache自身显存占用 —— kv cache量化、分布式kv cache存储、MQA/GQA等结构层面减少KV头数；
2. 优化系统管理方案，更高效地利用有限显存 —— PagedAttention减少碎片、KV Eviction驱逐不重要token、Prefix Caching复用相同前缀；
3. 减少需要缓存的token总量 —— 滑动窗口注意力只保留近期token、投机解码减少decode步数。%%

# KV Cache 量化

> [!Note]
>
> 目标：解决kv cache每一个token大小的问题，减少他在显存中的占用。

将 FP16 / BF16 等高精度的 Key 和 Value 缓存映射为 FP8、INT8、INT4 甚至 INT2 等低精度表示，从而成倍降低显存占用并减少访存带宽压力。

%%r:量化的核心思路：在可接受精度损失的范围，用更紧凑的数据格式来表示原始数据，也就是将浮点数映射到整数里面%%

## 量化数学原理

```text
//量化映射，有点类似预测未防止极端数据，会进行一定的缩放
scale = (max_val-min_val)/(2^bits - 1)
zero_point = round(-min_val/scale)
q = clamp(round(x/scale+zero_point),0,2^bits-1)

//反量化
x' = (q - zero_point)*scale

//scale:比例因子，决定量化精度
// Zero-point：零点，处理非对称范围
// Bit-width：位宽，决定熟知范围
```

ok，那根据这个数学原理，我们知道需要知道 max_val 和 min_val, 跟预测问题一样，我要么根据历史数据得知，但是这会有数据便宜，要么实时计算，这样量化又可以分为动态量化和静态量化


#  引擎里的 KV 存储技术

> [!Note]
>
> 目标：解决“存哪里、怎么存、存不下怎么办”的问题。

## 传统显存管理

KV Cache 是指大模型推理时，把 Transformer 每一层 Attention 中已经计算好的 Key 和 Value 向量缓存下来。通常我们会将其放置在gpu的HBM中，但是这里的大小有限。这时候就需要我们更好的利用整体的空间大小

> 模型中不是每一层都是attention操作啊，query，MLP 中间状态，LayerNorm / Activation 状态

最初，HBM分配是seq到来找到一块%%b:连续的空闲空间%%，这会造成大量的内存碎片。

![past_hbm_manage](images/past_hbm_manage.png)



## PagedAttention --vllm

PagedAttention（SOSP 2023）借鉴操作系统虚拟内存分页管理 KV Cache，解决传统方案的三大内存浪费：

- 内部碎片：按”最大可能序列长度”预分配连续内存，实际输出远短于上限；
- 外部碎片：请求长度各异，内存块之间产生无法组合的空隙；
- 冗余复制：并行采样、beam search 下多条序列共享相同前缀却各存一份。

核心设计：把逻辑连续的 KV Cache 切成固定大小的块（block），通过页表做逻辑到物理的映射，KV Cache 在物理显存中可以非连续存储。调度器每步显式分配/回收块；不同请求共享相同前缀（如 system prompt）时，前缀页指向同一物理块，用引用计数管理生命周期，写入时才触发 Copy-on-Write。显存利用率从 20%–40% 提升到 >90%，同延迟下吞吐提升 2–4×。

代价：注意力 kernel 必须感知分页（block table 间接寻址），编程复杂且有 kernel 性能损失
> 如何理解：
> ```python
> // 传统：KV 是连续数组，直接算地址
> kv_ptr = base_ptr + token_id * head_dim;
> // PagedAttention 下变成：
> // 分页：先查 block table，再算块内偏移
> block_id = token_id / block_size;
> block_ptr = block_table[block_id];  // ← 间接寻址！
> offset = (token_id % block_size) * head_dim;
> kv_ptr = block_ptr + offset;
> ```

![pageattention](images/pageattention.png)

%%r:vLLM V1 的完整链路：%%调度器维护三队列（WAITING → RUNNING → SWAPPED）。显存不足时触发抢占，两种策略——RECOMPUTE（释放全部 KV 块、下次重算）与 SWAP（KV 从 GPU 换出到 CPU、之后换回），这是引擎级 GPU↔CPU 分层存储的实现。执行层的 blocks_to_swap_in/out/copy 在 forward 前完成，其中 blocks_to_copy 正是 prefix cache 命中后共享块的写时复制路径。V1 还集成了 Automatic Prefix Caching（APC）——基于 token 块哈希的前缀复用。

> [!important]
> 补充代码
```python
# kv_cache 申请
kv_cache = malloc(seq_len * layer * head * dim * sizeof(dtype));
# 直接去寻址
for i in (0, seq_len):
    attention(kv_cache[i])
```

## RadixAttention --SGlang

RadixAttention（2023）的核心目标是跨请求的自动 KV 复用，数据结构是基数树（Radix Tree）：

- 树的每条边标记一段 token 序列，节点持有 KV pool 的 slot 索引；
- 新请求到来时做最长前缀匹配：命中部分直接复用其 KV，未命中部分追加为新节点；
- LRU 等策略淘汰旧节点释放显存。
![radix_attention](images/radix_attention.png)

%%r:实现细节（sglang mem_cache/，main 分支）——三个设计值得注意：%%

1. 树只存索引、不存张量：TreeNode.value 指向 KV pool 的 slot 索引，与 PagedAllocator（页粒度分配/回收）咬合；命中前缀时 value 就是可直接喂给 attention backend 的物理位置。insert 只登记索引不分配 slot，cache_finished_req 在请求完成时插树并去重释放已被树覆盖的前缀 slot。
2. 锁沿父链传播：只要某请求持有叶子节点的 lock（lock_ref），整条前缀链都不可淘汰；evict 从增量维护的 evictable_leaves 集合按策略建堆逐个释放，复杂度与淘汰量成正比而非树大小。淘汰策略有 7 种（LRU/LFU/FIFO/MRU/FILO/priority/SLRU），默认 LRU。
3. 统一锚点贯穿多层存储：TreeNode 同时持有 value（GPU slot 索引）、host_value（CPU slot 索引）、hash_value（每页一个哈希，作外部存储 key）——一棵树天然索引三层存储，是后文 HiCache 的基础。
extra_key（LoRA id / cache salt）进入键空间实现命名空间隔离；树只在 page 边界分裂节点，page 对齐贯穿全栈。

缺点：CPU 侧树操作开销随命中率上升；收益依赖负载的前缀分布，低命中负载反受其累

> [!important]
> 补充代码

```python

```

## vAttention：把分页做进 MMU
PagedAttention 把虚拟内存布局从连续变为非连续，代价是注意力 kernel 必须感知分页（block table 间接寻址），编程复杂且损失 kernel 性能。vAttention（Microsoft，arXiv 2405.04437）走另一条路：用 CUDA Virtual Memory Management API 解耦虚拟/物理内存分配——虚拟地址空间预先分配连续大块，物理显存按需映射。物理层碎片通过延迟映射解决，而虚拟连续性保留：原生兼容未改造的注意力 kernel（FlashAttention 等），吞吐比 PagedAttention kernel 最高 1.23×。

两条路线的对比本质是：PagedAttention 把分页做进 kernel，vAttention 把分页做进 MMU。后者依赖 CUDA VM API 且物理碎片仅缓解不消除，但思路代表了”OS 虚拟内存思想进一步下沉”。

## 架构演进对 KV 存储的冲击：MLA 与 KDA
MLA（DeepSeek）：把多头 K/V 压缩为低秩 latent 向量c_KV ，只缓存 latent、推理时解压。%%r:DeepSeek-V3 的 KV 维度从 GQA 的 2048 降到 576，压缩约 3.6×%%。==代价是 TP 切分困难：每个设备都需加载完整 latent，内存优势被并行度抵消==
> 如何解读：
> 
TPLA（ASPLOS 2026）用 Hadamard/PCA 重参数化使 latent 可沿特征轴切分，免训练加载 MLA checkpoint，32K 上下文加速 1.79×/1.93×。

KDA（Kimi K3）打破 per-token 假设：K3（2.8T 参数、1M 上下文）的注意力骨架是”三 KDA + 一 MLA”交替（93 层中 69 层 KDA）。%%r:KDA 不存逐 token KV%%，历史递归压缩为固定大小的 Recurrent State（$S_t \in \mathbb{R}^{128 \times 128}$与序列长度无关）+ 卷积窗口，KV Cache 降低约 75%。但这根本改变了缓存性质：

- 递归状态无法任意截断回退——token 级前缀匹配成功 ≠ 状态可恢复，必须在选定前缀边界显式存 checkpoint；
- 单个 KDA checkpoint ≈ 0.4 GiB；128 token 间隔下 1M 上下文需 ~3TB，10K 间隔也要 ~40GB——checkpoint 越稀疏，命中后重算的后缀越长；
- MLA KV 与 KDA State 必须在同一前缀边界对齐。

![kv_arch_boom](images/kv_arch_boom.png)
> Full Attention 的 KV 随序列线性增长、可任意截断复用；KDA 只存固定状态 + 稀疏 checkpoint，命中需从最近 checkpoint 分叉并重算后缀
---
%%r:SGLang 的解法（MambaRadixCache）%%：树节点同时持有 value（MLA per-token KV 索引）和 mamba_value（KDA per-prefix 状态 slot），两套独立 LRU/引用计数；节点分裂时 mamba_value = None（状态无子结构可切）。关键机制：track/donate/branching 闭环（递推中定期快照状态 → 零拷贝”过户”给基数树 → 命中时从最深状态节点分叉重算尾巴）、ping-pong track buffer（2 个 track slot 交替写/取，给 donate 2 轮 chunk 余量，是 overlap 调度的前提）、INT8 冷池（per-(head,k-channel) 对称量化，同显存缓存约 4× 前缀状态）。实测 60K 上下文下 KDA 混合 cache 比纯 MLA 省 70%。vLLM 社区（K3 团队贡献）则用三粒度解耦：hash 粒度（前缀匹配探测）× 物理状态块（摊销存储）× 调度对齐粒度，4,096-token 的物理块内部可在任意位置记录命中点直接恢复。TokenSpeed 的 Flat KV 更进一步：一页既可存 1,536 token 的 MLA latent，也可存一个完整 KDA 快照——统一页分配器与地址空间，prefix cache / CoW / PD 传输协议全部复用。

# Attention Store系统
单卡 HBM 装不下 KV Cache 时，就需要把它扩展到 CPU DRAM、SSD 和分布式存储——这类系统统称 Attention Store。核心思想是“以存换算”：用更多的存储资源换取更少的计算。

## 存储层次
| 层级 | 介质 | 典型容量 | 典型带宽 |
|---|---|---|---|
| L1 | GPU HBM | 80-96 GB | 2-3.3 TB/s |
| L2 | CPU DRAM | 0.5-2 TB | 50 GB/s |
| L3 | NVMe SSD | 4-100 TB | 7 GB/s(单盘) ~25 GB/s(RAID) |
| L4 | 分布式池/远端对象存储 | PB级 | 10-100 Gbps |

![store_layer](images/store_layer.png)

带宽落差是数量级的（NVMe→HBM 差约 480×），因此所有系统都在做同一件事：用预取、重叠、粒度设计把搬运藏进计算。三个反复出现的技术母题：

1. 重叠隐藏：逐层 load/store 与计算重叠（Mooncake/LMCache/CachedAttention/LayerKV）、多 CUDA stream 并行、预取窗口；
2. 未来信息 > 历史信息：用调度器队列的前瞻而非 LRU 的历史做放置与驱逐，命中率差距是数量级的（86% vs 48–58%）；
3. 粒度选择：传输粒度要大（聚合到 chunk 才能饱和带宽），管理粒度要细（页/层/会话各取所需）。

## CachedAttention / AttentionStore：会话级 KV 外存的奠基
CachedAttention（USENIX ATC 2024，NUS/上交/华为云）盯住多轮对话：ShareGPT 中 73% 为多轮，每轮结束后 KV 被丢弃，下一条消息要对全部历史重新 prefill——历史 token 占比超 99%，重复计算占 TTFT 的 99%。

架构：Controller（调度器）+ AttentionStore Manager（DRAM+SSD 两级）+ GPU 集群（含专用读写 buffer），Load/Save 走独立 CUDA stream，Fetch/Evict 走独立 IO 线程。四大机制：
- 逐层预载：算第$i$层时并发预载第$i+1$层；
- 异步保存：prefill 产生的 KV 逐层与 decode 重叠写回；
- 调度器感知放置/驱逐（本质创新）：用 job queue 的”未来信息”替代 LRU/FIFO 的历史信息，窗口内即将被访问的 KV 豁免驱逐——命中率 86% vs LRU 58% / FIFO 48%；
- 位置编码解耦：存不含 RoPE 的 K/V，加载时再嵌入新位置编码，上下文截断不使缓存失效。


效果：KV 命中率 71–90%，TTFT 降低 61–87%，prefill 吞吐提升 2.6–7.8×，成本最多降 70%（存储开销仅占总成本 9–16.4%）。HBM-only 命中率约 0%，加 10TB SSD 后达 71–90%——跨 DRAM/SSD 分层是命中率的根本来源。%%r:局限：单机本地、无跨节点路由（Mooncake 沿此方向工业化）。%%

## Mooncake：KVCache-Centric 分离架构的生产标杆
Mooncake（FAST 2025 最佳论文，月之暗面+清华）是 Kimi 的生产推理平台，核心洞察：调度请求本质上是调度 KVCache 在哪里、谁来算、传给谁。架构四件套：
- Prefill 池（max cache reuse，s.t. TTFT SLO）+ Decode 池（max throughput，s.t. TBT SLO）；
全局分布式 KVCache 池：把 GPU 服务器闲置的 CPU/DRAM/SSD/NIC 池化，通过 GPUDirect RDMA 传输（Messenger 组件）；
- Conductor 全局调度器：以前缀命中长度、实例负载、传输时间、SLO 统一权衡路由——TTFT = 排队 + prefill（+ 传输）；SLO 不可达直接 HTTP 429 提前拒绝；
- 热点迁移：请求路由到非最佳实例时顺带拉取 KV 复制热块（trace 显示 >50% 缓存块从未复用、最热块被访问数万次——复制热块收益极高）。

![Mooncake_kv](images/Mooncake_kv.png)

> Mooncake 四件套——Conductor 全局调度 + Prefill/Decode 分离池 + 分布式 KVCache 池，五步请求工作流
数据按 512-token 块组织，前缀哈希级联去重；Layer-wise Prefill 让逐层 load/store 与计算重叠，prefill 时间 ≈ max(加载, 计算)。长上下文流水（CPP）把多节点 prefill 流水化。

效果：已在数千节点部署、日处理超千亿 token；长上下文模拟吞吐 +50%~+525%；真实负载多处理 75% 请求且 TBT 达标率 ~100%（vLLM 57%）；平均 TTFT 6.26s vs 负载均衡 60.41s。

工程视角的双重身份（SGLang 集成）：Mooncake 既可以是 HiCache 的 L3 持久后端（共享缓存池，跨实例/重启复用，--hicache-storage-backend mooncake），也可以是 PD 分离的 KV 传输通道（请求级一次性搬运，--disaggregation-transfer-backend mooncake）——前者是”可复用资产”，后者是”过路数据”。Kimi K3 编程场景靠这套分离架构做到 >90% 缓存率。

## LMCache：引擎中立的 KV 存储传输层
LMCache（arXiv 2510.09665，TensorMesh+芝加哥大学）不是引擎，而是横跨引擎与存储后端的可插拔中间件——把 KV Cache 从”GPU 显存里的临时中间状态”升级为可持久化、可迁移、可调度的基础设施资源。

它抓住了一个被忽视的物理约束：PagedAttention 的页粒度 I/O 太小，无法饱和带宽——16-token 页仅 62.5KB，而实测消息 ≥16MB 才能饱和 400Gbps NIC。

> 传输粒度决定带宽——页级小消息只能吃到 88Gbps，聚合成 16MB 大消息才能饱和 400Gbps

核心机制：

- 批量化搬运：多页多层聚合为 chunk（默认 256 token），定制聚合/打散 CUDA kernel + DMA，延迟批写；
-计算-IO 重叠：逐层流水（只需单层 KV 大小的 GPU 缓冲）+ 排队期预取；
- 零拷贝（引用计数）与三指针动态 offloading 状态机；
- 标准化接口：KV Connector API（7 个函数，已被 NVIDIA Dynamo / llm-d / AIBrix / vLLM Production Stack 采纳）+ 分布式 Cache Controller（lookup/move/pin/batched_evict），支撑 KV-aware 路由与跨实例共享。

**效果:** CPU→GPU 加载带宽 400 Gbps（vLLM 原生 88 Gbps——根源就是页粒度 vs chunk 粒度）；TTFT 降低 1.9–8.1×、吞吐提升 2.3–14×；PD 分离下 mean TTFT 再降 1.53–1.84×。

**设计哲学:** 主要缓存 Prefill KV（system prompt、文档片段——下次请求必然复用）而非 Decode KV（复用率极低且频繁写盘造成 I/O 竞争）。生产数据：Agent 负载（Codex SWE-bench Pro trace）input/output 比 131:1、缓存命中率 94.2%、轮间隔中位数仅几秒——这是 LMCache 们的目标画像。

## HiCache：分层基数树（SGLang/阿里云 Tair）
HiCache 把 RadixAttention 的基数树扩展为 HiRadixTree（分层基数树）：L1 GPU HBM（热数据）→ L2 CPU 锁页内存（--hicache-ratio 2.0，L2 = ratio × L1）→ L3 外部存储（file/mooncake/hf3fs/nixl 等后端）。一个 TreeNode 的 value / host_value / hash_value 三字段分别锚定三层。

关键机制：
- 双 CUDA stream：write_stream（device→host DMA）与 load_stream（host→device，按 layer 分步）分离，配合 ack 队列 + TP all_reduce 同步；
- write_through 策略：平时异步备份到 host，evict 时只释放 device 槽、保留 host 数据（evict_device()）；命中已驱逐节点时沿父链收集并 DMA 回来（load_back）；
- L3 预取状态机：先探测命中长度、命中页数 ≥ 256 token（prefetch_threshold）才分配 host 内存并真正预取——避免为小命中浪费带宽与内存。

![HiRadixTree](images/HiRadixTree.png)

> HiRadixTree——同一个 TreeNode 的三个字段锚定三层存储，前缀在层间升降级而不换数据结构
> 
最易误解的点：SGLang 中 L3 有两条互斥集成路径——HiCache 的存储后端（HiCacheStorage，page 哈希寻址、同步 batch get/set）与 RadixCache 装饰器（LMCache/FlexKV 以 RadixCache 子类整体替代 L2+L3，只覆盖 match_prefix/init_load_back/cache_finished_req/evict 四个方法，用 token 序列寻址 + LOOKUP→RETRIEVE 两阶段）。一个进程只实例化一个 cache 类，不能自由组合。

阿里云 Tair 团队加入后演进为分布式版本：真实生产（Novita AI）中命中率 40%→80%，TTFT 降 56%，QPS 翻倍。

## 端到盘与直通存储：KVSwap 与 Phoenix
KVSwap（arXiv 2511.11907）面向端侧：%%r:统一内存设备上 GPU→CPU 卸载无意义，必须下盘%%，而盘带宽比内存低 2–3 个量级。做法是全量 KV 存盘 + 内存只留”低秩 K 摘要（只做预测、不参与计算，可激进压缩 σ 高达 32）+ 分组关键 KV 预测（对齐磁盘块读）+ 滚动/复用缓冲”。Jetson Orin 上 vs 最强基线提速 1.8×（NVMe）/4.1×（eMMC）。

Phoenix（http://xPU.IO，SC’25 + LMCache PR #4673）：SSD 是 Agent 时代唯一划算的大容量介质（单机 100TB 级），但传统路径”盘→主机内存→显存”两次穿 PCIe + CPU 全程参与——实测 4×NVMe RAID0 裸盘 25GB/s，端到端只剩 10.2GB/s；Codex 回放中 TTFT 1101ms 里加载占 600ms、真正 prefill 只 313ms。Phoenix 基于 Linux 标准 I/O 栈 + ZONE_DEVICE 为显存建 struct page，让 GPU 显存指针成为合法 I/O 地址，直接走 pread/io_uring：8×H20 生产 trace 下，平均 TTFT 是 SSD 中转的 2.2×、纯 DRAM 的 5.6×；TTFT 5s 红线下并发 7（DRAM-only）→13（SSD 中转）→37（Phoenix）。机制收益的本质：加载快于计算后，”KV 加载与前一批推理 overlap”的调度重新生效。

## 其他系统一览
- NVIDIA CMX：DPU 做多租户调度 + SSD 大容量，KV 管理从引擎解耦为专用存储栈；
- 英特尔 KV Shrink：QAT 硬件加速 + 分层卸载，L1(HBM)→L2(DDR)→L3/L4(SSD/分布式) 四级；
- 中科曙光 ParaCache：依托 ParaStor 的四级缓存，”从存数据到存智能”，KV 作为可复用数据资产管理；
- 华为 UCM：昇腾 NPU 的 KV 池化，经 KV Connector 对接 Mooncake 等后端，灵衢总线跨节点传输。
- EMS 缓存：

## 命中率经济学：Attention Store 的商业闭环
Attention Store 的价值最终由%%r:命中率 × 定价%%决定，工业界给出了几个反直觉结论：

**命中部分定价仅为未命中的 10%–25%**（无问芯穹，服务 GLM/Kimi）。由此推出浴盆模型：TPS_hit 随命中率单调上升，TPS_miss 近似线性下降；命中率 40%→70% 时未命中吞吐再降 20%，而命中部分提升不足 3 倍——中段（”偶然失效区间”）可能命中率越高越亏钱；且存在 20%/40%/70% 奇异点（末尾残块不满 chunk 独立计算）。优化方向：把浴盆变浅、加大 cache 容量逃离盆底、cache-aware 负载均衡。

![basin_model](images/basin_model.png)

> 浴盆模型——命中部分吞吐上升、未命中部分吞吐下降，中段”偶然失效区间”可能越命中越亏；20/40/70% 为残块奇异点

命中率天花板由 trace 决定，容量只是入场券（kvcache.ai 模拟器逆向）：Mooncake 真实 trace 的复用天花板（无限容量）仅 59.1%——Belady 最优在 16GiB 就到顶（有效复用工作集极小，仅 ~3.7% 唯一块有未来复用价值），LRU 却要 256GiB 才追平（分不清”将复用”与”死块”）。trace 的复用结构决定上限，模型结构只决定多少 GiB 换多少块。 Agent 负载改变命中率来源：传统对话输入 8K、命中率 50%；Agent 任务输入 64–128K、输出 ≤1K、命中率 90%。但前缀易变字段破坏命中：Agent ID、工作目录、时间戳、工具 schema 每次微变即 miss。TokenPilot（浙大）证明仅把易变字段替换成静态占位符 + 摄入边界确定性整理，就把宏观命中率 38.7%→79.2%、费用降 61–87%——上下文管理与 KV 存储是同一枚硬币的两面。









# 常用模型kv cache 计算
## glm53 计算

## DeepSeek-V4-Pro

第一步：每层存什么。 V4-Pro 有 61 层注意力，每层 = 一个局部滑窗（SWA）分支 + 一个全局压缩分支；全局分支按压缩比分两种——30 层 CSA（压缩比 4，记 C4）+ 31 层 HCA（压缩比 128，记 C128），30 个 C4 层还各带一个 lightning indexer。落到显存上是 7 个 cache pool，”单 slot”一列是该 pool 的一条条目跨全部层的字节数：

| Cache Pool | dtype | 条目宽 | 层数 | 单 slot |
|:---|:---|:---|:---|:---|
| SWA-KV-Cache | uint8（1B） | 584 | 61（MTP 态 62） | 35,624 / 36,208 B |
| C128-Cache | uint8（1B） | 584 | 31 | 18,104 B |
| C4-Cache | uint8（1B） | 584 | 30 | 17,520 B |
| C4-index-Cache | uint8（1B） | 132 | 30 | 3,960 B |
| C128-State | FP32（4B） | 512 × 2 | 31 | 126,976 B |
| C4-State | FP32（4B） | 512 × 2 × 2 | 30 | 245,760 B |
| C4-index-State | FP32（4B） | 128 × 2 × 2 | 30 | 61,440 B |

第二步：固定存储项。 固定项只由两件事构成——滑窗内驻留的 SWA KV，以及三个 State 环形缓冲。环深分别是 window_size = 128、C4_ring_size = 8、C128_ring_size = 128：

KV_fixed = SWA_slot × window_size
         + (C4_State_slot + C4_index_State_slot) × C4_ring_size
         + C128_State_slot × C128_ring_size

         = 35,624 × 128  +  (245,760 + 61,440) × 8  +  126,976 × 128
         = 4,559,872     +  2,457,600               +  16,252,928
         = 23,270,400 B  ≈ 22.2 MiB（文档记作 22.2MB）
拆占比：C128-State 环 15.5 MiB（70%）、SWA 滑窗 4.35 MiB（20%）、C4 两个 State 环 2.34 MiB（10%）。这里藏着一个反直觉的事实：C128-State 环（15.5 MiB）比它服务的 C128 压缩 KV 本身还大——100K 上下文下 C128-Cache 全长只有 13.8 MiB（交叉点约在 11.5 万 token）。固定开销的大头不是”要缓存的历史”，而是”维护压缩状态所需的无损缓冲”。

第三步：线性增长存储项。 7 个池里只有三个全局池随上下文增长，压缩比决定它们每多少 token 才落一条条目：

KV_linear = C128_slot × CTX/128 + (C4_slot + C4_index_slot) × CTX/4
          = 18,104 × CTX/128    + (17,520 + 3,960) × CTX/4
          = 141.4 × CTX         + 5,370 × CTX
          = 5,511 B / token     ≈ CTX × 5.4KB
两项相加，就是那条官方线性公式：


第四步：100K 上下文（102,400 token）一条请求的账。 固定项 22.2 MiB，线性项 538.2 MiB，合计 ≈560 MiB（≈587.6 MB）：

线性部分 538.2 MiB（564.4 MB = 102,400 × 5,511 B）
   C4-Cache：       17,520 B × 25,600（= CTX/4）  = 448.5 MB  （79.5%）
   C4-index-Cache：  3,960 B × 25,600             = 101.4 MB  （18.0%）
   C128-Cache：     18,104 B ×    800（= CTX/128）=  14.5 MB  （ 2.6%）
固定部分 22.2 MiB（23,270,400 B）
   SWA 滑窗 128 token × 35,624 B                  =   4.35 MiB
   C4 + C4-index State 环 8 槽                    =   2.34 MiB
   C128 State 环 128 槽                           =  15.50 MiB
合计 ≈ 560 MiB / 请求

# 相关学习资料

https://github.com/ForceInjection/AI-fundamentals/blob/main/09_inference_system/kv_cache/README.md

https://zhuanlan.zhihu.com/p/2082615179595073265
