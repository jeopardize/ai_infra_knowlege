

<!--Q id=q-1780884692048-amnn ts=1791274056380 topic=kv-cache-->

**问题 (zh)**

什么是KV Cache

**问题 (en)**

what is KV Cache

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780884756742-pal8 ts=1791274056373 topic=kv-cache-->

**问题 (zh)**

大模型推理时，“KV 缓存（KV Cache）”是降低计算量、提升响应速度的核心机制。请解释 KV 缓存的存储原理，以及在长序列对话场景（如上下文长度超 4k）中，如何优化 KV 缓存的存储结构以减少内存占用？

**问题 (en)**

In large model inference, the "KV cache" is a core mechanism for reducing computational load and improving response speed. Please explain the storage principle of the KV cache, and how to optimize the storage structure of the KV cache to reduce memory usage in long-sequence dialogue scenarios (such as context length exceeding 4k).

**答案 (zh)**

- KV Cache 存储的是每层每个 head 的历史 K、V 张量，用于避免自回归解码时的重复计算。

- 它通常存放在 GPU HBM 中，但由于显存有限，长序列下会出现严重内存压力。

- 在工程层面，PageAttention 通过 block table 将 KV Cache 分页管理，解决连续分配带来的内外碎片问题。
- 在算法层面，可以通过量化（INT8/4）、GQA/MQA、滑动窗口注意力等手段直接缩小 KV Cache 体积。
- 在系统层面，可以引入 HBM–DRAM–SSD 的多级缓存池，结合 LRU 等策略动态迁移数据，从而在超长上下文（>4k）场景中显著降低显存占用。

**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887362088-eomt ts=1791274056367 topic=speculative-decoding-->

**问题 (zh)**

投机采样的加速比分析公式，MoE模型的投机采样遇到的性能问题

**问题 (en)**

Speedup analysis formula for speculative sampling, performance issues encountered in speculative sampling of the MoE model.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887385961-ocf5 ts=1791274056361 topic=speculative-decoding-->

**问题 (zh)**

MTP和Eagle的区别，MTP head的prefill过程，MTP head的kv cache变化

**问题 (en)**

The differences between MTP and Eagle, the MTP head prefill process, and changes to the MTP head's key-value cache.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887452904-6g91 ts=1791274056355 topic=nccl-collectives-->

**问题 (zh)**

4机32卡H200训练从FSDP换成HSDP，通信时间会减少多少

**问题 (en)**

How much will communication time be reduced when switching from FSDP to HSDP for 4-machine, 32-card H200 training?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887465931-iuox ts=1791274056349 topic=nccl-collectives-->

**问题 (zh)**

nccl中的通信源语有哪些？all-reduce参数更新一次参数需要几次通信？

**问题 (en)**

What are the communication primitives in nccl? How many communications are required for an all-reduce parameter update?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887481850-855v ts=1791274056343-->

**问题 (zh)**

在 AI 训练场景中，分布式训练常面临通信瓶颈问题。请说明 AllReduce、Broadcast、ReduceScatter 这三种核心通信算子的适用场景差异，并分析在千亿参数大模型训练中，如何选择通信策略以降低延迟、提升吞吐量？

**问题 (en)**

In AI training scenarios, distributed training often faces communication bottlenecks. Please explain the differences in applicable scenarios for the three core communication operators: AllReduce, Broadcast, and ReduceScatter. Furthermore, analyze how to choose a communication strategy to reduce latency and improve throughput when training a large model with hundreds of billions of parameters.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887493942-wj9n ts=1791274056337-->

**问题 (zh)**

多GPU通信时NVSHMEM和NVLink的区别

**问题 (en)**

Differences between NVSHMEM and NVLink in multi-GPU communication

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887508039-jduf ts=1791274056332-->

**问题 (zh)**

megatron-lm中通信优化怎么做？

**问题 (en)**

How to optimize communication in megatron-lm?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887539797-obgm ts=1791274056327-->

**问题 (zh)**

在小数据量场景使用NVSHMEM，每个GPU直接读取其他GPU的数据，在本地reduce，相比ring all-reduce的好处

**问题 (en)**

In scenarios with small data volumes, NVSHMEM allows each GPU to directly read data from other GPUs and perform local reduction, offering advantages over ring all-reduce.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887657078-05hz ts=1791274056320-->

**问题 (zh)**

多维tensor的transpose kernel怎么设计，是否需要根据i, j的位置设计不同的kernel

**问题 (en)**

How should the transpose kernel of a multidimensional tensor be designed? Should different kernels be designed based on the positions of i and j?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887810461-n831 ts=1791274056314-->

**问题 (zh)**

GPU matrix transpose使用shared memory的好处

**问题 (en)**

The benefits of using shared memory in GPU matrix transpose

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887823673-gmov ts=1791274056295-->

**问题 (zh)**

AI 框架（如 TensorFlow/PyTorch）的算子优化是 AI Infra 的核心能力之一。假设需优化一个高频调用的“矩阵乘+激活函数（如 ReLU）”融合算子，从硬件适配（如 GPU CUDA 核心/TPU 脉动阵列）、数据布局（如 NHWC/NCHW）、指令调度三个维度，说明关键优化思路。

**问题 (en)**

Operator optimization in AI frameworks (such as TensorFlow/PyTorch) is one of the core capabilities of AI Infra. Suppose we need to optimize a frequently used "matrix multiplication + activation function (such as ReLU)" fusion operator. Explain the key optimization approaches from three dimensions: hardware adaptation (such as GPU CUDA cores/TPU systolic arrays), data layout (such as NHWC/NCHW), and instruction scheduling.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887848307-g9a5 ts=1791274056290-->

**问题 (zh)**

cuda的global memory和shared memory访存分别需要注意什么？

**问题 (en)**

What should be noted when accessing global memory and shared memory in CUDA?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887893929-yl18 ts=1791274056284-->

**问题 (zh)**

将Ampere架构的算子适配到Hopper架构的卡上，你会对哪些地方进行升级改造？

**问题 (en)**

If you were to adapt the Ampere architecture operators to the Hopper architecture card, what upgrades or modifications would you make?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887910836-q533 ts=1791274056278-->

**问题 (zh)**

在 AI Infra 底层硬件适配中，需考虑不同芯片架构（如 x86、ARM、RISC-V）的指令集差异。请以“卷积算子”为例，说明如何设计跨架构兼容的算子实现方案，同时兼顾不同硬件的性能优势？

**问题 (en)**

In AI Infrastructure hardware adaptation, the instruction set differences of different chip architectures (such as x86, ARM, RISC-V) need to be considered. Taking the "convolution operator" as an example, please explain how to design a cross-architecture compatible operator implementation scheme while taking into account the performance advantages of different hardware.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887925934-gaq3 ts=1791274056273-->

**问题 (zh)**

请说明“AI 编译优化”（如 TVM、TensorRT）的核心工作流程，对比“基于模板的编译”与“自动搜索的编译”两种技术路线的优缺点，以及在面对异构硬件（CPU+GPU+NPU）时的编译策略设计

**问题 (en)**

Please explain the core workflow of "AI compilation optimization" (such as TVM, TensorRT), compare the advantages and disadvantages of "template-based compilation" and "automatic search compilation", and design a compilation strategy when dealing with heterogeneous hardware (CPU+GPU+NPU).

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887987263-jrdf ts=1791274056267-->

**问题 (zh)**

量化（Quantization）是 AI 模型推理加速的关键技术，常见有 INT8、FP16、FP8 等精度。请说明模型量化过程中“校准（Calibration）”步骤的核心作用，以及在量化后出现精度下降时，可采取哪些技术手段（如混合精度、量化感知训练）进行优化？

**问题 (en)**

Quantization is a key technology for accelerating AI model inference, with common precision levels such as INT8, FP16, and FP8. Please explain the core role of the "calibration" step in the model quantization process, and what techniques (such as mixed precision or quantization-aware training) can be used to optimize precision when it decreases after quantization.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=kb-kv-basic ts=1791274056262 topic=kv-cache-->

**问题 (zh)**

KV cache 大小怎么算？

**问题 (en)**

How do you calculate KV cache size?

**答案 (zh)**

自回归生成时，每生成一个新 token，attention 需要它对**之前所有 token** 的 Q·K^T。

如果不缓存，每步都要重算前面所有 token 的 K/V → 复杂度 `O(L^2)` 每步、`O(L^3)` 总；
缓存后每步只需算新 token 的 K/V 并和缓存拼接 → `O(L)` 每步、`O(L^2)` 总。

代价是显存：cache 大小 = `2 (K+V) × num_layers × num_heads × head_dim × seq_len × batch × bytes`。LLaMA-7B 在 2048 长度下单 batch 约 **1GB**。

**答案 (en)**

In autoregressive generation, every new token's attention needs to compute Q·Kᵀ against **all previous tokens**.

Without caching, every step re-derives K/V for every prior token → `O(L²)` per step, `O(L³)` total.
With caching, every step only computes the new token's K/V and concatenates with the cache → `O(L)` per step, `O(L²)` total.

The trade-off is memory: cache size = `2 (K+V) × num_layers × num_heads × head_dim × seq_len × batch × bytes`. For LLaMA-7B at length 2048 with batch 1, that's about **1 GB**.

<!--/Q-->
<!--Q id=q-1780901766052-zma3 ts=1791274056256-->

**问题 (zh)**

讲一下 FlashAttention 的原理

**问题 (en)**

Let me explain the principle of FlashAttention.

**答案 (zh)**

FlashAttention 的核心不是“更快算 attention”，而是“避免把完整 attention matrix 落到显存里”。它通过分块计算、在线 softmax 和重计算，减少 HBM 访问量，把很多中间结果保留在更快的片上存储里。这样做的本质是用更少的内存带宽换更好的速度和更低的显存占用。长序列场景下，它的收益非常明显，因为传统 attention 的瓶颈往往就是读写中间矩阵。
除此之外，flash attention v1，v2，v3 分别都解决了不同的问题

**答案 (en)**



<!--/Q-->
<!--Q id=q-1780901854289-aj92 ts=1791274056248-->

**问题 (zh)**

什么 RMSNorm 在大模型里比 LayerNorm 更常见

**问题 (en)**

Why is RMSNorm more common than LayerNorm in large models.

**答案 (zh)**

RMSNorm 去掉了均值中心化，只保留方差归一化，计算更简单，开销更低，而且在大模型里通常足够稳定。相比 LayerNorm，它少了一部分计算和同步开销，尤其在大规模训练和推理里更有工程优势。很多模型选择 RMSNorm，不是因为它理论上一定更强，而是因为它在稳定性和效率之间更平衡。工程里常见的选择逻辑就是：只要效果差不多，就选更省的。

**答案 (en)**



<!--/Q-->
<!--Q id=q-1780925589053-k2g4 ts=1791274056243 topic=model-serving-->

**问题 (zh)**

如果推理时出现 OOM，你会怎么排查

**问题 (en)**

If an OOM (Out of Memory) error occurs during reasoning, how would you troubleshoot it?

**答案 (zh)**

先看是不是显存被参数、激活、KV Cache 或者临时张量占满，再判断是训练阶段还是推理阶段。推理阶段最常见的原因是上下文太长、batch 太大、KV Cache 没做合理管理，或者某些算子产生了额外的临时开销。排查时一般先缩 batch、缩 context、关掉不必要的 profiling，再逐步定位具体层。很多 OOM 不是模型太大，而是调度策略把显存峰值顶上去了。

**答案 (en)**



<!--/Q-->
<!--Q id=kb-prefix-cache ts=1791274056236 topic=kv-cache-->

**问题 (zh)**

Prefix caching 和 KV cache 有什么区别？

**问题 (en)**

How does prefix caching differ from plain KV cache?

**答案 (zh)**

多个 request 经常共享前缀（同一个 system prompt、相同的 few-shot 示例、agent 的对话历史等）。

实现：
- 用 prefix 的 hash 作为 key，把对应的 KV blocks 放进 LRU 池
- 新 request 来了先做最长前缀匹配，命中部分跳过 prefill
- 配合 PagedAttention 的 block 共享天然契合

收益场景：
- Chat 应用的多轮对话（每轮共享之前的全部历史）
- Agent 工具调用（每次都带相同的 system prompt + tools）
- 评测/批量任务（相同 prompt 模板）

**答案 (en)**

Many requests share a prefix (a shared system prompt, the same few-shot examples, an agent's running conversation history, etc.).

Implementation:
- Use the prefix hash as a key; place the corresponding KV blocks into an LRU pool
- For a new request, do longest-prefix matching; skip prefill on the matched portion
- Composes naturally with PagedAttention's block sharing

Where it pays off:
- Multi-turn chat (every turn shares the entire prior history)
- Agent tool-use (same system prompt + tools every time)
- Eval / batch jobs (identical prompt templates)

<!--/Q-->
