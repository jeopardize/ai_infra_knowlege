

<!--Q id=q-1780884756742-pal8 ts=1791276606895 topic=kv-cache-->

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
<!--Q id=q-1780884692048-amnn ts=1791276579390 topic=kv-cache-->

**问题 (zh)**

什么是KV Cache

**问题 (en)**

what is KV Cache

**答案 (zh)**



**答案 (en)**



<!--/Q-->
