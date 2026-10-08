

<!--Q id=kb-prefix-cache ts=1791362748736 topic=kv-cache-->

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
<!--Q id=kb-kv-basic ts=1791276643901 topic=kv-cache-->

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
