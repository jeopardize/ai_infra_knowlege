# 显存优化

显存是大模型推理中最刚性的资源：**权重常驻、KV Cache 随「上下文 × 并发」线性膨胀、激活峰值决定扛不扛得住长 prompt**。先对常见显卡建立量级感（消费级/国产卡迭代快，以官方口径为准）：

| **类别**      | **型号**        | **显存**               | **典型场景**          |
| :------------ | :-------------- | :--------------------- | :-------------------- |
| **训练·旗舰** | H100/H800       | 80 GB (HBM3)           | 超大规模训练·推理主力 |
|               | A100/A800       | 40/80 GB (HBM2e)       | 中大规模训练·推理     |
|               | H200            | **141 GB** (HBM3e)     | 显存密集型推理        |
|               | B200/B100       | **192 GB** (HBM3e)     | 万亿参数·下一代超算   |
|               | V100            | 16/32 GB (HBM2)        | 早期训练·遗留部署     |
| **推理卡**    | L40S/L40        | 48 GB (GDDR6)          | 云推理·图形混合       |
|               | A10 / L4        | 24 GB (GDDR6)          | 推理入门 / 轻量边缘   |
|               | T4              | 16 GB (GDDR6)          | 边缘·视频转码         |
| **国产 NPU**  | 昇腾 910C       | **64 GB/die × 2 die**  | 超节点·千亿~万亿模型  |
|               | 昇腾 950C       | 96 GB                  | 超节点·千亿~万亿模型  |
| **消费级**    | RTX 4090 / 5090 | 24 / 32 GB             | 个人·7B~13B 推理      |

# 显存构成估算

一次推理服务中，显存可以拆分成以下几个部分进行估算：

```
总显存 = 常驻(权重 + CUDA context/NCCL buffer + CUDA Graph pool)
       + 半常驻(KV Cache 池，动态增长但生命周期长)
       + 瞬时(激活值 / workspace，随 step 震荡)
       + 碎片化浪费(allocator 已 reserve 但无法分配)
```

**权重（常驻，刚性）**：

```
权重显存 = 参数量 × dtype_bytes            # BF16=2B, FP8=1B, INT4=0.5B
70B BF16 ≈ 140 GB   → 单卡 80 GB 放不下，至少 TP2
70B W4A16 ≈  35 GB  → 单卡 L40S (48GB) 可跑
MoE 注意：按总参数算（K2 总参 1T → BF16 ~2 TB），激活参数决定算力，不决定权重显存
```

**KV Cache（半常驻，随上下文 × 并发线性增长）**，单个 token 的 KV cache：

```
KV_per_token = 2 (K和V) × n_layers × n_kv_heads × head_dim × dtype_bytes

MLA（DeepSeek 系）：无独立 K/V，缓存的是压缩 latent
                   = n_layers × (kv_lora_rank + rope_dim) × dtype_bytes
```

常见模型单 token KV（BF16、head_dim=128，MB 按 2^20 计，均可代入公式验算）：

| **模型**               | **层数**        | **KV头数**    | **每 token KV** | **128K 上下文单请求** |
| :--------------------- | :-------------- | :------------ | :-------------- | :-------------------- |
| LLaMA-7B (MHA, 32头)   | 32              | 32            | 0.5 MB          | **64 GB**             |
| LLaMA-3-8B (GQA, 8头)  | 32              | 8             | 0.125 MB        | 16 GB                 |
| LLaMA-3-70B (GQA, 8头) | 80              | 8             | 0.3125 MB       | 40 GB                 |
| DeepSeek-V3 (MLA)      | 61              | latent 512+64 | 0.067 MB        | ~9 GB                 |
| **Kimi K2**            | 64              | GQA, 8        | 0.25 MB         | **32 GB**             |
| Kimi K2（1M 上下文）   | 64              | GQA, 8        | 0.25 MB         | **~256 GB**           |
| **GLM-5.2（60L 估）**  | 60              | GQA, 8        | 0.234 MB        | 30 GB                 |

从表里读出的结论：

- **MHA→GQA（KV 头 32→8）省 4 倍，GQA→MLA 再省 ~4 倍**，这就是 Llama-3 / Qwen / K2 清一色 GQA、DeepSeek 走 MLA 的原因；
- 长上下文场景显存大头在 KV 而非权重：K2 单请求 1M 上下文的 KV 要 256 GB，比多数模型的权重还大。

**激活 / workspace（瞬时，随 step 震荡）**：prefill 激活 ∝ batched_tokens × hidden（FlashAttention 已消除 O(T²) 注意力矩阵显存，但 MLP 中间态仍在）；decode 激活极小，大头是 logits 缓冲 `bs × vocab × 4B`。

**容量规划核心公式**（由 KV 池反推并发）：

```
可用 KV 池 ≈ 总显存 × gpu_memory_utilization - 权重 - 框架开销(~1-2 GB)
最大并发 ≈ KV 池 / (平均上下文长度 × KV_per_token)
```

例：80 GB 单卡跑 Llama-3-8B BF16（权重 16 GB），剩 ~55 GB KV 池；平均请求 8K token（8192 × 0.125 MB = 1 GB/请求），理论并发 ~55 路。vLLM 启动日志直接打印 `# GPU blocks` 与 `Maximum concurrency for xx tokens per sequence: xx`，就是这个估算的落地。

综上所述，有一下几个方向优化 
       1.针对dtype优化

       2.针对权重量化 

       3.针对kv cache量化

       4.针对激活参数量化

       5.

# dtype量化

# 针对权重量化
## checkpoint 
## 多节点显存策略

# 针对kv cache优化
> [!IMPORTANT]
>
> 解决的核心问题是：用更少的字节保存历史信息，能否在质量可接受的前提下，支持更长上下文和更多并发
## 降低单请求存储开销（如何让 Cache 更小）

*目标：通过算法或精度优化，减少单个序列占用的显存空间。*

### 架构层面的压缩

- **GQA / MQA (Grouped/Multi-Query Attention)**： 减少 Key/Value 的头数（Heads），迫使多个 Query 共享同一组 KV，直接成倍缩减 Cache 体积。
- **MLA (Multi-head Latent Attention)**： DeepSeek-V2/V3 核心创新，通过低秩联合压缩 KV，极大降低传输和存储负担。


### 长度压缩

- **滑动窗口注意力 (Sliding Window Attention)**： 限制每个 Token 只能关注局部窗口，丢弃过远的 KV Cache，以牺牲一定全局感知为代价换取线性复杂度。

### 低精度量化

- **KV Cache Quantization (INT8/INT4)**： 将原本 FP16/BF16 的 Cache 量化至更低比特，直接压缩内存占用，但需权衡精度损失。

一组浮点数共享

------

## 提高跨请求复用效率（如何更少地重复计算）

*目标：利用请求间的公共前缀，避免重复计算相同的 Context。*

### 前缀缓存 (Prefix Caching)

- **原理**：在多轮对话或 Agent 场景中，System Prompt 或工具调用描述通常占据大量前缀且固定不变。
- **实现**：通过哈希索引识别相同的前缀 Token，直接复用已计算好的 KV Cache Block，从而跳过 Prefill 阶段的重复计算。

### 多级缓存架构 (Tiered Caching)

- **背景**：单纯靠 GPU 显存（HBM）无法存下所有历史会话的 Cache，且随着 Batch Size 增大容易 OOM。
- **分级策略**：构建 **HBM (GPU) -> DRAM (CPU) -> SSD (磁盘)** 的三级缓存池。 **HBM**：存放当前正在推理的活跃 Cache，追求极致速度。 **DRAM/SSD**：存放冷数据或历史会话的 Cache，作为容量扩展。
- **挑战**：需要处理跨设备的数据迁移（Offloading）延迟，平衡 I/O 速度与存储容量。

### RadixAttention


## empty_cache


# 针对激活参数量化

## activationcompression





# CPU-GPU流水线重叠

# ROI评估


# offload
## ZeRo-offload
## DeepSpeed-Infinity