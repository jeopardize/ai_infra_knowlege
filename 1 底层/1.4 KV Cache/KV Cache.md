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

最初，HBM分配是seq到来找到一块连续的空闲空间，这会造成大量的内存碎片。

![](images/image-20260416091856559_副本.png)


## PagedAttention --vllm

这意味着：

- ✅ 用 **指针 + offset**
- ✅ 假设内存是 **一整块连续 buffer**
- ✅ kernel 内部不做复杂的索引/跳转

👉 一旦变成非连续：

- kernel 要维护 **block index**
- cache 命中模式被打乱
- 性能急剧下降

```python
# kv_cache 申请
kv_cache = malloc(seq_len * layer * head * dim * sizeof(dtype));
# 直接去寻址
for i in (0, seq_len):
    attention(kv_cache[i])
```

## RadixAttention --SGlang

##vAttention

# Attention Store系统


# 常用模型kv cache 计算

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
