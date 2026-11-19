 
> 2026/11/19:DeepInfer 版本0.1.0
> https://zhuanlan.zhihu.com/p/2097005233150889999

deepinfer 是 DeepSeek 基于开源的vllm框架自研的大模型推理框架。目前只能注册DeepSeek-V4.1-Flash一个模型。
首先了解一下deepseek v4.1 架构:

> 需要注意的是：相较于传统的transformer架构，他需要在moe后去过一个engram

# 它解决什么问题
官方文档说明：
1. 同一套引擎跑在不同厂商的加速器上。
2. Prefill和Decode按普通机器来组织：Prefill用普通8卡机器，Decode用64～128卡的超节点。
3. KV cache保留超过72小时，让长上下文尽量命中，少做重复Prefill。

个人认为：
- ttft 和 tpot的优化：只通过p/d 分离逐段优化, ttft: 提高kv缓存命中 tpot：dspark
- tps：通过moe 去组大batch
- 

# 模型推理流程
单层计算流大致逻辑：
注意力类型：

compress_ratios = 0的层是滑窗注意力（SWA，sparse_stage = 0）。
compress_ratios > 0的层做KV压缩，并带indexer（index_topk、index_n_heads）。Indexer用来在压缩后的KV里挑块。
第一个ratio = 1的层及其之前是Prefill / encoder侧；它后面的层是YoCo consumer。Decoder侧的block KV由最后一层encoder隐状态投影出来，所以Prefill不必把整段上下文再跑一遍decoder。
attn_module_list在主干层之后留一个空位给LM head，再接num_nextn_predict_layers个DFlash层。
V4.1里面的特性：

Engram：在指定层插入n-gram嵌入表。可以启用压缩词表，查找表会灌进C++调度器。Engram有自己的并行组engram_parallel_size。

Hyper-Connection（HC）：层与层之间用一组可学习的混合系数把残差流拆开再汇合，头上也有对应的HC head。V4.1默认hc_enabled。

多模态模型类是DeepseekVLForCausalLM。视觉塔是ViT，经downsample_mlp_gelu对齐到语言模型隐层。已经被KV完全盖住的图片前缀会换成等长的dummy文本，避免对缓存图像再跑ViT。


# Dspark
按置信度决定校验多长
