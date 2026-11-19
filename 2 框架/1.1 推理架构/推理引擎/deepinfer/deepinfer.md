 
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
- ttft 和 tpot的优化：只通过p/d 分离逐段优化
- tps：通过moe 去组打batch，以及通过高kv缓存命中提升效率

# 模型推理流程
单层计算流大致逻辑：



