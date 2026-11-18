# 20261118
## Evolving Infra: Agent-Adaptive LLM Serving at Scale
codehub: https://github.com/mit-han-lab/EvolvingInfra
> 标签：infra effective P/D split change

**背景**：现有存在pd配比都是固定好的，例如：A3 6p1d，当发现d变为瓶颈时候会从架构上变为4p1d，但是这种瓶颈的发现通常是针对一个特定话务模型摸测后的结果，而且上了现网后很难发生额外一些变更；
**问题**：1.但是现在一些ai 过度依赖，大家很容易就会想到用一些lstm这些轻量模型去做一些预测，这些是否过于沉重 2.我如果将中间一个节点突然从p转换为d，那么模型权重和预热的工作都需要全部重新做，kv cache的缓存也会同步丢失
