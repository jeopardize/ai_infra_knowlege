

<!--Q id=q-1791460215543-9evcyk ts=1791460215543-->

**问题 (zh)**

EP专家并行场景下通信开销怎么算

**问题 (en)**



**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887481850-855v ts=1791425068046-->

**问题 (zh)**

在 AI 训练场景中，分布式训练常面临通信瓶颈问题。请说明 AllReduce、Broadcast、ReduceScatter 这三种核心通信算子的适用场景差异，并分析在千亿参数大模型训练中，如何选择通信策略以降低延迟、提升吞吐量？

**问题 (en)**

In AI training scenarios, distributed training often faces communication bottlenecks. Please explain the differences in applicable scenarios for the three core communication operators: AllReduce, Broadcast, and ReduceScatter. Furthermore, analyze how to choose a communication strategy to reduce latency and improve throughput when training a large model with hundreds of billions of parameters.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780887508039-jduf ts=1791362716665-->

**问题 (zh)**

megatron-lm中通信优化怎么做？

**问题 (en)**

How to optimize communication in megatron-lm?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761656-ukvf ts=1791339314496-->

**问题 (zh)**

通信｜什么情况下 PD 分离不划算？（提示：短 prompt，传输固定开销 > 干扰收益。答出"最快的 KV 传输是你永远不必做的那次"会很加分）

**问题 (en)**

Communications | Under what circumstances is PD separation not cost-effective? (Hint: For short prompts, the fixed transmission overhead outweighs the interference mitigation benefit. Answering with "The fastest KV transfer is the one you never have to make" will earn extra points)

**答案 (zh)**

短 prompt 场景：prefill 计算量本身很小，干扰可以忽略，而跨实例/跨节点传输 KV 的固定开销（PCIe/IB 延迟、拷贝开销）占比极高，得不偿失。
小 batch 场景：单卡上 prefill 和 decode 的冲突本身就少，分离带来的收益微弱。
prefill 占比极低的场景：比如纯对话类长 decode 负载，prefill 只在开头出现一次，分离的边际收益很小。

**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761657-eiha ts=1791275996836-->

**问题 (zh)**

通信｜为什么 NCCL 不适合做 KV 传输？至少说三点。

**问题 (en)**



**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761657-da2k ts=1791275996813-->

**问题 (zh)**

通信｜GPUDirect RDMA 的原理和限制条件是什么？

**问题 (en)**

Communication | What are the principles and limitations of GPUDirect RDMA?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761657-cnsi ts=1791275996803-->

**问题 (zh)**

通信｜怎么把 KV 传输延迟藏起来？说出 layer-wise 流水之外的至少一种。

**问题 (en)**

Communications | How can we hide KV transmission latency? Name at least one method other than layer-wise pipelining.

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761658-ycjt ts=1791275996779-->

**问题 (zh)**

通信｜同节点 PD 能否用 NVLink？K8s 下为什么不行？

**问题 (en)**



**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761658-t3um ts=1791275996758-->

**问题 (zh)**

通信｜EP（专家并行）的 AllToAll 和 TP 的 AllReduce，通信量和频率差多少？对网络要求的区别？

**问题 (en)**

Communications | What is the difference in communication volume and frequency between AllToAll in EP (Expert Parallelism) and AllReduce in TP? What are the differences in network requirements?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761658-kzt9 ts=1791275996738-->

**问题 (zh)**

通信｜传输进行中 decode 节点挂了，请求怎么处理？

**问题 (en)**

Communication | If a decode node crashes during transmission, how should the request be handled?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761659-72wg ts=1791275996695-->

**问题 (zh)**

通信｜全局 KV 索引该放哪？router 预测 vs worker 上报，你选哪个，为什么？

**问题 (en)**

Communication | Where should the global KV index be placed? Would you choose router prediction or worker reporting, and why?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1788674761659-87uf ts=1791275996637-->

**问题 (zh)**

通信｜控制面和数据面为什么必须分离？

**问题 (en)**

Communications | Why must the control plane and data plane be separated?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
