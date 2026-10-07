

<!--Q id=q-1780901766052-zma3 ts=1791362164614-->

**问题 (zh)**

讲一下 FlashAttention 的原理

**问题 (en)**

Let me explain the principle of FlashAttention.

**答案 (zh)**

FlashAttention 的核心不是“更快算 attention”，而是“避免把完整 attention matrix 落到显存里”。它通过分块计算、在线 softmax 和重计算，减少 HBM 访问量，把很多中间结果保留在更快的片上存储里。这样做的本质是用更少的内存带宽换更好的速度和更低的显存占用。长序列场景下，它的收益非常明显，因为传统 attention 的瓶颈往往就是读写中间矩阵。
除此之外，flash attention v1，v2，v3 分别都解决了不同的问题

**答案 (en)**



<!--/Q-->
