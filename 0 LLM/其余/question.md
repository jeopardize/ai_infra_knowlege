

<!--Q id=q-1780901854289-aj92 ts=1791425111507-->

**问题 (zh)**

什么 RMSNorm 在大模型里比 LayerNorm 更常见

**问题 (en)**

Why is RMSNorm more common than LayerNorm in large models.

**答案 (zh)**

RMSNorm 去掉了均值中心化，只保留方差归一化，计算更简单，开销更低，而且在大模型里通常足够稳定。相比 LayerNorm，它少了一部分计算和同步开销，尤其在大规模训练和推理里更有工程优势。很多模型选择 RMSNorm，不是因为它理论上一定更强，而是因为它在稳定性和效率之间更平衡。工程里常见的选择逻辑就是：只要效果差不多，就选更省的。

**答案 (en)**



<!--/Q-->
