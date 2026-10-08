# 开发过程中遇到的坑

# Qwen和ds的部署形态不一致



# Kimi-k2 tokenizer多进程加载

## 问题背景：

同时会存在多个进程同时下去执行，Autokenizer.from_pretrained, 其他进程都没有什么问题，只有kimi在大集群下会出现竞态问题

<!--Q id=q-1791458326920-hqz7m1 ts=1791458326920-->

**问题 (zh)**

怎么知道一个batch应该塞多少token？

**问题 (en)**



**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1791425313345-yusdz2 ts=1791425313345-->

**问题 (zh)**

CUDA Graph 和 TP 同时开启卡死的问题，如何解决

**问题 (en)**

How to solve the issue of the system freezing when both CUDA Graph and TP (Tensor Parallelism) are enabled?

**答案 (zh)**



**答案 (en)**



<!--/Q-->
<!--Q id=q-1780925589053-k2g4 ts=1791425145458-->

**问题 (zh)**

如果推理时出现 OOM，你会怎么排查

**问题 (en)**

If an OOM (Out of Memory) error occurs during reasoning, how would you troubleshoot it?

**答案 (zh)**

先看是不是显存被参数、激活、KV Cache 或者临时张量占满，再判断是训练阶段还是推理阶段。推理阶段最常见的原因是上下文太长、batch 太大、KV Cache 没做合理管理，或者某些算子产生了额外的临时开销。排查时一般先缩 batch、缩 context、关掉不必要的 profiling，再逐步定位具体层。很多 OOM 不是模型太大，而是调度策略把显存峰值顶上去了。

**答案 (en)**



<!--/Q-->
