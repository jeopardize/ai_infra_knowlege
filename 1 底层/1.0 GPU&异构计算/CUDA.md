# GPU架构

cuda 和 算子什么关系？
CUDA：是 NVIDIA 推出的并行计算平台+编程模型，提供 GPU 的底层计算能力，用来编写 GPU 程序，调度 GPU 线程、内存、执行计算。
算子（Operator）：深度学习里的算子，是完成一个具体计算操作的单元，比如卷积、矩阵乘法、ReLU、Softmax 等，是算法层面的概念。
两者的联系
算子需要 CUDA 来实现硬件加速：很多深度学习算子（卷积、矩阵乘）会写 CUDA kernel，把算子计算放到 GPU 上并行执行，利用 GPU 大量 CUDA Core 加速运算。
CUDA 是算子的底层实现工具：算子是一个逻辑计算功能；CUDA 是把这个算子逻辑映射到 GPU 硬件上运行的手段。
举例：PyTorch 的 torch.nn.Conv2d 卷积算子，底层可以调用 CUDA 编写的卷积 kernel，在 GPU 上跑；如果不用 CUDA，算子只能在 CPU 上串行执行。

## SM & warp



warp详细讲解文档： https://www.cnblogs.com/1024incn/p/4541313.html



## Tensor Core

## HBM/cache

## 



# CUDA

# 性能优化

## Memory bandwidth



