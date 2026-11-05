> [1]黄曼蒂,李韬,杨惠,等.以太网RDMA网卡综述[J].计算机研究与发展,2025,62(5):1262-1289.
> https://zhuanlan.zhihu.com/p/1982525738998588869


1. 背景：以太网的传输速率 突破400Gps，朝着800Gps，1.6Tps发展，cpu的处理速度无法跟上
2. 定义：== RDMA （remote direct memory access）采用零拷贝（zero copy），内核旁路（kernel bypass）和处理器功能卸载（cpu offload）等思想 ==，可以远程直接内存访问，不需要CPU参与的情况下直接读下远端内存，实现高带宽低延迟的数据传输
3. 实现方式：IB（InfiniBand），RoCE（RDMA over coveraged Ethernet）和 iWarp   --*本质是网络协议和架构*
  - IB 性能最好，但自定义了全链路层的整套规范，需要专用网卡和交换机支持，不兼容以太网且价格昂贵
  - ROCE v1 将 IB 的链路层替换为以太网协议，ROCE v2 在 v1 基础上进一步将网络层替换为UDP/IP协议，== 允许使用标准以太网交换机进行跨网路由，只需要额外增设一个专用网卡支持 ==
  - iWarp 同样兼容以太网，但需将整个TCP/IP协议栈卸载到硬件
  
== 在RDMA理念中，核心的两个抽象是HCA（host channel adapter，也就是网卡）和QP（队列对）。其中HCA指的是硬件，QP则是软件。HCA内部通过硬件独立处理网络协议栈，内存注册，数据包发送和接收。HCA可以通过PCIe总线连接到host或者device的内存，应用程序通过HCA暴露的内存映射寄存器（MMIO）控制HCA。==

== QP这个软件概念，一般是由两部分组成，SQ和RQ（send queue和receive queue）。通信之前，机器之间通过某些建联机制建立起QP，交换各自的QP信息，SQ和RQ的关系并非是想象那样从SQ出发的数据进入对方的RQ，这两个queue之间没有直接的数据交流，而是通过实际传输的网络包构建的逻辑关系。发送方对自己的SQ放入信息，告诉网卡要发送的数据在哪里，网卡把数据打包发出去。接收方则是把自己要接收的数据信息放在RQ告诉网卡，网卡会预留内存等待数据到达，这个就是双边通信。单边通信则是不需要接收方在RQ中放WR。==

== 应用程序和HCA之间沟通的结构称为WR（work request），而HCA实际操控硬件单元完成各种操作，需要的是更贴近硬件的数据结构，称为WQE（work queue entry）。==


	

# RDMA 传输流程

## RDMA 传输准备阶段

1. rdma 建立连接，rdma 允许本端用户态应用程序快速写远端内存。
> 对比
> - 无rdma
> a gpu → gpu memory → cudaMemcpy → host memory → TCP/IP stack → NIC → network → NIC → host memory → cudaMemcpy → remote gpu memory
> - 没有GPUDirect + rdma：
> gpu memory → cudaMemcpy → host pinned memory → RDMA NIC → 网络 → 远端 NIC → 远端 host memory
> - 有GPUDirect+rdma：
> gpu memory → NIC → 网络 → NIC → 远端 gpu memory

因此，在正式通信之前，rdma本端引用程序需要首先申请注册内存区域（memory region， MR）用于存放收发数据，并与通信远端进行建链协商，提前获取远端主机信息和内存访问权限等（需要cpu参与）

> iWARP 和 ROCE v1 基于 CM（communication manager）接口建联，ROCE v2 同时机遇 CM 接口和Socket接口2中建链方式
> ** CM 和 Socket 区别：**
> socket需要自己写数据传输和控制层，而CM 已经封装好
> 

2.RDMA通信原语，RDMA通过verbs 变成编程接口为上层应用提供两种通信原语，分别为双边通信原语和单边通信原语
- 双边通信原语 RECV 和 SEND ：需要远端CPU在本端SEND发送消息前提前下发RECV制定接受地址
- 单边通信原语 READ，WRITE，ATOMIC则不涉及远端CPU参与，可以获得更低的延迟和更高的带宽


> verbs 就是 RDMA 的标准底层编程接口（libibverbs），给应用程序一套原语，直接操控 RNIC（RDMA 网卡）硬件，绕开内核协议栈，实现远端内存直接读写。
> 此外，SEND 和 WRITE可以携带额外的4B的数据，改数据不直接写入内存，而是作为状态信息返回给上层应用（CPU参与）

## RDMA正式传输：
待链路成功建立后，随即进入 RDMA 数据传输工作流程：

1.应用程序通过RDMA以工作请求（work request，WR）的形式下发post远端访问请求，报文如[@fig:ROCEv2_write]所示
![Rocev2_write](images/Rocev2_write.png){#fig:ROCEv2_write}

2. 对于应用下发的 WR，由用户态驱动将其转化为工作队列元素 （work queue element，WQE）并放入队列对（queue pair，QP）中（[@fig:Rocev2_process] 中①）。
> QP 是 RDMA 双端通信的基本单位，由发送队列 （send queue，SQ）和接收队列（receive queue，RQ）组成，
> 具体来说：由 SQ 接收应用程序下发的 SEND，READ，WRITE 等请求，由 RQ 接收应用程序下发的 RECV（对应 SEND 请求）。 对于 RDMA WRITE 操作，由网卡在获取 RDMA 传输请（[@fig:Rocev2_process] 中②）后取出本端内存数据（[@fig:Rocev2_process] 中③）并执行远端内存写入（[@fig:Rocev2_process] 中④。


QP可以通过一下四种方式进行通信，包括

|类型 | 描述 | 消息大小 | 适用 |
| --- | --- | --- | --- |
|Reliable Connected（RC）| 可靠、有序、点对点 |	<= 2^21| 最常用
|Unreliable Connected（UC）| 不可靠、有序、会丢包 | <= 2^21|	
|Reliable Datagram（RD）	| 可靠、无序 | | multicast
|Unreliable Datagram（UD）| 不可靠、无序|<= 4 | 路由管理|

3. 数据完成传输后，网卡生成完成队列元素（completion queue element，CQE）并将之放入与 QP 关联的完成队（completion queue，CQ）中（[@fig:Rocev2_process] 中⑤），最终应用程序由网卡上送（poll）的工作完成 （work completion，WC）获知 RDMA 传输完成[@fig:Rocev2_process] 中⑥）。

![Rocev2_process](images/Rocev2_process.png){#fig:Rocev2_process}

# RDMA 网卡架构
包含一下7个部件，架构图如{@fig:rdma_arch}：
- 用于连接主机端的CPU和内存的PCIe部件
- 用于连接网络端的MAC部件和PHY部件
- 用于处理本端或远端不同通信原语对应的RDMA请求和响应，进行保温封装和解析的保温处理逻辑部件
- 用于直接访问主机内存的虚实地址转换，DMA部件
- 用于支持RC传输的可靠性传输部件
- 用于存储数据和元数据信息的网卡SRAM（static random access memory）缓存
- 不同网卡架构中还可能有其他可选部件，可以在存储资源，可靠传输，应用方面进行优化
![rdma_arch](images/rdma_arch.png){#fig:rdma_arch}

# RDMA和host/device的交互
RDMA虽然想要bypass CPU，但是不能完全不和CPU交互。和CPU交互主要有两个机制：反馈和内存注册。反馈通过CQ（completion queue）完成。SQ/RQ上的WQE被完成了，HCA都会生成一个CQE（completion queue element）放在CQ中，用于记录哪个QP上的操作完成了。应用程序想要知道自己的通信是否完成，就要不断地poll CQ查看是否有新的CQE（CPU自旋在这个等待上）。HCA也支持中断事件，在CQ出现新的CQE的时候就去中断一下CPU（CPU之前可以异步作别的事情）。两种方法各有优缺点，polling方式浪费CPU，但是获取CQE延迟最小。中断方式解放CPU，但是获取CQE延迟大。

RDMA需要能访问host内存，内存注册（memory registration, MR）提供了安全的访问机制。注册通过HCA驱动完成，这块内存需要物理上被锁住（禁止换页），这块内存对应生成多个TPT（translation and protection table）条目，记录虚拟地址、物理地址、访问权限。注册后，会生成一个Memory Key返回给应用程序。Key有两个：LKey（local key）和RKey（remote key）,LKey是自己访问自己的内存，需要放在WR里的，RKey是远端访问本地内存需要提供的key。

对于device来说，也可以直接由HCA注册内存，这样通信就无需把数据从device内存搬运到host内存。比如NV提供的GPUDirect支持，就是一项允许HCA直接访问GPU内存的技术。

# RDMA编程基础

通信时，资源的创建必须按照顺序：

|顺序 | 资源名称 | 作用 | API/概念 |
| --- | --- | --- | --- |
1 | 上下文 (Context)	| 表示 HCA 本身，是所有操作的锚点。| ibv_context (通过 ibv_open_device 获取) |
2 | 保护域 (PD) | 内存安全沙箱，所有内存和队列资源必须绑定到 PD。| ibv_alloc_pd |
3 | 完成队列 (CQ) | HCA 用来报告事件的队列 | ibv_create_cq |
4 | 内存区域 (MR) | 注册 GPU 或主机内存，实现零拷贝。	| ibv_reg_mr |
5 | 队列对 (QP) | 双向通信通道 (SQ + RQ)。| ibv_create_qp |





# RDMA 网卡优化
分为三大类：
1. 存储资源优化
> rdma 硬件卸载和内核旁路等特性要求在网卡中缓存QPC，MTT等元数据信息以维护QP连接状态和进行直接内存访问等。

2. 可靠性优化
   - 拥塞优化：基于ECN，RTT，INT
   - 流量控制
   - 重传控制
3. 应用相关优化
   - 分布式存储：存储类型，存储应用
   - 数据中心应用：性能隔离，通用优化

# 网卡评估
指标：带宽，延迟，每秒消息数（messages per second， MPS），cpu利用率
- 运行一组代表性应用程序工作负载
- 使用专业性能评估工具，perftest
