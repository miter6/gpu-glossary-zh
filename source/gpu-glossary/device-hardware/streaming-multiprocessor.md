# 什么是流式多处理器？（SM）

当我们 [对 GPU 进行编程](/gpu-glossary/host-software/cuda-software-platform) 时，会生成供其流式多处理器 (Streaming Multiprocessor) 执行的[指令序列](/gpu-glossary/device-software/streaming-assembler)。

![](light-gh100-sm.svg)

> H100 GPU 流式多处理器的内部架构示意图。图中绿色部分为 GPU 核心，褐红色部分为其他计算单元，橙色部分为调度单元，蓝色部分为内存。修改自 NVIDIA 的 [H100 白皮书](https://modal-cdn.com/gpu-glossary/gtc22-whitepaper-hopper.pdf)。

NVIDIA GPU 的流式多处理器（SMs）大致类似于 CPU 的核心。也就是说，SMs 既执行计算，又在寄存器中存储可供计算使用的状态，并配有相关缓存。与 CPU 核心相比，GPU SMs 是简单、功能较弱的处理器。SMs 中指令的执行采用流水线方式（类似于 20 世纪 90 年代以来的几乎所有 CPU），但没有推测执行或指令指针预测（这与所有当代高性能 CPU 不同）。

不过，GPU 的流式多处理器（SMs）能够并行执行更多 [线程](/gpu-glossary/device-software/thread)。

作为对比：一颗 [AMD EPYC 9965](https://www.techpowerup.com/cpu-specs/epyc-9965.c3904) CPU 的最大功耗为 500 W，拥有 192 个核心，每个核心最多可以同时为两个线程执行指令，总共可并行执行 384 个线程，每个线程的运行功耗约为 1.25 W。

而一颗 H100 SXM GPU 最大功耗为 700 W，配置 132 个 SM，每个 SM 有四个 [线程束调度器 (Warp Scheduler)](/gpu-glossary/device-hardware/warp-scheduler)，每个调度器每个时钟周期可以向 32 个线程（也称为一个 [线程束 (Warp)](/gpu-glossary/device-software/warp)） 并行发出指令，总共可并行运行超过 128 × 132 > 16,000 个线程，每个线程的功耗约为 5 cW。请注意，这是真正的并行：16,000 个线程中的每一个线程都可以在每个时钟周期执行操作。

GPU 的流式多处理器（SMs） 还支持大量 *并发* 线程 —— 即指令交错执行的执行线程。

H100 上的单个 SM 最多可并发执行多达 2048 个并发线程，这些线程分布在 64 个线程组中，每个线程组包含 32 个线程。配置 132 个 SM 后，GPU 总共可支持超过 250,000 个并发线程。

CPU 也能并发运行大量线程。但是 [线程束 (Warp)](/gpu-glossary/device-software/warp) 之间的切换速度仅需一个时钟周期（比 CPU 的上下文切换快 1000 倍以上），这一能力由 SM 的 [线程束调度器 (Warp Scheduler)](/gpu-glossary/device-hardware/warp-scheduler) 支持。丰富的可用 [线程束 (Warp)](/gpu-glossary/device-software/warp) 资源与快速的 [线程束切换 (Warp Switch)](/gpu-glossary/device-hardware/warp-scheduler) 速度有助于掩盖内存读取、线程同步或其他高开销指令带来的 [延迟 (Latency Hiding)](/gpu-glossary/perf/latency-hiding)，从而确保由 [CUDA 核心 (CUDA Core)](/gpu-glossary/device-hardware/cuda-core) 和 [张量核心 (Tensor Core)](/gpu-glossary/device-hardware/tensor-core) 提供的 [算术带宽 (Arithmetic Bandwidth)](/gpu-glossary/perf/arithmetic-bandwidth) 得到充分利用。

这种 [延迟隐藏 (Latency Hiding)](/gpu-glossary/perf/latency-hiding) 正是 GPU 优势的核心。 CPU 则通过维护大型硬件管理缓存和复杂的指令预测来向终端用户和程序员隐藏延迟，而这些额外硬件会占用 CPU 的硅片面积、功耗和散热预算，从而限制了可用于计算的资源比例。

![](light-cpu-vs-gpu.svg)

> 与 CPU 相比，GPU 将更多的芯片面积用于计算（绿色部分），而用于控制和缓存（橙色和蓝色部分）的面积则更少。修改自 [Fabien Sanglard 博客](https://fabiensanglard.net/cuda) 中的图表，该图表本身可能修改自 [CUDA C 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 中的图表。

对于神经网络推理或数据库顺序扫描这类程序或函数，程序员可以相对容易地 [描述](/gpu-glossary/device-software/cuda-programming-model) [缓存](/gpu-glossary/device-hardware/l1-data-cache) 的行为（例如，存储每个输入矩阵的一部分并将其在缓存中保留足够长时间以计算相关输出），其结果是显著更高的吞吐量。