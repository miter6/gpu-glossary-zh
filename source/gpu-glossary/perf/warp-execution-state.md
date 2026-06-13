# 什么是线程束执行状态？

运行 [内核](/gpu-glossary/device-software/kernel) 的 [线程束](/gpu-glossary/device-software/warp) 状态可通过多个非互斥的形容词来描述：活跃的（active）、停滞的（stalled）、就绪的（eligible）和已选择的（selected）。

![](light-cycles.svg)

> 线程束的执行状态通过颜色标识。图表灵感来源于 GTC 2025 的 [*CUDA Techniques to Maximize Compute and Instruction Throughput*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) 演讲。

[线程束](/gpu-glossary/device-software/warp) 从其 [线程](/gpu-glossary/device-software/thread) 开始执行到所有 [线程](/gpu-glossary/device-software/thread) 都从 [内核](/gpu-glossary/device-software/kernel) 退出为止，该 [线程束](/gpu-glossary/device-software/warp) 均被认为是 *活跃的 (active)*。活跃的 [线程束](/gpu-glossary/device-software/warp) 构成了一个资源池，[线程束调度器](/gpu-glossary/device-hardware/warp-scheduler) 每个周期从中选择候选者来发射指令（即放入某个发射槽中）。

每个 [流式多处理器 (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor) 上活跃的 [线程束](/gpu-glossary/device-software/warp) 最大数量因 [架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 而异，具体可参考 [NVIDIA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=compute%2520capability#compute-capabilities) 中的 [计算能力](/gpu-glossary/device-software/compute-capability) 章节。例如，在具有 [计算能力](/gpu-glossary/device-software/compute-capability) 9.0 的 H100 SXM GPU 上，每个 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 最多可容纳 64 个活跃的 [线程束](/gpu-glossary/device-software/warp)（2048 个线程）。需要注意的是，活跃的 [线程束](/gpu-glossary/device-software/warp) 不一定正在执行指令。在上图中，除了一个槽位+周期外，其余所有槽位+周期都有活跃的 [线程束](/gpu-glossary/device-software/warp) —— 这表明了高 [占用率](/gpu-glossary/perf/occupancy)。

*就绪的 (eligible)* [线程束](/gpu-glossary/device-software/warp) 是指准备好发射下一条指令的活跃的 [线程束](/gpu-glossary/device-software/warp)。要使一个 [线程束](/gpu-glossary/device-software/warp) 变成就绪状态，必须满足以下所有条件：

- 已获取下一条指令，
- 所需的执行单元可用，
- 所有指令依赖关系已解析，并且
- 无同步屏障阻碍执行。

就绪的 [线程束](/gpu-glossary/device-software/warp) 是 [线程束调度器](/gpu-glossary/device-hardware/warp-scheduler) 可以立即进行指令发射的候选对象。在上图中，除了 n + 2 周期之外的所有周期均存在就绪的 [线程束](/gpu-glossary/device-software/warp)。若多个周期内没有就绪的 [线程束](/gpu-glossary/device-software/warp) 可能会对性能造成负面影响，特别是当您主要使用像 [CUDA 核心](/gpu-glossary/device-hardware/cuda-core) 这样的低延迟算术单元时。

*停滞的 (stalled)* [线程束](/gpu-glossary/device-software/warp) 是指因未解决的依赖关系或资源冲突而无法发射其下一条指令的活跃的 [线程束](/gpu-glossary/device-software/warp)。[线程束](/gpu-glossary/device-software/warp) 停滞的原因多种多样，包括：

- 执行依赖，必须等待先前算术指令的结果，
- 内存依赖，必须等待先前内存操作的结果，
- 流水线冲突，执行资源当前被占用。

当线程束因访问共享内存或因执行长时间运行的算术指令而停滞时，我们称其停滞在"短计分板（short scoreboard）"上。当因访问 GPU 内存而停滞时，则称为停滞在"长记分板（long scoreboard）"。 这两种停顿都被称为记分板停滞 ([Scoreboard Stalls](/gpu-glossary/perf/scoreboard-stall))。

在上图中，每个周期的多个槽位中都出现了停滞的 [线程束](/gpu-glossary/device-software/warp)。停滞的 [线程束](/gpu-glossary/device-software/warp) 本身并不一定是坏事——大量并发停滞的 [线程束](/gpu-glossary/device-software/warp) 可能是 [隐藏延迟](/gpu-glossary/perf/latency-hiding) 所必需的，这些延迟来自长时间运行的指令，如内存加载或像 `HMMA` 这样的 [张量核心](/gpu-glossary/device-hardware/tensor-core) 指令，这些指令 [可能运行数十个周期](https://arxiv.org/abs/2206.02874)。

*已选择的 (selected)* [线程束](/gpu-glossary/device-software/warp) 是指在当前周期已被 [线程束调度器](/gpu-glossary/device-hardware/warp-scheduler) 选中接收指令的就绪 [线程束](/gpu-glossary/device-software/warp) 。每个周期，[线程束调度器](/gpu-glossary/device-hardware/warp-scheduler) 都会查看其就绪 [线程束](/gpu-glossary/device-software/warp) 资源池，如果存在任何符合条件的线程束，则选择一个并向其发射一条指令。每个存在就绪 [线程束](/gpu-glossary/device-software/warp) 的周期中，都有一个已选择的 [线程束](/gpu-glossary/device-software/warp)。在 [活跃周期](/gpu-glossary/perf/active-cycle) 中，某个 [线程束](/gpu-glossary/device-software/warp) 被选中并发射指令的比例就是 [发射效率](/gpu-glossary/perf/issue-efficiency)。