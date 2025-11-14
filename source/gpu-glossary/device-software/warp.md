# 什么是线程束？

线程束 (Warp) 是一组被共同调度并并行执行的线程 [线程](/gpu-glossary/device-software/thread)。一个线程束中的所有 [线程](/gpu-glossary/device-software/thread) 都被调度到单个 [流式多处理器 (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor) 上执行。而单个 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 通常会执行多个线程束，至少会运行来自同一个 [协作线程数组](/gpu-glossary/device-software/cooperative-thread-array)（也称为[线程块](/gpu-glossary/device-software/thread-block)）的所有线程束。

线程束是 GPU 上典型的执行单元。在正常执行过程中，线程束的所有 [线程](/gpu-glossary/device-software/thread) 会并行执行相同的指令，这就是所谓的 "单指令多线程" 或 SIMT 模型。当线程束中的 [线程](/gpu-glossary/device-software/thread) 因执行不同指令而产生分支（也称为 [线程束分歧](/gpu-glossary/perf/warp-divergence)）时，性能通常会急剧下降。

从技术上讲，线程束大小是一个依赖于硬件的常数，但在实际应用中（以及本术语表的其他部分），其大小为32。

当一个线程束被分配指令后，通常无法在单个时钟周期内获得结果，因此无法立即执行依赖该结果的后续指令。这一点在从 [全局内存](/gpu-glossary/device-software/global-memory) （通常位于 [芯片之外](/gpu-glossary/device-hardware/gpu-ram)）读取数据时尤为明显，但某些算术指令也存在类似情况（请参阅 [CUDA C++ 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions) 了解具体指令的时钟周期结果表）。

若线程束的下一条指令因操作数未就绪而延迟执行，则称其处于 [停滞(stalled)](/gpu-glossary/perf/warp-execution-state) 状态。

当多个线程束被调度到同一个 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 上时，[线程束调度器](/gpu-glossary/device-hardware/warp-scheduler) 不会等待前一条指令的结果返回，而是选择另一个可执行的线程束执行。这种 [延迟隐藏](/gpu-glossary/perf/latency-hiding) 机制是 GPU 实现高吞吐量并确保执行期间所有核心始终有任务可处理的方式。因此，最大化调度到每个 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 上的线程束数量通常是有益的，确保始终有 [符合条件的](/gpu-glossary/perf/warp-execution-state) 线程束供 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 运行。线程束被分配指令的周期比例称为 [发射效率](/gpu-glossary/perf/issue-efficiency)。线程束调度中的并发程度称为 [占用率](/gpu-glossary/perf/occupancy)。

线程束实际上并不是 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 的 [线程层次结构](/gpu-glossary/device-software/thread-hierarchy) 中的一部分。相反，它们是在 NVIDIA GPU 上实现该模型的实现细节。从这个意义上说，它们有点类似于 CPU 中的 [缓存行](https://www.nic.uoregon.edu/~khuck/ts/acumem-report/manual_html/ch03s02.html)：一种您不直接控制且不需要考虑程序正确性的硬件特性，但对于实现 [最佳性能](/gpu-glossary/perf/index.rst) 很重要。

“Warp” 一词的命名源自编织工艺，根据 [Lindholm 等人，2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf) 论文所述，编织是 "最早的并行线程技术"。其他 GPU 编程模型中与线程束等效的概念包括 WebGPU 中的[subgroups](https://github.com/gpuweb/gpuweb/pull/4368)、DirectX 中的[waves](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html) 以及 Metal 中的[simdgroups](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups#2928931)。