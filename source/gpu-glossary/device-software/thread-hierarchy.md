# CUDA 线程层次结构是什么？

![](light-cuda-programming-model.svg)  

> [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 的线程层次结构从单个 [线程](/gpu-glossary/device-software/thread) 到 [线程块](/gpu-glossary/device-software/thread-block) 再到 [线程块网格](/gpu-glossary/device-software/thread-block-grid)（左侧），映射到硬件上则从 CUDA [核心](/gpu-glossary/device-hardware/core) 到 [流式多处理器](/gpu-glossary/device-hardware/streaming-multiprocessor) 再到整个 GPU（右侧）。改编自 NVIDIA 的 [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) 和 NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) 中的图表。

线程层次结构是 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 的核心抽象概念之一，与 [内存层次结构](/gpu-glossary/device-software/memory-hierarchy) 并列。它通过多个层级组织并行程序的执行，从单个线程一直扩展到整个GPU设备。

层次结构中最底层是单个 [线程](/gpu-glossary/device-software/thread)。与 CPU 上的执行线程类似，每个 [CUDA 线程](/gpu-glossary/device-software/thread) 执行一系列指令 (a stream of instructions) 。负责执行算术和逻辑指令的硬件资源称为 [核心](/gpu-glossary/device-hardware/core) 或有时称为 "流水线"。线程由 [线程束调度器](/gpu-glossary/device-hardware/warp-scheduler) 选择执行。

中间层级由 [线程块](/gpu-glossary/device-software/thread-block) 组成，在 [PTX](/gpu-glossary/device-software/parallel-thread-execution) 和 [SASS](/gpu-glossary/device-software/streaming-assembler) 中也称为 [协作线程数组](/gpu-glossary/device-software/cooperative-thread-array)。每个 [线程](/gpu-glossary/device-software/thread) 在其所属的 [线程块](/gpu-glossary/device-software/thread-block) 中都有唯一标识符。这些标识符基于索引，便于根据输入或输出数组的索引为线程分配任务。一个块内的所有线程会被同时调度到同一个 [流式多处理器 (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor) 上。它们可以通过 [共享内存](/gpu-glossary/device-software/shared-memory) 进行协调，并通过屏障进行同步。

在最高层，由多个 [线程块](/gpu-glossary/device-software/thread-block) 被组织成一个跨越整个 GPU 的 [线程块网格](/gpu-glossary/device-software/thread-block-grid)。[线程块](/gpu-glossary/device-software/thread-block) 在协调和通信方面受到严格限制。网格内的块以并发方式执行，没有固定的执行顺序。[CUDA 程序](/gpu-glossary/device-software/cuda-programming-model) 必须确保块的任何执行顺序（从完全串行到完全并行）都是有效的。这意味着 [线程块](/gpu-glossary/device-software/thread-block) 之间不能通过屏障同步。与 [线程](/gpu-glossary/device-software/thread) 类似，每个 [线程块](/gpu-glossary/device-software/thread-block) 都有一个唯一的、基于索引的标识符，以支持基于数组索引的分配任务。

这种层次结构直接映射到 [GPU 硬件](/gpu-glossary/device-hardware)： [线程](/gpu-glossary/device-software/thread) 在单个 [核心](/gpu-glossary/device-hardware/core) 上执行， [线程块](/gpu-glossary/device-software/thread-block) 被调度到 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 上， 而 [网格](/gpu-glossary/device-software/thread-block-grid) 则利用设备上所有可用的 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor)。