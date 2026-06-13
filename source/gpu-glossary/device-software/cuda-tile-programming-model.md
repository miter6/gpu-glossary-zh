# 什么是 CUDA Tile 编程模型？

CUDA Tile 编程模型是一种面向 NVIDIA GPU 的、基于分块（Tile-based）的编程模型。

传统的 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 向用户程序展现了[线程层次结构](/gpu-glossary/device-software/thread-hierarchy)和[内存层次结构(/gpu-glossary/device-software/memory-hierarchy)]。用户程序接收指针并并发执行，从而使用这些指针来改变内存状态。由于相同的指令被并行地发射给多个线程 ([threads](/gpu-glossary/device-software/thread))，因此这种编程模型被称为 “单指令多线程”（SIMT）编程模型。例如，在 [CUDA C/C++](/gpu-glossary/host-software/cuda-c) 以及早期面向 NVIDIA GPU 的程序所使用的 [PTX](/gpu-glossary/device-software/parallel-thread-execution) 中间表示（IR）中，采用的正是这种编程模型。

该编程模型是为“统一”的[硬件底层架构](/gpu-glossary/device-hardware/cuda-device-architecture)（即 CUDA 中的 “U”，Unified）而定义的。也就是说，由同质的流式多处理器 ([Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)) 和同质的 CUDA 核心 ([CUDA Cores](/gpu-glossary/device-hardware/cuda-core)) 来实现绝大多数操作，而不是像 CUDA 问世前的通用图形编程那样，由性质各异的专用核心组成并进行异构编程。

该编程模型不适合最新的 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 架构的 GPU。在这些新型 GPU 中，绝大部分的算术带宽 [Arithmetic Bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) 都集中在张量核心 ([Tensor Cores](/gpu-glossary/device-hardware/tensor-core)) 上。Tensor Core 只能执行矩阵乘法，并且必须通过线程（[thread](/gpu-glossary/device-software/thread)）级的指令和异步性来进行编程，这与用于对硬件其余部分进行编程的线程束 ([warp](/gpu-glossary/device-software/warp)) 级异步性大不相同。

在 CUDA Tile 编程模型中，程序是在 Tile 核函数 (Tile-kernels) 的级别上进行表达的。这些核函数是程序实例，在由 Tile 块 (Tile blocks) 组成的网格（Grid）上并发运行，其中每个 Tile 块都是一个独立的执行线程。在理想情况下（Happy path），Tile 核函数运行在 结构化指针 (Structured pointers) 上，这种指针将传统指针与数组的信息结合在了一起：包括其总维度（Shape）和访问模式（Stride）。这与 [CuTe](/gpu-glossary/host-software/cute) 类型系统中用于 `Layout` 和 `Tensor` 的机制非常相似。

与 CUDA C/C++ 和 PTX IR 中的传统 “CUDA SIMT” 一样，这种编程模型同样由高级语言和中间表示所共享——在这里，对应的中间表示是 [Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html).。

截至 2026 年中旬撰写本文时，CUDA Tile 编程模型还是一项新技术，它将在多大程度上取代现有的 “CUDA SIMT” 编程模型目前尚不明朗。
CUDA Tile 编程模型目前可参考 [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) 。此外，它也以实验性版本集成在 [cuTile BASIC](/gpu-glossary/host-software/cutile-basic) 和 [cuTile Rust](https://github.com/nvlabs/cutile-rs) 中。