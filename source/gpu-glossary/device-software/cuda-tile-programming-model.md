# 什么是 CUDA Tile 编程模型？

CUDA Tile 编程模型是一种面向 NVIDIA GPU 的、基于 tile 的编程模型。

传统的 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model)向用户程序暴露[线程层次结构](/gpu-glossary/device-software/thread-hierarchy)和[内存层次结构](/gpu-glossary/device-software/memory-hierarchy)。用户程序接收指针，并发执行，并相对于这些指针修改内存。同一条指令会并行发射给多个[线程](/gpu-glossary/device-software/thread)，因此这种编程模型是一种“单指令多线程”（SIMT, single-instruction, multiple thread）编程模型。例如，[CUDA C/C++](/gpu-glossary/host-software/cuda-c) 以及面向 NVIDIA GPU 的 CUDA Tile 之前程序所使用的 [PTX](/gpu-glossary/device-software/parallel-thread-execution) IR，都采用这一编程模型。

这一编程模型是为[“统一”的硬件基础](/gpu-glossary/device-hardware/cuda-device-architecture)定义的，也就是 “CUDA” 中的 “U”。换言之，大多数操作由同构的[流式多处理器（SM）](/gpu-glossary/device-hardware/streaming-multiprocessor)以及同构的 [CUDA Core](/gpu-glossary/device-hardware/cuda-core) 实现，而不是像 CUDA 出现之前的图形编程通常那样，由设备中的专用核心以异构方式编程完成。

对于最新 [SM 架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)的 GPU 来说，这种编程模型并不十分契合，因为绝大多数[算术带宽](/gpu-glossary/perf/arithmetic-bandwidth)都来自 [Tensor Core](/gpu-glossary/device-hardware/tensor-core)。Tensor Core 只能执行矩阵乘法，并且必须通过[线程](/gpu-glossary/device-software/thread)级指令和异步机制来编程，而不是通过用于编程其他硬件部分的 [warp](/gpu-glossary/device-software/warp) 级异步机制。

在 CUDA Tile 编程模型中，程序以 _tile-kernel_ 的层级表达。tile-kernel 是在 _tile block_ 网格上并发运行的程序实例，其中每个 tile block 都是一条单独的执行线程。在理想路径下，tile-kernel 操作的是 _结构化指针_：它将指针与数组信息结合起来，包括数组的整体范围（shape）和访问模式（stride）。这与 [CuTe](/gpu-glossary/host-software/cute) 类型系统中的 `Layout` 和 `Tensor` 很相似。

与 CUDA C/C++ 和 PTX IR 中传统的 “CUDA SIMT” 一样，这一编程模型同时被高级语言和中间表示共享；在这里，中间表示是 [Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html)。

截至 2026 年中本文撰写时，CUDA Tile 编程模型仍然很新，它会在多大程度上取代现有的 “CUDA SIMT” 编程模型尚不明确。目前，CUDA Tile 编程模型可通过 [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) 使用。它也可以通过 [cuTile BASIC](/gpu-glossary/host-software/cutile-basic) 和 [cuTile Rust](https://github.com/nvlabs/cutile-rs) 使用，不过后两者仍处于实验形态。
