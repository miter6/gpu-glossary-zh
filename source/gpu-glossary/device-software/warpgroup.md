# 什么是线程束组 ？

Warpgroup（线程束组）是指四个连续 [线程束 (warps)](/gpu-glossary/device-software/warp) 的集合，且该组中第一个线程束的秩（warp-rank）必须是 4 的倍数。


在分发 Warpgroup 级别的指令时，系统会协同 128 个 [线程](/gpu-glossary/device-software/thread) —— 即每个 Warpgroup 包含 4 个线程束，每个线程束包含 32 个线程。以更大的粒度进行操作，消除了显式线程束间同步的需求，并允许单条指令处理更大规模的问题，特别是更大的矩阵乘法。更大规模的矩阵乘法能够更容易地使近期数据中心 GPU 中 [Tensor Core (张量核心)](/gpu-glossary/device-hardware/tensor-core) 的海量 [算术带宽](/gpu-glossary/perf/arithmetic-bandwidth) 达到饱和。

Warpgroup 引入于 NVIDIA 的 Hopper [流式多处理器（SM）架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 中，用于支持 Warpgroup 级别的矩阵乘法，例如 `wgmma.mma_async` 指令。欲深入了解，请参阅 [Colfax 的这篇博客文章](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)。Warpgroup 在高性能 Hopper 和 Blackwell [内核（kernels）](/gpu-glossary/device-software/kernel)（如 [Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)）的流水线组件组织中占据重要地位。


在 [并行线程执行 (PTX)](/gpu-glossary/device-software/parallel-thread-execution) 中间表示（IR）中，一个线程束的秩（warp-rank）定义为：

```cpp
int linearIdx = (%tid.x + %tid.y * %ntid.x  + %tid.z * %ntid.x * %ntid.y);
int warpRank = linearIdx / 32;
```

其中 `tid` 是线程索引，通过特殊的 PTX [寄存器](/gpu-glossary/device-software/registers) 进行访问。

因此，对于包含 8 个线程束的调度，有效的 Warpgroup 分别为：

- **Warpgroup 0**: 秩为 0, 1, 2 和 3 的线程束
- **Warpgroup 1**: 秩为 4, 5, 6 和 7 的线程束

据我们需要，关于这种线程束秩（warp-rank）对齐限制的目的尚无官方文档说明。但近期数据中心 GPU 的 [流式多处理器（SM）](/gpu-glossary/device-hardware/streaming-multiprocessor) 似乎包含四个（未命名的）子单元，每个子单元都拥有独立的 [线程束调度器](/gpu-glossary/device-hardware/warp-scheduler) 和 Tensor Core。
