# 什么是 L1 数据缓存？

L1 数据缓存是 [流式多处理器 (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor) 的私有内存。

![](light-gh100-sm.svg)  

> H100 流式多处理器（SM）的内部架构图。浅蓝色部分描绘的是 L1 数据缓存。修改自 NVIDIA 的 [H100 白皮书](https://modal-cdn.com/gpu-glossary/gtc22-whitepaper-hopper.pdf)。

每个 SM 会将该内存分配给调度到其上的 [线程块](/gpu-glossary/device-software/thread-block)。

L1 数据缓存与执行计算的组件（例如 [CUDA 核心](/gpu-glossary/device-hardware/cuda-core)）位于同一位置，其速度仅比这些组件慢约一个数量级。

它采用 SRAM 实现，这种基础半导体单元也用于 CPU 缓存、寄存器以及 [Groq LPU 内存子系统](https://groq.com/wp-content/uploads/2023/05/GroqISCAPaper2022_ASoftwareDefinedTensorStreamingMultiprocessorForLargeScaleMachineLearning-1.pdf)。L1 数据缓存由 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 的 [加载/存储单元](/gpu-glossary/device-hardware/load-store-unit) 访问。

CPU 同样设有 L1 缓存，但是在 CPU 中，该缓存完全由硬件管理。而在 GPU 中，该缓存主要由程序员管理，即使在使用高级语言（如 [CUDA C](/gpu-glossary/host-software/cuda-c)）时也是如此。

H100 的每个 SM 中的 L1 数据缓存可存储 256 KiB（2,097,152 位）。在 H100 SXM 5 的 132 个 SM 中，总计提供 33 MiB（242,221,056 位）的缓存空间。