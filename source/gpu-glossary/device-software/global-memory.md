# 什么是全局内存？

![](light-cuda-programming-model.svg)  

> 全局内存是 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 中 [内存层次结构](/gpu-glossary/device-software/memory-hierarchy) 的最高层级。它存储在 [GPU 显存](/gpu-glossary/device-hardware/gpu-ram) 中。修改自 NVIDIA 的 [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) 和 NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) 中的图表。

作为 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 的一部分，[线程层次结构](/gpu-glossary/device-software/thread-hierarchy) 的每个层级都可以访问 [内存层次结构](/gpu-glossary/device-software/memory-hierarchy) 中对应的内存。这些内存可用于协调和通信，并由程序员（而非硬件或运行时）管理。

内存层次结构的最高级别是 (Global Memory)。全局内存在其作用域和生命周期上都是全局的。也就是说， [线程块网格](/gpu-glossary/device-software/thread-block-grid) 中的每个 [线程](/gpu-glossary/device-software/thread) 都可以访问它，并且其生命周期与程序执行时间一样长。

类似于 CPU 内存的同步方式，对全局内存中数据结构的访问可以使用原子指令在所有访问者之间同步。在 [协作线程数组](/gpu-glossary/device-software/cooperative-thread-array) 内，可以通过屏障等方式进行更紧密的同步。

[内存层次结构](/gpu-glossary/device-software/memory-hierarchy) 中的这一层级通常在 [GPU 显存](/gpu-glossary/device-hardware/gpu-ram) 中实现，由主机（host）通过 [CUDA Driver API](/gpu-glossary/host-software/cuda-driver-api) 或 [CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api) 提供的内存分配器进行分配。

需要注意的是，"全局" 这个术语与 [CUDA C/C++](/gpu-glossary/host-software/cuda-c) 中的 `__global__` 关键字产生了冲突，该关键字用于标注在主机端启动但在设备端运行的函数（[内核](/gpu-glossary/device-software/kernel)），而全局内存仅位于设备端。早期的 CUDA 架构师 Nicholas Wilt 在他的 [_CUDA Handbook_](https://www.cudahandbook.com/) 中讽刺地指出，这一选择是"旨在给开发者制造最大程度的困惑"。