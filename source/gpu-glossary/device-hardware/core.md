# 什么是 GPU 核心？

核心是构成 [流式多处理器 (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor) 的主要计算单元。

![](light-gh100-sm.svg)  

> H100 GPU 流式多处理器的内部架构示意图。图中绿色部分为 GPU 核心，褐红色部分为其他计算单元，橙色部分为调度单元，蓝色部分为内存。修改自 NVIDIA 的 [H100 白皮书](https://modal-cdn.com/gpu-glossary/gtc22-whitepaper-hopper.pdf)。

GPU 核心类型的举例包括 [CUDA 核心](/gpu-glossary/device-hardware/cuda-core) 和 [Tensor 核心](/gpu-glossary/device-hardware/tensor-core)。

尽管 GPU 核心与 CPU 核心在 “执行实际计算的组件” 这一点上具有可比性，但这种类比可能存在较大误导性。或许，从 [量化计算机架构师](https://archive.org/details/computerarchitectureaquantitativeapproach6thedition) 的角度出发，将它们视为数据流入并返回转换后数据的"管道"可能更有帮助。从硬件角度来看，这些管道与特定的 [指令](/gpu-glossary/device-software/streaming-assembler) 相关联；从程序员角度来看，则对应不同的基础吞吐量特性（例如 [Tensor 核心](/gpu-glossary/device-hardware/tensor-core) 的浮点矩阵乘法算术吞吐量）。

相比之下， [流式多处理器 (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor) 更接近于 CPU 核心的等效概念，因为它们拥有存储信息的 [寄存器内存](/gpu-glossary/device-hardware/register-file)、用于转换数据的核心，以及用于指定和控制转换操作的 [指令调度器](/gpu-glossary/device-hardware/warp-scheduler)。