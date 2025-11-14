# 什么是计算能力？

[并行线程执行 (Parallel Thread Execution)](/gpu-glossary/device-software/parallel-thread-execution) 指令集中的指令仅与特定的物理 GPU 兼容。用于将物理 GPU 的细节从指令集和 [编译器 (compiler)](/gpu-glossary/host-software/nvcc) 中抽象出来的版本控制系统称为 "计算能力" （Compute Capability）。

大多数计算能力版本号包含两个组成部分：主版本号和次版本号。NVIDIA 承诺按照 [洋葱层模型](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-module-directives-target)，主版本号和次版本号均支持前向兼容性（旧的 [PTX](/gpu-glossary/device-software/parallel-thread-execution) 代码可在新 GPU 上运行）。

在 Hopper 架构中，NVIDIA 引入了额外的版本后缀，即 `9.0a` 中的 `a`，它包含偏离洋葱模型的特性：即使在同一主版本号内，这些特性的未来兼容性也不保证。

在 Blackwell 架构中，NVIDIA 引入了另一个版本后缀，即 `10.0f` 中的 `f`，该后缀同样偏离了洋葱模型，更接近 [语义化版本 (SemVer)](https://semver.org/)：次版本号之间保证兼容性，但主版本号之间不保证。

在调用 [NVIDIA CUDA 编译器驱动程序 (NVIDIA CUDA Compiler Driver)](/gpu-glossary/host-software/nvcc) `nvcc` 时，可以指定 [PTX](/gpu-glossary/device-software/parallel-thread-execution) 编译的目标计算能力。默认情况下，编译器还会为匹配的 [流式多处理器架构 (Streaming Multiprocessor architecture)](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 生成优化的 [SASS](/gpu-glossary/device-software/streaming-assembler) 代码。 [`nvcc`](/gpu-glossary/host-software/nvcc) 的 [文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures) 将计算能力称为 "虚拟 GPU 架构"，与 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 版本表示的 "物理 GPU 架构" 形成对比。

每个计算能力版本的技术规格可以在 [NVIDIA CUDA C 编程指南的计算能力部分](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) 找到。