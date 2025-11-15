# 什么是 CUDA C++ 编程语言？

CUDA C++ 是 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 的一种实现，作为 C++ 编程语言的扩展。

为实现 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model)，CUDA C++ 为 C++ 语言添加了若干特性，具体如下：

- 使用 **`__global__`** 定义 [**内核 (Kernel)**](/gpu-glossary/device-software/kernel)。CUDA [**内核 (Kernel)**](/gpu-glossary/device-software/kernel) C++ 函数的形式实现，这类函数接收指针作为参数、返回类型为 void，并使用该关键字进行标注。
- 使用 **`<<<>>>`** 启动 [**内核 (Kernel)**](/gpu-glossary/device-software/kernel)。[**内核 (Kernel)**](/gpu-glossary/device-software/kernel) 通过三重括号语法从 CPU 主机端执行，该语法用于设置 [线程块网格 (Thread Block Grid)](/gpu-glossary/device-software/thread-block-grid) 的维度。
- 使用 `shared` 关键字分配 [**共享内存 (Shared Memory)**](/gpu-glossary/device-software/shared-memory)，使用内置函数 `__syncthreads()` 实现 **屏障同步**，以及使用 `blockDim` 和 `threadIdx` 内置变量进行 [线程块 (Thread Block)](/gpu-glossary/device-software/thread-block) 和 [线程 (Thread)](/gpu-glossary/device-software/thread) 的索引。

CUDA C++ 程序的编译由主机端 C/C++ 编译器（如 `gcc`）与 [NVIDIA CUDA 编译器驱动](/gpu-glossary/host-software/nvcc) `nvcc` 共同编译。

有关如何在 [Modal](https://modal.com) 上使用 CUDA C++ 的信息，请参阅[本指南](https://modal.com/docs/guide/cuda)。