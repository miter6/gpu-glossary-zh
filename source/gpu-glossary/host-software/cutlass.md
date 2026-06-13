# 什么是 CUTLASS ?

CUDA 线性代数子程序与求解器模板库 (CUDA Templates for Linear Algebra Subroutines and Solvers, CUTLASS) 是一个抽象库，用于在 [CUDA](/gpu-glossary/device-software/cuda-programming-model) 核函数 [kernels](/gpu-glossary/device-software/kernel) 中实现高性能线性代数计算。

与 [cuBLAS](/gpu-glossary/host-software/cublas) 类似，CUTLASS 的命名参考了基础线性代数子程序 [Basic Linear Algebra Subprograms (BLAS)](https://netlib.org/blas/blast-forum/) 标准，该标准定义了用于线性代数计算的底层例程。但与 cuBLAS 不同的是，CUTLASS 是一个用于构建核函数的工具包，而不是一个包含现成可调用例程的库。CUTLASS 主要用于 BLAS 层次结构中的第三层：通用矩阵乘法（"GEMMs"）。

正如其名称所示，CUTLASS 包含了一系列 [CUDA C++](/gpu-glossary/host-software/cuda-c) 模板抽象。模板 [Templates](https://en.cppreference.com/cpp/language/templates) 是 C++ 对 参数多态性 [parametric polymorphism](https://bartoszmilewski.com/2014/09/22/parametricity-money-for-nothing-and-theorems-for-free/) 的实现，您在其他语言中可能以 泛型 [generics](https://doc.rust-lang.org/rust-by-example/generics.html) 的形式接触过这一概念。多态函数只需编写一次，便可操作不同输入类型的数据。

现代 CUTLASS 的核心是 [CuTe](/gpu-glossary/host-software/cute) 库，它定义了 `Layout` 和 `Tensor` 类型，用于以可组合（Composable）的方式描述和操作由 [数据](/gpu-glossary/device-software/memory-hierarchy) 与 [线程](/gpu-glossary/device-software/thread-hierarchy) 构成的张量。请注意，不要将它与 CuTe DSL 混淆，后者是在 Python 中通过领域特定语言（DSL）来封装并提供 CuTe/CUTLASS 模板的功能。

在 CuTe 之上，CUTLASS 提供了一个仅包含头文件的 CUDA C++ 库，该库在三个级别上运行：整个设备（`device`）、单个核函数 [kernel](/gpu-glossary/device-software/kernel) 或 [threads](/gpu-glossary/device-software/thread) 集合（`collective`，通常为一个 [thread block](/gpu-glossary/device-software/thread-block)）。在 `collective` 层中，矩阵与矩阵的乘法通常被分为“主循环 (Mainloops)”和“尾部处理 (Epilogues)”。
主循环负责表达核心算法（例如分块策略）；尾部处理则描述后处理步骤，例如应用缩放因子或标量非线性变换（这在神经网络中非常流行）。

CUTLASS 经常被用来编写一些性能极高的核函数，尤其是针对较新 [Streaming Multiprocessor architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 硬件上的矩阵乘法。这些核函数需要对张量核心 [Tensor Cores](/gpu-glossary/device-hardware/tensor-core) 进行精细的编程，才能达到接近峰值的性能 [performance](/gpu-glossary/perf/index.rst)。

CUTLASS 是 开源的，可以在 [GitHub](https://github.com/nvidia/cutlass) 上获取。该库还包含了许多使用 CUTLASS 实现的高性能开源核函数，这些函数经常作为其他开源核函数开发时的参考。我们强烈推荐 [Colfax International 的 Jay Shah 撰写的热门教程](https://research.colfax-intl.com/)，它们详细解释了如何利用 CUTLASS 的核心组件来实现最大性能。不过需要注意的是，与大多数 C++ 模板元编程一样，CUTLASS 绝非浅尝辄止者所能轻易掌握的！