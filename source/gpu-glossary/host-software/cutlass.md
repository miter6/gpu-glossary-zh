# 什么是 CUTLASS？

CUDA Templates for Linear Algebra Subroutines and Solvers（CUTLASS）是一个抽象库，用于在 [CUDA](/gpu-glossary/device-software/cuda-programming-model) [内核](/gpu-glossary/device-software/kernel)中实现高性能线性代数。

和 [cuBLAS](/gpu-glossary/host-software/cublas) 一样，CUTLASS 的命名也参考了用于底层线性代数计算例程的 [Basic Linear Algebra Subprograms（BLAS）](https://netlib.org/blas/blast-forum/)标准。不同于 cuBLAS，CUTLASS 是一个用于构造内核的工具包，而不是一组可直接调用例程的库。CUTLASS 主要与 BLAS 层次结构的第三级，即通用矩阵乘法（GEMM）相关。

顾名思义，CUTLASS 包含一组 [CUDA C++](/gpu-glossary/host-software/cuda-c) 模板抽象。[模板](https://en.cppreference.com/w/cpp/language/templates)是 C++ 对[参数多态](https://bartoszmilewski.com/2014/09/22/parametricity-money-for-nothing-and-theorems-for-free/)的实现；你可能在其他语言中以[泛型](https://doc.rust-lang.org/rust-by-example/generics.html)的形式接触过类似概念。多态函数只需编写一次，却可以作用于不同类型的输入。

现代 CUTLASS 的核心是 [CuTe](/gpu-glossary/host-software/cute) 库。CuTe 定义了 `Layout` 和 `Tensor` 类型，用于以可组合方式描述和操作由[数据](/gpu-glossary/device-software/memory-hierarchy)与[线程](/gpu-glossary/device-software/thread-hierarchy)组成的张量。不要将其与 [CuTe DSL](/gpu-glossary/host-software/cute-dsl) 混淆；后者通过 Python 中的领域专用语言（DSL）暴露 CuTe/CUTLASS 模板。

在 CuTe 之上，CUTLASS 暴露了一个 header-only CUDA C++ 库，可以在三个层级上工作：整个 `device`、单个 [`kernel`](/gpu-glossary/device-software/kernel)，或一个[线程](/gpu-glossary/device-software/thread)的 `collective`（通常是一个[线程块](/gpu-glossary/device-software/thread-block)）。在 `collective` 层，矩阵-矩阵乘法通常被拆分为 “mainloop” 和 “epilogue”。mainloop 表达核心算法，例如 tiling 策略；epilogue 描述后处理步骤，例如应用缩放因子或标量非线性函数（在神经网络中很常见）。

CUTLASS 非常常用于编写一些性能最高的内核，尤其是在较新的[流式多处理器架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)硬件上执行的矩阵-矩阵乘法。这些内核需要对 [Tensor Core](/gpu-glossary/device-hardware/tensor-core) 进行细致编程，才能接近峰值[性能](/gpu-glossary/perf/index)。

CUTLASS 是[开源的，并可在 GitHub 上获取](https://github.com/nvidia/cutlass)。该库还包含许多使用 CUTLASS 实现的高性能开源内核，这些实现经常被其他开源内核开发工作作为参考。我们强烈推荐 Colfax International 的 Jay Shah 撰写的[系列热门教程](https://research.colfax-intl.com/)，其中详细解释了如何使用 CUTLASS 的关键组件来实现最高性能。不过需要注意，和大多数 C++ 模板元编程一样，CUTLASS 并不适合畏难者！
