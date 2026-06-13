# 什么是 CuTe DSL

CuTe DSL 是一种基于 Python 的领域特定语言（Domain-Specific Language, DSL），用于在保持高开发效率的同时，高性能地编写和动态编译核函数 [kernel](/gpu-glossary/device-software/kernel)。

CuTe DSL 是 [CUTLASS](/gpu-glossary/host-software/cutlass) 的一部分，而 CUTLASS 是一个由 [CUDA C++](/gpu-glossary/host-software/cuda-c) 模板和 DSLs 组成的集合。与 [cuBLAS](/gpu-glossary/host-software/cublas) 或 [cuDNN](/gpu-glossary/host-software/cudnn) 这种为常规操作提供现成可调用核函数的库不同，CUTLASS 技术栈提供的工具允许开发者以可组合的方式来定义高性能的核函数。

CuTe DSL 的核心抽象包括布局 (Layouts)、张量 (Tensors)、硬件原子 (Hardware Atoms) 以及分块操作 (Tiled Operations)。布局描述了数据在内存中以及在线程之间的组织方式；张量将数据指针或迭代器与布局元数据结合在一起；原子代表了底层的硬件操作，例如矩阵乘加（MMA）或内存拷贝；分块操作则描述了如何将这些原子应用到整个线程块 [thread blocks](/gpu-glossary/device-software/thread-block) 和 线程束 [warps](/gpu-glossary/device-software/warp) 中。关于其底层细节，请参阅 CuTe。

从 Python 启动 CuTe DSL 核函数时，Python 程序会调用一个带有 `@cute.jit` 修饰器的函数，而该函数会进一步启动一个带有 `@cute.kernel` 修饰器的函数。

`@cute.jit` 修饰器声明了一个即时编译（JIT）函数，该函数可以从 Python 或其他 CuTe DSL 函数中调用。@cute.kernel` 修饰器则定义了一个 GPU 核函数，该函数可以从 `@cute.jit` 函数中启动。Python 代码无法直接调用 @cute.kernel` 函数。

例如，让我们来看一个用于两个一维张量逐元素相加的朴素（未优化）CuTe DSL 核函数。 这是 GPU 编程的 “Hello World” 示例，其历史可以追溯到 [Ian Buck 的 Brook 框架](https://graphics.stanford.edu/papers/brookgpu/brookgpu.pdf) （该框架早于并启发了 [CUDA](/gpu-glossary/device-software/cuda-programming-model) 编程模型）。您可以使用 [Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-Vnwf5bQck2WSSETJUPk2UD) 来使用并编辑此核函数并在 B200 GPU 上运行它。

```
import cutlass.cute as cute
import torch

Tensor = cute.Tensor | torch.Tensor


@cute.kernel
def elem_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    block_x, _, _ = cute.arch.block_idx()
    block_dim_x, _, _ = cute.arch.block_dim()
    thread_x, _, _ = cute.arch.thread_idx()

    i = block_x * block_dim_x + thread_x

    if i < out.shape[0]:
        out[i] = a[i] + b[i]


@cute.jit
def elem_add(a: Tensor, b: Tensor, out: Tensor):
    n = out.shape[0]
    threads_per_block = 128
    blocks = (n + threads_per_block - 1) // threads_per_block

    elem_add_kernel(a, b, out).launch(
        grid=(blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )
```

其中 `elem_add_kernel` 函数就是核函数 ([kernel](/gpu-glossary/device-software/kernel))。每个线程 ([thread](/gpu-glossary/device-software/thread)) 计算一个输出元素。全局元素索引 `i` 是通过线程块索引、块内线程数以及块内线程索引计算得出的：

```
i = block_x * block_dim_x + thread_x
```

`elem_add` 函数计算出覆盖输出张量所需的线程块数量，并使用一维的线程块网格 ([thread block grid](/gpu-glossary/device-software/thread-block-grid)) 启动该核函数。

这个例子仅用于教学，并未经过优化。即便如此，它也展示了一个良好的基础访问模式：相邻的线程读取 `a` 和 `b` 的相邻元素，然后写入 `out` 的相邻元素。这是对全局内存 [global memory](/gpu-glossary/device-software/global-memory) 进行合并访问所需的模式；参见内存合并 [memory coalescing](/gpu-glossary/perf/memory-coalescing)。

布局（Layout）问题是 CuTe DSL 在编写高性能核函数时大显身手的原因之一。针对性能 [performance](/gpu-glossary/perf/index.rst) 进行工程设计非常困难，因为核函数必须与硬件紧密映射：哪些线程处理哪些数据、内存如何访问、工作如何分块，以及生成的代码应该使用哪些硬件操作。CuTe DSL 允许程序员显式地表达这些映射，同时可以在各种不同的形状（Shapes）和流式多处理器架构 [Streaming Multiprocessor architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 之间复用大部分相同的核函数代码。

对于其他领域关注性能的工程师来说，这可能会令人感到惊讶——一个用 Python 这种解释型语言编写的程序，怎么可能指望与用编译型语言编写的程序并驾齐驱？

答案是，CuTe DSL 核函数是即时（Just-In-Time, JIT）编译的。Python 源代码会被转换为抽象语法树（AST），通过代理参数进行追踪，然后再进行编译。请注意，JIT 编译的代码仅支持 Python 语法语义的一个子集。

在撰写本文时，在 CUTLASS 4.x 中，编译技术栈在执行前会先通过多级中间表示 [Multi-Level Intermediate Representation (MLIR)](https://mlir.llvm.org/) 转换为 [PTX](/gpu-glossary/device-software/parallel-thread-execution) IR，然后再转换为特定于设备的 [SASS](/gpu-glossary/device-software/streaming-assembler) 汇编。

以 [FlashAttention-4](https://arxiv.org/abs/2603.05451) 核函数为例。我们 [文章](https://modal.com/blog/reverse-engineer-flash-attention-4) 中对该开源代码进行了拆解分析，详细介绍了它如何通过流水线化的序列线程束（Pipelined Warp Specialization）、张量核心 [Tensor Core](/gpu-glossary/device-hardware/tensor-core) 操作，以及张量内存 [Tensor Memory](/gpu-glossary/device-hardware/tensor-memory) 和 张量内存加速器 [Tensor Memory Accelerator](/gpu-glossary/device-hardware/tensor-memory-accelerator) 操作，直接利用 CuTe DSL 实现了顶级的尖端性能。

欲了解更多关于 CuTe DSL 的细节，请参阅 NVIDIA 的 [CuTe DSL documentation](https://docs.nvidia.com/cutlass/4.4.2/media/docs/pythonDSL/cute_dsl.html) 以及 [CuTe DSL overview blog](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)。