# 什么是 CuTe DSL？

CuTe DSL 是一种基于 Python 的领域专用语言（DSL），用于以高性能和较高开发效率编写并动态编译[内核](/gpu-glossary/device-software/kernel)。

CuTe DSL 是 [CUTLASS](/gpu-glossary/host-software/cutlass) 的一部分；CUTLASS 是一组 [CUDA C++](/gpu-glossary/host-software/cuda-c) 模板和 DSL。与为常见操作提供可直接调用内核的 [cuBLAS](/gpu-glossary/host-software/cublas) 或 [cuDNN](/gpu-glossary/host-software/cudnn) 不同，CUTLASS 栈提供的是一组可组合定义高性能内核的工具。

CuTe DSL 的核心抽象包括 layout、tensor、hardware atom 和 tiled operation。layout 描述数据在内存中以及在线程之间的组织方式。tensor 将数据指针或迭代器与 layout 元数据结合起来。atom 表示基本硬件操作，例如矩阵乘加（MMA）或内存拷贝。tiled operation 描述如何在[线程块](/gpu-glossary/device-software/thread-block)和 [warp](/gpu-glossary/device-software/warp) 上应用 atom。底层细节可参阅 [CuTe](/gpu-glossary/host-software/cute)。

从 Python 启动 CuTe DSL 内核时，Python 程序会调用一个 `@cute.jit` 函数，而该函数再启动一个 `@cute.kernel` 函数。

`@cute.jit` 装饰器声明一个 JIT 编译函数，它既可以从 Python 调用，也可以从其他 CuTe DSL 函数调用。`@cute.kernel` 装饰器定义一个 GPU kernel 函数，该函数可以从 `@cute.jit` 函数中启动。Python 代码不能直接调用 `@cute.kernel` 函数。

例如，我们来看一个用于两个一维张量逐元素相加的 naive（未优化）CuTe DSL 内核。这是 GPU 编程中的 “hello world”，可以追溯到 Ian Buck 的 [Brook 框架](https://graphics.stanford.edu/papers/brookgpu/brookgpu.pdf)，后者早于并启发了 [CUDA](/gpu-glossary/device-software/cuda-programming-model)。你可以使用[这个 Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-Vnwf5bQck2WSSETJUPk2UD) 在 B200 GPU 上编辑并执行这个内核。

```python
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

`elem_add_kernel` 函数就是[内核](/gpu-glossary/device-software/kernel)。每个[线程](/gpu-glossary/device-software/thread)计算一个输出元素。全局元素索引 `i` 由[线程块](/gpu-glossary/device-software/thread-block)索引、块内线程数以及块内线程索引计算得到：

```python
i = block_x * block_dim_x + thread_x
```

`elem_add` 函数计算覆盖输出张量所需的线程块数量，并用一维[线程块网格](/gpu-glossary/device-software/thread-block-grid)启动内核。

这个例子主要用于教学，并未优化。即便如此，它展示了一个良好的基础访问模式：相邻线程读取 `a` 和 `b` 中的相邻元素，然后写入 `out` 中的相邻元素。这正是对[全局内存](/gpu-glossary/device-software/global-memory)进行合并访问所需的模式；参见[内存合并](/gpu-glossary/perf/memory-coalescing)。

layout 相关问题正是 CuTe DSL 适合高性能内核的原因之一。面向[性能](/gpu-glossary/perf/index)的工程很难，因为内核必须紧密映射到底层硬件：哪些线程处理哪些数据、内存如何访问、工作如何 tiling，以及生成代码应该使用哪些硬件操作。CuTe DSL 允许程序员显式表达这些映射，同时在多种 shape 和[流式多处理器架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)之间复用大部分相同的内核代码。

这可能会让其他领域中关注性能的工程师感到意外：用 Python 这种解释型语言编写的程序，怎么可能与编译型语言编写的程序竞争？

答案是，CuTe DSL 内核会被即时编译（JIT）。Python 源码会被转换为抽象语法树（AST），再用代理参数进行追踪，然后编译。需要注意，JIT 编译代码只支持 Python 语义的一个子集。

截至本文撰写时，在 CUTLASS 4.x 中，编译栈会经过 [Multi-Level Intermediate Representation（MLIR）](https://mlir.llvm.org/)，再到 [PTX](/gpu-glossary/device-software/parallel-thread-execution) IR，然后到设备特定的 [SASS](/gpu-glossary/device-software/streaming-assembler)，最终执行。

以 [FlashAttention-4](https://arxiv.org/abs/2603.05451) 内核为例。我们的[文章](https://modal.com/blog/reverse-engineer-flash-attention-4)梳理了其开源代码如何使用流水线化的 warp specialization、[Tensor Core](/gpu-glossary/device-hardware/tensor-core) 操作，以及 [Tensor Memory](/gpu-glossary/device-hardware/tensor-memory) 和 [Tensor Memory Accelerator](/gpu-glossary/device-hardware/tensor-memory-accelerator) 操作，直接从 CuTe DSL 达到当前领先的性能。

更多 CuTe DSL 细节可参阅 NVIDIA 的 [CuTe DSL 文档](https://docs.nvidia.com/cutlass/4.4.2/media/docs/pythonDSL/cute_dsl.html)和 [CuTe DSL 概览博客](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)。
