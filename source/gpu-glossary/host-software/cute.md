# 什么是 CuTe ?

CUDA 模板库 (CUDA Templates, CuTe) 是 [CUTLASS](/gpu-glossary/host-software/cutlass) 内部的一个仅包含头文件（Header-only）的 [CUDA C++](/gpu-glossary/host-software/cuda-c) 库，用于描述和操作由 [数据](/gpu-glossary/device-software/memory-hierarchy) 与 [线程](/gpu-glossary/device-software/thread-hierarchy) 构成的张量。

顾名思义，CuTe 充分利用了 CUDA C++ 的模板 [templates](https://en.cppreference.com/cpp/language/templates) 机制。模板是 C++ 对参数多态性 [parametric polymorphism](https://bartoszmilewski.com/2014/09/22/parametricity-money-for-nothing-and-theorems-for-free/) 的实现，您在其他语言中可能以泛型 [generics](https://doc.rust-lang.org/rust-by-example/generics.html) 的形式接触过这一概念。多态函数只需编写一次，便可操作不同输入类型的数据。请注意，CuTe 不要与 [CuTe DSL](/gpu-glossary/host-software/cute-dsl) 混淆，后者是在 Python 中通过领域特定语言（DSL）来封装并提供 CuTe/CUTLASS 的功能。

CuTe 类型系统的核心是 `Layouts`。 `Layouts` 描述了对 CuTe `Tensors` 进行常规访问的模式。而 `Tensors` 则将 `Layouts` 与指向 [内存](/gpu-glossary/device-software/memory-hierarchy) 的指针结合在了一起。至关重要的是，这些 `Layouts` 具有可组合性 (Composable)——它们构成了一个范畴 [category](https://arxiv.org/abs/2601.05972)，并拥有丰富的代数结构 [rich algebra](https://arxiv.org/abs/2603.02298)，从而将强大的表达能力与结构化完美结合。需要说明的是，`Layout` 本身是由 `Shape` 和 `Stride` 元组构成的，用于描述内存的维度范围以及如何遍历它们。

CuTe 利用类型系统来编码关键的程序元数据（如内存组织形式、步长访问和分块等），使得 [编译器](/gpu-glossary/host-software/nvcc) 在应用优化时，能够检查正确性的方方面面并维持不变性。这使得开发者可以在不牺牲性能的前提下，对核函数 [kernels](/gpu-glossary/device-software/kernel) 进行非常高级的元编程。例如，同一个模板可以被编译成跨越多种流式多处理器架构 [Streaming Multiprocessor architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 的高性能优化核函数。由于布局在编译期就已经解析完毕，内存访问带来了“零”额外的运行期开销，否则这种开销可能会直接摧毁内存受限型 [memory-bound](/gpu-glossary/perf/memory-bound) 工作负载的性能 [performance](/gpu-glossary/perf/index.rst)。

欲了解更多细节，请参阅 NVIDIA 的 [CuTe 官方文档](https://docs.nvidia.com/cutlass/4.4.2/media/docs/cpp/cute/index.html)。

下方展示了一个基于 CuTe 的矩阵转置核函数（该代码基于 [Colfax International](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/) 的这篇文章中的初始“朴素”实现）。它演示了 CuTe 的核心特性与类型——模板、形状、布局和张量。您可以使用 [这个 Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-owEUD0kdSVeL4KeEX5sjh1) 在 H100 GPU 上运行它。

```
// one CuTe trick: transpose a row-major matrix just using Layouts
template <typename T>
__global__ void transpose_kernel(const T* __restrict__ d_S,
                                 T* __restrict__ d_D,
                                 int M, int N)
{
    // define the Shape of tiles worked on by thread blocks
    using b = Int<32>;
    auto block_shape = make_shape(b{}, b{});

    // define the Shape of input/output Tensors
    auto tensor_shape = make_shape(M, N);

    // define the Layout of the input and output Tensors in global memory
    auto gmemLayoutS  = make_layout(tensor_shape, GenRowMajor{}); // input:  row-major
    auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{}); // output: col-major

    // construct the Tensors
    auto tensor_S  = make_tensor(make_gmem_ptr(d_S), gmemLayoutS);
    auto tensor_DT = make_tensor(make_gmem_ptr(d_D), gmemLayoutDT);

    // define a tile-ing of the Tensors (as a "Tensor of Tensors")
    auto tiled_tensor_S  = tiled_divide(tensor_S,  block_shape);
    auto tiled_tensor_DT = tiled_divide(tensor_DT, block_shape);

    // pull out the tiles this thread block will be working on
    auto tile_S  = tiled_tensor_S (make_coord(_, _), blockIdx.x, blockIdx.y);
    auto tile_DT = tiled_tensor_DT(make_coord(_, _), blockIdx.x, blockIdx.y);

    // create a Layout for threads in the thread block
    auto thr_layout = make_layout(
        make_shape(Int<8>{}, Int<32>{}),
        GenRowMajor{}
    );

    // pull out the tile this thread will work on
    auto thr_tile_S  = local_partition(tile_S,  thr_layout, threadIdx.x);
    auto thr_tile_DT = local_partition(tile_DT, thr_layout, threadIdx.x);

    // define a "Tensor" in register memory
    auto rmem = make_tensor_like<T>(thr_tile_S);

    // copy tile into registers
    copy(thr_tile_S, rmem);
    // copy tile out of registers as though it were column-major
    copy(rmem, thr_tile_DT);
}
```