# 什么是 CuTe？

CUDA Templates（CuTe）是 [CUTLASS](/gpu-glossary/host-software/cutlass) 中的一个 header-only [CUDA C++](/gpu-glossary/host-software/cuda-c) 库，用于描述和操作[数据](/gpu-glossary/device-software/memory-hierarchy)与[线程](/gpu-glossary/device-software/thread-hierarchy)组成的张量。

顾名思义，CuTe 使用 CUDA C++ [模板](https://en.cppreference.com/w/cpp/language/templates)。模板是 C++ 对[参数多态](https://bartoszmilewski.com/2014/09/22/parametricity-money-for-nothing-and-theorems-for-free/)的实现；你可能在其他语言中以[泛型](https://doc.rust-lang.org/rust-by-example/generics.html)的形式接触过类似概念。多态函数只需编写一次，却可以作用于不同类型的输入。不要将 CuTe 与 [CuTe DSL](/gpu-glossary/host-software/cute-dsl) 混淆；后者通过 Python 中的领域专用语言（DSL）暴露 CuTe/CUTLASS。

CuTe 类型系统的核心是 `Layout`。`Layout` 描述对 CuTe `Tensor` 的规则访问模式。`Tensor` 则将 `Layout` 与指向[内存](/gpu-glossary/device-software/memory-hierarchy)的指针结合起来。关键在于，这些 `Layout` 是可组合的：它们构成[一个范畴](https://arxiv.org/abs/2601.05972)，并具有[丰富的代数结构](https://arxiv.org/abs/2603.02298)，因此同时具备表达能力和结构性。需要注意，`Layout` 本身由 `Shape` 和 `Stride` 元组组成，用于描述内存范围以及遍历方式。

CuTe 使用类型系统编码关键的程序元数据，例如内存组织方式、带 stride 的访问以及 tiling，使[编译器](/gpu-glossary/host-software/nvcc)能够检查许多正确性方面的问题，并在应用优化时保持不变量。这使得程序员可以对[内核](/gpu-glossary/device-software/kernel)进行非常高层次的元编程，同时不牺牲性能。例如，同一个模板可以针对多种[流式多处理器架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)编译成高度优化的内核。由于 layout 在编译期解析，内存访问不会带来额外运行时开销；否则，这类开销可能会扼杀[内存受限](/gpu-glossary/perf/memory-bound)工作负载的[性能](/gpu-glossary/perf/index)。

更多细节可参阅 NVIDIA 的 [CuTe 文档](https://docs.nvidia.com/cutlass/4.4.2/media/docs/cpp/cute/index.html)。

下面这个基于 CuTe 的矩阵转置内核，改编自 [Colfax International 这篇文章](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/)中的初始“naive”实现，展示了 CuTe 的核心特性和类型：模板、shape、layout 和 tensor。你可以通过[这个 Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-owEUD0kdSVeL4KeEX5sjh1) 在 H100 上运行它。

```cpp
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
