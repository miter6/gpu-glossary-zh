# 什么是 CUDA 内核？

![](light-cuda-programming-model.svg)  

> 单个内核启动对应于 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 中的 [线程块网格](/gpu-glossary/device-software/thread-block-grid)。改编自 NVIDIA 的 [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) 和 NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) 中的图表。

内核（Kernel）是程序员通常编写和组合的 [CUDA](/gpu-glossary/device-software/cuda-programming-model) 代码单元，类似于面向 CPU 的编程语言中的过程或函数。

与过程不同，内核被调用（"启动"）一次并返回一次，但会被执行多次--每个 [线程](/gpu-glossary/device-software/thread) 执行一次。这些执行通常是并发的（执行顺序不确定）且并行的（在不同的执行单元上同时发生）。

执行内核的所有线程集合被组织为内核网格——也称为 [线程块网格](/gpu-glossary/device-software/thread-block-grid)，这是 [CUDA 编程模型](/gpu-glossary/device-software/cuda-programming-model) 中 [线程层次结构](/gpu-glossary/device-software/thread-hierarchy) 中的最高级别。内核网格在多个 [流式多处理器 (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor) 间执行，因此其操作规模覆盖整个 GPU。与之对应的 [内存层次结构](/gpu-glossary/device-software/memory-hierarchy) 级别是 [全局内存](/gpu-glossary/device-software/global-memory)。

在 [CUDA C++](/gpu-glossary/host-software/cuda-c) 中，内核由主机（host）调用时会接收指向设备（device） [全局内存](/gpu-glossary/device-software/global-memory) 的指针，且不返回任何值——它们仅对内存进行修改存。

为了让你对 CUDA 内核编程有初步了解，让我们来看两个 CUDA 内核实现"hello world"的例子：两个矩阵 `A` 和 `B` 的矩阵乘法。这两种实现的区别在于如何将经典矩阵乘法算法映射到 [线程层次结构](/gpu-glossary/device-software/thread-hierarchy) 和 [内存层次结构](/gpu-glossary/device-software/memory-hierarchy) 上。

其中最简单的实现灵感来自于 [Programming Massively Parallel Processors](https://www.amazon.com/dp/0323912311)（第 4 版，图 3.11）中首个矩阵乘法内核的启发，每个 [线程](/gpu-glossary/device-software/thread) 负责计算输出矩阵的一个元素——依次将 `A` 的特定 `row` 行和 `B` 的特定 `col` 列的元素加载到 [寄存器](/gpu-glossary/device-software/registers) 中，将成对元素相乘后累加结果，并将总和放回 [全局内存](/gpu-glossary/device-software/global-memory)。

```cpp
__global__ void mm(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

在这个内核中，每个 [线程](/gpu-glossary/device-software/thread) 每次从 [全局内存](/gpu-glossary/device-software/global-memory) 读取时执行一次浮点运算 (FLOP)：包含一次乘法和一次加法；对应从矩阵 `A` 和矩阵 `B` 各加载一次数据。这种方式无法 [充分利用整个 GPU](https://modal.com/blog/gpu-utilization-guide) 的性能，因为 [CUDA 核心](/gpu-glossary/device-hardware/cuda-core) 的 [算术带宽](/gpu-glossary/perf/arithmetic-bandwidth) (以 FLOPs/s 为单位) 远高于 [GPU 内存](/gpu-glossary/device-hardware/gpu-ram) 和 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 之间的 [内存带宽](/gpu-glossary/perf/memory-bandwidth)。

我们可以通过更精细地将算法任务映射到 [线程层次结构](/gpu-glossary/device-software/thread-hierarchy) 和 [内存层次结构](/gpu-glossary/device-software/memory-hierarchy) 上来提高 [浮点运算与内存操作的比例](/gpu-glossary/perf/arithmetic-intensity)。在下面的 "分块 (tiled)" 矩阵乘法内核中，灵感来自 [Programming Massively Parallel Processors](https://www.amazon.com/dp/0323912311) 第 4 版图 5.9，我们将矩阵 `A` 和 `B` 的子矩阵的加载操作以及矩阵 `C` 的子矩阵的计算操作分别映射到 [共享内存](/gpu-glossary/device-software/shared-memory) 和 [线程块](/gpu-glossary/device-software/thread-block) 上。

```cpp
#define TILE_WIDTH 16

__global__ void mm(float* A, float* B, float* C, int N) {

    // 在共享内存中声明变量
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float c_output = 0;
    for (int m = 0; m < N/TILE_WIDTH; ++m) {

        // 每个线程从全局内存加载 A 的一个元素和 B 的一个元素到共享内存
        As[threadIdx.y][threadIdx.x] = A[row * N + (m * TILE_WIDTH + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + col];

        // 我们等待 16x16 块中的所有线程完成加载到共享内存
        // 这样它就包含两个 16x16 的块
        __syncthreads();

        // 然后我们遍历内部维度，
        // 每次从全局内存加载一对数据时执行 16 次乘法和 16 次加法
        for (int k = 0; k < TILE_WIDTH; ++k) {
            c_output += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        // 在所有线程开始将下一个块加载到共享内存之前，
        // 等待所有线程完成计算
        __syncthreads();
    }
    C[row * N + col] = c_output;
}
```

在外层循环的每次迭代中（该迭代会加载两个元素），线程会执行16次内层循环——每次内层循环包含一次乘法和一次加法运算，因此每次全局内存读取可对应16次浮点运算（FLOPs）。

不过，这距离完全优化的矩阵乘法内核仍有差距。[Anthropic 的 Si Boehm 的这篇工作日志](https://siboehm.com/articles/22/CUDA-MMM) 详细介绍了进一步优化方法，这些优化能进一步提高浮点运算与内存读取的比率，并让算法与硬件特性更紧密地匹配。我们目前讨论的内核类似于他文中的 Kernel 1 和 Kernel 3，而该工作记录共涵盖了十种内核优化方案。

需要注意的是，该工作记录和本文均只讨论在 [CUDA 核心](/gpu-glossary/device-hardware/cuda-core) 上执行的内核编写。实际上，速度最快的矩阵乘法内核运行在 [张量核心](/gpu-glossary/device-hardware/tensor-core) 上，后者具有更高的 [算术带宽](/gpu-glossary/perf/arithmetic-bandwidth)。