# 什么是内存合并？

内存合并（Memory Coalescing）是一种硬件技术，通过在单次 *物理* 内存访问中处理多个 *逻辑* 内存读取请求，来提高 [内存带宽](/gpu-glossary/perf/memory-bandwidth) 的利用率。

内存合并发生在 [全局内存](/gpu-glossary/device-software/global-memory) 访问过程中。关于 [共享内存](/gpu-glossary/device-software/shared-memory) 的高效访问，请参阅关于 [存储体冲突](/gpu-glossary/perf/bank-conflict) 的文章。

在 [CUDA](/gpu-glossary/device-hardware/cuda-device-architecture) GPU 中，[全局内存](/gpu-glossary/device-software/global-memory) 由 [GPU显存](/gpu-glossary/device-hardware/gpu-ram) 提供，该显存采用如 GDDR 或 HBM 等动态随机存取存储器（DRAM）技术构建。这些技术具有高 [内存带宽](/gpu-glossary/perf/memory-bandwidth)，但访问延迟也很长（即使与 CPU RAM 所使用的 DDR5 等同类技术相比也是如此）。DRAM 的访问延迟受限于小型电容器为访问线路充电的速度，而这本质上受到散热、功耗和尺寸的限制。正因为这种高延迟，如果所有逻辑内存访问都作为单独的物理访问来处理，GPU的 [内存带宽](/gpu-glossary/perf/memory-bandwidth) 就无法得到充分利用。

内存合并利用 DRAM 技术的内部机制，使特定的访问模式能够实现全带宽利用率。每次访问 DRAM 地址时，多个连续地址会在单个时钟周期内被并行获取。更多细节请参阅 [《大规模并行处理器编程》第四版](https://www.amazon.com/dp/0323912311) 的第6.1节；全面细节请参阅 Ulrich Drepper 的优秀文章 [_每个程序员应该了解的内存知识_](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)。这些连续内存位置的访问和传输被称为 *DRAM 突发传输*。如果多个并发的逻辑访问可以通过单次物理突发传输来完成，那么这种访问就被称为 *已合并的*。需要注意的是，物理访问是内存事务（memory transaction）的一部分，这一术语在其他关于内存合并的描述中也可能会遇到。

在 CPU 上，类似的将突发传输映射到缓存行的机制可以提高访问效率。但与 CPU 中由硬件自动管理缓存行为不同，在 GPU 编程中，内存合并通常需要程序员进行显式优化。

不过，这种优化并不像想象中那么困难，因为 DRAM 突发传输与 [CUDA PTX](/gpu-glossary/device-software/parallel-thread-execution) 的单指令多线程 (SIMT) 执行模型天然契合。也就是说，在正常执行中，一个 [线程束](/gpu-glossary/device-software/warp) 中的所有 [线程](/gpu-glossary/device-software/thread) 会同时执行相同的指令。这使得 [CUDA](/gpu-glossary/device-software/cuda-programming-model) 程序员很容易编写具有合并访问模式的程序，并且内存管理硬件也能轻松检测到可以合并的访问。通常，单次突发传输可以处理 128 字节——这并非巧合，刚好足够一个 [线程束](/gpu-glossary/device-software/warp) 中的 32 个 [线程](/gpu-glossary/device-software/thread) 各加载一个 32 位浮点数。

为了直观展示内存合并对性能的影响，我们来考虑一个按可变步长（`stride`，即访问元素之间的间隔）读取数组的 [内核](/gpu-glossary/device-software/kernel)。随着步长的增加，每个 [线程束](/gpu-glossary/device-software/warp) 的读取请求所需的 DRAM 突发传输次数会增多，这会导致每个逻辑访问对应更多的物理访问，从而降低内存吞吐量。

```cpp
__global__ void strided_read_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    size_t N, int stride)
{
    const size_t t  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t T  = gridDim.x * (size_t)blockDim.x;

    float acc = 0.f;

    for (size_t j = (size_t)t * (size_t)stride; j < N; j += (size_t)T * (size_t)stride) {
        // across a warp, addresses differ by (stride * sizeof(float))
        float v = in[j]; // perfectly coalesced for stride == 1
        acc = acc * 1.000000119f + v;  // force compiler to keep the load
    }

    // do one write per thread (negligible vs reads)
    if (t < N) out[t] = acc;
}
```

当我们在 Godbolt 上通过一个微基准测试运行这个内核时（您可以 [在此处](https://godbolt.org/z/KbWhEWjcb) 复现），我们观察到步长与吞吐量之间的预期关系：

```
# Device: Tesla T4 (SM 75)
# N = 67108864 floats (256.0 MB), iters = 10
stride        GB/s
    1       206.0
    2       130.5
    4        68.8
    8        33.8
   16        16.8
   32        15.2
   64        13.6
  128        11.2
```

也就是说，步长增加为 2 会使吞吐量减半，这是因为每个 [线程束](/gpu-glossary/device-software/warp) 请求所需的 DRAM 突发传输次数翻倍。将步长加倍到 4 再次使吞吐量减半。而当步长为 16 时，吞吐量降至初始值的 16 倍，此后性能衰减模式发生变化，这大概是由于其他内存子系统组件的可见性增加以及它们因局部性降低（例如设备上的 TLB 未命中）而导致的性能下降。

关于全局内存访问的更多最佳实践，请参阅 NVIDIA 开发者博客上的文章 [_如何在CUDA C/C++内核中高效访问全局内存_](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)。