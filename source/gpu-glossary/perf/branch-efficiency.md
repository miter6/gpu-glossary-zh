# 什么是分支效率？

分支效率用于衡量的是当遇到条件语句时，[线程束 (warp)](/gpu-glossary/device-software/warp) 中的所有 [线程 (thread)](/gpu-glossary/device-software/thread) 选择相同执行路径的频率。

分支效率的计算方式为：统一控制流决策数量与执行的分支指总令数之比。控制流统一性是在 [线程束 (warp)](/gpu-glossary/device-software/warp) 级别进行测量的，因此分支效率反映了 [线程束分歧 (warp divergence)](/gpu-glossary/perf/warp-divergence) 的缺失程度。

并非所有条件语句都会降低分支效率。大多数 [内核 (kernel)](https://godbolt.org/z/d1PsYYPnW) 中常见的 "边界检查" 代码片段

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
```

通常具有非常高的分支效率，因为大多数 [线程束 (warp)](/gpu-glossary/device-software/warp) 中的 [线程 (thread)](/gpu-glossary/device-software/thread) 往往具有相同的条件判断结果，仅在少数 [线程束 (warp)](/gpu-glossary/device-software/warp) 中的 [线程 (thread)](/gpu-glossary/device-software/thread) 索引会同时分布在 `n` 的上下两侧。

虽然 CPU 也关心分支行为的统一性，但其核心关注点是时间维度的分支行为一致性（作为硬件控制的分支预测和推测执行机制的一部分）。具体而言，随着CPU内部电路在程序执行过程中不断积累某条分支的历史数据，性能会逐步优化。

而 GPU 更关注空间维度的统一性。也就是说，[线程束 (warp)](/gpu-glossary/device-software/warp) 内部的一致性，[线程束 (warp)](/gpu-glossary/device-software/warp) 中的 [线程 (thread)](/gpu-glossary/device-software/thread) 在时间上并发执行，但映射到不同的数据，如果这些 [线程 (thread)](/gpu-glossary/device-software/thread) 能够统一分支，性能就会提高。