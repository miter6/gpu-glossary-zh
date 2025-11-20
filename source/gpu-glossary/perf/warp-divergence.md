# 什么是线程束分化？

线程束分化 (warp divergence) 发生在 [线程束 (warp)](/gpu-glossary/device-software/warp) 内的线程因控制流语句而执行不同路径时出现的现象。

例如，考虑以下 [内核 (kernel)](/gpu-glossary/device-software/kernel)：

```cpp
__global__ void divergent_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] > 0.5f) {
		    // A
            data[idx] = data[idx] * 4.0f;
        } else {
		    // B
            data[idx] = data[idx] + 2.0f;
        }
        data[idx] = data[idx] * data[idx];
    }
}
```

当 [线程束 (warp)](/gpu-glossary/device-software/warp) 内的 [线程 (threads)](/gpu-glossary/device-software/thread) 遇到数据相关的条件判断时，根据 `data[idx]` 的值，一些 [线程 (threads)](/gpu-glossary/device-software/thread) 必须执行代码块 A，而其他线程必须执行代码块 B。由于这种数据依赖性以及 [CUDA 编程模型 (CUDA programming model)](/gpu-glossary/device-software/cuda-programming-model) 及其在 [PTX 机器模型 (PTX machine model)](/gpu-glossary/device-software/parallel-thread-execution) 中实现的结构性约束，程序员或编译器无法避免 [线程束 (warp)](/gpu-glossary/device-software/warp) 内部出现这种控制流分裂。

此时，[线程束调度器 (warp scheduler)](/gpu-glossary/device-hardware/warp-scheduler) 必须处理这些发散代码路径的并发执行，具体通过 “屏蔽” 部分 [线程 (threads)](/gpu-glossary/device-software/thread) 使其不执行指令来实现这一点。这是通过使用 [谓词寄存器 (predicate registers)](/gpu-glossary/device-software/registers) 实现的。

让我们检查生成的 [SASS (Streaming Assembler)](/gpu-glossary/device-software/streaming-assembler) ([Godbolt 链接](https://godbolt.org/z/EGWKb5oWr)) 来理解执行流程：

```nasm
LDG.E.SYS R4, [R2]                       // L1 load data[idx]
FSETP.GT.AND P0, PT, R4.reuse, 0.5, PT   // L2 set P0 to data[idx] > 0.5
FADD R0, R4, 2                           // L3 store 2 + data[idx] in R0
@P0 FMUL R0, R4, 4                       // L4 in some threads, store 4 * data[idx] in R0
FMUL R5, R0, R0                          // L5 store R0 * R0 in R5
STG.E.SYS [R2], R5                       // L6 store R5 in data[idx]
```

将数据加载到 `R4` (`L1` 行) 后，[线程束 (warp)](/gpu-glossary/device-software/warp) 中的所有 32 个 [线程 (threads)](/gpu-glossary/device-software/thread) 并发执行 `FSETP.GT.AND` (`L2` 行)，每个 [线程 (thread)](/gpu-glossary/device-software/thread) 根据 `R4` 中的 `data` 值获得自己的 `P0` 值。随后是 [编译器 (nvcc)](/gpu-glossary/host-software/nvcc) 的一个巧妙设计之处：在 `L3` 行，*所有* [线程 (threads)](/gpu-glossary/device-software/thread) 都执行代码块 A 的代码，写入 `R0`。只有那些 `P0` 为真的线程会继续执行代码块 B 的代码 (`L4` 行)，覆盖在 `L3` 行中写入 `R0` 的值。此时，[线程束 (warp)](/gpu-glossary/device-software/warp) 被称为处于 "分化" 的状态。在 `L5` 行，所有 [线程 (threads)](/gpu-glossary/device-software/thread) 回到同一执行路径。当 [线程束调度器 (warp scheduler)](/gpu-glossary/device-hardware/warp-scheduler) 通过在同一时钟周期发射相同指令使线程重新对齐后，线程束就 "收敛" 了。

这种实现可能比将分支直接编码为 [SASS (Streaming Assembler)](/gpu-glossary/device-software/streaming-assembler) 中更高效，后者会对 `L3` 行 和 `L4` 行两行都进行谓词判断 — 我们有理由相信 [编译器 (nvcc)](/gpu-glossary/host-software/nvcc) 的优化，并且启发式地看，这种设计是以廉价且足够的 [CUDA 核心 (CUDA Core)](/gpu-glossary/device-hardware/cuda-core) 计算来换取更昂贵的流控制操作。正如在 GPU 编程中常见的那样，即使只是简单的谓词化，浪费计算资源（每次执行 `L4` 行时进行一次不必要的 `FADD`）也往往比增加控制流复杂度要好！

编译器可能积极避免分化的一个原因是，在早期（Volta 架构之前）的 GPU 中，分化的 [线程束 (warps)](/gpu-glossary/device-software/warp) 总是完全串执行的。虽然线程束分化仍然会降低效率，但具有独立线程调度的现代 GPU 不一定会经历完全串行化带来的性能损失。