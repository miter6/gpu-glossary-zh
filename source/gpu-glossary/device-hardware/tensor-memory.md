# 什么是 Tensor Memory？

Tensor Memory（张量内存）是某些 GPU（如 [B200](https://modal.com/blog/introducing-b200-h200)）的 [流式多处理器 (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor) 中的一种专用内存，用于存储 [Tensor Core](/gpu-glossary/device-hardware/tensor-core) 的输入和输出。

张量内存的访问受到严格限制: 数据必须由线程束组（warpgroup）中的四个 [线程束 (warp)](/gpu-glossary/device-software/warp) 协同移动，并且仅支持特定模式的内存操作，例如在张量内存与 [寄存器 (register)](/gpu-glossary/device-software/registers) 之间传输数据、将 [共享内存 (shared memory)](/gpu-glossary/device-software/shared-memory) 写入张量内存，或向 [Tensor Core](/gpu-glossary/device-hardware/tensor-core) 发送矩阵乘加（MMA）指令以使用张量内存存储特定操作数。这便是 ["统一计算"设备架构](/gpu-glossary/device-hardware/cuda-device-architecture) 的复杂之处！

具体来说，对于计算 `D += A @ B` 的 `tcgen05.mma` [并行线程执行 (Parallel Thread eXecution)](/gpu-glossary/device-software/parallel-thread-execution) 指令若要使用张量内存，"累加器"矩阵 `D` *必须* 位于张量内存中，左侧矩阵 `A` *可以* 位于张量内存或 [共享内存 (shared memory)](/gpu-glossary/device-software/shared-memory) 中，而右侧矩阵 B *必须* 位于 [共享内存 (shared memory)](/gpu-glossary/device-software/shared-memory) 中，而非张量内存。这种设计虽复杂但并非随意——在矩阵乘法运算期间，累加器的访问频率远高于分块矩阵（tiles），因此更能从专用硬件中获益（例如 [Tensor Core](/gpu-glossary/device-hardware/tensor-core) 与张量内存之间更短、更简单的布线）。请注意，所有矩阵都不在 [寄存器 (register)](/gpu-glossary/device-software/registers) 中。

注意：张量内存与 [Tensor Memory Accelerator](/gpu-glossary/device-hardware/tensor-memory-accelerator) 没有直接关系，后者是将数据加载到 [L1 数据缓存](/gpu-glossary/device-hardware/l1-data-cache) 中。大致流程为：仅当执行张量核心 [Tensor Core](/gpu-glossary/device-hardware/tensor-core) 操作时，数据才会从 L1 缓存移至张量内存，后续需显式移出以进行后续处理，例如神经网络中矩阵乘法之后的非线性激活。

有关张量内存及其在矩阵乘法中的使用模式详情，请参见 [GTC 2025 的《使用 CUTLASS 编程 Blackwell Tensor Core》演讲](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/)。