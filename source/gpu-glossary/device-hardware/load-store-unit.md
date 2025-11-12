# 什么是加载/存储单元？（LSU）

加载/存储单元 (Load/Store Unit, LSU) 负责向 GPU 的内存子系统分派加载或存储数据的请求。

![](light-gh100-sm.svg)  

> H100 GPU 流式多处理器的内部架构示意图。 [加载/存储单元 (Load/Store Units)](/gpu-glossary/device-hardware/load-store-unit) 与特殊功能单元（SFUs）一同以粉色标注。修改自 NVIDIA 的 [H100 白皮书](https://modal-cdn.com/gpu-glossary/gtc22-whitepaper-hopper.pdf)。

对于 [CUDA 程序员](/gpu-glossary/host-software/cuda-software-platform) 而言，最重要的是加载/存储单元直接与[流式多处理器 (Streaming Multiprocessor)](/gpu-glossary/device-hardware/streaming-multiprocessor) 的片上 SRAM [L1 数据缓存](/gpu-glossary/device-hardware/l1-data-cache) 交互，并间接与片外设备上的 [全局内存 (global RAM)](/gpu-glossary/device-hardware/gpu-ram) 交互，这两者分别对应 [CUDA 编程模型 (CUDA programming model)](/gpu-glossary/device-software/cuda-programming-model) 中 [内存层次结构 (memory hierarchy)](/gpu-glossary/device-software/memory-hierarchy) 的最低层和最高层。