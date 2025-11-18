# 计算受限的含义是什么？

计算受限的 [内核 (Kernel)](/gpu-glossary/device-software/kernel) 受限于 [CUDA 核心 (CUDA Core)](/gpu-glossary/device-hardware/cuda-core) 或 [张量核心 (Tensor Core)](/gpu-glossary/device-hardware/tensor-core) 的 [算术带宽 (arithmetic bandwidth)](/gpu-glossary/perf/arithmetic-bandwidth)。

![](light-roofline-model.svg)

> 在上述的 [屋顶线图 (roofline diagram)](/gpu-glossary/perf/roofline-model) 中，位于蓝线以下的 [内核 (kernel)](/gpu-glossary/device-software/kernel) 属于计算受限。图表改编自 [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)。

计算受限内核的特征是具有高 [算术强度 (arithmetic intensity)](/gpu-glossary/perf/arithmetic-intensity)（每加载或存储一字节内存需要执行大量算术运算）。其性能瓶颈在 [算术流水线利用率 (Utilization of arithmetic pipes)](/gpu-glossary/perf/pipe-utilization)。

从技术角度而言，计算受限性仅针对单个 [内核 (kernel)](/gpu-glossary/device-software/kernel) 定义，作为 [屋顶线模型 (roofline model)](/gpu-glossary/perf/roofline-model) 的一部分，但稍作引申后，可以将其推广到构成典型工作负载的多个 [内核 (kernel)](/gpu-glossary/device-software/kernel)。

大型扩散模型的推理工作负载通常是计算受限的。当代大型语言模型的推理工作负载在批量预填充/提示处理阶段通常是计算受限的。此时每个权重可以加载到 [共享内存 (shared memory)](/gpu-glossary/device-software/shared-memory) 中一次，然后多个令牌重复使用。

让我们基于 [kipperrii](https://twitter.com/kipperrii) 的 [Transformer 推理算术](https://kipp.ly/transformer-inference-arithmetic) 框架做一个简单估算，对计算受限的 Transformer 语言模型推理的最小令牌间延迟（inter-token latency，即每个输出令牌的生成时间）进行简单估算。假设某个模型有 5000 亿参数，以 16 位精度存储，总计 1 TB。每个批处理元素需执行约1万亿次浮点运算（每个参数一次乘法和一次累加）。在具有 16 位矩阵运算的 1 petaFLOP/s [算术带宽 (arithmetic bandwidth)](/gpu-glossary/perf/arithmetic-bandwidth) 的 GPU 上运行，在计算受限的假设下，每个批处理元素的最小令牌间延迟为 1 毫秒。

需要注意的是，要使该 GPU 在批次大小为 1 时达到计算受限，需要具备 1 PB/s 的 [内存带宽 (memory bandwidth)](/gpu-glossary/perf/memory-bandwidth)（以便在 1 毫秒内加载全部 1 TB 权重）。当代 [内存带宽 (memory bandwidth)](/gpu-glossary/perf/memory-bandwidth) 在 TB/s 量级，因此需要数百个输入组成的批处理才能提供足够的 [算术强度 (arithmetic intensity)](/gpu-glossary/perf/arithmetic-intensity) 使执行过程进入计算受限状态。。

有关 LLM 推理的更多信息，请参阅我们的 [LLM 工程师指南](https://modal.com/llm-almanac/summary)。