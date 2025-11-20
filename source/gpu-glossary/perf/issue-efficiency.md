# 什么是发射效率？

发射效率 (Issue efficiency) 用于衡量 [线程束调度器](/gpu-glossary/device-hardware/warp-scheduler) 通过从 [就绪线程束](/gpu-glossary/perf/warp-execution-state) 发射指令，使执行流水线保持忙碌状态的效率。

![](light-cycles.svg)

> 在该示意图的四个时钟周期中，有三个周期发射了指令，因此发射效率为75%。该图灵感来源于GTC 2025上的 [*CUDA Techniques to Maximize Compute and Instruction Throughput*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/)演讲。

发射效率为 100% 意味着每个 [调度器](/gpu-glossary/device-hardware/warp-scheduler) 在每个周期都发射了一条指令，表明每个周期至少有一个 [就绪线程束](/gpu-glossary/perf/warp-execution-state)。低于 100% 的数值表明，在某些周期内，所有 [活跃线程束](/gpu-glossary/perf/warp-execution-state) 都处于 [停滞状态](/gpu-glossary/perf/warp-execution-state)——正在等待数据、资源或依赖关系——因此 [调度器](/gpu-glossary/device-hardware/warp-scheduler) 处于空闲状态，整体指令吞吐量下降。