# 什么是开销？

开销延迟是指未执行有用工作所花费的时间。

与在 [计算受限](/gpu-glossary/perf/compute-bound) 或 [内存受限](/gpu-glossary/perf/memory-bound) 下的 [性能瓶颈](/gpu-glossary/perf/performance-bottleneck) 时 GPU 以最大速度工作的情况不同，由开销引起的延迟代表 GPU 处于等待接收任务的状态。

开销通常来自 CPU 端的瓶颈，这些瓶颈阻止 GPU 及时接收任务。例如，每个内核启动的 CUDA API 调用开销大约为 10 微秒。此外，像 PyTorch 或 TensorFlow 这样的框架需要时间决定启动哪个 [内核](/gpu-glossary/device-software/kernel)，这可能花费许多微秒。我们通常将这类开销通常被称为 [(主机开销) "host overhead"](https://modal.com/blog/host-overhead-inference-efficiency) 尽管术语尚未完全标准化。 [CUDA Graphs](/gpu-glossary/host-software/cuda-graph) 是解决此类开销的常用方案，它将多个设备端 [内核](/gpu-glossary/device-software/kernel) 整合为单次主机端启动。更多信息，请参阅 [GTC 2025 上的《最大化并发和系统利用率的 CUDA 技术》演讲](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72686/)。

"内存开销" 或 "通信开销" 是在 CPU 与 GPU 之间或 GPU 与 GPU 之间来回搬运数据所产生的延迟。但当通信带宽成为限制因素时，更适合将其视为一种 [内存受限](/gpu-glossary/perf/memory-bound) 的形式，其中"内存"分布在多台机器上。