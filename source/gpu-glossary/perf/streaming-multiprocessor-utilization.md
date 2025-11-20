# 什么是 SM 利用率？

SM 利用率衡量 [流式多处理器 (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor) 执行指令的时间百分比。

SM 利用率类似于 [`nvidia-smi`](/gpu-glossary/host-software/nvidia-smi) 报告中更为人熟知的 [内核利用率](https://modal.com/blog/gpu-utilization-guide)，但粒度更细。它并非报告 [内核](/gpu-glossary/device-software/kernel) 在 GPU 任意位置执行的时间占比，而是报告所有 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 执行 [内核](/gpu-glossary/device-software/kernel) 的时间比例。如果一个 [内核](/gpu-glossary/device-software/kernel) 仅使用一个 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor)（例如，因为它只有一个 [线程块](/gpu-glossary/device-software/thread-block)），那么在其活动期间，GPU 利用率可达到 100%，但 SM 利用率最多仅为 SM 总数的倒数——在 H100 GPU 中这一比例不足 1 %。

[与 GPU 利用率类似但不同于 CPU 利用率](https://modal.com/blog/gpu-utilization-guide)，SM 利用率应保持较高水平，甚至达到 100%。

但尽管 SM 利用率比 GPU 利用率粒度更细，它仍不足以精确反映 GPU 计算资源的使用效率。如果 SM 利用率很高，但性能仍不理想，程序员应检查 [流水线利用率](/gpu-glossary/perf/pipe-utilization)，该指标衡量每个 SM 使用其内部功能单元的利用效率。高 SM 利用率伴随低 [流水线利用率](/gpu-glossary/perf/pipe-utilization) 表明您的 [内核](/gpu-glossary/device-software/kernel) 正在许多 SM 上运行，但未能充分利用每个 SM 内的计算资源。