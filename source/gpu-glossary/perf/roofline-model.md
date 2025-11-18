# 什么是屋顶线模型？

屋顶线模型是一种简化的、可视化的性能模型，用于快速判断程序是受 [内存带宽](/gpu-glossary/perf/memory-bandwidth) 限制还是[算术带宽](/gpu-glossary/perf/arithmetic-bandwidth) 限制。

![](light-roofline-model.svg)

> 在该模型中，位于脊点左侧的 [内核](/gpu-glossary/device-software/kernel) 受 [内存子系统带宽限制](/gpu-glossary/perf/memory-bound)，位于脊点右侧的 [内核](/gpu-glossary/device-software/kernel) 时受 [算术子系统带宽限制](/gpu-glossary/perf/compute-bound)。图表改编自提出屋顶线模型的 [Williams、Waterman 和 Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)。

屋顶线模型中，两条由硬件特性决定的 “屋顶线” 构成了性能的上限：

- "计算屋顶线"——目标硬件（[CUDA 核心](/gpu-glossary/device-hardware/cuda-core) 或 [张量核心](/gpu-glossary/device-hardware/tensor-core)）的 [峰值速率](/gpu-glossary/perf/peak-rate)，也称为 [算术带宽](/gpu-glossary/perf/arithmetic-bandwidth)
- "内存屋顶线"——目标硬件的峰值内存吞吐量，也称为 [内存带宽](/gpu-glossary/perf/memory-bandwidth)

将这些屋顶线绘制在一个平面上，其中 x 轴表示 [算术强度](/gpu-glossary/perf/arithmetic-intensity)（单位：操作数/字节），y 轴表示性能（单位：操作数/秒）。"计算屋顶线" 是一条水平线，高度等于 [算术带宽](/gpu-glossary/perf/arithmetic-bandwidth)。"内存屋顶线" 是一条斜线，斜率等于 [内存带宽](/gpu-glossary/perf/memory-bandwidth)。斜率是 "垂直变化量除以水平变化量"，因此该线的单位是字节/秒（即操作数/秒除以操作数/字节）。

通过某个 [内核](/gpu-glossary/device-software/kernel) 的 x 坐标，可立即判断其本质上是 [计算受限](/gpu-glossary/perf/compute-bound)（位于水顶下方）还是 [内存受限](/gpu-glossary/perf/memory-bound)（位于斜顶下方）。由于 [开销](/gpu-glossary/perf/overhead) 的影响，[内核] (/gpu-glossary/device-software/kernel) 很少能真正触及这两条屋顶线。

屋顶线上的边界，即斜屋顶和平屋顶相交处，称为 "脊点"。其 x 坐标是摆脱内存 [性能瓶颈](/gpu-glossary/perf/performance-bottleneck) 所需的最小 [算术强度](/gpu-glossary/perf/arithmetic-intensity)。脊点越靠左的计算机系统越容易实现最大性能；但随着时间推移，内存性能相对于计算性能的扩展速度较慢，导致系统的脊点普遍向右移动。

计算和内存屋顶线只需针对每个子系统推导一次（但需注意，它们因子系统而异，而非仅由系统决定；例如 [张量核心](/gpu-glossary/device-hardware/tensor-core) 就比 [CUDA 核心](/gpu-glossary/device-hardware/cuda-core) 具有更高的 FLOPS）。

NVIDIA 用于 [内核](/gpu-glossary/device-software/kernel) 性能工程的 NSight Compute 工具在分析 [内核](/gpu-glossary/device-software/kernel) 性能时自动执行屋顶线分析。

屋顶线模型看似简单，实则不然。例如，系统延迟并未出现在图表中的任何位置，仅包含带宽和吞吐量。它之所以简单，是因为它具有很强的倾向性，理解这些倾向性及其背后的原理是理解屋顶线模型的威力和正确应用的关键。

屋顶线模型由 Samuel Williams、Andrew Waterman 和 David Patterson 在 [这篇 2008 年的论文](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) 中提出。他们提出该模型时，硬件领域的几大趋势已深刻影响着系统架构的发展。

首先，正如 Patterson 在 2004 年一篇著名论文中单独观察到的，[延迟的提升滞后于带宽](https://dl.acm.org/doi/pdf/10.1145/1022594.1022596)。更具体地说，在计算、内存和存储等子系统中，延迟的线性改进历史上一直伴随着带宽的二次方提升。这表明未来的系统将像 GPU 一样，以吞吐量为导向。

其次，正如长期以来观察到的，计算子系统（如处理器核心）的性能扩展速度远快于内存子系统，如 [缓存](/gpu-glossary/device-hardware/l1-data-cache) 和 [DRAM](/gpu-glossary/device-hardware/gpu-ram)。这在 1994 年被 Wulf 和 McKee 普及为 [内存墙](https://www.eecs.ucf.edu/~lboloni/Teaching/EEL5708_2006/slides/wulf94.pdf)。

最后，21 世纪初，由于晶体管固定漏电流带来的功耗和散热问题，[登纳德缩放比例定律](https://en.wikipedia.org/wiki/Dennard_scaling)（即在同等功耗下提高时钟速度）宣告终结。此前，提高时钟频率一直是CPU等通用延迟导向型系统性能提升的主要手段。而这一放缓并未伴随着 [摩尔定律](https://en.wikipedia.org/wiki/Moore%27s_law)（即每芯片晶体管数量增加）的停滞。面对晶体管数量过剩但功耗受限的困境，硬件专业化成为架构上的解决方案：将计算机拆分为专注于特定任务的组件。关于一个记录详尽的例子，请参见 [Pixel Visual Core](https://blog.google/products/pixel/pixel-visual-core-image-processing-and-machine-learning-pixel-2/) 图像协处理器，该处理器在 Hennessy 和 Patterson 的第六版 [_计算机体系结构_](https://archive.org/details/computerarchitectureaquantitativeapproach6thedition/page/n13/mode/2up) 第 7 章中有详细解释。

综合这些趋势，作者们准确预测：未来的系统将以吞吐量为导向，并且在各种起作用的带宽中，[内存子系统的带宽](/gpu-glossary/perf/memory-bandwidth) 将是主要的 [性能瓶颈](/gpu-glossary/perf/performance-bottleneck)。因此，希望在这些系统上达到峰值性能的应用程序，应用程序需针对硬件的特定操作具备高运算强度——对于 GPU 而言，即针对 [张量核心](/gpu-glossary/device-hardware/tensor-core) 的[算术强度](/gpu-glossary/perf/arithmetic-intensity)，也就是说需要执行大规模的矩阵乘法。