.. gpu-glossary documentation master file, created by
   sphinx-quickstart on Mon Nov 10 20:07:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

性能
==========================

当应用程序在通用硬件上的性能不足时，就会使用 GPU。这使得为 GPU 编程与大多数其他编程形式截然不同。
对于传统的计算机应用程序，如数据库管理系统或 Web 服务器，正确性是首要关注点。如果应用程序丢失数据或返回错误结果，则意味着应用程序失败。性能常常被忽略。
在为GPU 编程时，正确性的定义通常很模糊。"正确"输出可能只定义到一定数量的有效位，或者仅针对某些不确定的"表现良好"输入子集。而且，正确性充其量只是必要条件而非充分条件。如果应用程序的程序员无法实现卓越的性能（每秒性能、每美元性能或每瓦特性能），那么应用程序就失败了。
GPU 编程过于困难且受限，运行成本又高，因此情况只能如此。
在 NVIDIA，这一事实被概括为一句精辟的口号："性能即产品"。
本部分 GPU 术语表收集并定义了优化 GPU 上运行程序性能所需理解关键术语。
大致来说，它应涵盖您在使用 [NSight Compute](https://developer.nvidia.com/nsight-compute) 调试 GPU [内核](/gpu-glossary/device-software/kernel) 性能问题时遇到的所有术语。%                                                                                                      

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   index.rst

   性能瓶颈<performance-bottleneck.md>
   屋顶线模型<roofline-model.md>
   计算受限<compute-bound.md>
   内存受限<memory-bound.md>
   算术强度<arithmetic-intensity.md>
   开销 (Overhead) <overhead.md>
   利特尔定律<littles-law.md>
   内存带宽<memory-bandwidth.md>
   算术带宽<arithmetic-bandwidth.md>
   延迟隐藏<latency-hiding.md>
   Warp执行状态<warp-execution-state.md>
   活跃周期<active-cycle.md>
   占用率 (Occupancy)<occupancy.md>
   流水线利用率<pipe-utilization.md>
   峰值速率<peak-rate.md>
   发射效率<issue-efficiency.md>
   SM利用率<streaming-multiprocessor-utilization.md>
   Warp分歧<warp-divergence.md>
   分支效率<branch-efficiency.md>
   内存合并<memory-coalescing.md>
   Bank冲突<bank-conflict.md>
   寄存器压力<register-pressure.md>
