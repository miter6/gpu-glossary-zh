.. gpu-glossary documentation master file, created by
   sphinx-quickstart on Mon Nov 10 20:07:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

性能
==========================

GPU 的使用场景往往是通用硬件性能不足以支撑应用需求时，这使得 GPU 编程与其他多数编程形式有显著差异。

对于数据库管理系统或 Web 服务器等传统计算机应用，正确性是首要考量——若应用丢失数据或返回错误结果，便意味着失败，性能问题常被搁置。  

而 GPU 编程中，“正确性”的定义通常较为模糊：“正确”输出可能仅要求达到一定有效位数，或仅针对部分“行为良好”的输入子集有效。更重要的是，正确性最多只是必要条件而非充分条件。如果开发者无法实现更优性能（无论是每秒运算量、每美元成本还是每瓦能耗），应用就失去了存在的意义——毕竟 GPU 编程难度高、限制多，硬件运行成本也不低，没有理由在性能上妥协。  

NVIDIA 用一句精炼的口号概括了这一点: “性能即产品” (performance is the product)。  

GPU 术语表的这一部分汇集并定义了优化 GPU 程序性能所需理解的关键术语。

大致来说，它应涵盖您在使用 `NSight Compute <https://developer.nvidia.com/nsight-compute>`_ 调试 GPU `内核 </gpu-glossary/device-software/kernel.html>`_ 性能问题时遇到的所有术语。                                                                                                      

.. toctree::
   :maxdepth: 2
   :caption: Contents:

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
   Warp分化<warp-divergence.md>
   分支效率<branch-efficiency.md>
   内存合并<memory-coalescing.md>
   Bank冲突<bank-conflict.md>
   寄存器压力<register-pressure.md>
