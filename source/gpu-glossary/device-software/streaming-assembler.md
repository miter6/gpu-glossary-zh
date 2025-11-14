# 什么是流式汇编器？ (SASS)

[流式汇编器 (Streaming ASSembler, SASS)](https://stackoverflow.com/questions/9798258/what-is-sass-short-for) 是在 NVIDIA GPU 上运行的程序的汇编格式。它是编写人类可读代码的最低级别格式，是 `nvcc`（[NVIDIA CUDA 编译器驱动程序](/gpu-glossary/host-software/nvcc)）输出的格式之一，与 [PTX](/gpu-glossary/device-software/parallel-thread-execution) 格式并列。它在执行过程中，会被转换为特定于设备的二进制微码。 "流式汇编器"中的"流式"大概指的是该汇编语言程序所面向的是 [流式多处理器](/gpu-glossary/device-hardware/streaming-multiprocessor) (SM)。

SASS 是有版本的，并且与特定的 NVIDIA GPU [SM 架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 相关联。另请参阅 [计算能力](/gpu-glossary/device-software/compute-capability)。

以下是 Hopper GPU 的 SM90a 架构的一些 SASS 示例指令：
- `FFMA R0, R7, R0, 1.5 ;` - 执行 `Fused Floating point Multiply Add`（融合浮点乘加运算），将寄存器 R7 和寄存器 R0 的内容相乘，再加上 1.5，并将结果存储在寄存器 R0 中。
- `S2UR UR4, SR_CTAID.X ;` - 将 [协作线程数组](/gpu-glossary/device-software/cooperative-thread-array) (CTA) 索引的 `X` 值从其 `Special Register`（特殊寄存器）复制到 `Uniform Register`（通用寄存器） 4。

与 CPU 相比，手动编写这种"GPU 汇编器"更为罕见。在性能分析和编辑高级 [CUDA C/C++](/gpu-glossary/host-software/cuda-c) 代码或内联 [PTX](/gpu-glossary/device-software/parallel-thread-execution) 时，查看编译器生成的 SASS [更为常见](https://docs.nvidia.com/gameworks/content/developertools/desktop/ptx_sass_assembly_debugging.htm)，尤其是在开发高性能内核时。[Godbolt](https://godbolt.org/z/5r9ej3zjW) 平台支持同时查看 [CUDA C/C++](/gpu-glossary/host-software/cuda-c)、SASS 和 [PTX](/gpu-glossary/device-software/parallel-thread-execution) 代码。若想了解更多以性能调试流程为重点的SASS细节，请参阅 Arun Demeure 的[这个演讲](https://www.youtube.com/watch?v=we3i5VuoPWk)。

SASS 的文档记录 *非常* 少——虽然 [NVIDIA CUDA 二进制工具文档](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref) 中列出了相关指令，但未定义其语义。从 ASCII 汇编到二进制操作码及操作数的映射完全没有公开文档，但在某些情况下已被逆向工程（[Maxwell](https://github.com/NervanaSystems/maxas)、[Lovelace](https://kuterdinel.com/nv_isa_sm89/)）。