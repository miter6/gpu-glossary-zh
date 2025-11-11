<!--
原文: 文件路径: gpu-glossary/host-software/cuda-software-platform.md
翻译时间: 2025-11-06 18:29:14
-->

---
 CUDA 软件平台是什么？
---

CUDA 全称是 _Compute Unified Device Architecture_（计算统一设备架构）。  
根据上下文，"CUDA"可以指代多个不同的事物：一种[总体设备架构](/gpu-glossary/device-hardware/cuda-device-architecture)，或是针对该[架构设计的一种并行编程模型](/gpu-glossary/device-software/cuda-programming-model)，或是扩展高级语言（如 C 语言）以添加该编程模型的[软件平台](/gpu-glossary/host-software/cuda-software-platform)。

CUDA 的愿景在 [Lindholm 等人于 2008 年](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)发布的白皮书中进行了阐述。我们强烈推荐这篇论文，它是 NVIDIA 文档中许多论断、图表乃至特定措辞的原始来源。

在这里，我们重点介绍 CUDA _软件平台_。

CUDA 软件平台是用于开发 CUDA 程序的一系列软件集合。
虽然也存在针对其他语言（如 FORTRAN）的 CUDA 软件平台，但我们将重点介绍主流的[CUDA C++](/gpu-glossary/host-software/cuda-c) 版本。

该平台大致可分为用于 _构建_ 应用程序的组件
（如[NVIDIA CUDA 编译器驱动](/gpu-glossary/host-software/nvcc) 工具链），以及在应用程序中 _使用_ 或 _调用_ 的组件（如[CUDA Driver API](/gpu-glossary/host-software/cuda-driver-api)和[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api)），如下图所示。

![](light-cuda-toolkit.svg)  

> CUDA 工具包。改编自《Professional CUDA C Programming Guide》。

在这些 API 之上构建的是用于为通用和特定领域构建优化[内核 (kernel)](/gpu-glossary/device-software/kernel)的库，例如用于线性代数的[cuBLAS](/gpu-glossary/host-software/cublas)和用于深度神经网络的[cuDNN](/gpu-glossary/host-software/cudnn)。