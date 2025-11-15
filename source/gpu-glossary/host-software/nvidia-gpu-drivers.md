# 什么是 NVIDIA GPU 驱动程序？

NVIDIA GPU 驱动程序负责协调主机程序或主机操作系统与 GPU 设备之间的交互。  
对于应用程序而言，与 GPU 驱动程序交互的主要接口依次为 [CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api) 和 [CUDA Driver API](/gpu-glossary/host-software/cuda-driver-api)。

![](light-cuda-toolkit.svg)  

> CUDA 工具包示意图。而 NVIDIA GPU 驱动程序是唯一直接与 GPU 通信的组件。改编自《Professional CUDA C Programming Guide》。  

NVIDIA 已开源其 Linux Open GPU [内核模块](/gpu-glossary/host-software/nvidia-ko) 的[源代码](https://github.com/NVIDIA/open-gpu-kernel-modules)。