# 什么是 CUDA Driver API？

[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) 是 NVIDIA CUDA 驱动的用户空间组件。  
它为熟悉 C 标准库的用户提供了实用工具：例如，用于在 GPU 设备上分配[内存](/gpu-glossary/device-software/global-memory)的函数 `cuMalloc` 。

![](light-cuda-toolkit.svg)

> CUDA 工具包。CUDA Driver API 位于应用程序或其他工具包组件与 GPU 之间。改编自《Professional CUDA C Programming Guide》。

很少有 CUDA 程序直接使用 CUDA Driver API 编写，它们通常使用[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api)。请参阅 CUDA Driver API 文档中的[章节](https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html#driver-vs-runtime-api)。

CUDA Driver API 通常不进行静态链接，而是动态链接，在 Linux 系统上通常名为 [libcuda.so](/gpu-glossary/host-software/libcuda)。

CUDA Driver API 具有二进制兼容性：针对旧版 CUDA Driver API 编译的应用程序可以在装有新版 CUDA Driver API 的系统上运行。也就是说，操作系统的二进制加载器可以加载新版 CUDA Driver API，而程序功能保持不变。

有关发布 [CUDA C/C++](/gpu-glossary/host-software/cuda-c) 应用程序的详细信息，请参阅 NVIDIA 的 [CUDA C/C++ 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide)。

CUDA Driver API 是闭源的，您可以在 [此处](https://docs.nvidia.com/cuda/cuda-driver-api/index.html) 找到其文档。

尽管不常用，但存在一些项目试图提供或使用 CUDA Driver API 的开源替代方案，例如 [LibreCuda](https://github.com/mikex86/LibreCuda) 和 [tinygrad](https://github.com/tinygrad)。详情请参阅[其源代码](https://github.com/tinygrad/tinygrad/blob/77f7ddf62a78218bee7b4f7b9ff925a0e581fcad/tinygrad/runtime/ops_nv.py)。