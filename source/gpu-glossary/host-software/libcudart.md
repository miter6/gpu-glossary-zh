# 什么是 libcudart.so？

在 Linux 系统中，实现 [CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api) 的二进制共享对象文件通常名为 `libcudart.so`。已部署的 CUDA 二进制文件通常会静态链接此文件，但基于 CUDA 工具包构建的库和框架（如 PyTorch）通常会动态加载它。