# 什么是 nvidia.ko？

`nvidia.ko` 是 Linux 系统下 [NVIDIA GPU 驱动程序](/gpu-glossary/host-software/nvidia-gpu-drivers) 的核心二进制 [内核模块](https://wiki.archlinux.org/title/Kernel_module) 文件。

与其他内核模块类似，它以特权模式运行，并代表用户直接与硬件通信 —— 在此场景下，硬件特指 GPU。

Linux Open GPU 内核模块是 [开源的](https://github.com/NVIDIA/open-gpu-kernel-modules)。