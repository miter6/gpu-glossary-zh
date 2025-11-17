# 什么是 cuDNN？

NVIDIA的cuDNN（CUDA Deep Neural Network，CUDA深度神经网络）是一个用于构建GPU加速深度神经网络的基元库。

cuDNN 提供了针对神经网络中频繁出现的运算的高度优化内核 [内核](/gpu-glossary/device-software/kernel)。包括卷积、自注意力（包括缩放点积注意力，又称 "Flash Attention"）、矩阵乘法、各种归一化、池化等。

cuDNN 是 [CUDA 软件平台](/gpu-glossary/host-software/cuda-software-platform) 应用层的关键库，与其姊妹库 [cuBLAS](/gpu-glossary/host-software/cublas) 并列。像 PyTorch 这样的深度学习框架通常利用 [cuBLAS](/gpu-glossary/host-software/cublas) 进行通用线性代数运算，例如构成稠密（全连接）层核心的矩阵乘法。而依赖 cuDNN 实现更专业的基元操作（如卷积层、归一化例程和注意力机制）。

在现代 cuDNN 代码中，计算通过操作图（operation graphs）表示，可以使用开源的 [Python 和 C++ 前端 API](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/overview.html) 通过声明式的 [Graph API](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.14.0/developer/graph-api.html) 来构建。

发者可通过该 API 将一系列操作定义为图，cuDNN 随后可以分析该图以执行优化，其中最重要的是操作融合（operation fusion）。在操作融合中，像"卷积 + 偏置 + ReLU"这样的操作序列被合并（"融合"）成一个单一操作，作为单个 [内核](/gpu-glossary/device-software/kernel) 运行。融合操作通过在整个操作序列中将程序中间结果保留在 [共享内存](/gpu-glossary/device-software/shared-memory) 中，从而减少对 [内存带宽](/gpu-glossary/perf/memory-bandwidth) 的需求。

前端通过闭源的底层 [C 后端](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html) 交互，该后端为传统用例或直接的 C FFI 暴露了一个 API 接口。

对于任何给定的操作，cuDNN 维护多个底层实现，并通过内部启发式算法（具体细节未公开）为目标 [流式多处理器 (SM) 架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)、数据类型和输入大小选择性能最佳的实现。

cuDNN 最初成名是因为在安培 [SM 架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) GPU 上加速了卷积神经网络。对于在 Hopper 尤其是 Blackwell [SM 架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 上的 Transformer 神经网络，NVIDIA 倾向于更强调 [CUTLASS](https://github.com/NVIDIA/cutlass) 库。

有关 cuDNN 的更多信息，请参阅 [官方 cuDNN 文档](https://docs.nvidia.com/deeplearning/cudnn/) 和 [开源前端 API](https://github.com/NVIDIA/cudnn-frontend)。