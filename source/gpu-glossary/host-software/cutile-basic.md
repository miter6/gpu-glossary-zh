# 什么是 cuTile BASIC？

cuTile BASIC 是 [CUDA Tile 编程模型](/gpu-glossary/device-software/cuda-tile-programming-model)在 [BASIC 编程语言](https://modal-cdn.com/BASIC_Oct64.pdf)中的一种实现。

BASIC 是 Beginner's All-purpose Symbolic Instruction Code 的缩写。BASIC 是一种诞生于 20 世纪 60 年代的编程语言，设计目标是易用和交互式编程。它曾在早期微型计算机程序员中很流行，例如 William Gates III。

cuTile BASIC 是[作为愚人节玩笑发布的](https://developer.nvidia.com/blog/cuda-tile-programming-now-available-for-basic/)。不过它仍然是这一编程模型的真实实现，只是偏玩具性质，同时也展示了该模型的通用性。你可以使用[这个 Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-151VgRNHYEDuKSfxJRjV5N) 在 B200 GPU 上运行下面的向量加法 cuTile BASIC 内核。cuTile BASIC 的开发也部分借助了这类 Notebook。

```basic
10 REM Vector Add: C = A + B
20 INPUT N, A(), B()
30 DIM A(N), B(N), C(N)
40 TILE A(128), B(128), C(128)
50 LET C(BID) = A(BID) + B(BID)
60 OUTPUT C
70 END
```
