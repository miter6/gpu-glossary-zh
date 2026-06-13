# 什么是 cuTile BASIC ?
cuTile BASIC 是 [CUDA Tile 编程模型](/gpu-glossary/device-software/cuda-tile-programming-model) 在 [BASIC 编程语言](https://modal-cdn.com/BASIC_Oct64.pdf) 中的一种实现。

BASIC 是 “Beginner's All-purpose Symbolic Instruction Code”（初学者通用符号指令代码）的缩写。BASIC 是一种设计于 20 世纪 60 年代的编程语言，旨在提供易用性和交互式编程体验。它曾在早期的微型计算机程序员（如威廉·盖茨三世，即比尔·盖茨）中风靡一时。

cuTile BASIC 是 作为一个 [愚人节玩笑](https://developer.nvidia.com/blog/cuda-tile-programming-now-available-for-basic/) 发布的。尽管它只是个玩具级的项目，但它确实是该编程模型的一个真实实现，并展示了该模型的通用性。
您可以使用这个 [Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-151VgRNHYEDuKSfxJRjV5N) 在 B200 GPU 上运行下方展示的 cuTile BASIC vector-addition 核函数。cuTile BASIC 的部分开发工作正是通过此类 Notebook 完成的。

```
10 REM Vector Add: C = A + B
20 INPUT N, A(), B()
30 DIM A(N), B(N), C(N)
40 TILE A(128), B(128), C(128)
50 LET C(BID) = A(BID) + B(BID)
60 OUTPUT C
70 END 
```