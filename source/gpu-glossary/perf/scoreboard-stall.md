# 什么是 scoreboard stall？

当一条指令由于依赖前一条指令的结果而无法发射时，就会发生 scoreboard stall。

scoreboard 是一种硬件结构，用于跟踪哪些[寄存器](/gpu-glossary/device-software/registers)正在等待飞行中的指令写入。当 [warp](/gpu-glossary/device-software/warp) 处于[停滞状态](/gpu-glossary/perf/warp-execution-state)时，它无法继续前进。

scoreboard stall 可以分为两类：short scoreboard stall 和 long scoreboard stall。

short scoreboard stall 发生在一条指令等待可变延迟指令的结果时，而该可变延迟指令不会离开[流式多处理器（SM）](/gpu-glossary/device-hardware/streaming-multiprocessor)。这包括[特殊函数单元](/gpu-glossary/device-hardware/special-function-unit)上的慢速数学指令，例如 `MUFU.EX2` 和 `MUFU.SQRT`，也包括 [Tensor Core](/gpu-glossary/device-hardware/tensor-core) 上的矩阵乘法，例如 `MMA`。此外还包括[共享内存](/gpu-glossary/device-software/shared-memory)操作，例如 `LDS` 和 `STS`。

long scoreboard stall 发生在一条指令等待会离开 [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) 的内存操作结果时，例如全局内存加载（`LDG`）或存储（`STG`）。long scoreboard stall 通常主导[内存受限](/gpu-glossary/perf/memory-bound)代码。

一个 [warp](/gpu-glossary/device-software/warp) 有 6 个 scoreboard，编译器会使用它们跟踪指令之间的数据依赖。

部分 scoreboard 信息可以在[流式汇编器（SASS）](/gpu-glossary/device-software/streaming-assembler)中读出。例如，下面是使用 `cuobjdump` 的 `--dump-sass` 标志时可能看到的内容：

```text
[barrier:  :  :  :  ]  /*line*/  INSTRUCTION Ri, [Rj] ; # Format: scoreboard info, line number, instruction, operands
[B------:R-:W2:-:S04]  /*00f0*/  LDG.E.SYS R0, [R2] ;   # Sets scoreboard 2
[B------:R-:W2:-:S01]  /*0100*/  LDG.E.SYS R5, [R4] ;   # `ptxas` intelligently reuses scoreboard 2
...
[B--2---:R-:W-:Y:S08]  /*0150*/  IMAD R0, R0, c[0x0][0x160], R5 ;  # Waits on scoreboard 2
```

可以看到，我们的 `IMAD` 指令在 scoreboard 2 上有一个 barrier（`B--2---`），表示它必须等待该 bit flag 清除后才能发射。两个 `LDG` 指令在发射时都会递增（`W2` write）scoreboard 2，这样 `IMAD` 指令执行前，寄存器 `R0` 和 `R5` 中就会有正确的值。

一条指令可能需要等待多个 scoreboard，例如 `B01--4-` 表示需要等待 scoreboard 0、1、4 全部清除。当数据依赖被满足后，相应的 scoreboard 会递减。

scoreboard 复用可能导致 Nsight Compute 给出的 stall 分类不准确，因为 long scoreboard stall 和 short scoreboard stall 如果使用了同一个 scoreboard，就可能被混在一起。

用于动态指令调度中依赖跟踪的 [scoreboarding](https://www.cs.umd.edu/~meesh/411/website/projects/dynamic/scoreboard.html) 可以追溯到“第一台超级计算机” [Control Data Corporation 6600](https://en.wikipedia.org/wiki/CDC_6600)，其中一台机器曾在 1966 年[推翻了欧拉幂和猜想](https://www.ams.org/journals/bull/1966-72-06/S0002-9904-1966-11654-3/S0002-9904-1966-11654-3.pdf)。与 CPU 不同，GPU 中的 scoreboarding 不用于[线程](/gpu-glossary/device-software/thread)内部的乱序执行（指令级并行），而只用于线程之间（线程级并行）；参见[这项 NVIDIA 专利](https://patents.google.com/patent/US7676657)。

关于 GPU 上 scoreboard 实现的更多细节，可参阅 [Matthew D. Sinclair 教授的课件](https://pages.cs.wisc.edu/~sinclair/courses/cs758/fall2019/handouts/lecture/cs758-fall19-gpu_uarch2.pdf)。
