# 什么是 cuBLAS？

cuBLAS (CUDA Basic Linear Algebra Subroutines，CUDA 基础线性代数子程序) 是 NVIDIA 对 [基础线性代数子程序 (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) 标准的高性能实现。它是一个专有软件库，提供了针对常见线性代数运算高度优化的 [内核 (kernel)](/gpu-glossary/device-software/kernel)。

开发者无需从头编写和优化像矩阵乘法这样的常见运算，而是可以直接从他们的主机代码中调用 cuBLAS 函数。该库包含大量内核，每个内核都针对特定的数据类型（例如 FP32、FP16）、矩阵大小和 [流式多处理器 (Streaming Multiprocessor, SM) 架构](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) 进行了精细调优。在运行时，cuBLAS 会使用内部启发式算法（具体细节未公开）选择性能最佳的内核及其最优启动参数。因此，cuBLAS 成为了在 NVIDIA GPU 上进行大多数 [高性能 (high-performance)](/gpu-glossary/perf/index) 数值计算的基础，并被 PyTorch 等深度学习框架广泛使用，用于加速其核心运算，同时使用的还有更专门的 [内核 (kernel)](/gpu-glossary/device-software/kernel) 库，例如 [cuDNN](/gpu-glossary/host-software/cudnn)。

使用 cuBLAS 时最常见的一个错误来源是矩阵数据布局。由于历史原因，并且为了保持与原始 BLAS 标准（用 Fortran 编写）的兼容性，cuBLAS 期望矩阵采用 [列优先 (column-major order)](https://en.wikipedia.org/wiki/Row-_and_column-major_order)。 这与 C、C++ 和 Python 中常用的行优先相反。此外，BLAS 函数不仅需要知道运算规模（例如 `M`, `N`, `K`），还需要知道如何在内存中定位每一列的起始位置——这由前导维度（leading dimension，如lda）指定。前导维度是连续列之间的步长：当处理整个已分配矩阵时，前导维度等于行数；而处理子矩阵时，前导维度则为从中提取子矩阵的更大父矩阵的行数。

幸运的是，对于GEMM（通用矩阵乘法）等计算密集型内核，无需将矩阵从行优先重新排序为列优先。相反，我们可以利用数学上的恒等式：如果 `C = A @ B`，那么 `C^T = B^T @ A^T`。核心思路是：以行优先存储的矩阵，其内存布局与按列优先存储的转置矩阵完全相同。因此，如果我们向 cuBLAS 提供我们的行优先矩阵 `A` 和 `B`，但交换它们在函数调用中的顺序（并调整维度参数），cuBLAS 将计算 `C^T` 并以列优先输出它。当这段内存按行优先解读时，恰好就是我们所需的矩阵 `C`。这种技巧可通过以下函数示例说明：

```cpp
#include <cublas_v2.h>

// 在行主序矩阵上使用 cublasSgemm 执行单精度运算 C = alpha * A @ B + beta * C
void sgemm_row_major(cublasHandle_t handle, int M, int N, int K,
                     const float *alpha,
                     const float *A, const float *B,
                     const float *beta,
                     float *C) {

  // A 是 M x K (行主序), cuBLAS 将其视为 A^T (K x M, 列主序),
  //   A^T 的主维度是 K
  // B 是 K x N (行主序), cuBLAS 将其视为 B^T (N x K, 列主序),
  //   B^T 的主维度是 N
  // C 是 M x N (行主序), cuBLAS 将其视为 C^T (N x M, 列主序),
  //   C^T 的主维度是 N

  // 注意交换了 A 和 B 的位置，以及交换了 M 和 N 的位置
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K,
              alpha,
              B, N,  // B^T 的主维度
              A, K,  // A^T 的主维度
              beta,
              C, N); // C^T 的主维度
}
```

此示例的一个完整可运行版本可在 [Godbolt](https://godbolt.org/z/axzYb75ro) 上找到。

`CUBLAS_OP_N` 标志用于指示内核按矩阵的原始形式使用（从其视角来看，无需额外的转置操作）。

要使用 cuBLAS 库，必须在编译时进行链接（例如，在使用 [nvcc](/gpu-glossary/host-software/nvcc) 编译时使用 `-lcublas` 标志）。其函数通过 `cublas_v2.h` 头文件暴露。

有关 cuBLAS 的更多信息，请参阅 [官方 cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/)。
