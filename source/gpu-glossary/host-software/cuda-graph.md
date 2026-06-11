# 什么是 CUDA Graph？

CUDA Graph 是由[内核](/gpu-glossary/device-software/kernel)启动以及其他工作组成的图，主机可以将其一次性提交给设备执行。

CUDA Graph 的主要用途，是降低主机在短时间内识别、配置和提交大量[内核](/gpu-glossary/device-software/kernel)所带来的[开销](/gpu-glossary/perf/overhead)。每次启动大约需要微秒量级的时间，因此如果需要在毫秒级时间内启动数百个[内核](/gpu-glossary/device-software/kernel)，这部分开销就会非常明显。这种情况常见于[低延迟 LLM 推理](https://modal.com/docs/guide/high-performance-llm-inference)。

CUDA Graph 最常见的创建方式，是使用 [CUDA Runtime](/gpu-glossary/host-software/cuda-runtime-api) 中的 stream capture API。它允许捕获发生在单个 CUDA stream 上的所有操作，并在之后像下面这样重放：

```cpp
// capture
cudaStreamBeginCapture(stream);
kernelGemm<<<{32, 20},64,19200,stream>>>(a, b, c);
kernelEpilogue<<<{256,2},{8,32},0,stream>>>(c, c);
cudaStreamEndCapture(stream, &graph);

// launch
cudaGraphInstantiate(&graphExec, graph, flags);
cudaGraphLaunch(graphExec, stream);
```

NVIDIA 在[这里](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)记录了 [CUDA Runtime](/gpu-glossary/host-software/cuda-runtime-api) 中的 CUDA Graph 接口。

PyTorch 对这一 API 进行了封装，例如通过 `torch.cuda.graph` 上下文管理器完成封装；神经网络训练和推理中的 CUDA Graph 通常就是通过这种方式捕获的。

下面是一个示例 CUDA Graph，它来自 B200 GPU 执行 `torch.Linear` 层时的捕获结果：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          NODE 0: KERNEL                           │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │  ID:         0 (topoId: 1)                                        │  │
│  │  Kernel:     cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_      │  │
│  │              64x32x16_1x1x1_3_tnn_align1_bias_f32_relu            │  │
│  │              <<<{32,20},64,19200>>>                               │  │
│  │  Node handle: 0x0000564604539520                                  │  │
│  │  Func handle: 0x0000564603AFCC00                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          NODE 1: KERNEL                           │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │  ID:         1 (topoId: 0)                                        │  │
│  │  Kernel:     _ZN8cublasLt8epilogue4impl12globalKernelILi8E...     │  │
│  │              <<<{256,2},{8,32},0>>>                               │  │
│  │  Node handle: 0x0000564604539C88                                  │  │
│  │  Func handle: 0x00005646044770F0                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

注意，[内核](/gpu-glossary/device-software/kernel)是通过指针标识的，例如 `0x564603AFCC00`。输入和输出也由指针定义。这些对设备资源的引用以及其他类似引用，会阻止 CUDA Graph 被序列化，并使其不具备可移植性，除非完整地[检查点保存并恢复主机与设备内存](https://modal.com/docs/guide/memory-snapshots)。
