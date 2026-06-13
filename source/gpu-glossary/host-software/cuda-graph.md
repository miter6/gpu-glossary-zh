# 什么是 CUDA Graph ?

CUDA 图 (CUDA Graph) 是一个由核函数 ([kernel](/gpu-glossary/device-software/kernel)) 启动操作以及其他工作组成的图结构，主机端（Host）可以将其一次性整体提交给设备端（Device）执行。

CUDA 图的核心应用场景是减少由主机端在短时间内进行识别、配置和提交大量核函数所带来的开销 ([overhead](/gpu-glossary/perf/overhead))。
每次启动核函数通常需要微秒（$\mu s$）级别的耗时，因此如果需要在几毫秒内启动数百个核函数，这种开销就会变得非常明显。这在 [低延迟大语言模型 (LLM) 推理](https://modal.com/docs/guide/high-performance-llm-inference) 中是一个非常普遍的现象。

CUDA 图通常通过 CUDA 运行时 ([CUDA Runtime](/gpu-glossary/host-software/cuda-runtime-api)) 中的流捕获 API (Stream Capture API) 来创建。它允许将单个 CUDA 流（Stream）上发生的所有操作捕获下来，并在随后进行重放，示例如下：

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

NVIDIA 在 [此处](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) 对 [CUDA Runtime](/gpu-glossary/host-software/cuda-runtime-api) 中的 CUDA Graphs 接口进行了详细记录。

该 API 也被 PyTorch 进行了封装（例如通过 `torch.cuda.graph` 上下文管理器），这也是在神经网络训练和推理中通常捕获 CUDA 图的方式。

以下是一个 CUDA 图的示例，它是在 B200 GPU 上执行 `torch.Linear` 层时捕获到的：

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

请注意，核函数是通过指针来识别的（例如 `0x564603AFCC00`），输入和输出同样由指针定义。
这些以及其他对设备资源的引用，导致 CUDA 图无法被直接序列化，同时也使其失去了可移植性——除非通过完整的[主机与设备内存检查点设置及恢复技术](https://modal.com/docs/guide/memory-snapshots) 来实现。