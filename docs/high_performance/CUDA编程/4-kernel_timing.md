# 给核函数计时

:::note

在内核的性能转换过程中，了解核函数的执行需要多长时间是很有
帮助并且十分关键的。衡量核函数性能的方法有很多。最简单的方法是
在主机端使用一个 CPU 或 GPU 计时器来计算内核的执行时间。（摘自《CUDA C 编程权威指南》）

:::

## 用 CPU 计时

在接触 CUDA 之前，我们一般使用 `clock()` 函数来计算程序的执行时间，但是在 CUDA 中，我们不能使用 `clock()` 函数来计算程序的执行时间，因为 `clock()` 函数返回的是 CPU 的时钟周期数。 我们需要使用 `gettimeofday` 函数来计算程序的执行时间。下面是一个使用 `gettimeofday` 函数计算程序执行时间的例子。

首先我们需要引入头文件 `sys/time.h` ，然后我们定义一个函数 `cpuSecond()` ，这个函数用来获取当前时间，然后我们在主函数中调用这个函数两次，第一次调用这个函数的时候，我们将获取的时间保存到变量 `start` 中，第二次调用这个函数的时候，我们将获取的时间保存到变量 `end` 中，然后我们就可以计算程序的执行时间了。

```c

```c
#include <sys/time.h>
...

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
```

有了这个函数之后，我们就可以计算程序的执行时间了。

```c
int main(int argc, char **argv)
{
    ...
    double iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    ...
}
```

这里我们用上文中的向量加法的例子来计算程序的执行时间。 需要注意的是，我们需要在调用 `sumArraysOnHost()` 函数之后调用 `cudaDeviceSynchronize()` 函数。因为核函数是异步执行的，如果我们不调用 `cudaDeviceSynchronize()` 函数，那么程序会立即执行下一条语句，这样就会导致程序的执行时间计算错误。

## 用 nvprof 计时

nvprof 是 NVIDIA 提供的一个用来分析 CUDA 程序性能的工具，我们可以使用 nvprof 来计算程序的执行时间。下面是一个使用 nvprof 计算程序执行时间的例子。

这里我们用上文中的向量加法的例子来计算程序的执行时间。 

```bash
nvprof ./vector_add
```

这里有时候会出现下面的错误。

```bash
nvprof is not supported on devices with compute capability 8.0 and higher.
Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling. Refer https://developer.nvidia.com/tools-overview for more details.
```

其实很多显卡 nvprof 都已经不支持了，以后我们会介绍如何使用 NVIDIA Nsight Systems 和 NVIDIA Nsight Compute 来计算程序的执行时间。

## 总结

在本节中，我们介绍了如何使用 `gettimeofday` 函数来计算程序的执行时间，以及如何使用 nvprof 来计算程序的执行时间。在后面的章节中，我们会介绍如何使用 NVIDIA Nsight Systems 和 NVIDIA Nsight Compute 来计算程序的执行时间。这些更高级的工具可以帮助我们更好的分析程序的性能。



## 参考资料

1. [CUDA C 编程权威指南](https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_idx=1&tn=baidu&wd=CUDA%20C%E7%BC%96%E7%A8%8B%E6%9D%83%E5%A8%81%E6%8C%87%E5%8D%97&fenlei=256&rsv_pq=0xfed4a61a000e3772&rsv_t=0d02lKS%2Blx%2BdvIVO447ej8nu1F1JZ2R2sUUEGNoSYLiNj3M8QV7s%2FscVGcDD&rqlang=en&rsv_enter=1&rsv_dl=tb&rsv_sug3=2&rsv_sug1=2&rsv_sug7=101&rsv_sug2=0&rsv_btype=i&prefixsug=%2526lt%253BUDA%2520%2526lt%253B%25E7%25BC%2596%25E7%25A8%258B%25E6%259D%2583%25E5%25A8%2581%25E6%258C%2587%25E5%258D%2597&rsp=9&inputT=4428&rsv_sug4=4428)
2. [【CUDA 基础】2.2 给核函数计时](https://face2ai.com/CUDA-F-2-2-%E6%A0%B8%E5%87%BD%E6%95%B0%E8%AE%A1%E6%97%B6/)



