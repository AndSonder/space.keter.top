# 为Paddle优化Histogram在GPU上的计算性能

:::tip

飞桨黑客松赛题，【Hackathon 4 No.33】为 Paddle 优化 Histogram OP在GPU上的性能

相关PR: https://github.com/PaddlePaddle/Paddle/pull/53112

:::

## API 介绍

计算输入 Tensor 的直方图。以 min 和 max 为 range 边界，将其均分成 bins 个直条，然后将排序好的数据划分到各个直条(bins)中。如果 min 和 max 都为 0，则利用数据中的最大最小值作为边界。

```python
import paddle

inputs = paddle.to_tensor([1, 2, 1])
result = paddle.histogram(inputs, bins=4, min=0, max=3)
print(result) # [0, 2, 1, 0]
```

## 关键模块与性能提升点

关键是使用__global__ kernel的方式实现了KernelMinMax，加速Histogram确定直方图边界的计算部分，从而提高Histogram算子在GPU上的计算性能。预期能够平均提升2倍以上。

Paddle 和 Pytorch 在实现 Histogram 算子时，基本代码都一致，主要差距在于确定直方图边界的计算部分，Paddle 使用了 Eigen 的方式实现，对优化前 Paddle 的算子进行 GPU 计算分析。在分析结果中，总共有超过90%的GPU计算时间使用在 Eigen 的计算中。

## 老代码分析

首先，我们定义了几个用于 CUDA 编程的重要元素和数据类型。这些包括：

```cpp
using IndexType = int64_t;
using phi::PADDLE_CUDA_NUM_THREADS;
```

- `IndexType` 是一个自定义的整数类型，用于表示索引。
- `phi::PADDLE_CUDA_NUM_THREADS` 是一个表示 CUDA 线程数量的符号常量。

### 获取块数的函数

接下来，我们定义了一个用于计算块数的辅助函数 `GET_BLOCKS`：

```cpp
inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}
```

该函数用于根据待处理的数据点数量来确定启动 CUDA 核函数所需的块数。这有助于有效地利用 GPU 并行性。

### 直方图计算核函数

这段代码中的核心部分是直方图计算的 CUDA 核函数。核函数定义如下：

```cpp
// CUDA 核函数，用于计算输入数据的直方图
template <typename T, typename IndexType>
__global__ void KernelHistogram(const T* input,              // 输入数据数组
                                const int total_elements,   // 输入数据中的总元素数
                                const int64_t nbins,        // 直方图的分箱数
                                const T min_value,           // 数据的最小值
                                const T max_value,           // 数据的最大值
                                int64_t* output) {           // 直方图的输出数组

  // 声明一个共享内存数组，用于保存每个线程块计算的直方图分箱统计
  extern __shared__ int64_t buf_hist[];

  // 初始化共享内存为零
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    buf_hist[i] = 0;
  }

  // 等待所有线程块完成初始化
  __syncthreads();

  // 使用 CUDA_KERNEL_LOOP 迭代遍历输入数据
  CUDA_KERNEL_LOOP(input_index, total_elements) {
    // 获取输入值
    const auto input_value = input[input_index];

    // 检查输入值是否在指定的范围内
    if (input_value >= min_value && input_value <= max_value) {
      // 计算输入值所属的分箱索引
      const IndexType output_index =
          GetBin<T, IndexType>(input_value, min_value, max_value, nbins);

      // 使用原子操作将计数增加到对应的分箱
      phi::CudaAtomicAdd(&buf_hist[output_index], 1);
    }
  }

  // 等待所有线程块完成直方图计算
  __syncthreads();

  // 使用原子操作将每个线程块的计数合并到全局输出数组中
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
    phi::CudaAtomicAdd(&output[i], buf_hist[i]);
  }
}
```

这个核函数的任务是计算输入数据的直方图。核函数的输入包括：

- `input`：输入数据数组。
- `total_elements`：输入数据中的总元素数。
- `nbins`：直方图的分箱数。
- `min_value` 和 `max_value`：数据的最小值和最大值。
- `output`：直方图的输出数组。

这个核函数使用共享内存 `buf_hist` 来保存每个线程块计算的直方图分箱统计。它首先将共享内存初始化为零，然后遍历输入数据并根据数据值将计数增加到相应的分箱中。最后，通过原子操作将每个线程块的计数合并到全局输出中。

### 直方图计算函数

最后，我们有一个直方图计算函数 `HistogramKernel`：

```cpp
// CUDA 直方图计算函数
template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,            // 设备上下文
                     const DenseTensor& input,         // 输入数据
                     int64_t bins,                     // 直方图分箱数
                     int min,                          // 最小值
                     int max,                          // 最大值
                     DenseTensor* output) {            // 输出直方图

  auto& nbins = bins;                                 // 获取分箱数的引用
  auto& minval = min;                                 // 获取最小值的引用
  auto& maxval = max;                                 // 获取最大值的引用

  const T* input_data = input.data<T>();               // 获取输入数据的指针
  const int input_numel = input.numel();              // 获取输入数据的元素个数

  // 为输出分配内存并将其初始化为零
  int64_t* out_data = dev_ctx.template Alloc<int64_t>(output);
  phi::funcs::SetConstant<Context, int64_t>()(dev_ctx, output, static_cast<int64_t>(0));

  // 如果输入数据为空，直接返回
  if (input_data == nullptr) return;

  T output_min = static_cast<T>(minval);              // 将最小值转换为 T 类型
  T output_max = static_cast<T>(maxval);              // 将最大值转换为 T 类型

  // 如果最小值和最大值相等，需要根据数据中的最小和最大值重新计算
  if (output_min == output_max) {
    auto input_x = phi::EigenVector<T>::Flatten(input);

    DenseTensor input_min_t, input_max_t;
    input_min_t.Resize({1});
    input_max_t.Resize({1});
    auto* input_min_data = dev_ctx.template Alloc<T>(&input_min_t);
    auto* input_max_data = dev_ctx.template Alloc<T>(&input_max_t);
    auto input_min_scala = phi::EigenScalar<T>::From(input_min_t);
    auto input_max_scala = phi::EigenScalar<T>::From(input_max_t);

    auto* place = dev_ctx.eigen_device();
    input_min_scala.device(*place) = input_x.minimum();
    input_max_scala.device(*place) = input_x.maximum();

    DenseTensor input_min_cpu, input_max_cpu;
    phi::Copy(dev_ctx, input_min_t, phi::CPUPlace(), true, &input_min_cpu);
    phi::Copy(dev_ctx, input_max_t, phi::CPUPlace(), true, &input_max_cpu);

    output_min = input_min_cpu.data<T>()[0];
    output_max = input_max_cpu.data<T>()[0];
  }

  // 如果最小值和最大值仍然相等，调整它们以避免问题
  if (output_min == output_max) {
    output_min = output_min - 1;
    output_max = output_max + 1;
  }

  // 检查最小值和最大值是否有效
  PADDLE_ENFORCE_EQ((std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max)) ||
                     std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max))),
                    false,
                    phi::errors::OutOfRange("min 和 max 的范围不是有限值"));

  PADDLE_ENFORCE_GE(
      output_max,
      output_min,
      phi::errors::InvalidArgument(
          "max 必须大于等于 min。如果 min 和 max 都为零，则使用数据的最小值和最大值。但是接收到的 max 是 %d，min 是 %d",
          maxval,
          minval));

  auto stream = dev_ctx.stream();

  // 调用直方图计算核函数 KernelHistogram
  KernelHistogram<T, IndexType><<<GET_BLOCKS(input_numel),
                                  PADDLE_CUDA_NUM_THREADS,
                                  nbins * sizeof(int64_t),
                                  stream>>>(
      input_data, input_numel, nbins, output_min, output_max, out_data);
}
```

这个函数是代码的主要入口点。它接受输入数据 `input`、分箱数 `bins`、最小值 `min` 和最大值 `max`，并计算直方图并将结果存储在输出 `output` 中。

函数首先分配所需的内存，然后检查输入数据是否为空。接下来，它计算输出的最小值和最大值，并确保它们是有限的且 `max` 大于等于 `min`。最后，它调用核函数 `KernelHistogram` 来执行直方图计算。

可以看到这里面有一个对于边界进行处理的操作使用了Eigen的方式实现。这个操作具有很大的优化空间。

### 边界代码分析

边界代码是处理最小值（output_min）和最大值（output_max）相等的情况下进行的特殊处理。这种情况下，直方图的分箱范围无法正确确定，因此需要根据输入数据的实际最小值和最大值来重新计算这些值。

下面一步步解释这段代码的执行过程：

1. 首先，代码检查 `output_min` 和 `output_max` 是否相等：

```cpp
if (output_min == output_max) {
  // 如果最小值和最大值相等，执行以下代码块
  // ...
}
```

2. 如果它们相等，代码创建一个 Eigen 向量（`input_x`），该向量从输入数据中抽取所有元素。这个向量是 Eigen 库中的一种数据结构，用于进行数学运算。

```cpp
auto input_x = phi::EigenVector<T>::Flatten(input);
```

3. 接下来，代码创建了两个大小为 1 的 DenseTensor 对象，`input_min_t` 和 `input_max_t`，用于存储计算出的最小值和最大值。

```cpp
DenseTensor input_min_t, input_max_t;
input_min_t.Resize({1});
input_max_t.Resize({1});
```

4. 代码使用 `dev_ctx` 上下文分配了用于存储最小值和最大值的内存，并获取了对应的指针 `input_min_data` 和 `input_max_data`。

```cpp
auto* input_min_data = dev_ctx.template Alloc<T>(&input_min_t);
auto* input_max_data = dev_ctx.template Alloc<T>(&input_max_t);
```

5. 接下来，代码通过 `phi::EigenScalar<T>::From` 将 `input_min_t` 和 `input_max_t` 转换为 Eigen 标量（scalar）类型，即单一的数值。

```cpp
auto input_min_scala = phi::EigenScalar<T>::From(input_min_t);
auto input_max_scala = phi::EigenScalar<T>::From(input_max_t);
```

6. 代码获取当前设备上下文的设备（`place`），然后使用 Eigen 的设备对象来计算输入数据的最小值和最大值。

```cpp
auto* place = dev_ctx.eigen_device();
input_min_scala.device(*place) = input_x.minimum();
input_max_scala.device(*place) = input_x.maximum();
```

7. 接下来，代码创建了两个大小为 1 的 DenseTensor 对象，`input_min_cpu` 和 `input_max_cpu`，用于将计算出的最小值和最大值从 GPU 复制到 CPU。

```cpp
DenseTensor input_min_cpu, input_max_cpu;
phi::Copy(dev_ctx, input_min_t, phi::CPUPlace(), true, &input_min_cpu);
phi::Copy(dev_ctx, input_max_t, phi::CPUPlace(), true, &input_max_cpu);
```

8. 最后，代码从 CPU 版本的 `input_min_cpu` 和 `input_max_cpu` 中提取最小值和最大值，并将它们赋值给 `output_min` 和 `output_max`，以确保它们正确地更新为输入数据的实际最小值和最大值。

```cpp
output_min = input_min_cpu.data<T>()[0];
output_max = input_max_cpu.data<T>()[0];
```

总之，这段代码的作用是在最小值和最大值相等的情况下，使用 Eigen 库来计算输入数据的实际最小值和最大值，并将它们更新为正确的值，以确保直方图计算正确。这种处理方式可以应对输入数据分布特殊情况的情形。

## 新代码分析

新代码主要就是对于边界的处理进行了优化，用了一个自定义的KernelMinMax来实现。

首先先给出优化后的代码：

```cpp
// CUDA 核函数，用于计算输入数组的最小值和最大值
template <typename T>
__global__ void KernelMinMax(const T* input,          // 输入数据数组
                             const int numel,         // 输入数据的总元素数
                             const int block_num,     // 线程块的数量
                             T* min_ptr,              // 存储最小值的数组指针
                             T* max_ptr) {            // 存储最大值的数组指针

  // 计算当前线程在整个数据数组中的索引
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t i = index;
  
  // 初始化最小值和最大值为输入数据的第一个元素或零
  T min_value = static_cast<T>(i < numel ? input[i] : input[0]);
  T max_value = static_cast<T>(i < numel ? input[i] : input[0]);

  // 循环迭代数据并更新最小值和最大值
  for (; i < numel; i += blockDim.x * gridDim.x) {
    T value = static_cast<T>(input[i]);
    min_value = value < min_value ? value : min_value;
    max_value = value > max_value ? value : max_value;
  }

  // 合并线程块内的最小值和最大值
  if (max_ptr && min_ptr) {
    __syncthreads();  // 同步线程块内的线程
    
    // 使用 BlockReduceMin 和 BlockReduceMax 函数计算线程块内的最小值和最大值
    T block_min_value = phi::funcs::BlockReduceMin<T>(min_value, FINAL_MASK);
    T block_max_value = phi::funcs::BlockReduceMax<T>(max_value, FINAL_MASK);

    // 如果当前线程是线程块内的第一个线程，则将最小值和最大值存储在对应的数组中
    if (threadIdx.x == 0) {
      min_ptr[blockIdx.x] = block_min_value;
      max_ptr[blockIdx.x] = block_max_value;
    }
  }

  __syncthreads();  // 再次同步线程块内的线程

  // 如果当前线程是所有线程中的第一个线程，则合并所有线程块的最小值和最大值
  if (index == 0) {
    if (min_ptr && max_ptr) {
      min_value = min_ptr[0];
      max_value = max_ptr[0];

      // 循环遍历所有线程块的最小值和最大值，并选择其中的最小值和最大值
      for (int64_t i = 1; i < block_num; i++) {
        min_ptr[0] = min_ptr[i] < min_value ? min_ptr[i] : min_value;
        max_ptr[0] = max_ptr[i] > max_value ? max_ptr[i] : max_value;
      }

      // 如果最小值和最大值相等，将它们调整一个单位以避免问题
      if (min_ptr[0] == max_ptr[0]) {
        min_ptr[0] = min_ptr[0] - 1;
        max_ptr[0] = max_ptr[0] + 1;
      }
    }
  }
}

// 用于指定最小值和最大值的特殊版本的 CUDA 核函数
template <typename T>
__global__ void KernelMinMax(const T min_value,        // 指定的最小值
                             const T max_value,        // 指定的最大值
                             T* min_ptr,              // 存储最小值的数组指针
                             T* max_ptr) {            // 存储最大值的数组指针

  // 如果当前线程是第一个线程块的第一个线程，则将指定的最小值和最大值存储在对应的数组中
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    min_ptr[0] = min_value;
    max_ptr[0] = max_value;
  }
}
```

下面让我们通过逐步解释这个核函数的代码来深入理解它的工作原理以及为什么需要这么设计。

首先，核函数为每个线程分配一个索引，这个索引由线程的ID和线程块的ID组合而成。这意味着每个线程都有自己的任务，可以独立地处理一部分数据。这充分利用了GPU的并行计算能力，允许同时处理大量数据。

```cpp
int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
int64_t i = index;
```

接下来需要初始化最大值和最小值：

```cpp
T min_value = static_cast<T>(i < numel ? input[i] : input[0]);
T max_value = static_cast<T>(i < numel ? input[i] : input[0]);
```

每个线程初始化两个变量，min_value 和 max_value，它们分别用于存储最小值和最大值。这里使用了三元条件运算符来处理数据的边界情况，确保即使输入数组很短，也能正常工作。

接下来，代码使用循环迭代数据并更新最小值和最大值：

```cpp
for (; i < numel; i += blockDim.x * gridDim.x) {
    T value = static_cast<T>(input[i]);
    min_value = value < min_value ? value : min_value;
    max_value = value > max_value ? value : max_value;
}
```

这是核函数的核心部分。每个线程独立地遍历一部分输入数据，计算这部分数据的最小值和最大值。这个循环实际上将数据分成若干块，每个线程块中的线程处理自己的块。这样，不同线程块之间不会互相干扰，提高了并行性。

然后，将每个线程块的最小值和最大值合并到全局输出中：

```cpp
if (max_ptr && min_ptr) {
    __syncthreads();
    T block_min_value = phi::funcs::BlockReduceMin<T>(min_value, FINAL_MASK);
    T block_max_value = phi::funcs::BlockReduceMax<T>(max_value, FINAL_MASK);

    if (threadIdx.x == 0) {
        min_ptr[blockIdx.x] = block_min_value;
        max_ptr[blockIdx.x] = block_max_value;
    }
}
```

每个线程块内的线程需要合并各自的块级最小值和最大值。这里使用了 `BlockReduceMin` 和 `BlockReduceMax` 函数来实现。这些函数会在线程块内合并最小值和最大值，然后将结果存储在 block_min_value 和 block_max_value 中。最后，只有每个线程块的第一个线程（threadIdx.x == 0）会将块级最小值和最大值存储到全局内存中，以备后续合并。

最后，在全局范围内，只有一个线程（index == 0）会合并所有线程块的最小值和最大值。这个过程使用一个循环来找到全局的最小值和最大值，确保最终结果正确。如果最小值和最大值相等，它们将被微调以避免问题。

```cpp
__syncthreads();
if (index == 0) {
    if (min_ptr && max_ptr) {
        min_value = min_ptr[0];
        max_value = max_ptr[0];
        for (int64_t i = 1; i < block_num; i++) {
            min_ptr[0] = min_ptr[i] < min_value ? min_ptr[i] : min_value;
            max_ptr[0] = max_ptr[i] > max_value ? max_ptr[i] : max_value;
        }
        if (min_ptr[0] == max_ptr[0]) {
            min_ptr[0] = min_ptr[0] - 1;
            max_ptr[0] = max_ptr[0] + 1;
        }
    }
}
```

这个 CUDA 核函数的设计使得在GPU上高效计算输入数组的最小值和最大值成为可能。通过充分利用并行计算和适当的数据分割，它能够在处理大规模数据时提供卓越的性能。

## 总结

Histogram这个算子的优化，其实就是对于边界的处理进行了优化，用了一个自定义的KernelMinMax来实现。这个操作具有很大的优化空间。通过充分利用并行计算和适当的数据分割，它能够在处理大规模数据时提供卓越的性能。

通过这个算子的优化，我们可以看到，对于GPU上的算子优化，其实就是对于GPU的并行计算能力的充分利用，以及对于数据分割的合理利用。这样才能够在GPU上获得更好的性能。这也要求我们清楚GPU的架构以及CUDA编程的基本原理，才能够更好的进行GPU上的算子优化。




