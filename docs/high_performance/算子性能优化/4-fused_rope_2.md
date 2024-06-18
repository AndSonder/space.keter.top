# Rope 算子实现调研

## 1. Paddle 中的实现

FusedRopeKernel 是 PaddlePaddle 框架中的一个用于在 GPU 上高效地应用旋转位置嵌入（rotary positional embeddings）的核函数。它能够处理输入张量 q，并在需要时处理可选的输入张量 k 和 v，从而为 transformer 模型中的注意力机制提供位置编码。

FusedRopeKernel 的实现位于 `paddle/phi/kernels/fusion/gpu/fused_rope_kernel.cu` 文件中。下面是该文件中的核函数的声明：


```cpp
void FusedRopeKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const paddle::optional<DenseTensor>& k,
                     const paddle::optional<DenseTensor>& v,
                     const paddle::optional<DenseTensor>& sin,
                     const paddle::optional<DenseTensor>& cos,
                     const paddle::optional<DenseTensor>& position_ids,
                     bool use_neox_rotary_style,
                     bool time_major,
                     float rotary_emb_base,
                     DenseTensor* out_q,
                     DenseTensor* out_k,
                     DenseTensor* out_v) 
```

FusedRopeKernel 中主要是对数据的初始化和格式检查，具体实现在 VectorizedFusedRopeCudaKernelFunc 函数中。VectorizedFusedRopeCudaKernelFunc 函数是 FusedRopeKernel 的实现函数，它是一个 GPU 核函数，用于在 GPU 上执行旋转位置嵌入的计算。

FusedRopeKernel 中一个线程块（block）处理一个 batch 的数据，一个线程（thread）处理一个 head 的数据。在每个线程中，首先计算当前线程需要处理的数据索引和步长，然后根据索引读取输入数据，计算正弦和余弦值，最后对输入数据进行旋转操作。一次迭代处理两个相邻元素，通过正弦和余弦值将输入数据旋转到新的位置编码。这种方法增强了模型对序列中元素位置信息的感知能力，特别适用于处理长序列的数据。

### 1.1 参数初始化与格式检查

在核函数中，首先对输出张量进行内存分配，并确定批量大小（batch size）、序列长度（sequence length）和头部维度（head dimension）。

```cpp
int64_t numel = q.numel();
if (numel <= 0) return;
dev_ctx.template Alloc<T>(out_q);

phi::Array<int64_t, 3> inputs_num_heads;

auto batch_size = time_major ? q.dims()[1] : q.dims()[0];
auto seq_len = time_major ? q.dims()[0] : q.dims()[1];
inputs_num_heads[0] = q.dims()[2];
auto head_dim = q.dims()[3];

PADDLE_ENFORCE_EQ(head_dim % 2,
                  0,
                  phi::errors::InvalidArgument(
                      "The head_dim of input must be a multiple of 2."));
```

接下来，配置 GPU 核函数的启动参数，包括网格大小（grid size）和块大小（block size）

```cpp
constexpr const int vec_size = 2;
auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);
int64_t grid = config.block_per_grid.x;
int64_t block = config.thread_per_block.x;
auto stream = dev_ctx.stream();
```

接下来，初始化输入和输出数据指针的数组，并为 q，k 和 v 分配指针。

```cpp
phi::Array<T*, 3> outs_data;
phi::Array<const T*, 3> ins_data;
phi::Array<const T*, 2> sin_cos_data;
const int64_t* position_ids_data = NULL;

ins_data[0] = q.data<T>();
outs_data[0] = out_q->data<T>();
int num_inputs = 1;

if (k) {
    dev_ctx.template Alloc<T>(out_k);
    ins_data[num_inputs] = k->data<T>();
    outs_data[num_inputs] = out_k->data<T>();
    inputs_num_heads[num_inputs] = k->dims()[2];
    num_inputs++;
}

if (v) {
    dev_ctx.template Alloc<T>(out_v);
    ins_data[num_inputs] = v->data<T>();
    outs_data[num_inputs] = out_v->data<T>();
    inputs_num_heads[num_inputs] = v->dims()[2];
    num_inputs++;
}
```

接下来检查是否提供了正弦和余弦张量，并验证它们的维度是否正确。如果提
供了位置 ID 张量，也需要进行相应的验证。在 FusedRopeKernel 中，检查正弦和余弦张量的维度是为了确保输入数据的一致性和正确性，这对于后续的计算至关重要。具体的检查步骤如下：

1. **检查正弦和余弦张量的维度是否一致**
2. **检查正弦和余弦张量的维度大小**: 正弦和余弦张量的维度应该是2或4
3. **检查特定维度的大小**: 对于4维的情况，正弦和余弦张量的第1维和第3维应该都是1
4. **检查序列长度和头部维度是否匹配**: 根据提供的位置 ID 张量进行进一步检查。如果位置 ID 张量存在，需要检查正弦和余弦张量的序列长度和头部维度是否匹配, 如果位置 ID 张量不存在，直接检查正弦和余弦张量的序列长度和头部维度



### 1.2 VectorizedFusedRopeCudaKernel 核函数实现

在 PaddlePaddle 框架中，FusedRopeKernel 使用了两种不同的 CUDA 核函数来实现旋转位置嵌入：`VectorizedFusedRopeWithRotateEveryTwoKernel` 和 `VectorizedFusedRopeWithRotateHalfKernel`。这两种方法分别实现了不同的旋转操作，选择的依据是 `use_neox_rotary_style` 标志。接下来，我们详细介绍这两种核函数的具体实现。

:::note

Neox Rotary Style 是一种特定的旋转位置嵌入方法，最早在 EleutherAI 的 GPT-NeoX 模型中使用。它在处理自注意力机制中的位置编码时，通过对**每两个元素进行旋转**来实现位置嵌入，从而增强模型对位置信息的感知能力。

:::

```cpp
VectorizedFusedRopeCudaKernelFunc<T, MPType, vec_size> kernel_func =
    use_neox_rotary_style
        ? VectorizedFusedRopeWithRotateEveryTwoKernel<T, MPType, vec_size>
        : VectorizedFusedRopeWithRotateHalfKernel<T, MPType, vec_size>;
```

#### 1.2.1 VectorizedFusedRopeWithRotateEveryTwoKernel

首先，我们来看一下 `VectorizedFusedRopeWithRotateEveryTwoKernel` 的定义：

```cpp
template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateEveryTwoKernel(
    phi::Array<const T*, 3> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t batch_stride,
    int64_t seq_stride,
    int num_inputs,
    MPType div_c,
    float rotary_emb_base,
    phi::Array<T*, 3> outs_data) {
```

接下来计算当前线程需要处理的数据索引和步长：

```cpp
int64_t index = (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x) * VecSize;
int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x) * VecSize;
int64_t size = batch_size * seq_len * num_heads * head_dim;
MPType sin_value[VecSize];
MPType cos_value[VecSize];
```

这里，index 是当前线程处理的起始索引，stride 是线程间处理的数据步长，size 是总数据大小。

接下来检查 rotary_emb_base 是否接近 kDefaultRotaryBase，以选择不同的正弦和余弦值计算方法：

```cpp
if (fabs(rotary_emb_base - static_cast<float>(kDefaultRotaryBase)) < Epsilon) {
  for (; index < size; index += stride) {
    VectorizedGetSinCos<T, MPType, VecSize, kDefaultRotaryBase>::run(
        sin_cos_data,
        position_ids_data,
        flag_sin_cos,
        index,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        batch_stride,
        seq_stride,
        div_c,
        rotary_emb_base,
        sin_value,
        cos_value);
    rotate_every_two<T, MPType, VecSize>(
        ins_data, num_inputs, index, sign, sin_value, cos_value, outs_data);
  }
} else {
  for (; index < size; index += stride) {
    VectorizedGetSinCos<T, MPType, VecSize>::run(
        sin_cos_data,
        position_ids_data,
        flag_sin_cos,
        index,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        batch_stride,
        seq_stride,
        div_c,
        rotary_emb_base,
        sin_value,
        cos_value);
    rotate_every_two<T, MPType, VecSize>(
        ins_data, num_inputs, index, sign, sin_value, cos_value, outs_data);
  }
}
```

VectorizedGetSinCos 函数用于计算给定索引处的正弦和余弦值。如果 rotary_emb_base 接近 kDefaultRotaryBase，则使用特化版本的 VectorizedGetSinCos，否则使用通用版本。主要作用就是预处理出来，公式里面的 sin 和 cos 值。

$$
\left(\begin{array}{c}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\cos m \theta_0 \\
\cos m \theta_0 \\
\cos m \theta_1 \\
\cos m \theta_1 \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\cos m \theta_{d / 2-1}
\end{array}\right)+\left(\begin{array}{c}
-q_1 \\
q_0 \\
-q_3 \\
q_2 \\
\vdots \\
-q_{d-1} \\
q_{d-2}
\end{array}\right) \otimes\left(\begin{array}{c}
\sin m \theta_0 \\
\sin m \theta_0 \\
\sin m \theta_1 \\
\sin m \theta_1 \\
\vdots \\
\sin m \theta_{d / 2-1} \\
\sin m \theta_{d / 2-1}
\end{array}\right)
$$



rotate_every_two 函数对每两个相邻元素进行旋转操作。其具体实现如下：

```cpp
template <typename T, typename MPType, int VecSize = 2>
__device__ __forceinline__ void rotate_every_two(
    phi::Array<const T*, 3> ins_data,
    int num_inputs,
    int64_t index, // 当前线程处理的数据起始索引
    int sign, // 用于确定旋转方向
    MPType* sin_value, // 正弦值
    MPType* cos_value, // 余弦值
    phi::Array<T*, 3> outs_data) {

    using VecType = phi::AlignedVector<T, VecSize>;
    constexpr int kVectorsPerThread = VecSize / 2;
    MPType result[VecSize];
    T store[VecSize];

    #pragma unroll
    for (int iter = 0; iter < 3; iter++) {
        if (iter >= num_inputs) break;
        const T* input = ins_data[iter] + index;
        VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

        #pragma unroll
        for (int nx = 0; nx < kVectorsPerThread; ++nx) {
            int pr_index = nx * 2;
            int ls_index = pr_index + 1;

            MPType p0 = static_cast<MPType>(input[pr_index]);
            MPType p1 = static_cast<MPType>(input[ls_index]);

            if (sign == 1) {
                result[pr_index] = cos_value[pr_index] * p0;
                result[pr_index] -= sin_value[pr_index] * p1;
                result[ls_index] = sin_value[ls_index] * p0;
                result[ls_index] += cos_value[ls_index] * p1;
            } else if (sign == -1) {
                result[pr_index] = cos_value[pr_index] * p0 + sin_value[ls_index] * p1;
                result[ls_index] = cos_value[ls_index] * p1 - sin_value[pr_index] * p0;
            }

            store[pr_index] = static_cast<T>(result[pr_index]);
            store[ls_index] = static_cast<T>(result[ls_index]);
        }
        out[0] = *(reinterpret_cast<VecType*>(store));
    }
}
```

`rotate_every_two` 的主要作用是对每两个相邻元素进行旋转，以实现位置嵌入。旋转操作通过应用正弦和余弦值，将输入数据变换到新的位置编码。这种方法增强了模型对序列中元素位置信息的感知能力，特别适用于处理长序列的数据。

为了更好地理解 `rotate_every_two` 函数的作用，下面通过一个具体的输入输出示例来说明其工作原理。

假设我们有一个简单的输入张量 input 和预先计算好的正弦和余弦值 `sin_value` 和 `cos_value`，我们将展示 rotate_every_two 函数如何对这些输入进行旋转操作。

模拟一些输入数据：


```python
input = [1.0, 2.0, 3.0, 4.0]
sin_value = [0.5, 0.5]
cos_value = [0.866, 0.866]
```

我们使用 rotate_every_two 函数对输入张量进行旋转操作。假设 VecSize = 2，num_inputs = 1，sign = 1，具体步骤如下：

1. 从输入张量中读取两个相邻元素，例如 `p0 = 1.0` 和 `p1 = 2.0`
2. 计算旋转后的结果，例如 `result[0] = cos_value[0] * p0 - sin_value[0] * p1` 和 `result[1] = sin_value[1] * p0 + cos_value[1] * p1`
3. 将结果存储到输出张量中，例如 `out[0] = [result[0], result[1]]`
4. 重复上述步骤，直到处理完所有的输入数据

#### 1.2.2 VectorizedFusedRopeWithRotateHalfKernel

`VectorizedFusedRopeWithRotateHalfKernel` 和 `VectorizedFusedRopeWithRotateEveryTwoKernel` 的实现类似，主要区别在于旋转操作的方式。，它对每一对元素的前半部分和后半部分分别进行旋转操作。与 VectorizedFusedRopeWithRotateEveryTwoKernel 类似，每个线程负责处理一小部分数据。具体地说，每个线程根据索引计算正弦和余弦值，并对输入向量的每对元素进行旋转操作。

与 rotate_every_two 不同，rotate_half 函数对输入向量的前半部分和后半部分进行旋转操作。具体实现如下：

```cpp
template <typename T, typename MPType, int VecSize = 2>
__device__ __forceinline__ void rotate_half(
    phi::Array<const T*, 3> ins_data,
    int num_inputs,
    int64_t head_dim,
    int64_t index,
    int sign,
    MPType* sin_value,
    MPType* cos_value,
    phi::Array<T*, 3> outs_data) {

    MPType result[VecSize];
    T store[VecSize];
    using VecType = phi::AlignedVector<T, VecSize>;
    constexpr int kVectorsPerThread = VecSize / 2;
    int64_t stride_r = head_dim / 2;

    #pragma unroll
    for (int iter = 0; iter < 3; iter++) {
        if (iter >= num_inputs) break;

        // 获取 value_index 和 rotate_half_index
        int64_t index_v = index;
        int64_t index_r = (index % head_dim) < stride_r ? (index + stride_r) : (index - stride_r);
        MPType sign_r = (index % head_dim) < stride_r ? static_cast<MPType>(-1) : static_cast<MPType>(1);
        const T* input_v = ins_data[iter] + index_v;
        const T* input_r = ins_data[iter] + index_r;
        VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

        #pragma unroll
        for (int nx = 0; nx < VecSize; ++nx) {
            MPType p0 = static_cast<MPType>(input_v[nx]);
            MPType p1 = static_cast<MPType>(input_r[nx]);

            result[nx] = cos_value[nx] * p0 + sign * sign_r * sin_value[nx] * p1;
            store[nx] = static_cast<T>(result[nx]);
        }
        out[0] = *(reinterpret_cast<VecType*>(store));
    }
}
```

VectorizedFusedRopeWithRotateHalfKernel 与 VectorizedFusedRopeWithRotateEveryTwoKernel 的主要区别在于旋转操作的实现方式。rotate_half 函数对输入向量的前半部分和后半部分进行旋转操作，而 rotate_every_two 函数则对每两个相邻元素进行旋转操作。通过这种方式，FusedRopeKernel 实现了高效的并行计算，增强了模型对位置信息的感知能力。


## 1.3 是否支持 2-D 数据

Paddle 的 FusedRopeKernel 实现主要针对一维旋转位置嵌入（RoPE）。在 VectorizedFusedRopeWithRotateEveryTwoKernel 内核中，正弦和余弦值的计算方法是 get_sin_cos_by_passed_values 和 get_sin_cos_by_rotary_base，这些方法计算的是沿着单一维度的旋转角度。计算的是沿着序列长度维度（seq_len）的旋转角度。计算正弦和余弦值的公式中，pos_seq 只涉及一个维度的序列位置。

rotate_every_two 函数对两个相邻元素进行旋转操作的实现也是基于一维数据的。

**扩展到二维 Rope 的潜在问题**

- 正弦和余弦值的计算：目前的实现仅计算一个维度的旋转角度。对于二维 RoPE，需要分别计算两个维度的旋转角度。
- 旋转操作：目前的旋转操作仅对两个相邻元素进行旋转。对于二维 RoPE，需要对两个维度分别进行旋转操作，这将涉及更复杂的数据访问和计算逻辑。


## 2. OneFlow 中的实现


OneFlow 中的实现在 `oneflow/user/kernels/fused_attention_kernels.cu` 中，OneFlow 中实现旋转位置嵌入（RoPE）逻辑的是 `FusedApplyRotaryEmbKernel`。 

在 OneFlow 中，实现旋转位置嵌入（RoPE）的核心逻辑主要集中在 PlaneKernel 和 IntervalKernel 中。前面的代码已经涵盖了输入输出数据的获取、参数初始化和格式检查。下面我们详细介绍 PlaneKernel 和 IntervalKernel 中的实现。

### 2.1 PlaneKernel

PlaneKernel 是用于处理平面数据的 CUDA 内核，假设输入数据是连续存储的。

```cpp
template<typename T, typename PositionType, typename IndexType, size_t num_dims, size_t rotary_emb_dim>
__global__ void PlaneKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims, rotary_emb_dim> param) {
```

其中 

内核首先计算每个线程处理的数据起始位置 offset，并遍历数据块中的所有元素：


```cpp
for (IndexType offset = threadIdx.x + blockIdx.x * blockDim.x; offset < param.num_elements;
     offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, 2>; // 使用 2 元素打包的加载方式
    IndexType temp_offset = offset;
    IndexType index[num_dims];
#pragma unroll
    for (int i = 0; i < num_dims - 1; i++) {
        IndexType ref_stride = param.ref_stride[i];
        IndexType idx = temp_offset / ref_stride; // 计算每个维度的索引
        index[i] = idx;
        temp_offset = temp_offset - idx * ref_stride;
    }
    index[num_dims - 1] = temp_offset;
```

接下来根据 temp_offset 计算各个维度的索引值

```cpp
    const IndexType b_index = index[0], m_index = index[1], k_index = index[num_dims - 1];
    const IndexType position_rotate_index = (k_index >= param.k0) ? 1 : 0;
    const IndexType position_id_offset = b_index * param.position_b_stride
                                         + position_rotate_index * param.position_rotate_stride
                                         + m_index;
    const PositionType position =
        param.position_ids ? param.position_ids[position_id_offset] : m_index;
    const IndexType actual_k_index = k_index % param.actual_rotary_size;
    const IndexType sinuous_offset = position * param.k + actual_k_index;
```

接下来，根据计算得到的索引值，计算正弦和余弦值：

```cpp
    T cos_val, sin_val, out_val;
    if (param.cos && param.sin) {
        cos_val = *(param.cos + sinuous_offset);
        sin_val = *(param.sin + sinuous_offset);
    } else {
        T val = position * expf(2.0f * static_cast<float>(k_index % (param.actual_rotary_size >> 1))
                                * param.inv_actual_rotary_size * logf(param.theta));
        cos_val = cosf(val);
        sin_val = sinf(val);
    }
```

根据计算的正弦和余弦值对输入数据进行旋转操作，并将结果写入输出张量：

```cpp
    LoadPack x_vec;
    IndexType x_offset = param.x_offset;
    IndexType out_offset = 0;
#pragma unroll
    // 计算输入和输出的偏移量
    for (int i = 0; i < num_dims; i++) {
        x_offset = x_offset + param.x_stride[i] * index[i];
        out_offset = out_offset + param.out_stride[i] * index[i];
    }
    // 对输入数据进行旋转操作
    if (k_index < param.k0) {
        x_vec.elem[0] = *(param.x + x_offset);
        x_vec.elem[1] = (param.k0 - k_index > param.rotate_stride)
                        ? static_cast<T>(-*(param.x + x_offset + param.rotate_stride))
                        : *(param.x + x_offset - param.rotate_stride);
        out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
    } else if (k_index < param.k1) {
        x_vec.elem[0] = *(param.x + x_offset);
        x_vec.elem[1] = (param.k1 - k_index > param.rotate_stride)
                        ? static_cast<T>(-*(param.x + x_offset + param.rotate_stride))
                        : *(param.x + x_offset - param.rotate_stride);
        out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
    } else {
        out_val = *(param.x + x_offset);
    }
    *(param.out + out_offset) = out_val;
  }
}
```

### 2.2 IntervalKernel

IntervalKernel 是用于处理间隔存储数据的 CUDA 内核，适用于数据不是连续存储的情况。和 PlaneKernel 的主要区别在于：

1. 数据加载和存储方式：IntervalKernel 需要根据索引计算具体的加载和存储位置，适用于数据不是连续存储的情况
2. 使用 PackSize 元素打包的加载方式：IntervalKernel 使用 PackSize 元素打包的加载方式，以提高数据访问效率。

内核首先计算每个线程处理的数据起始位置 packed_offset 和实际数据偏移量 offset，并遍历数据块中的所有元素：

```cpp
for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
     packed_offset < param.num_elements; packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, PackSize>; // 使用 PackSize 元素打包的加载方式
    IndexType offset = packed_offset * PackSize;
    IndexType index[num_dims];  // b, m, h, k
    IndexType temp_offset = offset;
```

接下来根据 temp_offset 计算各个维度的索引值：

```cpp
    for (int i = 0; i < num_dims - 1; i++) {
        IndexType ref_stride = param.ref_stride[i];
        IndexType idx = temp_offset / ref_stride; // 计算每个维度的索引
        index[i] = idx;
        temp_offset = temp_offset - idx * ref_stride;
    }
    index[num_dims - 1] = temp_offset;
```

根据多维索引计算输入和输出数据的实际偏移量：

```cpp
    IndexType x_offset = param.x_offset;
    IndexType out_offset = 0;
#pragma unroll
    for (int i = 0; i < num_dims; i++) {
        x_offset = x_offset + param.x_stride[i] * index[i];
        out_offset = out_offset + param.out_stride[i] * index[i];
    }
    const LoadPack x_vec = *reinterpret_cast<const LoadPack*>(param.x + x_offset);
```

最后根据根据索引计算正弦和余弦值，并进行旋转操作：

```cpp
const IndexType k_index = index[num_dims - 1];
// 获取当前索引的最后一个维度（k 维度）的索引

// 判断 k_index 是否小于旋转嵌入的尺寸
if (k_index < param.rotary_size) {
    // 确定旋转索引的位置，如果 k_index 大于等于 k0，则 position_rotate_index 为 1，否则为 0
    const IndexType position_rotate_index = (k_index >= param.k0) ? 1 : 0;
    
    // 获取批量索引 b 和 序列索引 m
    const IndexType b_index = index[0], m_index = index[1];
    
    // 计算位置 ID 的偏移量
    const IndexType position_id_offset = b_index * param.position_b_stride
                                         + position_rotate_index * param.position_rotate_stride
                                         + m_index;
    // 如果 position_ids 存在，则获取对应位置 ID，否则使用 m_index
    const PositionType position =
        param.position_ids ? param.position_ids[position_id_offset] : m_index;
    // 计算实际的 k 索引（取模实际旋转嵌入尺寸）
    const IndexType actual_k_index = k_index % param.actual_rotary_size;
    // 计算正弦余弦值的偏移量
    const IndexType sinuous_offset = position * param.sinuous_m_stride + actual_k_index;
    // 声明加载的打包数据
    LoadPack cos_vec, sin_vec, out_vec;
    if (param.cos && param.sin) {
        // 如果 cos 和 sin 存在，直接加载对应的值
        cos_vec = *reinterpret_cast<const LoadPack*>(param.cos + sinuous_offset);
        sin_vec = *reinterpret_cast<const LoadPack*>(param.sin + sinuous_offset);
    } else {
        // 如果 cos 和 sin 不存在，计算正弦和余弦值
        const IndexType actual_ndim = param.rotary_size / rotary_emb_dim;
#pragma unroll
        for (int i = 0; i < PackSize / 2; i++) {
            T val = position
                    * expf(2.0f * static_cast<float>(((actual_k_index >> 1) + i))
                           * param.inv_actual_rotary_size * logf(param.theta));
            T cos_val = cosf(val);
            T sin_val = sinf(val);
            cos_vec.elem[i * 2] = cos_val;
            cos_vec.elem[i * 2 + 1] = cos_val;
            sin_vec.elem[i * 2] = sin_val;
            sin_vec.elem[i * 2 + 1] = sin_val;
        }
    }

#pragma unroll
    for (int i = 0; i < PackSize / 2; i++) {
        // 对输入数据进行旋转计算
        out_vec.elem[i * 2] =
            x_vec.elem[i * 2] * cos_vec.elem[i * 2] - x_vec.elem[i * 2 + 1] * sin_vec.elem[i * 2];
        out_vec.elem[i * 2 + 1] = x_vec.elem[i * 2 + 1] * cos_vec.elem[i * 2 + 1]
                                  + x_vec.elem[i * 2] * sin_vec.elem[i * 2 + 1];
    }
    // 将旋转后的数据写入输出
    *(reinterpret_cast<LoadPack*>(param.out + out_offset)) = out_vec;
    
} else {
    // 如果 k_index 大于等于旋转嵌入的尺寸，直接将输入数据写入输出
    *(reinterpret_cast<LoadPack*>(param.out + out_offset)) = x_vec;
    
}
```

### 2.3 支持 2-D 数据





## 3. Torch 中的实现

Pytorch 直接用 python 实现了 RoPE 的逻辑。在 PyTorch 中，旋转位置嵌入（RoPE）通过一个名为 RotaryPositionalEmbeddings 的模块实现。这个模块负责初始化和缓存正弦和余弦值，并在前向传播时将这些值应用于输入张量。

```python
from typing import Optional
import torch
from torch import nn, Tensor

class RotaryPositionalEmbeddings(nn.Module):
    """
    该类实现了旋转位置嵌入（RoPE），其方法提出于论文 https://arxiv.org/abs/2104.09864。

    参考实现（用于正确性验证）可见：https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450

    在此实现中，我们在初始化期间缓存每个位置的嵌入，最多缓存到 `max_seq_len`。

    参数:
        dim (int): 嵌入维度，通常设置为注意力模块中每个头的维度，即 `embed_dim // num_heads`
        max_seq_len (int): 模型的最大预期序列长度，如果超出此长度，将重新计算缓存的频率
        base (int): 用于计算旋转角度的几何级数的基数
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # 我们需要显式定义 reset_parameters 以进行 FSDP 初始化
    # 详见 https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        # 初始化 theta，用于计算旋转嵌入的角度
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # 创建位置索引 `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # theta 和位置索引的外积，输出张量形状为 [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # 缓存包括 cos 和 sin 分量，因此输出形状为 [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        参数:
            x (Tensor): 输入张量，形状为 [bsz, seq_len, num_heads, head_dim]
            input_pos (Optional[Tensor]): 包含当前 token 位置的可选张量，仅在推理期间使用，默认为 None

        返回:
            Tensor: 应用 RoPE 后的输出张量

        张量形状的符号说明:
            - b: 批大小
            - s: 序列长度
            - n_h: 注意力头数量
            - h_d: 每个头的维度

        TODO: 以下实现可在推理过程中更高效
        """
        # 输入张量的形状为 [b, s, n_h, n_d]
        seq_len = x.size(1)

        # 根据是否设置 input_pos 提取值。在设置 input_pos 时处于推理模式
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # 重塑输入张量，最后一个维度用于计算输出
        # 张量形状为 [b, s, n_h, n_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # 重塑缓存以进行广播
        # 张量形状为 [1, s, 1, n_d // 2, 2]
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

        # 张量形状为 [b, s, n_h, n_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # 张量形状为 [b, s, n_h, n_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
```

## 4. GPT-Neox/apex 中的实现

GPT-NeoX 仓库的实现代码在 `megatron/fused_kernels/fused_rotary_positional_embedding.h` 中，主要逻辑在 `fused_rope_forward` 中。

:::note

GPT Neox 中的实现源自于 Nvidia 的 apex 库

:::

### 4.1 fused_rope_forward 函数

```cpp
template <typename scalar_t>
__global__ void fused_rope_forward(const int h,
                                   const int d,
                                   const int d2,
                                   const int stride_s,
                                   const int stride_b,
                                   const int stride_h,
                                   const int stride_d,
                                   const int o_stride_s,
                                   const int o_stride_b,
                                   const int o_stride_h,
                                   const int o_stride_d,
                                   const scalar_t* src,
                                   const float* freqs,
                                   scalar_t* dst)
```

首先，通过 blockIdx.x 和 blockIdx.y 获取当前线程块在s和b维度的索引，即 s_id 和 b_id。接着，计算当前线程块在源张量中的起始偏移量，`offset_block = s_id * stride_s + b_id * stride_b`，以及在目标张量中的起始偏移量，`offset_block_dst = s_id * o_stride_s + b_id * o_stride_b`。

```cpp
    int s_id = blockIdx.x, b_id = blockIdx.y;
    int offset_block = s_id * stride_s + b_id * stride_b;
    int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
```

接下来在d2范围内使用线程的x维度索引遍历。sincosf 函数计算给定频率对应的正弦和余弦值，存储在 v_sin 和 v_cos 中。

```cpp
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
        float v_cos, v_sin;
        sincosf(freqs[s_id * d2 + d_id], &v_sin, &v_cos);
```

sincosf 函数是 CUDA 数学库的一部分，用于计算正弦和余弦值。它在 cuda_runtime.h 头文件中定义。

下面在 h 范围内使用线程的 y 维度索引遍历。计算当前元素在源张量和目标张量中的偏移量，分别为 offset_src 和 offset_dst。从源张量中读取当前元素的值存储在 v_src 中。计算旋转后的值 v_src_rotate，根据 d_id 的位置确定旋转方向。将旋转后的值存储到目标张量中，使用正弦和余弦值进行线性组合。

```cpp
#pragma unroll
        for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
            int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
            int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
            scalar_t v_src = src[offset_src];
            scalar_t v_src_rotate = (d_id + d2 / 2 < d2)
                                        ? -src[offset_src + (d2 / 2) * stride_d]
                                        : src[offset_src + (d2 / 2 - d2) * stride_d];
            dst[offset_dst] = v_src * (scalar_t)v_cos + v_src_rotate * (scalar_t)v_sin;
        }
```

最后处理剩余的深度值（如果 d 大于 d2），在h范围内使用线程的y维度索引遍历。计算当前行在源张量和目标张量中的起始偏移量，分别为 offset_head 和 offset_head_dst。在剩余的深度范围内使用线程的x维度索引遍历，直接将源张量的值复制到目标张量中。

```cpp
    if (d > d2) {
#pragma unroll
        for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
            int offset_head = offset_block + h_id * stride_h;
            int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
            for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
                dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d];
            }
        }
    }
}
```





## 4. 总结

各框架在实现RoPE时采用了不同的方法，PaddlePaddle和OneFlow更注重底层的CUDA优化，以提高GPU上的计算效率，而PyTorch则通过Python层面的实现，简化了实现复杂度，同时利用其高效的张量操作库保证性能。

