# Paddle C++ 算子开发

:::tip

[官方教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)的笔记

:::


## 开发流程

新增一个 C++ 算子大概需要以下几个步骤：

1. 确定算子的功能、输入、输出、属性等信息，并在相应的 `Yaml` 文件中进行配置；
2. 实现算子的 `InferMeta` 函数，用于推导输出结果的维度、数据类型等静态信息；
3. 实现算子在不同硬件设备上的 `Kernel` 函数，用于完成输出结果的数值计算；
4. 封装 `Python API`，用于用户调用算子；
5. 添加单元测试，用于验证算子的正确性。


用户使用飞桨开发神经网络模型时使用的 `Python` 接口(如 `paddle.add()`, `paddle.relu()`等) 我们一般称都为飞桨的 `Python API`，每个运算类的 `Python API` 在框架内部都会对应到一个或者多个 `C++` 端算子，每个算子在不同硬件设备上（`CPU`, `GPU` 等）实现的运算逻辑代码又被称为 `Kernel`, 这里主要是由于不同硬件设备提供的编程接口不同，所以虽然同一个算子的不同硬件设备 `Kernel` 都实现了相同的数学运算逻辑，但在代码实现上却有所差异。算子 `InferMeta` 函数是在算子 `kernel` **执行前先将输出结果的维度、数据类型等信息进行处理**，由于计算量较小所以可以直接在 `CPU` 上计算，因此每个算子只需要实现一个 **InferMeta** 函数，而不必像 `Kernel` 一样在不同硬件上实现多个。

### 算子的执行过程

算子的执行主要包括两个过程：

1. 执行算子 `InferMeta` 函数完成输出结果的维度、数据类型等静态信息的推导。
2. 根据输入变量的设备信息选择对应的硬件设备来执行算子 `Kernel`，完成输出结果的数值计算。


下面我们按照：新增算子描述及定义、 新增算法 `Kernel`、新增算子 `Python API`、新增算子单元测试四个部分分别介绍如何开发一个算子。

## 新增算子描述及定义

算子描述及定义主要是定义算子的基本属性，包括算子的输入、输出以及各项非计算逻辑的配置，这些都是设备无关的。

### 算子 Yaml 文件配置

在 `paddle/phi/api/yaml/ops.yaml` 和 `paddle/phi/api/yaml/backward.yaml` 文件中对算子进行描述及定义，在框架编译时会根据 `YAML` 文件中的配置自动生成 C++ 端的相关代码接口以及内部实现

[paddle/phi/api/yaml/ops.yaml](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html#:~:text=Yaml%20%E9%85%8D%E7%BD%AE%E8%A7%84%E5%88%99%EF%BC%9A-,paddle/phi/api/yaml/ops.yaml,-%E4%B8%AD%20trace%20%E7%9B%B8%E5%85%B3):

```yaml
- op : trace # 算子名称
  args : (Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1) # 输入参数
  output : Tensor(out) # 输出类型
  infer_meta :
    func : TraceInferMeta # 调用的 InferMeta 函数
  kernel : # 算子的 Kernel 配置
    func : trace # 函数注册名
  backward : trace_grad 
```

[paddle/phi/api/yaml/backward.yaml](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html#:~:text=backward%20%3A%20trace_grad-,paddle/phi/api/yaml/backward.yaml,-%E4%B8%AD%20trace%20%E7%9B%B8%E5%85%B3):

```yaml
- backward_op : trace_grad # 反向算子名称
  forward : trace (Tensor x, int offset, int axis1, int axis2) -> Tensor(out) # 对应的前向算子名称
  args : (Tensor x, Tensor out_grad, int offset, int axis1, int axis2) # 输入参数
  output : Tensor(x_grad) # 输出类型
  infer_meta :
    func : UnchangedInferMeta # 与前向算子的 InferMeta 函数相同
    param : [x] 
  kernel :
    func : trace_grad # 与前向算子的 Kernel 函数相同
    data_type : x
  no_need_buffer : x # 不需要缓存的输入参数 [可选]
```

### 实现 InferMeta 函数

`InferMeta` 函数是根据输入参数，推断算子输出 `Tensor` 基本信息的函数，推断的信息包括输出 Tensor 的 `shape`、`data` `type`，同时它也承担了检查输入数据维度、类型等是否合法的功能。

`trace` 算子的 `InferMeta` 函数 实现如下：

```cpp
void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out) {
	/**
	x: the input tensor
	offset: the offset of the diagonal from the main diagonal
	axis1: the first axis with respect to take a diagonal
	axis2: the second axis with respect to take a diagonal
	out: the output tensor
	**/
  int dim1 = axis1;
  int dim2 = axis2;

  auto x_dims = x.dims();
  // convert negative axis to positive
  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::OutOfRange(
          "Input's dim is out of range (expected at least 2, but got %ld).",
          x_dims.size()));
  PADDLE_ENFORCE_LT(
      dim1_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(dim1) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim1));
  PADDLE_ENFORCE_LT(
      dim2_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(dim2) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim2));
  PADDLE_ENFORCE_NE(
      dim1_,
      dim2_,
      phi::errors::InvalidArgument("The dimensions should not be identical "
                                   "%ld vs %ld.",
                                   dim1,
                                   dim2));
  
  auto sizes = vectorize(x_dims);
  // calulate the output shape
  if (x_dims.size() == 2) {
    sizes.clear();
    sizes.push_back(1);
  } else {
    sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
    sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
  }
  // set output shape
  out->set_dims(phi::make_ddim(sizes));
  // set output data type
  out->set_dtype(x.dtype());
}
```

#### InferMeta 的实现位置

InferMeta 的文件放置规则（`paddle/phi/infermeta` 目录下，以 Tensor 输入个数为判定标准）

- `nullary.h`: 没有输入 Tensor 参数的函数
- `unary.h`: 仅有一个输入 Tensor 参数的函数
- `binary.h`: 有两个输入 Tensor 参数的函数
- `ternary.h`: 有三个输入 Tensor 参数的函数
- `multiary.h`: 有四个及以上输入 Tensor 参数的函数
- `backward.h`: 反向算子的 InferMeta 函数, 不受前序的规则约束

#### InferMeta 的编译时与运行时

在静态图模型中，`InferMeta` 操作在 编译时(`compile time`)和运行时(`run time`) 都会被调用，在 `compile time` 时，由于真实的维度未知，框架内部用 -1 来表示，在 `run time` 时，用实际的维度表示，因此维度的值在 `compile time` 和 `run time` 时可能不一致，如果存在维度的判断和运算操作，`InferMeta` 就需要区分 `compile time` 和 `run time`。

对于此类 `InferMeta` 函数，需要在 `InferMeta` 函数声明的参数列表末尾增加 `MetaConfig` 参数，例如：

```cpp
void ConcatInferMeta(const std::vector<MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());
```

然后在函数体中，使用 `config.is_runtime` 判断处于编译时还是运行时。

##  新增算子 Kernel

新增算子 `Kernel` 在 `paddle/phi/kernels` 目录中完成，基本目录结构如下：

```
paddle/phi/kernels
./ (根目录放置设备无关的 kernel 声明和实现)
./cpu（仅放置 cpu 后端的 kernel 实现）
./gpu（仅放置 gpu 后端的 kernel 实现）
./xpu（仅放置百度 kunlun 后端的 kernel 实现）
./gpudnn
./funcs（放置一些支持多设备的、在多个 kernel 中使用的公共 functor 和 functions）
...
```




