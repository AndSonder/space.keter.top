# 为Paddle优化Lerp在GPU上的计算性能

:::tip

飞桨黑客松赛题，【Hackathon4 No.34】为 Paddle 优化 Lerp OP在GPU上的性能

相关PR: https://github.com/PaddlePaddle/community/pull/513

:::

## API 介绍

paddle 的 lerp api 主要功能为基于给定的 weight 计算 x 与 y 的线性插值。

$$
\operatorname{lerp}(x, y, \text { weight })=x+\text { weight } *(y-x)
$$

## 关键模块与性能提升点

1. 优化读取、写入以及线程配置方法，预计整体性能将提升20%以上
2. 使用飞桨充分优化的Elementwise Kernel，预计受Broadcast影响而性能较差的Case（如Case 2、Case 4）的速度将提升4倍以上

:::tip

当下飞桨有很多高性能的Kernel，使用这些Kernel要比使用 Eigen 这样的第三方计算库更加高效。

:::


## 老代码分析

首先我们先看一下优化之前的代码：

```cpp
// 模板函数,处理rank为D的tensor的lerp计算
template <typename Context, typename T, size_t D>  
static void LerpFunction(const Context& ctx, // 执行上下文
                         const DenseTensor& x, // 输入tensor x
                         const DenseTensor& y, // 输入tensor y
                         const DenseTensor& weight, // 权重tensor
                         DenseTensor* out) { // 输出tensor

  // 为输出tensor分配内存
  ctx.template Alloc<T>(out);   

  // 获取输出tensor的维度
  const auto& out_dims = out->dims();   

  // 将x,y,weight的维度扩展到rank D
  auto x_dims = phi::funcs::ExtendDims2Rank(x.dims(), D);
  auto y_dims = phi::funcs::ExtendDims2Rank(y.dims(), D);
  auto w_dims = phi::funcs::ExtendDims2Rank(weight.dims(), D);

  // 计算broadcast的维度
  Eigen::DSizes<int, D> x_bcast_dims;
  Eigen::DSizes<int, D> y_bcast_dims; 
  Eigen::DSizes<int, D> w_bcast_dims;
  phi::funcs::GetBroadcastDims<D>(x_dims, out_dims, &x_bcast_dims);
  phi::funcs::GetBroadcastDims<D>(y_dims, out_dims, &y_bcast_dims);
  phi::funcs::GetBroadcastDims<D>(w_dims, out_dims, &w_bcast_dims);

  // 创建Eigen tensor
  auto eigen_x = phi::EigenTensor<T, D>::From(x, x_dims);
  auto eigen_y = phi::EigenTensor<T, D>::From(y, y_dims);
  auto eigen_w = phi::EigenTensor<T, D>::From(weight, w_dims);
  auto eigen_out = phi::EigenTensor<T, D>::From(*out);

  // 定义乘法精度类型
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  
  // 获取当前设备
  auto& place = *ctx.eigen_device();

  // 进行lerp操作
  eigen_out.device(place) =  
      (eigen_x.broadcast(x_bcast_dims).template cast<MPType>() +
       eigen_w.broadcast(w_bcast_dims).template cast<MPType>() *
           (eigen_y.broadcast(y_bcast_dims).template cast<MPType>() -
            eigen_x.broadcast(x_bcast_dims).template cast<MPType>()))
          .template cast<T>();
}

// 专门处理0阶tensor的情况  
template <typename Context, typename T>
static void LerpFunctionZero(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& weight,
                             DenseTensor* out) {

  // 为输出tensor分配内存
  ctx.template Alloc<T>(out);

  // 标量的1阶维度
  auto dim = make_ddim(std::vector<int64_t>(1, 1));

  // 创建Eigen标量
  auto eigen_x = phi::EigenTensor<T, 1>::From(x, dim);
  auto eigen_y = phi::EigenTensor<T, 1>::From(y, dim);
  auto eigen_w = phi::EigenTensor<T, 1>::From(weight, dim);
  auto eigen_out = phi::EigenTensor<T, 1>::From(*out, dim);

  // 定义乘法精度类型 
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // 获取当前设备
  auto& place = *ctx.eigen_device();

  // 进行lerp操作
  eigen_out.device(place) =   
      (eigen_x.template cast<MPType>() +
       eigen_w.template cast<MPType>() *
           (eigen_y.template cast<MPType>() - eigen_x.template cast<MPType>()))
          .template cast<T>();
}

// 主要的kernel函数
template <typename T, typename Context>  
void LerpKernel(const Context& ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                const DenseTensor& weight,
                DenseTensor* out) {

  // 验证输入tensor不为空
  PADDLE_ENFORCE_GT(x.numel(), 0, ...);
  PADDLE_ENFORCE_GT(y.numel(), 0, ...);

  // 获取输出tensor的rank
  int rank = out->dims().size();

  // 验证rank
  PADDLE_ENFORCE_GE(rank, 0, ...);
  PADDLE_ENFORCE_LE(rank, 6, ...);

  // 调用相应rank的lerp函数
  switch (rank) {
    case 0:
      LerpFunctionZero<Context, T>(ctx, x, y, weight, out);
      break;
    case 1:  
      LerpFunction<Context, T, 1>(ctx, x, y, weight, out);
      break;
    ...
  }
}
```

可以看到，老代码主要是通过Eigen实现的，但是Eigen的性能并不是很好，使用飞桨充分优化的Elementwise Kernel可以提升性能。

## 新代码分析

```cpp
template <typename T>
struct BroadcastMinElementWiseDirectCUDAFunctor {
  // 广播求最小值的 CUDA Functor
  HOSTDEVICE inline T operator()(const T min) const {
    return min; 
  }
};

template <typename T>  
struct LerpElementWiseDirectCUDAFunctor {
  // 逐元素线性插值的 CUDA Functor
  HOSTDEVICE inline T operator()(const T x, const T y, const T weight) const {
    return x + weight * (y - x);
  }
};

template <typename T>
struct LerpScalarDirectCUDAFunctor {
  // 标量线性插值的 CUDA Functor
  const T* weight_; // 权重指针
  
  HOSTDEVICE inline LerpScalarDirectCUDAFunctor(const T* weight) 
    : weight_(weight) {}
    
  HOSTDEVICE inline T operator()(const T x, const T y) const {
    return x + weight_[0] * (y - x); 
  }
};

template <typename T, typename Context>
void LerpKernel(const Context& ctx, 
                const DenseTensor& x,
                const DenseTensor& y,
                const DenseTensor& weight, 
                DenseTensor* out) {
  
  // 检查x是否为空
  PADDLE_ENFORCE_GT(x.numel(), 0, 
                    phi::errors::InvalidArgument("x不能为空")); 
                    
  // 检查y是否为空                
  PADDLE_ENFORCE_GT(y.numel(), 0,
                    phi::errors::InvalidArgument("y不能为空"));  

  // 获取out的秩
  int rank = out->dims().size();  
  
  // 检查out的秩>=0
  PADDLE_ENFORCE_GE(rank, 0, 
                    phi::errors::InvalidArgument("out的秩必须>=0"));
                    
  // 为out分配内存
  ctx.template Alloc<T>(out);
  
  // 定义outputs
  std::vector<DenseTensor*> outputs = {out};
  
  // 定义inputs
  std::vector<const DenseTensor*> inputs;
  
  // 权重是标量
  if (weight.numel() == 1) {
    
    // 获取权重数据指针
    const T* weight_ptr = weight.data<T>();
    
    // 设置inputs
    inputs.reserve(2);
    inputs.emplace_back(&x);
    inputs.emplace_back(&y);
    
    // 创建标量插值Functor
    auto functor = LerpScalarDirectCUDAFunctor<T>(weight_ptr);
    
    // 广播调用Functor
    phi::funcs::BroadcastKernel<T>(ctx, inputs, &outputs, functor);
    
  // 权重是张量  
  } else {

    // 设置inputs 
    inputs.reserve(3);
    
    // 创建逐元素插值Functor
    auto functor = LerpElementWiseDirectCUDAFunctor<T>();
    
    // 创建用于broadcast的最小值张量
    DenseTensor b_min = phi::EmptyLike<T>(ctx, *out);  
    
    // 输入形状不匹配
    if (x.dims().size() != y.dims().size() && 
        weight.dims().size() != y.dims().size()) {
        
      // 定义broadcast最小值的inputs
      std::vector<const DenseTensor*> broadcast_min_inputs;
      broadcast_min_inputs.reserve(1);
      
      // 定义broadcast最小值的outputs    
      std::vector<DenseTensor*> broadcast_min_outputs = {&b_min};
      
      // 创建广播最小值Functor
      auto broadcast_min_functor = BroadcastMinElementWiseDirectCUDAFunctor<T>();
      
      // 广播x
      if (x.dims().size() < y.dims().size() &&
          x.dims().size() < weight.dims().size()) {
          
        broadcast_min_inputs.emplace_back(&x);
        phi::funcs::BroadcastKernel<T>(ctx, 
                                       broadcast_min_inputs, 
                                       &broadcast_min_outputs,
                                       broadcast_min_functor);
                                       
        inputs.emplace_back(&b_min);
        inputs.emplace_back(&y);
        inputs.emplace_back(&weight);
        
      // 广播y    
      } else if (y.dims().size() < weight.dims().size()) {
        
        broadcast_min_inputs.emplace_back(&y);
        phi::funcs::BroadcastKernel<T>(ctx,
                                       broadcast_min_inputs, 
                                       &broadcast_min_outputs,
                                       broadcast_min_functor);
                                       
        inputs.emplace_back(&x);
        inputs.emplace_back(&b_min);
        inputs.emplace_back(&weight);
        
      // 广播weight  
      } else {
        
        broadcast_min_inputs.emplace_back(&weight);
        phi::funcs::BroadcastKernel<T>(ctx,
                                       broadcast_min_inputs,
                                       &broadcast_min_outputs,
                                       broadcast_min_functor);
                                       
        inputs.emplace_back(&x);
        inputs.emplace_back(&y);
        inputs.emplace_back(&b_min);
      }
      
    // 输入形状匹配  
    } else {
    
      inputs.emplace_back(&x);
      inputs.emplace_back(&y);
      inputs.emplace_back(&weight);
    }
    
    // 广播调用Functor
    phi::funcs::BroadcastKernel<T>(ctx, inputs, &outputs, functor);
  }
}
```

## 总结

Lerp 这个算子的优化，主要是通过使用飞桨的Elementwise Kernel，以及使用Broadcast Kernel来实现的。这两个Kernel都是飞桨的高性能Kernel，可以提升算子的性能。


