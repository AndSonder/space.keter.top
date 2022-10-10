# 基于 NF 的密度估计算法

:::tip

Normalizing Flow 在密度估计方面上有广泛的应用，其中根据不同的 Transformator 算法和 Conditioner 算法。 本文针对使用神经网络作为 Transformator 算法以及使用掩码自回归作为 Conditioner 算法的 Normalizing Flow 进行介绍。 本文文章的前置知识可以在 [这里](https://space.keter.top/docs/deep_learning/%E8%A7%86%E8%A7%89%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B/%E5%9F%BA%E4%BA%8EFlow%E7%9A%84%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B/NF%E7%BB%BC%E8%BF%B0#%E5%90%84%E7%A7%8D-conditioner-c-%E7%9A%84%E5%AE%9E%E7%8E%B0) 查看

:::


## Masked Autoregressive Flow for Density Estimation

### 论文简介

本文通过堆叠自回归模型的方式构建出了 Masked Autoregressive Flow (MAF) 模型 [1]，该模型可以用于密度估计。 这个结构和 IAF[2] 很相似，同时也可以看做 Real NVP[3] 模型的一个拓展。 

### 实现方法

### 论文总结




## 参考文献

[1] George Papamakarios, Theo Pavlakou, & Iain Murray (2017). Masked Autoregressive Flow for Density Estimation neural information processing systems.

[2] Improved Variational Inference with Inverse Autoregressive Flow.

[3] Laurent Dinh, Jascha Sohl-Dickstein, & Samy Bengio (2016). Density estimation using Real NVP Learning.


