# DifferNet：基于标准流的半监督异常检测

## 论文简介


本篇论文提出了一种基于 normalize flow（标准流）的半监督异常检测算法。 Flow是一种效果类似 GAN 的算法，但是要比GAN更加好训练。DifferNet的主要思想是首先用CNN进行特征提取，然后再用 `Normalize Flow` 进行特征估计。他的核心思路如下图所示：

![图 2](images/2b386eaa28bd15187a9691b282c65ecd8db022d6e2659803cbb2df20ece306c2.png)  

DifferNet会根据训练数据去估计出一个分布，正常数据会落入到分布数值较大，异常得分较低的区域，异常数据会落入到异常得分较高的区域。同时 Differnet 可以通过将`似然损失`反向传播后得到的`梯度映射`来识别缺陷区域。 **这篇文章主要还是聚焦于异常样本检测而不是异常样本定位。**

## 关键技术

![图 3](images/c0b2a7dcff2169f5c9f5bb98adbd4f499bcac8bfc234dd2b8fca230e9c7bca74.png)  

上图展示了DiiferNet的基本流程，首先为了提取多尺度的特征和让模型训练出来更加的鲁棒，会先对图像进行缩放和旋转等变换，然后会送入到一个特征提取器当中。这个特征提取器选择的AlexNet，本个工作也尝试了其他更加复杂的网络比如Vgg和Resnet但是发现效果都差不多，所以就选择了更加简单的AlexNet（剃刀原则）。将三个不同大小的图像提取了特征之后将这些特征concat起来送入到 normalizeing flow (NF) 模块当中通过极大似然训练的方式得到图像的特征密度估计。

### Normalizing Flow 

Normalizing flow 通过仿射变换的方式完成图像从特征空间到隐空间的双向变换。特征会被转换为隐空间中的一个正态分布 $z$, 其中 $p_{Z}(z)$ 为 $z$ 的概率密度函数。

Different中使用的Normalizing flow结构是 Real-NVP 中梯度的coupling layer结构。这个结构 $f_{NF}$ 是由很多个block组成的，每个block的结构如下所示：

![图 4](images/e149d24bfb58f779bf28f48e6605772c165302203c2e26bfd4d61f6ad20bf7be.png)  

每一个block在开始之前首先会对特征进行一个预定义的随机化，这可以让每一个维度可以影响到其他的位置。为了应用scale和shift，$y_{in}$ 被分为了两个部分 $y_{in,1}$ 和 $y_{in,2}$。scale和shift的操作可以用如下公式表示：

$$
\begin{array}{r}
y_{\text {out }, 2}=y_{\text {in }, 2} \odot e^{s_{1}\left(y_{\text {in }, 1}\right)}+t_{1}\left(y_{\text {in }, 1}\right) \\
y_{\text {out }, 1}=y_{\text {in }, 1} \odot e^{s_{2}\left(y_{\text {out }, 2}\right)}+t_{2}\left(y_{\text {out }, 2}\right)
\end{array}
$$

为了保持非零的特性，在进行缩放之前会使用一个指数函数。 内部的函数 $s$ 和 $t$ 可以是任意的函数，在这篇论文里面 $s$ 和 $t$ 是一个全连接网络。 同时作者在这里还对s使用了`soft-clamping`去确保模型的稳定性。 `soft-clamping` 的定义如下：

$$
\sigma_{\alpha}(h)=\frac{2 \alpha}{\pi} \arctan \frac{h}{\alpha}
$$

对于最后一层的$s$, 为了防止组件的数字过大，$s(y)$ 会被限制到$(-\alpha, \alpha)$。

### 训练

训练的主要目的是让$f_{NF}$能够学习到参数，让$$

