# 基于Embedding的异常检测

:::tip
搬运于：https://blog.csdn.net/qq_36560894/article/details/121589041?spm=1001.2014.3001.5502

写的太好了就搬运过来了，方便后续添加自己的笔记
:::


## 前言

异常检测领域中，基于Embedding的方法指的是：`将图像送入模型，提取特征，并在特征空间中构造分界面/评分规则`。与重构方法的主要不同在于，`其不在RGB图像空间而是在高维的特征空间中进行异常检测`；与自监督的方法不同，`其不过于依赖额外的负样本的构造 / 代理任务的设计，主要考虑的特征空间中的差异`。

## 基于预训练模型的

### SPADE (CVPR2021)

```
SPADE本身不是CVPR的工作，很早就挂在arxiv，但之后一直没有正式发表，最后与其他的创新点结合后在CVPR上发表:
论文：PANDA: Adapting Pretrained Features for Anomaly Detection and Segmentation (以色列,耶路撒冷希伯来大学)
但这篇文章其实跟SPADE方法本身关系不大了，要了解SPADE方法本身的话，最好参考下面这篇文章：
论文：Sub-Image Anomaly Detection with Deep Pyramid Correspondences
原文地址：https://arxiv.org/abs/2005.02357
代码地址：https://github.com/byungjae89/SPADE-pytorch?utm_source=catalyzex.com
```

SPADE指的是`Semantic Pyramid Anomaly Detection`，方法本身非常简单，但对于异常定位的效果却异常的好。

SPADE首先在ImageNet预训练的网络对所有正常样本(训练集)都进行特征提取，并存储所有特征(特征池)。在测试时，用同样的网络提取到特征后，在特征池中检索到K个最近的模板特征，利用下式得到异常得分：

$$
d(y)=\frac{1}{K} \sum_{f \in N_{K}\left(f_{y}\right)}\left\|f-f_{y}\right\|^{2}
$$

对于异常定位任务，需要更精细的像素级特征，因此使用相同的网络(Wide ResNet50x2)得到多层级特征(前3个stage的特征图)，将对应位置的特征进行`concat`（编码局部特征和全局特征）作为像素级特征。同样，提取所有正常样本的像素级特征构造特征池。在测试时，**对测试样本的每个像素级特征进行KNN索引，计算每个像素点的异常得分，上采样到原图尺寸得到最终的anomaly map**：

$$
d(y, p)=\frac{1}{\kappa} \sum_{f \in N_{\kappa}(F(y, p))}\|f-F(y, p)\|^{2}
$$

#### 优缺点

优点：方法简单，效果好。只使用固定的预训练网络提取特征，无需训练。
缺点：测试时，时间复杂度与数据量称线性关系，训练采用的正常图像越多，存储的特征也就越多，测试时的KNN复杂度越高。

### PaDiM（ICPR2020）

```
论文：PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
论文地址：https://arxiv.org/pdf/2011.08785.pdf
代码地址：https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
```

基于SPADE之上，PaDiM主要的改进就是不再构造特征池，执行KNN进行异常检测。而是提取每个位置上的多层级特征（和SPADE一样，concat三个stage的特征），为每个位置上估计一个分布（多元高斯分布）。

:::tip
每个位置"指的是$(i,j) \in [H, W]$，HxW是网络提取的最大特征图的尺寸
:::

具体来说，训练集中有 N 张正常图像，那么每个位置上可以收集到 N 个特征，根据这 N 个特征可以估计一个协方差矩阵，构造多元高斯分布。测试时，**用同样的网络提取得到一个HxW的特征图，计算每个位置上的特征与对应分布之间的马氏距离**，作为该位置的异常得分，从而可以得到Anomaly map。对于图像级的异常得分，选取anomaly map中的最大值。

特征维度越高，估计分布所需的时间就会越长，因此作者也探讨了PCA和random reduce两种降维策略，维持性能的同时，降低了训练时间，最终的结果是两种降维策略得到的结果相似。

#### 优缺点

优点：引入了训练步骤(估计分布)，图像级的异常检测性能得到提升，并且大大减少了测试的复杂度。类似模板匹配的思想：为每个位置构造一个正常模板(分布)。
缺点：PaDiM为HxW个位置单独估计分布，但是每个位置上的像素并不是严格对齐的，比如screw这一类，每张训练图像的朝向都不一样。

### PatchCore（2021）

```
论文：Towards Total Recall in Industrial Anomaly Detection (图宾根大学，亚马逊)
论文地址：https://arxiv.org/pdf/2106.08265v1.pdf
代码地址：https://github.com/hcw-00/PatchCore_anomaly_detection
```

PatchCore主要解决了SPADE测试速度太慢的问题，并且在特征提取部分做了一些探索。

![图 11](images/b2746df61ddc4ad41756a891d44e9febcc54af8807e045da53b24056b95dc607.png)  

如上图所示，**训练阶段只使用stage2、stage3的特征图**(SPADE使用了stage1,2,3)，但同样使用了多层级特征融合机制。如下图的消融实验所示，使用2+3的操作在检测和定位任务上表现最好。

此外，引入了`local neighborhood aggregation`操作，其实就是`average pooling`，实现采用一个窗口大小为3，步长为1，padding为1 的 AvgPool2d 实现，`不损失空间分辨率的同时增大感受野`。如下图的消融实验所示，引入`local neighborhood aggregation`操作后，两个任务上都提升了不少性能。（可以提取更深层的特征图来增大感受野，比如引入stage3/4/5的感受野，但这也会损失空间分辨率并且引入更多的ImageNet class bias）。

:::tip
实验说明，感受野对于异常检测任务来说是非常重要的，但ImagNet预训练的网络在深层会提取到一些高级语义特征，具有ImageNet class bias，不益于异常检测
:::

PatchCore在测试仍然采用`KNN`来衡量异常得分，为了提升检测效率，作者采用`greedy coreset subsampling`来选取最具代表性的特征点，缩减特征池的大小。算法流程大致如下，并证明只保留1%也能获得很好的检测性能，但大大缩减了测试时间。

![图 12](images/9109477ea1baf15934f54f56329fa6556fdc05bfc9c41f4fcaeebe4116e8d6e4.png)  

还有一点改进是在计算`Image-Level`的异常得分时，采用了`re-weighting`策略，这比直接取max要更鲁棒。详见原文。

#### 优缺点：

优点：简单高效，进行了一些结构上的探索，采用coreset selection优化测试速度
缺点：思想不够新颖，总体来说是一个较工程化的工作

### Focus Your Distribution（2021）

```
论文：Focus Your Distribution: Coarse-to-Fine Non-Contrastive Learning for Anomaly Detection and Localization (商汤)
论文地址：https://arxiv.org/pdf/2110.04538.pdf
```
![图 13](images/beba1121abdd4b3054726c450eb2ec57c5e429b47b5d8f0c0328c041fb31c6de.png)  

SPADE，PaDiM都是使用预训练模型提取特征，而这篇文章会根据异常定位的任务进行`fine-tuning`。针对`PaDiM`存在特征不对齐的问题，第一个改进点就是引入`STN模块`进行粗对齐，如上图所示：

:::tip
STN是一个空间转换模块，核心的思想是让网络去学习生成矩阵参数，从而学会空间转换

![图 14](images/c5f24e3997c49c6ae514b7ab340e2a647a3fe7961f62c589389b5d0e051d05f6.png)  

STN模块的整体结构如上图所示，其由`localisation net`、`Grid generator`和`Sampler`三部分组成，输入特征图U（也可以直接是RGB图像）经过空间变换模块得到输出特征图 V。

:::

图像级的粗对齐(ICA)，会额外引入一个L2 loss：

$$
\mathcal{L}_{I C A}\left(\mathcal{D} ; \theta_{h}, \mathcal{T}_{\theta}\right)=\sum_{A, B \in \mathcal{D}} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1}\left\|h_{\mathcal{T}_{\theta}}\left(A_{i, j}\right)-h_{\mathcal{T}_{\theta}}\left(B_{i, j}\right)\right\|_{2}
$$

特征级的粗对齐(FCA)，在预训练网络中的每个stage引入一个`STN模块`，引导特征图对齐，但这里没有引入额外的损失(只受到Fine Alignment模块的监督)



