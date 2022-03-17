# YOLO系列对比整理

## YOLOv1
### 论文核心思路

1) 将一幅图像分成SxS个网格(grid cell), 如果某个object的中心落在这个网格 中，则这个网格就负责预测这个object

![](https://gitee.com/coronapolvo/images/raw/master/20220309220454.png)

2) 每个网格要预测B个bounding box, 每个bounding box 除了要预测位置之外，还要附带预测一个confidence值，每个网格还要预测C个类别的分数。 confidence表示该位置有物体的概率

![](https://gitee.com/coronapolvo/images/raw/master/20220309220632.png)

3) 网络后处理，使用nms对多余的box去除；

![](https://gitee.com/coronapolvo/images/raw/master/20220309224307.png)


### YOLOV1网络结构



![](https://gitee.com/coronapolvo/images/raw/master/20220309225308.png)


YOLOv1的网络结构可以说是比较简单了，网络结构主要由卷积、池化和全连接三部分组成。最终的输出结构为7x7x30。其中7x7表示一个图片会被划分为7x7的网格，下图更加详细的说明了30的组成部分：

![](https://gitee.com/coronapolvo/images/raw/master/20220309230236.png)

30分别由两个预测框信息，两个置信度信息和20个类别置信度组成（Pascal VOC数据集）

### 损失函数

![](https://gitee.com/coronapolvo/images/raw/master/20220309230500.png)

YOLOv1的损失函数主要采用了平方距离的方式。需要注意的是对于w和h的误差，yolov1采用了根号差平方的形式。

### 一些小问号
:::tip
YOLOv1有哪些创新点？
:::

1. 将整张图片作为网络的输入，直接在输出层回归bounding box的位置和所属的类别
2. 速度快，one-stage的开山之作


:::tip
为什么yolov1的损失函数中在求box的w和h的误差时要先开根号？
:::

因为如果不开根号对于大物体和小物体相同的w和h就会产生相同的误差，如下图所示：

![](https://gitee.com/coronapolvo/images/raw/master/20220309231426.png)

很明显上图的大目标的预测结果要比小目标的预测结果好，但是却具有相同的w。开根号能够使用具有相同w的大物体产生更小的误差，具有相同w的小物体产生更大的误差。

![](https://gitee.com/coronapolvo/images/raw/master/20220309231822.png)



:::tip
YOLOv1的损失函数由几部分组成？
:::

1.  bounding box 损失：box坐标的x和y坐标分别作差求平方后求和
2.  confidence 损失：confidence 作差求平方和
3.  class 损失：类别作差平方求和


:::tip
Yolov1的问题有哪些？
:::

1. 对于群体性的小目标检测结果很差：在YOLOv1的思想中每一个cell只预测两个bounding box而且这两个boxes都是同一个类别的。对于没有每一个cell都只预测一组同一个类别的目标。所以当小目标聚集在一起的时候就会检测效果很差；
2. 对于不同尺寸的同一个目标检测结果并不是很理想
3. 主要错误的原因都是源于定位不准确的问题：这是因为YOLOv1采用直接预测坐标的一个方式，而不是类似Faster RCNN一样采用一个anchor base的方法。


## YOLOv2

:::tip
YOLOv2是对v1的一个全方面的改进，其检测进度与检测效果达到了当时SOTA的水平
:::

### 让检测效果更好

#### Batch Normalization
BN层使得YOLOv2网络能够更好的收敛，且具有正则化的效果。使用了BN后可以舍去掉YOLOv1中使用的Dropout算法。

:::info
**为什么BN可以替换dropout？**
:::

在以前Dropout是一种防止模型过拟合的重要手段，但是在当下的网络里面Dropout出现的次数是越来越少了，主要原因有如下几点：
1. Dropout对于卷积层的正则化效果很差，因为卷积本身参数就很少
2. 当下的网络经常使用全连接网络代替全连接的结果，Dropout的效果远远没有BN好


:::info
**为什么BN可以加速模型的收敛？**
:::

BN在训练时候,会把每一层的Feature值约束到均值为0, 方差为1, 这样**每一层的数据分布**都会一样,数据被约束到均值为0 ,方差为1,相当于**把数据从饱和区（数据变换平缓的区域）约束到了非饱和区（数据变换不平缓的区域）**,这样**求得的梯度值会更大,加速收敛,也避免了梯度消失和梯度爆炸问题**.

:::info
**为什么BN可以防止模型过拟合？**
:::

在训练中，BN的使用使得一个mini-batch中的所有样本都被关联在了一起，因此网络不会从某一个训练样本中生成确定的结果。就是一个batch数据中**每张图片对应的输出都受到一个batch所有数据影响**,这样**相当于一个间接的数据增强**,达到防止过拟合作用.


#### High-Resolution Classifier
YOLOv2增加了输入图片的尺寸 （448 x 448），大图片可以提供更多的特征信息。检测效果提升很正常；

#### Anchor Boxes
YOLOv1采用直接生成box坐标的方式，但在YOLOv2中作者还是采用了根据anchor去预测偏移量的方式。原文中对于Anchor Box的描述是：

>Convolutional With Anchor Boxes. YOLO predicts the coordinates of bounding boxes directly using fully connected layers on top of the convolutional feature extractor.
   map. Predicting offsets instead of coordinates simplifies the problem and makes it 
   easier for the network to learn.

使用anchor并没有让yolo的map增加，但是让模型的召回率增加了（这使得模型有更多的优化空间）

:::tip
**YOLOv2的anchor box是如何得到的？**

使用k-means聚类的方式对数据集中的box框进行聚类所得
:::

:::tip
**YOLOv2中预测坐标偏移和FasterRCNN中的有什么区别？**

YOLOv2在预测box相对于cell顶点坐标的偏移的时，使用了sigmoid函数将偏移量限制到0-1中。这样就不会使得预测的box中心坐标跑到别的cell中的问题。也能使得模型的训练过程更加稳定。
:::

#### Fine-Grind Features;
我们知道YOLO系列的网络会把图像分割成nxn的网格，对于大目标的物体13x13的网格已经很够用了，但是对于小目标的问题来说，更细的网格划分策略会产生更好的效果。因此YOLOv2将26x26与13x13大小的网络进行了融合。作者采用Passthough Layer来完成这个效果；

这个Passthough Layer听起来挺玄乎的，但是本质其实就是特征重排，26x26x512的feature map分别按行和列隔点采样，可以得到4幅13x13x512的特征，把这4张特征按channel串联起来，就是最后的13x13x2048的feature map.还有就是，passthrough layer本身是不学习参数的，直接用前面的层的特征重排后拼接到后面的层，越在网络前面的层，感受野越小，有利于小目标的检测。

#### Multi-Scale Training
每隔几轮便改变模型输入尺寸，以使模型对不同尺寸图像具有鲁棒性。每个10 batches，模型随机选择一种新的输入图像尺寸（320,352,...608，32的倍数，因为模型下采样因子为32），改变模型输入尺寸，继续训练。


### 让检测速度更快
#### Darknet-19
作者设置了新的分类网络来提取特征，在YOLOv1中使用了类似google net的网络结构，计算量比VGG-16小，准确率比VGG16略低。作者设计了一个新的分类网络（Darknet-19）来作为YOLOv2的基础模型。Darnet-19 由卷积层和pooling 层组成

:::tip
**为什么Darknet-19后面要几个19？**

因为有19个卷积层 O(∩_∩)O 哈哈
:::

Darknet-19的结构

![](https://gitee.com/coronapolvo/images/raw/master/20220312172142.png)

### 网络结构

![](https://gitee.com/coronapolvo/images/raw/master/20220312172617.png)

## YOLOv3
YOLOv3总结了自己在YOLOv2的基础上做的一些尝试性改进，有的尝试取得了成功，而有的尝试 并没有提升模型性能。其中有两个值得一提的亮点，一个是`使用残差模型`，进一步加深了网络结
构；另一个是`使用FPN架构实现多尺度检测`。

### 创新点
1. 新的网络结构：Darknet-53
2. 融合FPN
3. 用逻辑回归代替softmax作为分类器

###  结构上的改进

`backbone的改进:` YOLOv3在Darknet-19的基础上引入了残差块，并进一步加深了网络，改进后的网络有53个卷积层，因此叫做Darknet-53，此外Darknet-53用卷积层代替了Darknet-19中的max pooling层。

:::info
用卷积进行下采样和用池化进行下采样的区别是什么？

卷积在下采样的同时可以对特征进行提取，运行速度会慢一些，参数需要学习

池化进行下采样只能进行特征降维，运行速度快，参数不需要学习
:::

`使用FPN的结构进行特征提取：` YOLOv3中借鉴了FPN的思想，从不同尺度提取特征。相比于YOLOv2，YOLOv3提取最后三层特征图，不仅在每个特征图上分别独立检测，同时通过将小特征图上采样到与大特征图进行拼接后做进一步的预测。用k-means聚类的思想聚类出9种尺度的anchor box，将9种尺度的anchor box均匀的分配给3种尺度的特征图。

### 正负样本均衡

在论文中论文中只选择了最好的预测框用于计算loss，但是实际上这样会导致正负样本不均衡的情况。在u版的代码中作者将所有iou大于阈值的boxes都作为正样本，这样可以一定程度上解决正负样本不均衡的问题。

### 网络结构图
![](https://gitee.com/coronapolvo/images/raw/master/20220317222055.png)

在上图中我们能够很清晰的看到三个预测层分别来自的什么地方，以及Concatenate层与哪个层进行拼接。**注意Convolutional是指Conv2d+BN+LeakyReLU，而生成预测结果的最后三层都只是Conv2d。**


