---
title: 「论文阅读」Conditional-Convolutions-for-InstanceSegmentation
date: 2020-10-30 14:01:02
tags: [论文阅读,实例分割]
categories: 论文阅读
katex: true
---

# 0x01 概要

Mask R-CNN等性能最好的实例分割方法依赖于ROI操作（通常是ROIPool或roalign）来获得最终的实例掩码。相比之下，我们建议从一个新的角度来解决站姿分割问题。我们不使用实例级roi作为固定权重网络的输入，而是使用基于实例的动态感知网络。CondInst有两个优点：1）通过完全卷积网络。
<!--more-->
CondInst有两个优点：
1）通过完全卷积网络解决实例分割，消除了ROI裁剪和特征对齐的需要；
2）由于动态生成条件卷积的能力大大提高，掩模头可以非常紧凑（例如，3转换层，每个只有8个通道），导致明显更快的推断。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201030000730950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)

# 0x02 Mask-RCNN的缺点

实例分割是计算机视觉中的一项基本而又具有挑战性的任务，它需要一种算法来预测图像中每个感兴趣的实例的每像素掩模。尽管最近提出了一些工作，但实例分割的主流框架仍然是两阶段方法Mask R-CNN，它将实例分割转化为两阶段的检测和分割任务。Mask R-CNN首先使用一个更快的对象检测器R-CNN来预测每个实例的边界框。然后在每个实例中，使用roiallign操作从网络的特征图中裁剪出感兴趣的区域（roi）。为了预测每个实例的最终掩模，将一个紧凑的全卷积网络（FCN）（即掩模头）应用到这些roi上，以执行前景/背景分割。

1） 由于ROI通常是轴对齐的边界框，对于形状不规则的对象，它们可能包含大量不相关的图像内容，包括背景和其他实例。这个使用旋转roi可以缓解这个问题，但代价是更复杂的管道。

2） 为了区分前景实例和背景内容或实例，掩码头需要一个相对较大的接受域来编码足够大的上下文信息。因此，在掩模头中需要一组3×3的卷积（例如，掩模R-CNN中有4个3×3卷积，256个通道）。它大大增加了掩模头的计算复杂度，导致推理时间在实例数上有显著变化。

3） ROI的大小通常不同。为了在现代深度学习框架中使用有效的批处理计算，通常需要调整大小操作来将裁剪区域调整为相同大小的补丁。例如，Mask R-CNN会将所有裁剪区域的大小调整为14×14（使用反褶积将采样率提高到28×28），这限制了实例分段的输出分辨率，因为大型实例需要更高的分辨率来保留边界的细节

# 0x03 为什么FCNs在实例分割上的效果不好

我们发现，将模糊神经网络应用于实例分割的主要困难在于相似图像的出现可能需要不同的预测，但FCNs难以实现这一点。例如，如果在一个input image中有两个外观相似的人A和B，那么在预测A的实例掩码时，FCN需要将B预测为background w.r.t.A，这可能很困难，因为它们在外观。因此，ROI操作用于裁剪感兴趣的人，即A；并过滤掉B，实例分段需要两种类型的in队形：

1） 用于对对象进行分类的外观信息；

2）用于区分属于同一类别的多个对象的位置信息。几乎所有的方法都依赖于ROI裁剪，对实例的位置信息进行显式编码。相比之下，CondInst通过使用敏感卷积滤波器以及显示在特征地图上的相对坐标来利用位置信息。

因此，我们提倡一种新的解决方案，即使用实例感知FCNs进行实例任务预测。换言之，与使用一组固定卷积滤波器的标准ConvNet作为掩码头来预测所有实例，而是根据要预测的实例来调整网络参数。在动态过滤网络[20]和CondConv[41]的启发下，对于每个实例，控制器子网络（见图3）动态生成掩码FCN网络参数（以实例的中心区域为条件），然后使用该参数预测该实例的掩码。预计网络参数可以对该实例的特征（例如相对位置、形状和外观）进行编码，并且只对该实例的像素进行激发，从而绕过了上述困难。这些条件掩模头被应用到整个特征映射中，消除了对ROI操作的需要。乍一看，这个想法可能行不通，因为如果某些图像包含多达几十个实例，则实例掩码头可能会产生大量的网络参数。然而，我们发现，一个非常紧凑的FCN掩模头和动态生成的滤波器已经可以优于先前基于ROI的mask R-CNN，从而大大降低了Mask-CNN中掩模头的每一瞬间的计算复杂度。

# 0x04 主要贡献

- 试图从一个新的角度来解决实例分割问题。为此，我们提出了CondInst实例分割框架，该框架比现有的Mask R-CNN等方法在提高实例分割速度的同时，提高了实例分割的性能。据我们所知，这是第一次一个新的实例分割框架在精确度和速度上都优于最新的技术
- CondInst是完全卷积的，并且避免了许多现有方法中使用的上述调整大小操作，因为CondInst不依赖ROI。不必调整特征图的大小，就可以获得高分辨率的分辨率，并具有更精确的边缘。
- 与以前的方法不同，一旦训练完所有实例，掩码头中的过滤器都是固定的，而我们的掩码头中的过滤器是动态生成的，并根据实例进行调整。因此，只需记住一个过滤器，就可以大大减少所要求的过滤器的负载。因此，Mask head 可以非常轻量，显著减少推理时间。与bounding box检测器FCOS相比，CondInst只需要多10%的计算时间，甚至可以处理每个图像的最大实例数（即100个实例）。

# 0x05 CondInst的实例分割
## 1x01 网络总体结构

回想一下mask R-CNN使用一个对象检测器来预测输入图像中实例的边界框。边界框实际上就是掩码R-CNN表示实例的方式。类似地，CondInst使用实例感知过滤器来表示实例。换句话说，CondInst没有将实例概念编码到边界框中，而是隐式地将其编码到掩码头的参数中，这是一种更灵活的方式。例如，它可以很容易地表示不规则的形状，而这些不规则形状是很难被表示的,）被边界框紧紧包围。这是CondInst相对于以前基于ROI的方法的优势之一。

与基于ROI的方法获取边界框的方式类似，实例感知滤波器也可以通过对象检测器获得。在这项工作中，由于CondInst的简单性和灵活性，CondInst在流行的目标检测器FCOS上构建CondInst。同时，在FCOS中消除锚盒也可以节省参数的数目和条件的计算量。如图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201030005947312.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
利用特征金字塔网络的特征映射{P3、P4、P5、P6、P7}，其下采样率分别为8、16、32、64和128。如图3所示，在FPN的每个特征层上，应用一些功能层（在虚线框中）来进行实例相关的预测。例如，目标实例的类和实例的动态生成的筛选器。从这个意义上讲，CondInst可以看作是Mask R-CNN，它们都是先处理图像中的实例，然后预测实例的像素级掩码（即实例优先），除了检测器，如图所示，还有一个掩模分支，**它提供了我们生成的掩码头作为输入来预测所需实例掩码的特征映射。**

特征图表示为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201030012049379.png#pic_center)
掩模分支连接到FPN level p3，因此其输出分辨率为输入图像分辨率的18。掩模分支在最后一层之前有四个3×3的卷积，有128个通道。之后，为了减少生成参数的数量，掩码分支的最后一层将信道数从128减少到8（即，Cmask＝8）。令人惊讶的是，使用cmask=8已经可以获得优异的性能，而使用更大的cmask（例如16）并不能提高性能，如我们的实验所示。更严重的是，使用cmask=2只会使mask AP的性能降低0.3%。此外，如图3所示，fmaski与坐标图相结合，这些坐标是fmask上所有位置到位置（x，y）的相对坐标（即，生成遮罩头的滤波器的位置）。然后，将组合发送到掩码头以预测实例掩码。如我们的实验所示，相对坐标为预测实例掩模提供了强有力的线索。此外，单个sigmoid被用作掩模头的最终输出，因此掩模预测是类无关的。**实例的类别由分类头与控制器并行预测。**

原始掩模预测的分辨率与F mask的分辨率相同，后者是输入图像分辨率的八分之一。为了产生高分辨率的实例掩模，使用双线性上采样将maskprediction上采样到4，得到400×512掩模预测（如果输入图像大小为800×1024）。我们将在实验中证明上采样对CondInst的最终瞬间分段性能至关重要。需要注意的是，该掩模的分辨率比掩模R-CNN（如前所述仅为28×28）高得多。

## 1x02 网络输出和训练标签

CondInst 具有如下的输出头：

- Classiofication Head：
分类头预测实例的分类，ground-truth标签为类别信息或者0（背景）即背景）。在FCOS中，网络预测一个C-D vector <img src="https://www.zhihu.com/equation?tex=p_{x,y}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">用于分类，每个元素在<img src="https://www.zhihu.com/equation?tex=p_{x,y}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">对应于一个二进制分类器，其中C表示类别的数量。
- Controller  Head
控制器头与上述分类头具有相同的结构，用于预测该位置实例的Mask head的参数。Mask head 预测这个物体Mask。为了预测参数，我们将滤波器的所有参数（即权重和偏差）串联在一起作为一个N-D向量 <img src="https://www.zhihu.com/equation?tex=\theta_{x,y}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">，其中N是参数的总数。因此，控制器头部没有输出信道。掩模头是一种非常紧凑的fcn结构，它有三个1×1的卷积，每个卷积有8个通道，除了最后一个外，都使用ReLU作为激活函数。这里没有使用诸如批处理规范化之类的规范化层。最后一层有1个输出信道，使用sigmoid来预测前景的概率。任务头总共有169个参数（#权重=（8+2）×8（conv1）+8×8（conv2）+8×1（conv3）和#biaes=8（conv1）+8（conv2）+1（conv3））。如前所述，生成的过滤器包含关于实例所在位置的信息，因此，理想情况下，带有过滤器的遮罩头将只对实例的像素点进行触发，甚至将整个特征映射作为输入。

- Center-ness  and  Box  Heads.
从概念上讲，CondInst可以消除box head，因为CondInst不需要ROIs。然而，我们发现如果使用基于Box的NMS，推理时间将大大减少。因此，我们仍然预测CondInst中的Box。我们要强调的是，预测框仅在NMS中使用，不涉及任何ROI操作。

## 1x03 损失函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201030062801683.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201030063548336.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
## 1x04 推理
给定一幅输入图像，通过网络进行转发，得到包含分类置信度的输出<img src="https://www.zhihu.com/equation?tex=p_{x,y}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">, 中心度得分，box预测 <img src="https://www.zhihu.com/equation?tex=t_{x,y}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">和生成的参数<img src="https://www.zhihu.com/equation?tex=\theta_{x,y}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">, 我们首先按照FCOS中的步骤来获得Box。然后，使用阈值为0.6的基于Box的NMS来消除重复检测，然后使用前100个框计算掩码。与FCOS不同，这些方框还与控制器生成的过滤器相关联。让我们假设在NMS之后还有K个box，因此我们有生成的K个过滤器组。这一组滤波器用于产生特定于实例的Mask-head。这些特定于实例的掩码头以FCNs的方式应用于与<img src="https://www.zhihu.com/equation?tex=F_{x,y}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">（即<img src="https://www.zhihu.com/equation?tex=F_{mask}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">,<img src="https://www.zhihu.com/equation?tex=O_{x,y}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">的组合）来预测实例的掩码。由于任务头是一个非常紧凑的网络（3个1×1卷积，共有8个信道和169个参数），计算掩码的开销非常小。例如，即使有100次检测（即MS-COCO上每个图像的最大检测次数），掩模头上总共也只有不到5毫秒的时间，这只给基本检测器FCOS增加了10%的计算时间。相比之下，Mask R-CNN的Mask head有4个3×3的256个通道，参数大于2.3M，计算时间较长。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201030065343641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3B5dGhvbl9MQ19ub2h0eXA=,size_16,color_FFFFFF,t_70#pic_center)
# 0x06 小结
提出了一个新的更简单的实例分割框架Condinst。不同于以往的方法，如Mask R-CNN，它使用固定权重的任务头，CondInst将掩码头设置在实例上，并动态生成掩码头的过滤器。这不仅降低了掩模头的参数和计算复杂度，而且消除了ROI操作，从而形成了一个更快、更简单的实例分割框架。

















































