---
title: 「论文阅读」Understanding-Object-Detection-Through-An-Adversarial-Lens
date: 2021-07-20 14:01:02
tags: [神经网络对抗攻击,综述]
categories: 神经网络对抗攻击
katex: true
---

# Abstract

>Deep neural networks based object detection models have revolutionized computer vision and fueled the development of a wide range of visual recognition applications. However, recent studies have revealed that deep object detectors can be compromised under adver- sarial attacks, causing a victim detector to detect no object, fake ob- jects, or mislabeled objects. With object detection being used perva- sively in many security-critical applications, such as autonomous vehi- cles and smart cities, we argue that a holistic approach for an in-depth understanding of adversarial attacks and vulnerabilities of deep object detection systems is of utmost importance for the research community to develop robust defense mechanisms. This paper presents a framework for analyzing and evaluating vulnerabilities of the state-of-the-art object detectors under an adversarial lens, aiming to analyze and demystify the attack strategies, adverse effects, and costs, as well as the cross-model and cross-resolution transferability of attacks. Using a set of quantitative metrics, extensive experiments are performed on six representative deep object detectors from three popular families (YOLOv3, SSD, and Faster R-CNN) with two benchmark datasets (PASCAL VOC and MS COCO). We demonstrate that the proposed framework can serve as a methodical benchmark for analyzing adversarial behaviors and risks in real-time ob- ject detection systems. We conjecture that this framework can also serve as a tool to assess the security risks and the adversarial robustness of deep object detectors to be deployed in real-world applications.

# 相关工作与问题陈述

目标检测是计算机视觉的核心任务，它以输入的图像或视频帧为对象，根据已知的类别检测出多个语义对象实例。尽管有些人可能将目标检测视为图像分类任务的概括，但深度目标检测器是一个多任务学习者，执行两个独特的学习任务，这使得攻击目标检测比图像分类更复杂，更具挑战性。（1）对象检测应该检测和识别封装在单个图像或视频帧中的多个语义对象的实例，而普通的图像分类器处理的是将每个图像分类为一个已知的类(2）目标检测是对单个图像中多个语义对象的多个实例进行局部化和分类，每个实例的定位精度会影响实例的分类精度。`因此，针对图像分类器的对抗性攻击技术不适用于攻击深目标探测器。`想要攻击目标检测网络则需要利用更复杂的攻击技术生成攻击目标检测模型的对抗性例子，通过同时迭代地最大化目标丢失、定位丢失和分类丢失，计算并向良性输入注入对抗性特征。

当下的目标检测网络主要可以分为单阶段目标检测网络与二阶段目标检测网络。二阶段的目标检测网络首先通过RPN层检测出目标区域，然后再使用分类器对区域进行分类。典型的例子有Faster R-CNN，R-CNN以及Mast R-CNN。单阶段的检测网络通过直接预测边界框的坐标来联合估计对象的边界框和类标签。这一块的代表就是YOLO和SSD。此外，不同的目标探测器，甚至来自同一个系列（例如，Faster R-CNN），可以使用不同的神经网络作为主干，另外一些还使用不同的输入分辨率来优化其检测性能。

白盒攻击，通过利用RPN来攻击Faster R-CNN，如DAG、UEA和其他类似方法[1,12]。例如，DAG首先（随机）为检测到的每个区域分配一个对抗性标签，然后执行迭代梯度反向传播来对提议进行错误分类。然而，以Faster R-CNN作为受害探测器的DAG攻击不能应用或推广到不使用RPN的单阶段探测器。类似于对图像分类器的黑盒转移攻击[18]，UEA[27]研究了攻击的可转移性，通过使用Faster R-CNN检测器生成的对抗性示例来攻击SSD检测器。

![image-20210722095014385](https://gitee.com/coronapolvo/images/raw/master/20210722095016image-20210722095014385.png)

# 攻击模块

下图中给出了proposed framework的概述。

![image-20210721174741351](https://gitee.com/coronapolvo/images/raw/master/20210721174744image-20210721174741351.png)

## 基于DNN的目标检测与对抗攻击

基于DNN的目标检测是一个多任务的学习问题，其目的是为了使目标的检测更加简单。这个任务的目标是最小化检测(1) object existence, (2) bounding boxes, and (3) class labels of detected objects 的误差。

下面是论文中对于训练一个目标检测神经网络的描述：

![image-20210721181103469](https://gitee.com/coronapolvo/images/raw/master/20210721181105image-20210721181103469.png)

那么我们的攻击对象的目标就应该是生成一个$x'$使得：

![image-20210721181140802](https://gitee.com/coronapolvo/images/raw/master/20210721181142image-20210721181140802.png)

尽管针对目标检测系统的对抗性攻击更加复杂，采用不同的公式，但它们通常利用公式（1）中一个或多个损失得出的梯度。这使得攻击算法能够细致地向输入图像注入扰动，使得输入中的微小变化将在受害者检测器的整个前向传播过程中被放大，并且变得足够大以改变一种或多种类型的预测结果（即，对象存在、边界框和类概率），下面我们分析了四种典型的目标检测系统攻击算法，了解了它们的特性，揭示了它们的工作原理。

## TOG

TOG论文的作者基于迭代梯度方法开发了TOG攻击家族。使用梯度下降得到的恶意扰动让检测器给出所需的错误检测。在适当设置指定的检测
$$
\mathcal{O}^{*}(\boldsymbol{x})
$$

以及攻击损失$\mathcal{L}^{*}$下, TOG一般可表述为：

![image-20210722094201877](https://gitee.com/coronapolvo/images/raw/master/2021072209474220210722094206image-20210722094201877.png)

TOG中提出了几种攻击方式，都是在上述公式的基础上做一些变形以达到不同的效果；

### Untargeted attacks

非目标攻击欺骗受害者检测器随机误判，而不针对任何特定对象。如果对抗性示例可以让检测器给出任何形式的错误结果，例如让对象随机消失、伪造或错误标记，则认为此类攻击成功。

TOG利用了$
\mathcal{L}_{\text {obj }}, \mathcal{L}_{\text {bbox }}, \text { and } \mathcal{L}_{\text {class }}
$的梯度，并制定了要执行的攻击：

![image-20210722094738935](https://gitee.com/coronapolvo/images/raw/master/20210722094747image-20210722094738935.png)

如表1第2列所示，检测器无法识别在第1列正常输入上检测到的任何正确对象，但具体的效果在不同的输入图像和攻击算法中有所不同。

### Object-vanishing attacks consistently

Object-vanishing attacks consistently使得受害者探测器不能定位和识别任何物体。TOG Vanising利用$\mathcal{L}_{\mathrm{obj}}$的梯度来确定对象是否存在，并将攻击公式化如下：
$$
\boldsymbol{x}_{t+1}^{\prime}=\prod_{\boldsymbol{x}, \epsilon}\left[\boldsymbol{x}_{t}^{\prime}-\alpha_{\mathrm{TOG}} \Gamma\left(\nabla_{\boldsymbol{x}_{t}^{\prime}} \mathcal{L}_{\mathrm{obj}}\left(\boldsymbol{x}_{t}^{\prime}, \varnothing ; \boldsymbol{\theta}\right)\right)\right]
$$
专门针对目标消失的攻击，如果攻击成功，将使受害者探测器无法检测到任何目标，如表1第3列所示，在这两个示例中均未检测到任何目标。

### Targeted object-mislabeling attacks consistently

故名思义，这个攻击方法就是让检测器将目标检测的类别检测错误。用恶意选择的标签去替换真实的标签，同时保留正确的检测框位置。在保持其他两个部分的渐变不变的情况下，TOG-mislabeling从$\mathcal{O}^{*}(x)$中的每个对象。Patch的更新公式如下：
$$
\boldsymbol{x}_{t+1}^{\prime}=\prod_{\boldsymbol{x}, \epsilon}\left[\boldsymbol{x}_{t}^{\prime}-\alpha_{\operatorname{TOG}} \Gamma\left(\nabla_{\boldsymbol{x}_{t}^{\prime}} \mathcal{L}\left(\boldsymbol{x}_{t}^{\prime}, \boldsymbol{\mathcal { O }}^{*}(\boldsymbol{x}) ; \boldsymbol{\theta}\right)\right)\right]
$$
例如，表1第5列中停车标志牌被错误检测成了语言。请注意，人和车仍然可以在此攻击下被检测到，因为它们不是攻击目标，只有停车标志会被错误检测。

由于TOG不会攻击对象检测器中的特殊结构 (例如RPN)，因此它适用于单阶段和二阶段的网络。受攻击图像分类器的通用扰动的启发 ，TOG还开发了单向扰动，以在对象消失或对象制造攻击方面攻击深度对象检测器 。通过在训练集和受害者检测器上训练通用扰动，可以在在线检测阶段将通用扰动应用于发送给受害者的任何输入。

## DAG

DAG是一种无目标的随机攻击，首先手动分配IOU阈值以0.90作为二阶段模型的RPN中的非最大抑制 (NMS)的阈值。此攻击设置要求一个proposal region与另一个proposal region高度重叠 (> 90%)，以便修剪。在随后的网络进行边界框和类标签预测的细化之后，DAG为每个proposal region分配一个随机选择的标签，然后执行迭代梯度攻击以使用以下公式对proposal进行错误分类:

![image-20210722104940346](https://gitee.com/coronapolvo/images/raw/master/20210722104943image-20210722104940346.png)

由于DAG需要操纵RPN以生成大量proposal，因此它只能直接适用于二阶段检测模型。

## RAP

RAP是一种无目标的随机攻击，其重点是在两阶段算法中让RPN的功能失效。它利用了来自 (i) 对象损失 (即Lobj) 的复合梯度，该objness损失使RPN无法返回前景对象，以及 (ii) 定位损失 (即Lbbox) 导致边界框估计不正确。

![image-20210722105331592](https://gitee.com/coronapolvo/images/raw/master/20210722105335image-20210722105331592.png)

## UEA

UEA 是一种无目标的随机攻击。它训练有条件的生成对抗网络 (GAN) 来制作对抗示例。在深度目标检测器中，骨干网在两阶段算法中的region proposals的特征提取或一阶段技术中的对象识别中起着重要作用。在实践中，它通常是在大规模图像分类中表现良好

的流行体系结构 (例如，VGG16) 之一，并且使用ImageNet数据集进行了预训练以进行转移学习。UEA设计了多尺度注意力损失，鼓励GAN创建对抗示例，这些示例可以破坏受害者检测器中骨干网提取的特征图。

![image-20210722110947902](https://gitee.com/coronapolvo/images/raw/master/20210722110949image-20210722110947902.png)

每当另一个检测器是相同的主干时，对抗性示例很可能是有效的。公式10与DAG (公式8)，都需要对RPN进行操纵。因此，它无法直接攻击一阶段算法。



































