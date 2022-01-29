---
title: >-
  「论文阅读」TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time
  Object Detection Systems
date: 2021-07-23 09:50:25
tags: 神经对抗攻击
categories: 神经网络对抗攻击
katex: true
---

>The rapid growth of real-time huge data capturing has pushed the deep learning and data analytic computing to the edge systems. Real-time object recognition on the edge is one of the representative deep neural network (DNN) powered edge systems for real-world mission-critical applications, such as autonomous driving and aug- mented reality. While DNN powered object detection edge systems celebrate many life-enriching opportunities, they also open doors for misuse and abuse. This paper presents three Targeted adversar- ial Objectness Gradient attacks, coined as TOG, which can cause the state-of-the-art deep object detection networks to suffer from object-vanishing, object-fabrication, and object-mislabeling attacks. We also present a universal objectness gradient attack to use ad- versarial transferability for black-box attacks, which is effective on any inputs with negligible attack time cost, low human perceptibil- ity, and particularly detrimental to object detection edge systems. We report our experimental measurements using two benchmark datasets (PASCAL VOC and MS COCO) on two state-of-the-art detection algorithms (YOLO and SSD). The results demonstrate serious adversarial vulnerabilities and the compelling need for de- veloping robust object detection systems.

Edge 数据分析和作为边缘服务的深度学习已经吸引了业界和学术界的一系列研究和开发工作[5,9]。开源的深度目标检测网络[8,12,13]推动了新的边缘应用和边缘系统的部署，例如自动驾驶车辆上的交通标志识别和智能监控系统上的入侵检测[3]。然而，很少有人对实时深度目标检测器的漏洞进行系统的研究，这些漏洞对于边缘安全和隐私至关重要。图1显示了一个典型的场景，边缘系统从传感器(例如，摄像机)接收输入图像或视频帧，并在边缘设备(例如，带有人工智能加速模块的 Raspberry Pi)上运行一个实时 DNN 对象检测模型(例如，YOLOv3[12])。在没有攻击的情况下，训练有素的物体探测器可以处理良性输入(上图)并准确地识别街对面行走的人。然后一个对抗性的例子(底部) （从人眼视觉上看并没有什么不同 ）同样的物体检测器将被愚弄，输出错误的检测结果。

![image-20210723102417527](https://gitee.com/coronapolvo/images/raw/master/20210723102419image-20210723102417527.png)

# 介绍

在本文中，我们通过开发3种针对对抗性对象梯度攻击，提出了 DNN 对象保护系统的3个漏洞，作为一系列针对实时目标检测系统的 TOG 攻击。文献[4]虽然对 DNN 图像分类器进行了大量的对抗性攻击，但这些攻击主要是通过不同的攻击策略来确定 DNN 分类器的位置和每像素摄动量，从而对良性输入图像进行注入，导致 DNN 分类器产生错误的分类结果。相比之下，深度目标检测网络可以检测和分割单个图像或视频帧中可能视觉上重叠的多个对象，并为每个检测到的对象提供一个类标签。因此，深入了解 DNN 图像分类器中深层目标检测器的各种漏洞比误分类攻击更为复杂，因为 DNN 目标检测器具有更大、更多样的攻击面，如目标存在、目标定位、目标类标签等，为各种攻击目标和复杂性的攻击提供了更多的机会。TOG 攻击是目标检测网络上第一种针对不同对象语义的攻击方法，如使对象消失、制造更多的对象、给部分或全部对象贴错标签等。这些攻击中的每一个都注入了一个人类无法察觉的对抗性的 pertur-bation，以欺骗实时对象检测器，使其以三种不同的方式行为不当，如图2所示。图2(b)中的对象消失攻击使所有对象从 YOLOv3[12]检测器中消失。图2(c)中的物体制造攻击导致探测器高可信度地输出许多虚假物体。图2(d)中的对象错误标记攻击欺骗探测器错误标记(例如，停车标志变成了一把伞)。我们进一步提出了一个高效的通用对抗扰动算法，它产生一个单一的通用扰动，可以扰乱任何输入以有效地欺骗受害者检测器。鉴于攻击是离线生成的，通过利用对抗传输性，可以使用 TOG 通用扰动发起一个黑盒攻击，其在线攻击代价非常低(几乎为零) ，这在实时边缘对象检测应用程序中尤其致命。

![image-20210723103029275](https://gitee.com/coronapolvo/images/raw/master/20210723103030image-20210723103029275.png)

# TOG ATTACKS

一般来说，深目标检测网络具有相似的输入与输出的结构。它们都采用输入图像或视频帧，通过bounding box技术为所有感兴趣的目标提供目标定位，并对每个检测到的目标进行分类，从而产生输出。TOG 攻击不受任何特定检测算法的限制，正如我们的实验评估所示。TOG 攻击不受任何特定检测算法的限制，正如我们的实验评估所示。给定一个输入图像 x，对象检测器首先检测大量的候选边框$\hat{\mathcal{B}}(\boldsymbol{x})=\left\{\hat{\boldsymbol{o}}_{1}, \hat{\boldsymbol{o}}_{2}, \ldots, \hat{\boldsymbol{o}}_{S}\right\}$​​是宽和高。（都是目标检测的东西这里都不多说了）

一个敌对的例子 x ′是由良性输入 x 不断更新生成的，旨在让检测器检测错误。对抗性示例的生成过程可以公式化为：
$$
\min \left\|x^{\prime}-x\right\|_{p} \quad  s.t. \hat{O}\left(x^{\prime}\right)=O^{*}, \hat{O}\left(x^{\prime}\right) \neq \hat{O}(x)
$$
其中 p 是距离度量，可以是L0范数，表示被改变的像素的百分比，L2范数计算欧几里得度量，或$L_{\infty}$表示目标攻击的目标检测值，或非目标攻击的目标检测值。

图3说明了使用 TOG 的对抗性攻击过程。首先给定一个输入源(例如，一个图像或视频帧) ，TOG 攻击模块使用敌方指定的配置来准备相应的对抗性扰动，这些扰动将被添加到输入中，导致检测器错误检测。TOG 中的前三个攻击: TOG-vanishing, TOG-fabrication,和 TOG-mislabeling，为每个输入量定制一个对抗性的扰动，而 TOG-universal 使用相同的通用扰动来破坏任何输入。

![image-20210723115520629](https://gitee.com/coronapolvo/images/raw/master/20210723115521image-20210723115520629.png)

深度神经网络的训练通常从模型权值的随机初始化开始，然后通过求损失函数$\mathcal{L}$的导数，使用下列方程，直到收敛为止:

$$
\boldsymbol{W}_{t+1}=\boldsymbol{W}_{t}-\alpha \frac{\partial \mathbb{E}_{(\tilde{\boldsymbol{x}}, \boldsymbol{O})}\left[\mathcal{L}\left(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W}_{t}\right)\right]}{\partial \boldsymbol{W}_{t}}
$$

其中 α 是控制更新步长的学习速率。通过固定输入图像 x 和逐步更新模型权重 w 来训练深度目标检测网络，TOG 通过逆转训练过程来进行对抗性攻击。我们确定了受害者检测器的模型权重，并通过以下一般方程迭代更新输入图像 x，使其朝向由攻击类型确定的目标:

<img src="https://gitee.com/coronapolvo/images/raw/master/20210723120954image-20210723120944632.png" alt="image-20210723120944632" style="zoom:50%;" />

目标检测网络的优化目标可以用如下公式表述：
$$
\begin{aligned}
\mathcal{L}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W})=& \mathcal{L}_{\mathrm{obj}}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W})+\lambda_{\text {noobj }} \mathcal{L}_{\text {noobj }}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W}) \\
&+\lambda_{\operatorname{loc}} \mathcal{L}_{\mathrm{loc}}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W})+\mathcal{L}_{\mathrm{prob}}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W})
\end{aligned}
$$
下面是对公式具体项的一些公式：

检测目标和未检测目标的损失项：
$$
\begin{aligned}
\mathcal{L}_{\mathrm{obj}}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W}) &=\sum_{i=1}^{S}\left[\mathbb{1}_{i} \ell_{\mathrm{BCE}}\left(1, \hat{C}_{i}\right)\right] \\
\mathcal{L}_{\mathrm{noobj}}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W}) &=\sum_{i=1}^{S}\left[\left(1-\mathbb{1}_{i}\right) \ell_{\mathrm{BCE}}\left(0, \hat{C}_{i}\right)\right]
\end{aligned}
$$

$位置检测损失项：$
$$
\begin{aligned}
\mathcal{L}_{\mathrm{loc}}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W}) &=\sum_{i=1}^{S} \mathbb{1}_{i}\left[\ell_{\mathrm{SE}}\left(b_{i}^{x}, \hat{b}_{i}^{x}\right)+\ell_{\mathrm{SE}}\left(b_{i}^{y}, \hat{b}_{i}^{y}\right)\right.\\
&\left.+\ell_{\mathrm{SE}}\left(\sqrt{b_{i}^{W}}, \sqrt{\hat{b}_{i}^{W}}\right)+\ell_{\mathrm{SE}}\left(\sqrt{b_{i}^{H}}, \sqrt{\hat{b}_{i}^{H}}\right)\right]
\end{aligned}
$$
类别信息损失项：

$$
\mathcal{L}_{\text {prob }}(\tilde{\boldsymbol{x}} ; \boldsymbol{O}, \boldsymbol{W})=\sum_{i=1}^{S} \mathbb{1}_{i} \sum_{c \in \text { classes }} \ell_{\mathrm{BCE}}\left(p_{i}^{c}, \hat{p}_{i}^{c}\right)
$$

>TOG的主要思路就是通过更改正常目标检测损失函数的输入，来达到扰乱检测网络的作用；

下面是他更改损失项的方法；

![image-20210723232359219](https://gitee.com/coronapolvo/images/raw/master/20210728142932image-20210723232359219.png)



















