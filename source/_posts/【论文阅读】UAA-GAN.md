---
title: 「论文阅读」UAA-GAN
date: 2021-07-27 14:01:02
tags: 神经网络对抗攻击
categories: 神经网络对抗攻击
katex: true
---



> Unsupervised Adversarial Attacks on Deep Feature-based Retrieval with GAN

这是一篇用GAN攻击神经网络的一篇文档，文中提出的UAA-GAN可以用于攻击图像检索网络如ReID等。

# 介绍

深层神经网络(Deep neural networks，dnn)是一种功能强大的特征表示学习工具，能够实现基于内容的图像检索(content-based image retrieval，CBIR)。近年来，深度特征正在迅速取代传统的图像特征，这些特征依赖于手工制作的关键点检测器和描述符。基于 dnn 的模型通过聚合预先训练的深层神经网络顶层的激活，生成一个图像的深层特征描述符，然后根据图像特征向量的欧几里得度量或余弦距离来确定图像之间的相似度(或距离)。据观察，这种方法比那些低层次的基于关键点的特征能够保存更多抽象和全局的语义信息。由于 DNN 具有良好的表示能力，许多研究者致力于通过学习鉴别特征表示来提高检索的准确性。然而，DNN 特征在检索过程中的鲁棒性和稳定性却被忽视了。

众所周知，基于 DNN 的分类系统很容易受到Adversarial image的影响: 通过对输入图像添加一些精心制作的微小扰动，目标 DNN 经常被误导，并以高置信度预测图像到错误的类别。对抗性的例子首先在文献[1]中介绍，扰动是通过调整输入以获得最大化的分类错误而产生的。很多注意力被吸引到对抗性攻击和防御之间的竞争。

基于深度特征的基于内容的图像检索(CBIR)系统及其衍生应用——人脸识别(ReID)和人脸搜索(face search)也容易受到对抗性攻击。也就是说，有可能篡改图像，使其在视觉上与原始形式几乎相同，但几乎不可能在图像检索系统中进行搜索。由于两个主要原因，攻击图像检索系统比攻击分类模型更具挑战性。

# 网络结构设计

其实本篇论文对于GAN的网络并没有太大的创新，`主要是改进点在于增加了生成器的损失项`。生成器的骨干部分使用的是CGAN。

下图是他的网络结构图：

![image-20210727141119457](https://gitee.com/coronapolvo/images/raw/master/20210728142849image-20210727141119457.png)

该框架的设计目标是通过生成一个带有特定扰动的对抗性示例，来攻击构建在特定目标特征提取网络上的检索系统。

## 生成器损失函数

生成器的损失函数主要由三部分组成：

第一部分用与衡量生成的图片与原有图片特征图上的不同，目的是让生成图片的特征图与原有图片的特征图差距越大越好。
$$
\begin{aligned}
\operatorname{maximize} & d\left(f_{\boldsymbol{x}}, f_{\tilde{\boldsymbol{x}}}\right) \\
\text { s.t. } &\|\delta\|_{\infty} \leq \epsilon, \\
& \tilde{\boldsymbol{x}} \in[0,1]
\end{aligned}
$$

$$
\mathcal{L}_{G A N_{-} G}=\mathbb{E}_{\tilde{\boldsymbol{x}} \sim p_{(\tilde{\boldsymbol{x}} \mid \boldsymbol{x})}}\left[(D(\tilde{\boldsymbol{x}})-1)^{2}\right]
$$

第二部分用与衡量原图与生成图像的差距，也就是生成的图像和攻击图像肉眼看起来有多大的不同。目的是让生成的攻击图像和原图像看起来几乎没有什么差距。
$$
\mathcal{L}_{\text {recon }}=\|\tilde{\boldsymbol{x}}-\boldsymbol{x}\|_{2}
$$
第三项的loss使用了 triplet loss和online hard negative mining的设计思路。现在假设$<\boldsymbol{x}, \tilde{\boldsymbol{x}}, \boldsymbol{x}^{\prime}>$​​之间的距离。可以用如下公式表示：
$$
d\left(f_{\boldsymbol{x}}, f_{\boldsymbol{x}^{\prime}}\right)+m+\leq d\left(f_{\boldsymbol{x}}, f_{\tilde{\boldsymbol{x}}}\right)
$$
所以第三项损失就可以写成：
$$
\mathcal{L}_{\text {metric }}=\max \left(d\left(f_{\boldsymbol{x}}, f_{\boldsymbol{x}^{\prime}}\right)+m-d\left(f_{\boldsymbol{x}}, f_{\tilde{\boldsymbol{x}}}\right), 0\right)
$$
生成器部分的完整损失就是：
$$
\mathcal{L}_{G}=\mathcal{L}_{G A N_{-} G}+\lambda_{r} \mathcal{L}_{\text {recon }}+\lambda_{m} \mathcal{L}_{\text {metric }}
$$

## 鉴别器损失函数

鉴别器损失函数就非常普通了，就是可以鉴别出来哪些图片是真的，哪些图片是假的即可。
$$
\mathcal{L}_{G A N_{-} D}=\mathbb{E}_{\boldsymbol{x} \sim p_{\operatorname{data}(\boldsymbol{x})}}\left[(D(x)-1)^{2}\right]+\mathbb{E}_{\tilde{\boldsymbol{x}} \sim p_{(\tilde{\boldsymbol{x}} \mid \boldsymbol{x})}}\left[(D(\tilde{\boldsymbol{x}}))^{2}\right]
$$













