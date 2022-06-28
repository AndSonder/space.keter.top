# 多模态对抗攻击「RGB-D人脸识别」

:::tip
RGB-D 人脸对抗攻击研究计划
:::

## 简介

如今人脸识别识别技术已经广泛地融入到人们的生活当中。深层神经网络是当下人脸识别系统中的核心技术点，但是越来越多的研究表明神经网络容易受到对抗性样本的干扰，这可能会导致基于DNN的人脸识别系统产生错误的识别结果。以手机扫脸解锁举例，攻击者可以使用一张照片制造出对抗性贴片，使得任何人贴上贴片后都可以解锁手机[1]。

​当下人脸识别系统中搭载的人脸识别算法可以分为两类。一类是基于二维图像的人脸识别算法。这一类算法使用FaceNet[2] 等网络将人脸映射到特征向量空间，通过比对特征向量的距离来进行人脸相似度对比。另一类人脸识别技术是采用RGB-D[3]或三维点云的三维人脸识别技术。对于基于二维图像的人脸识别算法，已经有研究成功的攻击了物理识别中的人脸识别算法[1, 4-5]。相较于基于二维图像的算法，基于RGB-D或三维点云的人脸识别算法具有更好的抗干扰性。这是因为基于RGB-D的人脸识别技术可以考虑除RGB模态外的其他特征，从而对利用RGB贴片进行的攻击具有鲁棒性。当下也有研究点扰动、点生成[6]或者删除关键点[7]的方式来攻击基于三维点云的检测器，但目前的方法主要是针对数字世界中的模拟攻击，无法攻击物理世界中的检测器且针对攻击三维人脸识别的工作较少。故RGB-D人脸识别算法的安全性是当下值得研究的工作。

​攻击基于RGB-D的人脸识别主要有两大难点：1、基于RGB-D的人脸识别算法往往融合了RGB和深度两个模态，攻击算法需要同时攻击两个模态并让对抗性样本与目标对象在特征空间上相近  2、在物理世界中攻击基于红外传感的深度信息比较困难。针对第一个难点可以采用模态分离的方法解决，分别攻击RGB模态与深度模态最后将两个攻击手段融合到一个攻击算法中。针对第二个难点，由于深度信息是传感器基于红外光的反射时间获取的，使用能够吸收红外光的材质制造贴片就可以在红外成像中模拟出空洞，即在物理世界中模拟出删除关键区域的效果。总而言之，之后的计划中主要聚焦于攻击RGB-D人脸识别算法，并尝试通过模态分离的方法分别攻击RGB模态和深度模态。通过非3D打印人脸的方式去攻击具有深度模态的人脸识别算法。

## 研究方法

​在如今人脸识别安全越来越受重视的背景下，很多研究聚焦于攻击2D人脸识别。AdvHat[1]通过在额头上贴上对抗性贴片的方式来攻击人脸识别算法。该算法使用梯度下降的方式生成对抗性贴片，且在生成贴片的时候考虑了额头的弧度并通过仿射变换的方式模拟出弧度，这使得AdvHat在物理世界中具有很好的鲁棒性，本计划拟使用AdvHat作为攻击RGB模态的基线方法。对于深度模态的攻击，本计划采取掩盖关键区域的方式，关键区域使用基于区域的启发式差分算法（RHDE）[5] 获取。该算法能够根据评估指标快速获取人脸中的关键区域。在使用RHDE获取关键区域后，在深度图中将关键区域深度置零以此来攻击深度模态。

​本计划拟采用IIIT-D[8]与CurtinFaces[9] 两个公开RGB-D人脸数据集，并使用 [10,11] 中提出的RGB-D人脸识别算法作为被攻击模型进行测试。研究首先验证攻击方法在两个数据集和两个RGB-D人脸识别算法上的有效性。接着验证分别攻击每个模态时的攻击效果并与同时攻击两个模态时的攻击成功率进行对比。在数字模拟阶段结束后将对物理世界攻击算法的可迁移性进行研究，通过使用传感器拍摄佩戴对抗性装饰的人，将拍摄所得的视频分帧后统计物理世界中的攻击成功率。

​本计划拟用模态分离的方法来攻击基于RGB-D的人脸识别算法，具体来说在人脸上贴上一个用于攻击RGB模态的贴片与多个用于攻击深度模态的贴片。由于攻击深度模态的贴片可能有多个，如果攻击RGB模态的贴片位置可以固定且给面部留出更多的空间就可以给深度模态贴片更大的位置选择空间。故本计划选取AdvHat作为攻击RGB模态的算法。但是由于AdvHat的区域较大可能会对深度模态产生一定的影响。针对深度模态的攻击，本计划拟首先使用RHDE获取人脸上的关键区域，接下来将对应深度图区域的深度置零来攻击深度模态。这样做的原因是很难在物理世界中干扰或控制传感器获取的深度信息，目前使用可吸收的红外光的材质制造贴片来干扰物理世界中的红外传感器是较为可行的方法。

## 总结

​人脸识别技术已经渗透到人们生活的方方面面，基于RGB-D的人脸识别技术更是有着广泛的应用。因此该类算法的安全性就十分重要。本计划拟提出了一种攻击RGB-D人脸识别算法的方案。本计划采用模态分离的方法进行攻击，分别考虑针对RGB模态与深度模态的攻击。对于RGB模态使用可打印的曲面AdvHat进行攻击；对于深度模态使用RHDE算法获取关键区域后通过消去目标区域的深度信息进行攻击，并使用可吸收红外光的材质实现物理世界中的攻击。该方法旨在使得基于RGB-D的人脸识别算法产生误识别，并进行类别诱导攻击以及能够通过攻击物理世界中的RGB-D人脸识别器。

## 参考文献

[1] Komkov S , Petiushko A . AdvHat: Real-World Adversarial Attack on ArcFace Face ID System[C] // International Conference on Pattern Recognition. IEEE, 2021.

[2] Schroff F , Kalenichenko D , Philbin J . FaceNet: A Unified Embedding for Face Recognition and Clustering[C]// 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2015.

[3] Jiang L , Zhang J , Deng B . Robust RGB-D Face Recognition Using Attribute-Aware Loss[J]. Institute of Electrical and Electronics Engineers (IEEE), 2020(10).

[4] Guo L , Zhang H . A white-box impersonation attack on the FaceID system in the real world[J]. Journal of Physics Conference Series, 2020, 1651:012037.

[5] Guo Y , Wei X , Wang G , et al. Meaningful Adversarial Stickers for Face Recognition in Physical World[J]. 2021.

[6] C. Xiang, C. R. Qi, and B. Li, “Generating 3d adversarial point clouds,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9136–9144, 2019.

[7] T. Zheng, C. Chen, J. Yuan, B. Li, and K. Ren, “Pointcloud saliency maps,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1598–1606, 2019.

[8] Goswami G , Bharadwaj S , Vatsa M , et al. On RGB-D face recognition using Kinect[C]// IEEE Sixth International Conference on Biometrics: Theory. IEEE, 2014.

[9] Li B , Mian A S , Liu W , et al. Using Kinect for face recognition under varying poses, expressions, illumination and disguise[C]// IEEE Workshop on Applications of Computer Vision. IEEE, 2013.

[10] Chowdhury A , Ghosh S , Singh R , et al. RGB-D face recognition via learning-based reconstruction[C]// IEEE International Conference on Biometrics Theory. IEEE, 2016.

[11] Uppal H , Sepas-Moghaddam A , Greenspan M , et al. Two-Level Attention-based Fusion Learning for RGB-D Face Recognition[C]// International Conference on Pattern Recognition. IEEE, 2021.