---
title: >-
  「论文阅读」On the Effectiveness of Interval Bound Propagation for Training
  Verifiably Robust Models
date: 2022-01-21 21:39:38
tags: [论文阅读,模型鲁棒性]
ketex: true
---

> `Absract`
> `研究领域概况`: 最近的工作说明了，训练一个可以抵抗范数有界的对抗性扰动的神经网络是可能的。大多数的方法都是在所有的对抗性扰动可能性下最小化最坏(worst-case)损失的上界(upper bound)。
> `当前研究的问题:` 当前的技术大多都使用了难以优化的方法，以至于难以应用了大规模的网络上（因为计算成本和复杂度都会大幅度上升）
> `简要说明论文的方法与结果：` 通过一个全面的分析，我们展示了一个简单的边界技术IBP(interval bound propagation)能够被用来训练大型可证明的具有鲁棒性的神经网络，以达到目前最好的认证准确率。
> `再具体说一下优化点：` 然而对于一般的网络，IBP计算的上界可能是非常脆弱的，但是我们证明了适当的损失函数和超参数可以让IBP边界变的紧缩。
> `实验结果：` 这使得学习算法快速且稳定，效果优于以往的方法且实现了在MNIT，CIFAR-10和SVHN上的最好的实验结果。
> `补充说明：` 这同样让我们可以在缩小版的IMAGENET上训练一个大型的模型以验证vacuous bounds.

# 以前工作和问题
很多以前的防御工作[4,8-10]都是针对与某个特定的防御算法且很容易被强大的对抗性网络攻破。
其中一种防御方式就是将对抗性样本加入到训练集当中进行训练，但是这样的防御方式不能抵御使用新的方式生成的对抗性样本。下图说明了为什么PGD (梯度下降攻击法)并不总是找到最脆弱的点进行攻击。
![](https://gitee.com/coronapolvo/images/raw/master/20220121223633.png)
对于上图，给定一个看似健壮的神经网络，最坏情况下扰动的大小为$\epsilon=0.1$ . 使用200轮PGD迭代生成的对抗性图片就可以使得神经网络输出错误的分类结果。然后通过暴力穷举（MIO）的方式生成的对抗性样本应该是2；

上述问题促使了对于形式验证（formal verification）：      保证神经网络与网络的所有可能输入的规格一致。相关的工作已经取得了一定的进展。使用 Satisfiability Modulo Theory(SMT) [14-16]和Mixed-Integer Programming (MIP) [13,17,18]      以解决验证问题的凸松弛为基础的不完备方法[19-26]. 对于提供了准确的鲁棒性边界的完备算法，它们往往是`复杂的且难以扩展到其他神经网络`当中。

# 核心方法
本文研究基于区间算术[14,15,28]的区间传播 ( interval bound propagation【IBP】)算法，这是一种训练可验证鲁棒分类器的不完全方法。
 IBP allows defining a loss to minimize an upper bound on the maximum difference between any pair of logits when the input can be perturbed within an $\ell_{\infty}$ norm-bounded ball.
##  Neural network
本文着重于`前馈神经网络`训练的分类任务。网络的输入为$x_0$, 网络的输出为一个预测向量(logits), 向量中包含了$x_0$属于每个类别的置信度。
为了更加清晰的表述，假设神经网络的第K层是由一个转换$h_k$实现的。
![](https://gitee.com/coronapolvo/images/raw/master/20220122220536.png)
##  Verification problem



# 实验结果




# References
[1] Ian Goodfellow, YoshuaBengio, and Aaron Courville, Deep Learning. MIT Press, 2016. 1
[2] NicholasCarliniandDavidWagner, “Adversarialexamples are not easily detected: Bypassing ten detection methods,” in Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security. ACM, 2017, pp. 3–14. 1
[3] ——,“Towardsevaluatingtherobustnessofneuralnetworks,” in 2017 IEEE Symposium on Security and Privacy. IEEE, 2017, pp. 39–57. 1, 2
[4] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy, “Explaining and harnessing adversarial examples,” arXiv preprint arXiv:1412.6572, 2014. 1
[5] Alexey Kurakin, Ian Goodfellow, and Samy Bengio, “Adversarial examples in the physical world,” arXiv preprint
arXiv:1607.02533, 2016.
[6] ChristianSzegedy, Wojciech Zaremba,IlyaSutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus, “Intriguing properties of neural networks,” arXiv preprint arXiv:1312.6199, 2013. 1
[7] Anish Athalye, Logan Engstrom, Andrew Ilyas, and KevinKwok, “Synthesizing robust adversarial examples,” in Interna- tional Conference on Machine Learning, 2018, pp. 284–293. 1
[8] Nicolas InternationalPaper not, Patrick Drew McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami, “Distillation as a defense to adversarial perturbations against deep neural networks,” in 2016 IEEE Symposium on Security and Privacy, SP 2016. Institute of Electrical and Electronics Engineers Inc., 2016, pp. 582–597. 1
[9] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu, “Towards deep learn- ing models resistant to adversarial attacks,” in International Conference on Learning Representations, 2018. 1, 2, 5, 8, 11
[10] HariniKannan, and Ian Goodfellow AlexeyKurakin,andIanGoodfellow,“Adversarial logit pairing,” arXiv preprint arXiv:1803.06373, 2018. 1
[11] JonathanUesato, BrendanODonoghue,PushmeetKohli,and Aaron Oord, “Adversarial risk and the dangers of evaluat- ing against weak attacks,” in International Conference on Machine Learning, 2018, pp. 5032–5041. 1
[12] AnishAthalye,NicholasCarlini,andDavidWagner,“Obfus- cated gradients give a false sense of security: Circumventing defenses to adversarial examples,” in International Confer- ence on Machine Learning, 2018, pp. 274–283. 1
[13] VincentTjeng,KaiY.Xiao,andRussTedrake,“Evaluating robustness of neural networks with mixed integer program- ming,” in International Conference on Learning Representa- tions, 2019. 1, 2, 5
[14] Guy Katz, Clark Barrett, David L Dill, Kyle Julian, and Mykel J Kochenderfer, “Reluplex: An efficient smt solver for verifying deep neural networks,” in International Conference on Computer Aided Verification. Springer, 2017, pp. 97–117. 2
[15] Ruediger Ehlers, “Formal verification of piece-wise lin- ear feed-forward neural networks,” in International Sympo- sium on Automated Technology for Verification and Analysis.
[16] Nicholas Carlini, Guy Katz, Clark Barrett, and David L Dill, “Ground-truth adversarial examples,” arXiv preprint arXiv:1709.10207, 2017. 2
[17] Rudy Bunel, Ilker Turkaslan, Philip HS Torr, Pushmeet Kohli, and M Pawan Kumar, “Piecewise linear neural net- work verification: a comparative study,” arXiv preprint arXiv:1711.00455, 2017. 2
[18] Chih-Hong Cheng, Georg Nu ̈hrenberg, and Harald Ruess, “Maximum resilience of artificial neural networks,” in Interna- tional Symposium on Automated Technology for Verification and Analysis. Springer, 2017, pp. 251–268. 2
[19] Tsui-WeiWeng,HuanZhang,HonggeChen,ZhaoSong,Cho- Jui Hsieh, Luca Daniel, Duane Boning, and Inderjit Dhillon, “Towards fast computation of certified robustness for relu networks,” in International Conference on Machine Learning, 2018, pp. 5273–5282. 2
[20] Matthew Mirman, Timon Gehr, and Martin Vechev, “Dif- ferentiable abstract interpretation for provably robust neural networks,” in Proceedings of the 35th International Confer- ence on Machine Learning, vol. 80, 2018, pp. 3578–3586. 2, 3, 8, 13, 14
[21] TimonGehr,MatthewMirman,DanaDrachsler-Cohen,Petar Tsankov, Swarat Chaudhuri, and Martin Vechev, “Ai 2: Safety and robustness certification of neural networks with abstract interpretation,” in IEEE Symposium on Security and Privacy, 2018.
[22] Krishnamurthy Dvijotham, Robert Stanforth, Sven Gowal, Timothy A Mann, and Pushmeet Kohli, “A dual approach to scalable verification of deep networks.” in UAI, 2018, pp. 550–559. 3
[23] Krishnamurthy Dvijotham, Sven Gowal, Robert Stanforth, Relja Arandjelovic, Brendan O’Donoghue, Jonathan Uesato, and Pushmeet Kohli, “Training verified learners with learned verifiers,” arXiv preprint arXiv:1805.10265, 2018. 2, 3, 5, 7, 8, 13
[24] Eric Wong and Zico Kolter, “Provable defenses against ad- versarial examples via the convex outer adversarial polytope,” in International Conference on Machine Learning, 2018, pp. 5283–5292. 2, 5, 7, 8
[25] EricWong,FrankSchmidt,JanHendrikMetzen,andJZico Kolter, “Scaling provable adversarial defenses,” in Advances in Neural Information Processing Systems, 2018, pp. 8400– 8409. 2, 5, 6, 8, 11, 12, 14
[26] Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, and Suman Jana, “Formal security analysis of neural net- works using symbolic intervals,” in 27th {USENIX} Security Symposium ({USENIX} Security 18), 2018, pp. 1599–1614. 2
[27] AditiRaghunathan,JacobSteinhardt,andPercyLiang,“Cer- tified defenses against adversarial examples,” in International Conference on Learning Representations, 2018. 2, 5
[28] TeruoSunaga,“Theoryofintervalalgebraanditsapplication to numerical analysis,” RAAG memoirs, vol. 2, no. 29-46, p. 209, 1958. 2
[29] KaiY.Xiao,VincentTjeng,NurMuhammad(Mahi)Shafiul- lah, and Aleksander Madry, “Training for faster adversarial robustness verification via inducing reLU stability,” in International Conference on Learning Representations, 2019. 2,7, 8
[30] EstebanReal,AlokAggarwal,YanpingHuang,andQuocVLe, “Regularized evolution for image classifier architecture search,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, 2019, pp. 4780–4789. 7
[31] Diederik P Kingma and Jimmy Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014. 11
[32] Mart ́ın Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard et al., “Tensorflow: a system for large-scale machine learning.” in OSDI, vol. 16, 2016, pp. 265–283. 11
[33] ShiqiWang,YizhengChen,AhmedAbdou,andSumanJana, “Mixtrain: Scalable training of formally robust neural net- works,” arXiv preprint arXiv:1811.02625, 2018. 14