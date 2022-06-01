# Auto-attack论文阅读笔记

## 摘要

> The field of defense strategies against adversarial attacks has significantly grown over the last years, but progress is hampered as the evaluation of adversarial defenses is often insufficient and thus gives a wrong impression of robustness. Many promising defenses could be broken later on, making it difficult to identify the state-of-the-art. Frequent pitfalls in the evaluation are improper tuning of hyperparameters of the attacks, gradient obfuscation or masking. In this paper we first propose two extensions of the PGD-attack overcoming failures due to suboptimal step size and problems of the objective function. We then combine our novel attacks with two complementary existing ones to form a parameterfree, computationally affordable and user-independent ensemble of attacks to test adversarial robustness. We apply our ensemble to over 50 models from papers published at recent top machine learning and computer vision venues. In all except one of the cases we achieve lower robust test accuracy than reported in these papers, often by more than 10%, identifying several broken defenses.

`提出当下的问题` 在过去几年里，针对对抗性攻击的防御策略领域有了显著增长，但由于对对抗性防御的评估往往不够充分，给人们留下了现在的防御策略很强大的错觉。很多“强大的”防御策略都在之后被攻破了，这使得很难判定哪种防御算法才是最好的。评估的方法中常见的不当是攻击的超参数调整不当，梯度混淆或者梯度掩码。

`提出文本的贡献` 在这篇工作中，我们第一次提出了对于PGD-attack的两个插件来克服由于步长不合适和目标函数导致的攻击效果不好的问题。然后我们结合我们的新型攻击与两个互补的现有攻击，形成一个无需参数，计算可承受和用户独立的攻击集合，以测试对抗性的鲁棒性。我们将我们的集成应用于超过50个来自顶级机器学习和计算机视觉领域发表论文的模型。几乎所有的模型（有一个除外）我们都获得了比这些文章中给出的鲁棒性测试更低的的准确度，通常都低了10%。






