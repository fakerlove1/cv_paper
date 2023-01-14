# 2020-CL-MedSeg-NeurIPS

> 论文题目：Contrastive learning of global and local features for medical image segmentation with limited annotations
>
> 中文题目：局部特征与全局特征的对比学习，用于有限标注的医学图像分割
>
> 论文链接：[https://arxiv.org/abs/2006.10511](https://arxiv.org/abs/2006.10511)
>
> 论文代码：[https://github.com/krishnabits001/domain_specific_cl](https://github.com/krishnabits001/domain_specific_cl)
>
> 发表时间：2020年6月
>
> 团队：苏黎世联邦理工学院计算机视觉实验室
>
> 引用：Chaitanya K, Erdil E, Karani N, et al. Contrastive learning of global and local features for medical image segmentation with limited annotations[J]. Advances in Neural Information Processing Systems, 2020, 33: 12546-12558.
>
> 引用数：238（目前时间：2022年12月26号）





## 1. 简介

### 1.1 摘要

监督式深度学习成功的一个关键要求是一个大的标记数据集——这是医学图像分析中难以满足的条件。

自监督学习(SSL)可以通过提供一种策略来使用未标记的数据预训练神经网络，然后对带有有限注释的下游任务进行微调，从而在这方面提供帮助。对比学习(SSL的一种特殊变体)是学习图像级表示的强大技术。

在这项工作中，我们提出了一些策略，通过利用特定领域和特定问题的线索，在具有有限注释的半监督环境中扩展对比学习框架，用于分割体积医学图像。

具体来说，我们提出了`新的对比策略`，利用体积医学图像之间的结构相似性(特定领域线索)和`对比损失的局部版本`，以学习对每像素分割有用的局部区域的独特表示(特定问题线索)。

我们对三个磁共振成像(MRI)数据集进行了广泛的评估。在有限的注释设置下，与其他自我监督和半监督学习技术相比，所提出的方法产生了实质性的改进。当与简单的数据增强技术相结合时，所提出的方法仅使用两个标记MRI体积进行训练，达到基准性能的8%以内，对应于用于训练基准的训练数据的4%(对于ACDC)。







## 2. 网络





## 3. 代码







