# 2022-IDEAL-CVPR

> 论文题目：IDEAL: IMPROVED DENSE LOCAL CONTRASTIVE LEARNING FOR SEMI-SUPERVISED
> MEDICAL IMAGE SEGMENTATION
>
> 中文题目：改进的密集局部对比学习用于半监督医学图像分割
>
> 论文链接：[https://arxiv.org/abs/2210.15075](https://arxiv.org/abs/2210.15075)
>
> 论文代码：
>
> 团队：石溪大学、2贾达浦尔大学、3多伦多大学、4庆北大学
>
> 发表时间

## 1. 简介

### 1.1 摘要

由于标记数据的稀缺，对比自监督学习(SSL)框架最近在一些医学图像分析任务中显示出巨大的潜力。然而，现有的对比机制由于不能挖掘局部特征，对于密集的像素级分割任务是次优的。为此，我们将度量学习的概念扩展到分割任务，使用密集(非)相似学习来预训练深度编码器网络，并使用半监督范式来微调下游任务。具体来说，我们提出了一种简单的卷积投影头来获得密集的像素级特征，并提出了一种新的对比损失来利用这些密集投影，从而改善局部表示。针对下游任务，设计了一种双向一致性正则化机制，包括两流模型训练。通过比较，我们的IDEAL方法在心脏MRI分割上优于SoTA方法。

### 1.2

## 2. 网络

## 3. 代码