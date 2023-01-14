# 2022-U2PL-CVPR

> 论文题目：Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels
>
> 论文链接：[https://arxiv.org/abs/2203.03884](https://arxiv.org/abs/2203.03884)
>
> 论文代码：[https://haochen-wang409.github.io/U2PL/](https://haochen-wang409.github.io/U2PL/)
>
> 团队：上海交通大学&香港中文大学&商汤科技
>
> 发表时间：2022年3月
>
> 引用：Wang Y, Wang H, Shen Y, et al. Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 4248-4257.
>
> 引用数：31



## 1. 简介

### 1.1 摘要

半监督语义分割的关键是为未标记图像的像素分配足够的伪标签。一种常见的做法是选择高置信度的预测作为伪真值，但这会导致一个问题，即大多数像素可能由于不可靠而未被使用。我们认为，每个像素对模型训练都很重要，即使它的预测是模糊的。直觉上，不可靠的预测可能会在顶级类别（即概率最高的类别）中混淆，但是，它应该对不属于其余类别的像素有信心。因此，对于那些最不可能的类别，这样一个像素可以令人信服地视为负样本。基于这一认识，我们开发了一个有效的管道，以充分利用未标记的数据。具体地说，我们通过预测熵来分离可靠和不可靠像素，将每个不可靠像素推送到由负样本组成的类别队列中，并设法用所有候选像素来训练模型。考虑到训练进化，预测变得越来越准确，我们自适应地调整可靠-不可靠划分的阈值。在各种基准和训练环境下的实验结果表明，我们的方法优于最先进的替代方法。



### 1.2 解决的问题



## 2. 网络





## 3. 代码





Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels





参考资料

[使用不可靠伪标签的半监督语义分割 - 简书 (jianshu.com)](https://www.jianshu.com/p/af3657a14c36)

[U2PL: 使用不可靠伪标签的半监督语义分割 (CVPR'22) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/474771549)