#  2021-MaskFormer  NeurIPS



> 论文题目: Per-Pixel Classification is Not All You Need for Semantic Segmentation
>
> 论文链接:[https://arxiv.org/abs/2107.06278](https://arxiv.org/abs/2107.06278)
>
> 论文代码: [https://github.com/facebookresearch/MaskFormer](https://github.com/facebookresearch/MaskFormer)
>
> 视频讲解: [https://www.bilibili.com/video/BV17f4y1A7XR](https://www.bilibili.com/video/BV17f4y1A7XR)



## 1. 简介

### 1.1 简介

现在的方法通常将**语义分割**制定为**per-pixel classification**任务，而**实例分割**则使用**mask classification**来处理。

本文作者的观点是：**mask classification完全可以通用**，即可以使用**完全相同的模型、损失和训练程序**以统一的方式解决语义和实例级别的分割任务。

据此本文提出了一个简单的mask classification模型——**MaskFormer**，预测一组二进制掩码，每个掩码都与单个全局类标签预测相关联，并且可以将任何现有的per-pixel classification模型无缝转换为mask classification。

MaskFormer 的性能优于SOTA的语义分割模型（ADE20K 上的 55.6 mIoU）和全景分割模型（COCO 上的 52.7 PQ），**特别是类别数量很大时**。







## 2. 网络

### 2.1 总体架构





## 3. 代码







参考资料

> https://blog.csdn.net/bikahuli/article/details/121998593
>
> [深度学习：语义分割：论文阅读（NeurIPS 2021）MaskFormer: per-pixel classification is not all you need_sky_柘的博客-CSDN博客_语义分割最新论文](https://blog.csdn.net/zhe470719/article/details/125067737)