# 【半监督医学图像分割 MICCAI】2022-SS-Net 

> 论文题目：Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation
>
> 中文题目：半监督医学图像分割中的平滑性和类分离方法研究
>
> 论文链接：[https://arxiv.org/abs/2203.01324v3](https://arxiv.org/abs/2203.01324v3)
>
> 论文代码：[https://github.com/ycwu1997/SS-Net](https://github.com/ycwu1997/SS-Net)
>
> 发表时间：2022年3月
>
> 论文团队：莫那什大学&南洋理工大学
>
> 引用：Wu Y, Wu Z, Wu Q, et al. Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation[J]. arXiv preprint arXiv:2203.01324, 2022.
>
> 引用数：3(截止时间2023年1月12号)



## 1. 简介





## 2. 网络





## 3. 代码









[ycwu1997/SS-Net: Official Code for our MICCAI 2022 paper "Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation" (github.com)](https://github.com/ycwu1997/ss-net)





半监督分割在医学成像中仍然具有挑战性，因为带注释的医学数据量通常很少，并且在粘合边缘或低对比度区域附近有许多模糊像素。为了解决这些问题，我们主张首先约束具有和不具有强扰动的像素的一致性以应用足够的平滑度约束，并进一步鼓励类级分离以利用低熵正则化进行模型训练。特别是，在本文中，我们通过同时探索像素级平滑度和类间分离，提出了用于[半监督医学图像](https://www.zhihu.com/search?q=半监督医学图像&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2778083734})分割任务的 SS-Net。像素级平滑度迫使模型在对抗性扰动下生成不变的结果。同时，类间分离鼓励各个类特征接近其对应的高质量原型，以使每个类分布紧凑并分离不同的类。我们针对公共 LA 和 ACDC 数据集上的五种最新方法评估了我们的 SS-Net。在两个半监督设置下的大量实验结果证明了我们提出的 SS-Net 模型的优越性，在两个数据集上实现了新的最先进 (SOTA) 性能。



作者：AIandR艾尔
链接：https://www.zhihu.com/question/269075961/answer/2778083734
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。