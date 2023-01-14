# 2022-VAN 

> 论文题目：Visual Attention Network
>
> 论文地址:[https://arxiv.org/abs/2202.09741](https://arxiv.org/abs/2202.09741)
>
> 论文代码：





## 1. 简介



### 1.1 简介

**2.1 CNN**
学习特征表示（feature representation）很重要， CNN因为使用了局部上下文信息和平移不变性，极大地提高了神经网络的效率。在加深网络的同时，网络也在追求更加轻量化。 本文的工作与MobileNet有些相似，把一个标准的卷积分解为了两个部分：一个depthwise conv，一个pointwise conv。本文把一个卷积分解成了三个部分：depthwise conv， depthwise and dilated conv 和pointwise conv。我们的工作将更适合高效地分解大核的卷积操作。我们还引入了一个注意力机制来获得自适应的特性。

**2.2 视觉注意力方法**
注意力机制使得很多视觉任务有了性能提升。 视觉的注意力可以被分为四个类别： 通道注意力、空间注意力、时间注意力和分支注意力。每种注意力机制都有不同的效果。
Self-attention 是一个特别的注意力，可以捕捉到长程的依赖性和适应性，在视觉任务中越来越重要。但是，self-attention有三个缺点

1. **它把图像变成了1D的序列进行处理，忽略了2D的结构信息。**
2. **对于高分辨率图片来说，二次计算复杂度太高。**
3. **它只实现了空间适应性却忽略了通道适应性。**

对于视觉任务来说，不同的通道经常表示不同的物体，通道适应性在视觉任务中也是很重要的。为了解决这些问题，我们提出了一个新的视觉注意力机制：LKA。 它包含了self-attention的适应性和长程依赖，而且它还吸收了卷积操作中利用局部上下文信息的优点。

**2.3 视觉MLP**
在CNN出现之前，MLP曾是非常知名的方法。但是由于高昂的计算需求和低下的效率，MLP的能力被限制了很长一段时间。 最近的一些研究成功地把标准的MLP分解为了spatial MLP和channel MLP，显著降低了计算复杂度和参数量，释放了MLP的性能。 与我们最相近的MLP是gMLP, 它分解了标准的MLP并且引入了注意力机制。 但是gMLP有两个缺点：

1. **gMLP对输入尺寸很敏感，只能处理固定尺寸的图像。**
2. **gMLP只考虑了全局信息而忽略了局部的信息。**

本文的方法可以充分利用它的有点并且避免它的缺点。



## 2. 网络





## 3. 代码
