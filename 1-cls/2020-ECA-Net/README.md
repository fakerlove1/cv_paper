# 2020 ECA-Net CVPR

> 论文名称：ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
>
> 论文地址：[https://arxiv.org/abs/1910.03151](https://arxiv.org/abs/1910.03151)
>
> 代码地址：[https://github.com/BangguWu/ECANet](https://github.com/BangguWu/ECANet)
>
> 发表时间：2019年10月
>
> 引用：Wang Q ,  Wu B ,  Zhu P , et al. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks[C]// 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020.
>
> 



## 1. 简介

近年来，通道注意机制在改善深度卷积神经网络（CNN）性能方面显示出巨大的潜力。然而，大多数现有方法致力于开发更复杂的注意模块，以获得更好的性能，这不可避免地增加了模型的复杂性。**为了克服性能和复杂性之间的矛盾**，本文提出了一种高效的通道注意（ECA）模块，该模块只涉及少量参数，同时带来明显的性能增益。**通过剖析SENet中的通道注意模块，我们实证地表明，避免维度缩减对于学习通道注意非常重要，适当的跨通道交互可以在显著降低模型复杂度的同时保持性能。因此，我们提出了一种无降维的局部交叉信道交互策略，该策略可以通过一维卷积有效地实现。**

## 2. 网络

说白了就是一句话，使用一维卷积代替了SENet的全连接层





![image-20220606162526334](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220606162526334.png)

1. **针对SEBlock的步骤(3)，将MLP模块(FC->ReLU>FC->Sigmoid)，转变为一维卷积的形式**，有效减少了参数计算量（我们都知道在CNN网络中，往往连接层是参数量巨大的，因此将全连接层改为一维卷积的形式）
2. **一维卷积自带的功效就是非全连接，每一次卷积过程只和部分通道的作用**，即实现了适当的跨通道交互而不是像全连接层一样全通道交互。



## 3. 代码

**给定通过平均池化(average pooling)获得的聚合特征$[C,1,1]$**,**ECA模块通过执行卷积核大小为k的一维卷积来生成通道权重，其中k通过通道维度C的映射自适应地确定。**

图中与SEBlock不一样的地方仅在于SEBlock的步骤(3)，用一维卷积替换了全连接层，其中一维卷积核大小通过通道数C自适应确定。

自适应确定卷积核大小公式：$k=|\frac{\log_2 C+b}{\gamma}|_{odd}$

其中k表示卷积核大小，C表示通道数,$||_{odd}$表示k只能取奇数,$\gamma$和$b$在论文中设置为2和1,用于改变通道数C和卷积核大小和之间的比例。

~~~python
import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


if __name__ == "__main__":
    features_maps = torch.randn((8, 54, 32, 32))
    model = ECABlock(54, gamma = 2, b = 1)
    model(features_maps)

~~~





参考链接

> https://blog.csdn.net/weixin_43913124/article/details/123113339