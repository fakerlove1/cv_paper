# 【图像分类】2021-MLP-Mixer NIPS

> 论文题目：MLP-Mixer: An all-MLP Architecture for Vision
>
> 论文链接：[https://arxiv.org/abs/2105.01601](https://arxiv.org/abs/2105.01601)
>
> 论文代码：[https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
>
> 论文作者：谷歌大脑的研究员( **原ViT团队** )
>
> 发表时间：2021年5月
>
> 引用：Tolstikhin I O, Houlsby N, Kolesnikov A, et al. Mlp-mixer: An all-mlp architecture for vision[J]. Advances in Neural Information Processing Systems, 2021, 34: 24261-24272.
>
> 引用数：464



## 1. 简介

### 1.1 简介

卷积神经网络（CNN）是计算机视觉的首选模型。 最近，基于注意力的网络（例如ViT）也变得很流行。 在本文中，我们表明，尽管卷积和注意力都足以获得良好的性能，但它们都不是必需的。

文章介绍了MLP-Mixer，这是一种仅基于多层感知机（MLP）的体系结构。 **MLP-Mixer仅仅依赖于在空域或者特征通道上重复实施的多层感知器；Mixer仅依赖于基础矩阵乘操作、数据排布变换(比如reshape、transposition)以及非线性层** 。

> 众所周知，CV领域主流架构的演变过程是 MLP->CNN->Transformer 。
>
> 难道现在要变成 MLP->CNN->Transformer->MLP ?
>
> 都说时尚是个圈，没想到你学术圈真的有一天也变成了学术“圈”。



### 1.2 摘要

本文是谷歌大脑的研究员( **原ViT团队** )在网络架构设计方面挖的新坑：MLP-Mixer。

 **无需卷积、注意力机制，MLP-Mixer仅需MLP即可达到与CNN、Transformer相媲美的性能** 。

比如，在JFT-300M数据集预训练+ImageNet微调后，所提Mixer-H/14取得87.94%的top1精度。

尽管所提方法性能并未达到最优，但本文的目的并不在于达成SOTA结果，而在于表明： **简简单单的MLP模型即可取得与当前最佳CNN、注意力模型相当的性能**



### 1.3 好处

为什么要用全连接层有什么好处呢？它的**归纳偏置（inductive bias）更低**。归纳偏置可以看作学习算法自身在一个庞大的假设空间中对假设进行选择的启发式或者“价值观”。下图展示了机器学习中常用的归纳偏置。

CNN 的归纳偏置在于卷积操作，只有感受野内部有相互作用，即图像的局部性特征。时序网络 RNN 的归纳偏置在于时间维度上的连续性和局部性。事实上，ViT 已经开始延续了一直以来想要在神经网络中移除手工视觉特征和归纳偏置的趋势，让模型只依赖于原始数据进行学习。

MLP 则更进了一步。原文提到说：`One could speculate and explain it again with the difference in inductive biases: self-attention layers in ViT lead to certain properties of the learned functions that are less compatible with the true underlying distribution than those discovered with Mixer architecture`。

![请添加图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/bc33530cd0df476196f828a203846270.png)



## 2. 网络

### 2.1 整体架构

**主要组成部分**

> MLP-Mixer主要包括三部分：Per-patch Fully-connected、Mixer Layer、分类器。
>
> 其中分类器部分采用传统的全局平均池化（GAP）+全连接层（FC）+Softmax的方式构成，故不进行更多介绍，

**主要流程**

* 先将输入图片拆分成patches，然后通过Per-patch Fully-connected将每个patch转换成feature embedding，
* 然后送入N个Mixer Layer，
* 最后通过Fully-connected进行分类。

![image-20220726112218608](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220726112218608.png)



### 2.2 Per-patch Fully-Connected



假设我们有输入图像$224 \times 224 \times 3$，首先我们切 patch，例如长宽都取 32，则我们可以切成$7 \times 7=49$ 个 patch，每个 patch 是$32 \times 32 \times 3$。我们将每个 patch 展平就能成为 49 个 3072 维的向量。通过一个全连接层（Per-patch Fully-connected）进行降维，例如 512 维，就得到了 49 个 token，每个 token 的维度为 512。然后将他们馈入 Mixer Layer。

![image-20220726164208649](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220726164208649.png)

### 2.3 MixerLayer

Mixer采用了两种类型的MLP层(注:这两种类型的层交替执行以促进两个维度间的信息交互)：

- **channel-mixing** MLP：不同通道之间进行交流，每个token独立处理，即采用每一行作为输入；
- **token-mixing** MLP：允许不同空间位置(tokens)进行交流，每个通道图例处理，即采用每一列作为输入。

这两种类型的layer是交替堆叠的，方便支持两个输入维度的交流。每个MLP由两层fully-connected和一个GELU构成。

> mlp-mixer在极端情况下，本文所提架构可视作一种特殊CNN，
>
> 它采用卷积进行 **channel mixing** ，
>
> 全感受野、参数共享的的单通道深度卷积进行 **token mixing** 。



![image-20220726163838899](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220726163838899.png)



## 3. 代码

![image](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f70726f746f6e782d636c6f75642d73746f726167652f43617074757265332e504e47.png)

代码非常的简单。

~~~python
import collections.abc

import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple([x] * 2)


class FeedForward(nn.Module):
    """
    mlp结构
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        """

        Args:
            dim: 输入维度
            num_patch: patch的数目
            token_dim:
            channel_dim:
            dropout: dropout的比例
        """
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):

    def __init__(self,
                 in_channels=3,
                 dim=512,
                 patch_size=16,
                 image_size=224,
                 depth=8,
                 mlp_ratio=(0.5, 4.0),
                 num_classes=1000, ):
        """

        Args:
            in_channels: 输入通道数
            dim: 分类头 中间通道数
            patch_size: patch大小
            image_size: 图片大小
            depth: block重复次数
            mlp_ratio: mix_block 中间维度 缩放数，用来计算token_dim, channel_dim
            num_classes: 分类数
        """
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # patch 的数目
        self.num_patch = (image_size // patch_size) ** 2

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        # 可以换成 linear projection
        # self.to_patch_embedding=nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
        #                                                 p1=patch_size, p2=patch_size),
        #               nn.Linear((patch_size ** 2) * in_channels, dim), )

        # 计算token_dim, channel_dim
        token_dim, channel_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x=[1,3,224,224]
        x = self.to_patch_embedding(x)  # [1,3,224,224] ->[1, 196, 512]
        # 3*224*224 使用patch_size 为 16 的话 。意思为下采样16倍 。变成 [1,dim,224/patch_size,224/patch_size]
        # [1,3,224,224] -> [1,512,14,14]
        # [1,512,14,14] -> [1, 14*14, 512]=[1, 196, 512]

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)  # [1, 196, 512]-> [1, 196, 512]
        x = x.mean(dim=1)  # [1, 196, 512]->[1, 512]

        return self.mlp_head(x)


def mixer_s32(**kwargs):
    """ Mixer-S/32
    """
    model = MLPMixer(patch_size=32,
                     depth=8,
                     dim=512,
                     **kwargs)
    return model


def mixer_s16(**kwargs):
    """ Mixer-S/16
    """
    model = MLPMixer(patch_size=16,
                     depth=8,
                     dim=512,
                     **kwargs)
    return model


def mixer_s4(**kwargs):
    """ Mixer-S/4
    """
    model = MLPMixer(patch_size=4,
                     depth=8,
                     dim=512,
                     **kwargs)
    return model


def mixer_b32(**kwargs):
    """ Mixer-B/32
    """
    model = MLPMixer(patch_size=32,
                     depth=12,
                     dim=768,
                     **kwargs)
    return model


def mixer_b16(pretrained=False, **kwargs):
    """ Mixer-B/16
    """
    model = MLPMixer(patch_size=16,
                     depth=12,
                     dim=768,
                     **kwargs)
    return model


def mixer_l32(**kwargs):
    """ Mixer-L/32
    """
    model = MLPMixer(patch_size=32,
                     depth=24,
                     dim=1024,
                     **kwargs)
    return model


def mixer_l16(**kwargs):
    """ Mixer-L/16 224x224
    """
    model = MLPMixer(patch_size=16,
                     depth=24,
                     dim=1024,
                     **kwargs)
    return model


if __name__ == "__main__":
    model = mixer_s16(num_classes=1000)

    #  计算参数量
    from thop import profile
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))

~~~



参考资料

> [(1条消息) 深度学习之图像分类（二十一）-- MLP-Mixer网络详解_木卯_THU的博客-CSDN博客_mlp 图像处理](https://blog.csdn.net/baidu_36913330/article/details/120526870)
>
> https://blog.csdn.net/baidu_36913330/article/details/120526870