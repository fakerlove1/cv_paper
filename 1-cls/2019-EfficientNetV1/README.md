# 【图像分类】2019-EfficientNetV1  ICML

> 原论文名称：`EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks`
>
> 论文下载地址：[https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
>
> 原论文提供代码：[https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
>
> 发表时间：2019年5月
>
> 引用：Tan M, Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.
>
> 引用数：7764



## 1. 简介

### 1.1 简介

本文是发表于ICML 2019的一篇论文，在各个参数量级上精度都超过了之前的SOTA。

在论文中提到，本文提出的EfficientNet-B7在Imagenet top-1上达到了当年最高准确率84.3%，与之前准确率最高的GPipe相比，参数数量（Params）仅为其1/8.4，推理速度提升了6.1倍（看上去又快又轻量，但个人实际使用起来发现很吃显存）。下图是EfficientNet与其他网络的对比（注意，参数数量少并不意味推理速度就快）。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-2d861b2fee02c0b3214679de1d68c9ce_720w.jpg)

### 1.2 思考

在之前的一些手工设计网络中(AlexNet，VGG，ResNet等等)经常有人问，为什么输入图像分辨率要`固定为224`，为什么卷积的个数要设置为这个值，为什么网络的深度设为这么深？这些问题你要问设计作者的话，估计回复就四个字——工程经验。

而这篇论文主要是用NAS（Neural Architecture Search）技术来搜索网络的图像分辨率`r`,网络的深度`depth`​以及​`channel`的宽度​`width`​三个参数的合理化配置。

在之前的一些论文中，基本都是通过改变上述3个参数中的一个来提升网络的性能，而这篇论文就是同时来探索这三个参数的影响。

在之前的一些论文中

* 有的会通过增加网络的width即增加卷积核的个数（增加特征矩阵的channels）来提升网络的性能如图(b)所示，
* 有的会通过增加网络的深度即使用更多的层结构来提升网络的性能如图(c)所示，
* 有的会通过增加输入网络的分辨率来提升网络的性能如图(d)所示。
* 而在本篇论文中会同时增加网络的width、网络的深度以及输入网络的分辨率来提升网络的性能如图(e)所示：



![modelscaling](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210306162502756.png)



* 根据以往的经验，**增加网络的深度depth能够得到更加丰富**、`复杂的特征`并且能够很好的应用到其它任务中。但网络的深度过深会面临梯度消失，训练困难的问题。
  The intuition is that deeper ConvNet can capture richer and more complex features, and generalize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing gradient problem
* **增加网络的width**能够获得`更高细粒度的特征`并且也`更容易训练`，但对于width很大而深度较浅的网络往往很难学习到更深层次的特征。
  wider networks tend to be able to capture more fine-grained features and are easier to train. However, extremely wide but shallow networks tend to have difficulties in capturing higher level features.
* **增加输入网络的图像分辨率**能够潜在得获得`更高细粒度的特征模板`，但对于非常高的输入分辨率，准确率的增益也会减小。并且`大分辨率图像会增加计算量`。
  With higher resolution input images, ConvNets can potentially capture more fine-grained patterns. but the accuracy gain diminishes for very high resolutions.



接下来看一下，分别改模型的 宽度，深度，分辨率，模型上的提升。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-24e7100ef25517b7699d972b23a14438_720w.jpg)

从上图我们可以看出，`黄蓝绿三条曲线分别对应以上三点，在达到80%左右就趋近饱和了`。

`红色的线是同时增加网络的深度、宽度和分辨率`。在达到80%的时候并未饱和而且继续增长。说明了同时改进网络的宽度、深度和分辨率，会得到更好的结果。在FLOPS(横轴)相同的情况下，同时改进三者，得到的效果会更好。

作者通过实验，发现同时增加网络的深度和图像分辨率最好，如下图红色曲线。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-f05419f96ee829c644c0bbef33389753_720w.jpg)

## 2. 网络结构

我们已经读到了论文的想法？？

那么问题来了，应该如何同时增加网络的深度、宽度和分辨率呢？我们来具体看一下EfficientNet的网络结构：

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-290388de2189a7342d8b460b2a537b5f_720w.jpg)

### 2.1 MBConv

MBConv的块结构如下，MBConv1指的是下面n=1，MBConv6指的是n=6。



![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-4ac0ff80997277dc43645a58fa829f78_720w.jpg)



### 2.2 SE模块

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-9155df77520e9cf7a85c838c686edcce_720w.jpg)





### 2.3 总体架构

论文给出了每个架构的代码

| Model          | input_size | width_coefficient | depth_coefficient | drop_connect_rate | dropout_rate |
| -------------- | ---------- | ----------------- | ----------------- | ----------------- | ------------ |
| EfficientNetB0 | 224x224    | 1.0               | 1.0               | 0.2               | 0.2          |
| EfficientNetB1 | 240x240    | 1.0               | 1.1               | 0.2               | 0.2          |
| EfficientNetB2 | 260x260    | 1.1               | 1.2               | 0.2               | 0.3          |
| EfficientNetB3 | 300x300    | 1.2               | 1.4               | 0.2               | 0.3          |
| EfficientNetB4 | 380x380    | 1.4               | 1.8               | 0.2               | 0.4          |
| EfficientNetB5 | 456x456    | 1.6               | 2.2               | 0.2               | 0.4          |
| EfficientNetB6 | 528x528    | 1.8               | 2.6               | 0.2               | 0.5          |
| EfficientNetB7 | 600x600    | 2.0               | 3.1               | 0.2               | 0.5          |
|                |            |                   |                   |                   |              |

- `input_size`代表训练网络时输入网络的图像大小
- `width_coefficient`代表`channel`维度上的倍率因子，比如在 EfficientNetB0中`Stage1`的`3x3`卷积层所使用的卷积核个数是32，那么在B6中就是$32\times 1.8=57.6$接着取整到离它最近的8的整数倍即56，其它`Stage`同理。
- `depth_coefficient`代表`depth`维度上的倍率因子（仅针对`Stage2`到`Stage8`），比如在EfficientNetB0中`Stage7`的$L_i=4$,那么在B6中就是$4\times 2.6 =10.4$接着向上取整即11.
- `drop_connect_rate`是在`MBConv`结构中dropout层使用的`drop_rate`，在官方keras模块的实现中`MBConv`结构的`drop_rate`是从0递增到`drop_connect_rate`的。还需要注意的是，这里的Dropout层是`Stochastic Depth`，即会随机丢掉整个block的主分支（只剩捷径分支，相当于直接跳过了这个block）也可以理解为减少了网络的深度。具体可参考`Deep Networks with Stochastic Depth`这篇文章。
- `dropout_rate`是最后一个全连接层前的`dropout`层（在`stage9`的Pooling与FC之间）的`dropout_rate`。



最后给出原论文中关于EfficientNet与当时主流网络的性能参数对比



![EfficientNetvsothers](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210306162051639.png)





## 3. 代码

### 3.1 _make_divisible

~~~python
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

~~~

>作用
>
>使ch 等于离divisor最近的整数倍，图中 即被8整除的最近的数。
>原因：对硬件更加友好。

我们做一个测试

~~~python
if __name__ == '__main__':
   print( _make_divisible(57.6)) # 答案是56
~~~





### 3.2 drop_path

DropPath/drop_path 是一种`正则化手段`，其效果是将深度学习模型中的多分支结构随机”`删除`“，python中实现如下所示：

~~~python
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    DropPath/drop_path 是一种正则化手段，其效果是将深度学习模型中的多分支结构随机”删除“
    Drop paths (随机深度) per sample (当应用于主路径的残差块时).
    "随机深度的深度网络", https://arxiv.org/pdf/1603.09382.pdf
    这个函数取自rwightman人的代码。如果有需要，请看下面的代码
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

~~~



我们做一个测试

~~~python
if __name__ == '__main__':
    drop = DropPath(0.1)
    input = torch.randn(size=(1, 3, 214, 214))
    out = drop(input)
    print(input.shape)
    print(out)
~~~

可以看到结果

~~~python
torch.Size([1, 3, 214, 214])
tensor([[[[-0.4348,  0.7865, -0.0499,  ...,  0.0834,  1.3408, -1.1010],
          [-0.7139,  0.9813, -0.1933,  ..., -0.6845,  0.5976,  0.5007],
~~~

还是正常的，接下来，让我们把随机丢率增加到0.9

~~~python
if __name__ == '__main__':
    drop = DropPath(0.9)
    input = torch.randn(size=(1, 3, 214, 214))
    out = drop(input)
    print(input.shape)
    print(out)
~~~

结果,可以看到，大部分数据，都变成了0.`丢失了`

~~~python
torch.Size([1, 3, 214, 214])
tensor([[[[0., 0., 0.,  ..., -0., -0., -0.],
          [-0., -0., 0.,  ..., 0., -0., 0.],
          [0., -0., 0.,  ..., -0., 0., 0.],
~~~





### 3.3 SE注意力模块

~~~python
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        # 用卷积层代替了全连接层
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


~~~





### 3.4 MBConv模块

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-48cd6e673098c2b7aa838f1d705ca291_720w.jpg)



总体代码如下

~~~python
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from functools import partial
import copy

import math


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    使ch 等于离divisor最近的整数倍，图中 即被8整除的最近的数。
    能够被8 整除的 最接近 的数
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# if __name__ == '__main__':
#    print( _make_divisible(57.6)) # 答案是56
#
#

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    DropPath/drop_path 是一种正则化手段，其效果是将深度学习模型中的多分支结构随机”删除“
    Drop paths (随机深度) per sample (当应用于主路径的残差块时).
    "随机深度的深度网络", https://arxiv.org/pdf/1603.09382.pdf
    这个函数取自rwightman人的代码。如果有需要，请看下面的代码
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# if __name__ == '__main__':
#     drop = DropPath(0.9)
#     input = torch.randn(size=(1, 3, 214, 214))
#     out = drop(input)
#     print(input.shape)
#     print(out)


class SqueezeExcitation(nn.Module):
    """
    SE注意力模块
    """

    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class ConvBNActivation(nn.Sequential):
    """
    返回卷积操作的一套流程
    卷积
    归一化
    激活函数
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: nn.Module = None,
                 activation_layer: nn.Module = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=(kernel_size, kernel_size),
                                                         stride=(stride, stride),
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())


class InvertedResidualConfig:
    """
    这里是转换配置的，根据base的模型，乘以倍率。
    """

    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,  # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,  # 1 or 2
                 use_se: bool,  # True
                 drop_rate: float,
                 index: str,  # 1a, 2a, 2b, ...
                 width_coefficient: float):
        """

        Args:
            kernel: 卷积核大小
            input_c: 输入通道数
            out_c: 输出通道数
            expanded_ratio: 扩张的被绿
            stride: 步长
            use_se: 是否使用SE注意力模块
            drop_rate:
            index: 该模块重复的次数
            width_coefficient:
        """

        # 乘完倍率的输入通道数= 输入通道数* 宽度扩张倍率。
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        # 扩张输出通道数(模型中间的通道数--，类似于瓶颈结构)= 乘完倍率的输入通道数 * 扩张的倍率
        self.expanded_c = self.input_c * expanded_ratio
        # 乘完倍率的输出通道数 = 原本输出通道数 * 扩张倍率
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se  # 是否使用SE注意力模块
        self.stride = stride  # 步长
        self.drop_rate = drop_rate  # 随机丢掉数据的比率
        self.index = index  # 该模块重复次数

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        "返回乘积后--> 能倍8除尽的"
        """
        比如 B0中，卷积核的数目为 32,
        B6的扩张倍率为1.8，32*1.8=57.6。
        所以找到能被8 除尽的最近参数为 56。
        """
        return _make_divisible(channels * width_coefficient, 8)


class MBConv(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: nn.Module):
        super(MBConv, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        #
        activation_layer = nn.SiLU  # alias Swish

        # 如果 扩张的输出通道数= 乘完倍率的输入通道数
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            """ 如果为True ,使用SE模块"""
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            # 否则就是正常的残差
            self.dropout = nn.Identity()

    def forward(self, x):
        result = self.block(x)
        result = self.dropout(result)
        #  如果使用了
        if self.use_res_connect:
            result += x

        return result


class efficientnet(nn.Module):

    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: nn.Module = None,
                 norm_layer: nn.Module = None
                 ):
        """

        Args:
            width_coefficient: 变宽的倍数
            depth_coefficient: 变深的倍数
            num_classes: 分类树
            dropout_rate: 随机dropout数目
            drop_connect_rate:
            block: 模块
            norm_layer:
        """
        super(efficientnet, self).__init__()
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        """
        取整，拓宽模块的深度。
        原本的repeats=3,depth_coefficient=2.2
        round_repeats(3) 结果是7
        """

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # 提前获取adjust_channels 这个方法
        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # 提前获取配置信息
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        # 计算每个模块的重复度
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))

        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # 第一个捐几块
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # 添加剩下的 MAConv模块
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return efficientnet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return efficientnet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return efficientnet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return efficientnet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return efficientnet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return efficientnet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return efficientnet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return efficientnet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)


if __name__ == '__main__':
    from thop import profile
    model = efficientnet_b0(1000)
    # from torchstat import stat
    # d = stat(model, (3, 1024, 2048))
    
    input = torch.randn(1, 3, 1024,2048)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops /1e9))
    print("params:{:.3f}M".format(params /1e6))

~~~











参考资料

> [EfficientNet代码和论文解析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/147058015)
>
> [efficientnet 代码_穹镜的博客-CSDN博客_efficientnet代码](https://blog.csdn.net/weixin_42890793/article/details/118483809)
>
> [【机器学习】详解 EfficientNet_花与人间事同的博客-CSDN博客_efficientnetb3](https://blog.csdn.net/qq_39478403/article/details/117414535)