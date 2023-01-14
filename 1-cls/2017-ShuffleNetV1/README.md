# 【图像分类】2017-ShuffleNetV1 CVPR

>论文题目：ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
>
>论文地址：[https://arxiv.org/abs/1707.01083](https://arxiv.org/abs/1707.01083)
>
>代码地址:[https://github.com/jaxony/ShuffleNet](https://github.com/jaxony/ShuffleNet)
>
>发表时间：2017年7月
>
>引用：Zhang X, Zhou X, Lin M, et al. Shufflenet: An extremely efficient convolutional neural network for mobile devices[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 6848-6856.
>
>引用数：4345

## 1. 简介

ShuffleNet V1 是旷视科技的张翔雨提出的一种适用于移动设备的轻量化网络。

为了解决主要的视觉识别任务，构建更深更大的卷积神经网络是一个基本的趋势，大多数准确率高的卷积神经网络通常都有上百层和上千个通道，需要数十亿的 FLOPS。这篇报告走的是另一个极端，在只有几十或者几百 FLOPS 的计算资源预算下，追求最佳的精度，目前的研究主要集中在剪枝、压缩和量化上。在这里我们要探索的是根据我们的计算资源设计一种高效的基本网络架构。



## 2. 网络

### 2.1 Channel Shuffle 

> 创新点就是把$1\times 1$的逐点卷积 变成了 `分组逐点卷积+ 通道重排`。
>
> $1\times 1$的逐点卷积-->分组逐点卷积+ 通道重排

Xception，ResNeXt，MobileNet 等网络都使用了`group conv`，他们有一个问题，是采用了密集的`1x1 pointwise conv`，这一步需要相当大的计算量。

为此，作者指出，一个非常自然的解决方案就是把`1x1 pointwise conv`同样应用`group conv`，这样就可以进一步降低计算量。但是，这又带来一个新的问题：“outputs from a certain channel are only derived from a small fraction of input channels”->> 某一通道的输出仅来自输入通道的一小部分。

为了解决这一问题，作者构建了 channel shuffle，如下图所示。

* 图(a)表示利用两个堆叠的 group conv 提取特征。
* 图(b)表示 channel shuffle ，对 group conv 之后的特征 “重组” ，接下来的 group conv 输入来自不同的组，信息可以在不同组间流转。
* 图(c) 是图(b) 的另一种表示方法。

![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/cdce1fa5aeb24f75b1a7503ef353378a.png)



步骤如下

\- 有g个组的卷积层进行划分使得输出有 gxn 个通道；(划分为g个组，每个组有n个通道)

\- feature map reshape为(g, n)；

\- 将维度为(g, n)的feature map转置为(n, g)；

\- 平坦化之后分组送入下一层；(这时到底划分为几个组就取决于下一个组卷积的组数了)

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-5635d74058cf9d2b6bcbe10461f4d9a1_720w.jpg)

而且，通道清洗是可微分的，这意味着模型可以进行 end-to-end 的训练；通道清洗操作使得使用多个组卷积层构建更强大的结构成为可能。

下面是实现通道混排的代码

~~~python
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward(self, x):
        batchsize,channels,height,width=x.data.size()
        channels_per_group=int(channels/self.groups)
        x = x.view(batchsize,self.groups,channels_per_group,height,width)
        # 这里进地了矩阵的转置，然后必须要使用.contiguous()
        # 使得张量在内存连续之后才能调用view函数
        x = x.transpose(1,2).contiguous()
        x = x.view(batchsize,-1,height,width)
        return x
~~~

或者定义一个方法

~~~python
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
~~~



### 2.2 ShuffleNet unit

ShuffleNet的基本单元和标准的 MobileNet 单元的区别如下图所示。 

* MobileNet 的基本单元如图(a)所示，首先是1x1的卷积降低 feature map 的通道数，然后用 3x3 的 depthwise conv 处理，然一用 1x1 的 pointwise conv 处理。
* 图 (b) 展示了改进思路，把第一个 1x1 的卷积用 group conv 替换，然后增加了一个 channel shuffle 操作。值得注意的是 3x3 卷积后没有加 channel shuffle，作者表示以这个单元中加一个 channel shuffle 就足够了。
* 图（c）表示的是起到pooling作用，stride=2 的 ShuffleNet 单元，把两个通路的结果百行拼接，而不是相加，作者的解释是 makes it easy to enlarge channel dimension with little extra computation cost。



![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20200721155223181.png)



### 2.3 总体架构

![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/2020072116054972.png)

## 3. 代码



~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    """3x3 convolution with padding
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add'):

        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm=True,
            relu=True
        )

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels, self.bottleneck_channels,
            stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            batch_norm=True,
            relu=False
        )

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
                              batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        # save for combining later with output
        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3,
                                    stride=2, padding=1)

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)

        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNet(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        """ShuffleNet constructor.
        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.
        """
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.num_classes = num_classes

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.in_channels,
                             self.stage_out_channels[1],  # stage 1
                             stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2)
        # Stage 3
        self.stage3 = self._make_stage(3)
        # Stage 4
        self.stage4 = self._make_stage(4)

        # Global pooling:
        # Undefined as PyTorch's functional API can be used for on-the-fly
        # shape inference if input size is not ImageNet's 224x224

        # Fully-connected classification layer
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(
            self.stage_out_channels[stage - 1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat'
        )
        modules[stage_name + "_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + "_{}".format(i + 1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add'
            )
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # global average pooling layer
        x = F.avg_pool2d(x, x.data.size()[-2:])

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    """Testing
    """
    from thop import profile

    model = ShuffleNet(num_classes=1000)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
~~~



参考资料

> [深度学习入门笔记之ShuffleNet_ysukitty的博客-CSDN博客](https://blog.csdn.net/ysukitty/article/details/123016846)
>
> 