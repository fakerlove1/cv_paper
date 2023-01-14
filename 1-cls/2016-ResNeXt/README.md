# 【图像分类】2016-ResNeXt CVPR

> 论文题目：Aggregated Residual Transformations for Deep Neural Networks
>
> 论文地址：[https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
>
> 代码地址：[https://github.com/facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt)
>
> 论文翻译: [ResNeXt论文翻译](https://www.pianshen.com/article/2768764286/)
>
> 发表时间：2016年11月
>
> 引用：Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1492-1500.
>
> 引用数：7439

## 1. 简介

### 1.1 简介

2016年，图像识别比赛的亚军的作品。

### 1.2 摘要

 我们为图像分类提出一个**简单**、**高度模块化**的网络结构。网络通过**重复一个block**来构建，这个block**聚合了一组有相同拓扑的转换**。我们的简单设计产生了一个**同构的**、**多分支的** 架构，它只需要设置几个超参数。这个策略揭示了一个**新的维度**，我们称之为"**基数-cardinality**"(这组转换的大小)，除了深度和宽度之外的一个关键因子。在ImageNet-1K数据集上，我们的实验表明，在**控制复制度**的受限情况下，**增加基数**可以**提升分类精度**。而且当**提升模型容量**时(即增加模型复杂度)，**提升基数比更深或更宽更加有效**。我们的模型叫ResNeXt，是参加ILSVRC 2016分类任务的基础，我们获得了第二名。我们进一步在ImageNet-5K 数据集和COCO检测数据集上研究ResNeXt,同样展示了比对应的ResNet更好的结果。代码和模型公布在



## 2. 网络

ResNet的创新点，`就是组卷积`。

### 2.1 Building block

优点：该网络具有相同的、多分支的结构，并且对应的超参数非常少；

首先作者想出了3种`分组卷积的思路`。

![image-20220519221255501](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220519221255501.png)

ResNeXt对ResNet进行了改进，采用了多分支的策略，

在论文中作者提出了`三种等价的模型结构`，最后的ResNeXt用了`C`的结构来构建我们的ResNeXt、

这里面和我们的Inception是不同的，在Inception中，每一部分的拓扑结构是不同的，比如一部分是1x1卷积，3x3卷积还有5x5卷积，而我们ResNeXt是用相同的拓扑结构，并在保持参数量的情况下提高了准确率。



![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/17b756bfca9d4ffebc6c1675697b12bb.png)



结论：由图可以看出，ResNeXt就是采用了多分支，并且每个分支是相同的结构；

1. **同stage中的block使用相同的width和filter size；**
2. **spatial size减小时，增加channel的数量。**



## 3. 代码

在ResNet18和RexNet34的基础上，ResNext没进行改进，因为用的是两层卷积。ResNext在3层的卷积上做的改进

所以ResNext只有50和101的版本

~~~python
'''
ResNeXt in PyTorch.
See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
这一部分借鉴官方的实现方式，改ResNet进行训练
'''
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # 最基础的Block的压缩为1，在ResNet18，34有效
    expansion = 1

    # downsample代表：是否进行下采样，或者说我们是否需要一个shortcut，也就是我们的下采样
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 首先是一个3x3的卷积层，这里的stride是我们设置的，因为对于我们的第一个卷积层，stride为2
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 进行残差连接
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    实验结果证明，这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    # 这里相对于简单的残差网络中多增加了两个参数，一个是groups和width_per_group,分别是组卷积的个数和每个组卷积的通道数
    # 默认值就是正常的ResNet
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        # 这里也可以自动计算中间的通道数，也就是3x3卷积后的通道数，如果不改变就是out_channels
        # 如果groups=32,with_per_group=4,out_channels就翻倍了
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 组卷积的数，需要传入参数
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # -----------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,  # 表示block的类型
                 blocks_num,  # 表示的是每一层block的个数
                 num_classes=1000,  # 表示类别
                 include_top=True,  # 表示是否含有分类层(可做迁移学习)
                 groups=1,  # 表示组卷积的数
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion  # 得到最后的输出
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def ResNet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ResNet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ResNet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


# 论文中的ResNeXt50_32x4d
def ResNeXt50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def ResNeXt101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    model=ResNeXt101_32x8d(num_classes=1000)
    y=model(x)
    print(y.shape)
~~~

