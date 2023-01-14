# 【图像分类】2021-BoTNet



> 论文题目: Bottleneck Transformers for Visual Recognition
>
> 论文地址：[https://arxiv.org/abs/2101.11605](https://arxiv.org/abs/2101.11605)
>
> 代码链接：[https://github.com/leaderj1001/BottleneckTransformers](https://github.com/leaderj1001/BottleneckTransformers)
>
> 发表时间: 2021年1月 
>
> 引用：Srinivas A, Lin T Y, Parmar N, et al. Bottleneck transformers for visual recognition[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 16519-16529.
>
> 引用数：326



## 1. 简介

### 1.1 简介

UC Berkeley 和 谷歌2021发表的一篇论文，属于`早期的结合CNN+Transformer`的工作。基于`Transformer的骨干网络`，同时使用卷积与自注意力机制来保持全局性和局部性。模型在ResNet最后三个BottleNeck中使用了`MHSA替换3x3卷积`。简单来讲Non-Local+Self Attention+BottleNeck = BoTNet

### 1.2 摘要

本篇文章首先在检测和分割任务上进行试验，因为检测和分割都需要高质量的全局信息，然后推广到分类视觉任务。作者认为，不同于分类任务输入图片较小$(224\times 224)$，检测任务输入较大$(1024\times 1024)$，这使得self-attention对内存和计算要求较高。为了解决这个问题，作者结采用了如下做法：

(1) 使用卷积提取有效的局部特征，降低分辨率
(2) 使用self-attention聚合全局信息（操作对象是feature map) 

综上，这是一个混合结构（卷积+self-attention）





## 2. 网络

### 2.1 分类情况

首先介绍一下，文章的总体内容。介绍了一下，这个BoTNet属于什么那一块领域。CNN+Transformer相结合的，并且是用在backbone里面的。



![image-20220607152135291](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220607152135291.png)





### 2.2 MHSA模块

![image-20220607152612372](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220607152612372.png)



关于MHSA的具体内容

![image-20220607152650064](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220607152650064.png)

上边的这个MHSA Block是核心创新点，其与Transformer中的MHSA有所不同：

- 由于处理对象不是一维的，而是类似CNN模型，所以有非常多特性与此相关。
- 归一化这里并没有使用Layer Norm而是采用的Batch Norm，与CNN一致。
- 非线性激活，BoTNet使用了三个非线性激活
- 左侧content-position模块引入了二维的位置编码，这是与Transformer中最大区别。

由于该模块是处理$B\times C\times H\times W$的形式。

这篇文章，只要是 借鉴了 2018年的Non Local这篇文章 论文地址: https://arxiv.org/abs/1711.07971
$$
y_i=softmax(\theta(x_i)^T\phi(x_j))g(x_j)=\frac{1}{\sum_{\forall j}e^{\theta(x_i)^T\phi(x_j)}}e^{\theta(x_i)^T\phi(x_j)}W_gx_j
$$
![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20200105163010813.png)







### 2.3 整体架构

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/6532a0c16aea43d193219895f949db45.png)

整体的设计和ResNet50几乎一样，唯一不同在于最后一个阶段中三个BottleNeck使用了MHSA模块。具体这样做的原因是Self attention需要消耗巨大的计算量，在模型最后加入时候feature map的size比较小，相对而言计算量比较小。



## 3. 代码

别人的代码，

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# reference
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, resolution=(224, 224), heads=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # for ImageNet
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, mhsa=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out) # for ImageNet

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet50(num_classes=1000, resolution=(224, 224), heads=4):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads)


def main():
    x = torch.randn([2, 3, 224, 224])
    model = ResNet50(resolution=tuple(x.shape[2:]), heads=8)
    print(model(x).size())
    print(get_n_params(model))


# if __name__ == '__main__':
#     main()
~~~







参考资料

> [BoTNet:Bottleneck Transformers for Visual Recognition_*pprp*的博客-CSDN博客](https://blog.csdn.net/DD_PP_JJ/article/details/122171200)
>
> [Bottleneck Transformers for Visual Recognition 阅读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/349014232)