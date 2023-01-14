# 2019-BiSeNetV1 CVPR

> 论文题目：BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
>
> 论文地址：[https://arxiv.org/abs/1808.00897](https://arxiv.org/abs/1808.00897)
>
> 代码地址:[https://github.com/CoinCheung/BiSeNet](https://github.com/CoinCheung/BiSeNet)
>
> 发表时间：2018年8月
>
> 引用：Yu C, Wang J, Peng C, et al. Bisenet: Bilateral segmentation network for real-time semantic segmentation[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 325-341.
>
> 引用数：1155

## 1. 简介

### 1.1 挑战

基于`轻量化网络模型`的设计作为一个热门的研究方法，许多研究者都在`运算量、参数量和精度之间寻找平衡`，希望使用尽量少的运算量和参数量的同时获得较高的模型精度。目前，轻量级模型主要有SqueezeNet、MobileNet系列和ShuffleNet系列等，这些模型在图像分类领域取得了不错的效果，可以作为基本的主干网络应用于语义分割任务当中。

然而，在语义分割领域，由于需要对输入图片进行逐像素的分类，运算量很大。通常，`为了减少语义分割所产生的计算量`，通常而言有两种方式：`减小图片大小和降低模型复杂度`。

* 减小图片大小可以最直接地减少运算量，但是图像会丢失掉大量的细节从而影响精度。
* 降低模型复杂度则会导致模型的特征提取能力减弱，从而影响分割精度。

所以，**如何在语义分割任务中应用轻量级模型，兼顾实时性和精度性能具有相当大的挑战性。**



### 1.2 背景

本文对之前的`实时性语义分割算法`进行了总结，发现当前主要有三种加速方法：

1. 通过`剪裁`或`resize`来限定输入大小，以降低计算复杂度。尽管这种方法简单而有效，空间细节的损失还是让预测打了折扣，尤其是边界部分，导致度量和可视化的精度下降；
2. 通过减少网络通道数量加快处理速度，尤其是在骨干模型的早期阶段，但是这会弱化空间信息。
3. 为追求极其紧凑的框架而丢弃模型的最后阶段（比如ENet）。该方法的缺点也很明显：由于 ENet 抛弃了最后阶段的下采样，模型的感受野不足以涵盖大物体，导致判别能力较差。

这些提速的方法**会丢失很多 Spatial Details 或者牺牲 Spatial Capacity，从而导致精度大幅下降。**

为了弥补空间信息的丢失，有些算法会采用 U-shape 的方式恢复空间信息。但是,U-shape会降低速度，同时`很多丢失的信息`并不能简单地通过融合浅层特征来恢复。

但是，这一技术有两个弱点：

1. 由于高分辨率特征图上额外计算量的引入，完整的U形结构拖慢了模型的速度。
2. 更重要的是，如图 1(b) 所示，绝大多数由于裁剪输入或者减少网络通道而丢失的空间信息无法通过引入浅层而轻易复原。换言之，U 形结构顶多是一个备选方法，而不是最终的解决方案。



### 1.3 结果

本文采用修改版的 Xception39 处理实时语义分割任务，并在 Cityscapes，CamVid 和 COCO-Stuff 三个数据集上对 BiSeNet 进行了评估，对比其他同类方法，给出了自己的结果。本节还对算法的速度与精度进行了着重分析。

#### 速度分析

本文实验在不同设置下做了完整的对比。首先给出的是 FLOPS 和参数的状态（见表 4）

![表 4：基础模型 Xception39 和 Res18 在 Cityscapes 验证集上的精度与参数分析。](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-67420911924d43491e5e8c8f35df1063_720w.jpg)

FLOPS 和参数表明在给定分辨率下处理图像所需要的操作数量。出于公平对比的考虑，本文选择 640×360 作为输入图像的分辨率。同时，表 5 给出了不同输入图像分辨率和不同硬件基准下本方法与其他方法的速度对比。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-455e78decc7f21075b0caa252796cc48_720w.jpg)

> 表 5：本文方法与其他方法的速度结果对比。1和2分别代表 backbone 是 Xception39 和 Res18 网络。

最后，本文给出了该方法在 Cityscapes 测试数据集上的速度及相应的精度结果。从表 6 可以看出，该方法相较于其他方法在速度和精度方面的巨大进展。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-3c5385828984cd774135ac9c9722a6b7_720w.jpg)



在评估过程中，本文首先把输入图像的分辨率从 2048×1024 缩至 1536×768，以测试速度和精度；同时，通过 online bootstrap 的策略计算损失函数。整个过程中本文不采用任何测试技巧，比如多尺度或多次裁剪测试。



#### 精度分析

事实上，BiSeNet 也可以取得更高的精度结果，甚至于可以与其他非实时语义分割算法相比较。这里将展示 Cityscapes，CamVid 和 COCO-Stuff 上的精度结果。同时，为验证该方法的有效性，本文还将其用在了不同的骨干模型上，比如标准的 ResNet18 和 ResNet101。



#### 结果

Cityscapes：如表 7 所示，该方法在不同模型上都取得了出色的结果。为提升精度，本方法首先随机裁切 1024x1024 大小的图作为输入。图 4 给出了一些可视化结果实例。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-d738971d22830b37d0966776a28c7e7f_720w.jpg)

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-b196a1b8fae9cd3818fc4fbdbda67c9a_720w.jpg)



CamVid：表 8 给出了 CamVid 数据集上统计的精度结果。对于测试，本文通过训练数据集和测试数据集训练模型。这里训练和评估使用的分辨率是 960×720。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-a257f85b3a39af2b4d56c96d6db65b71_720w.jpg)

COCO-Stuff：表 9 给出了 COCO-Stuff 验证集上的精度结果。在训练和验证过程中，本文把输入分辨率裁剪为 640×640。出于公平对比，本文不采用多尺度测试。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-b76f0097e3e3c3198e6cb741952a0937_720w.jpg)

## 2. 网络

### 2.1 设计思想

基于上述观察，本文提出了双向分割网络（Bilateral Segmentation Network/BiseNet），它包含两个部分：`Spatial Path (SP)` 和 `Context Path (CP)`。顾名思义，这两个组件分别用来`解决空间信息缺失`和`感受野缩小`的问题，其设计理念也非常清晰。

对于 Spatial Path，本文只叠加`三个卷积层`以获得 `1/8 特征图`，其保留着丰富的空间细节。对于 Context Path，本文在 Xception 尾部附加一个全局平均池化层，其中感受野是 backbone 网络的最大值。图 1(c) 展示了这两个组件的结构。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-d737d5de7b6d1b449c28d53ed77f5090_720w.jpg)



### 2.2 Spatial Path

> 很简单，就是3个卷积层。
>
> 每个卷积包括 conv+bn+relu

基于这一观察，本文提出`Spatial Path`以保留原输入图像的空间尺度，并编码丰富的空间信息。Spatial Path 包含三层，每层包含一个步幅（stride）为 2 的卷积，随后是批归一化和 ReLU。因此，该路网络提取相当于原图像 1/8 的输出特征图。由于它利用了较大尺度的特征图，所以可以编码比较丰富的空间信息。图 2(a) 给出了这一结构的细节。

![image-20220517155353363](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220517155353363.png)





### 2.3 Context Path

> 这个部分包括 ARM,FFM 模块

在语义分割任务中，`感受野对于性能表现至关重要`。为增大感受野，一些方法利用金字塔池化模块，金字塔型空洞池化（ASPP）或者 “large kernel”，但是这些操作比较耗费计算和内存，导致速度慢。

出于较大感受野和较高计算效率兼得的考量，本文提出 Context Path，它充分利用轻量级模型与全局平均池化以提供大感受野。

在本工作中，轻量级模型，比如 Xception，可以快速下采样特征图以获得大感受野，编码高层语义语境信息。接着，本文在轻量级模型末端添加一个全局平均池化，通过全局语境信息提供一个最大感受野。在轻量级模型中，本文借助 U 形结构融合最后两个阶段的特征，但这不是一个完整的 U 形结构。图 2(c) 全面展示了 Context Path。



#### ARM

Attention Refinment Module

注意力优化模块（ARM）：在 Context Path 中，本文提出`一个独特的注意力优化模块`，以优化每一阶段的特征。如图 2(b) 所示，ARM 借助全局平均池化捕获全局语境并计算注意力向量以指导特征学习。这一设计可以优化 Context Path 中每一阶段的输出特征，无需任何上采样操作即可轻易整合全局语境信息，因此，其计算成本几乎可忽略。

![image-20220517155428481](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220517155428481.png)





#### FFM

Feature Fusion Module

![image-20220517155509252](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220517155509252.png)



## 3. 代码



~~~python
import torch.nn as nn
import torch
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan,)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan, ),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs.
    channels_last corresponds to inputs with shape (batch_size, height, width, channels)
    while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvBNReLU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn = LayerNorm(out_channels)
        self.relu = nn.GELU()
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ARM(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(ARM, self).__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn_atten = LayerNorm(out_channels)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        """初始化权重信息"""
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = ARM(256, 128)
        self.arm32 = ARM(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        #  feat32 [1, 512, 7, 7]
        #  feat16 [1, 256, 14, 14]
        #
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        # print("feat32_up", feat16.shape)
        # print("feat32_up", feat32.shape)


        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)





        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up  # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, kernel_size=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FFM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FFM, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    """分类头"""

    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=(1, 1), bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNet(nn.Module):
    def __init__(self, n_classes):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FFM(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        # 返回两个特征图，一个是1/8的特征图和 一个1/16的特征图
        feat_cp8, feat_cp16 = self.cp(x)
        # feat_cp8 torch.Size([1, 128, 28, 28])
        # feat_cp16 torch.Size([1, 128, 14, 14])

        # print("feat_cp8",feat_cp8.shape)
        # print("feat_cp16", feat_cp16.shape)
        feat_sp = self.sp(x)
        #  这里进行融合
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


if __name__ == '__main__':
    from thop import profile
    model = BiSeNet(n_classes=10)
    model.eval()
    input = torch.randn(1, 3, 224, 224)
    # output=model(input)
    # print(output[0].shape)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops/1e9))
    print("params:{:.3f}M".format(params/1e6))
~~~







参考资料

> [语义分割之Bisenet - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/348383907)