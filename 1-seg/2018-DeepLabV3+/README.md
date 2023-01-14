# 【语义分割】2018-DeeplabV3+ ECCV

> 论文题目：Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
>
> 论文链接：[https://arxiv.org/abs/1802.02611](https://arxiv.org/abs/1802.02611)
>
> 论文代码：[https://github.com/jfzhang95/pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
>
> 发表时间：2018年2月
>
> 引用：Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 801-818.
>
> 引用数：7224

## 1. 简介

### 1.1 简介

deeplab-v3+是一个`语义分割`网络，它基于`deeplab-v3`，添加一个简单有效的`Decoder来细化分割结果`，尤其是沿着目标对象边界的分割结果，以及采用空间金字塔池模块或编解码结构二合一的方式进行实现。



### 1.2 改进

* v1：修改了VGG16引入空洞卷积，

* v2：设计ASPP模块，
* v3：串联ASPP与并联ASPP，讨论了3x3卷积可能会丢失部分信息，提出了1x1卷积的必要性。
* v3+：使用小幅度修改的xception主网络，结合deeplab v3+作为encoder，自己设计decoder）

> DeeplabV3+在Encoder部分**引入了大量的空洞卷积，在不损失信息的情况下，加大了感受野，让每个卷积输出都包含较大范围的信息**。
>
> DeeplabV3+被认为是语义分割的新高峰，因为这个模型的效果非常好。
> DeeplabV3+主要在模型的架构上作文章，引入了可任意控制编码器提取特征的分辨率，通过空洞卷积平衡精度和耗时。



![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20191111203311239.png)







## 2. 网络讲解

### 2.1 backbone-Xception

**简介**

> * 这里对应的是上面网络结构图中的DCNN（深度卷积神经网络）部分
>
> * Xception结构由keras的作者François Chollet发表于2016年（论文下载：[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) ) 其在3.5亿张图像、17000个分类的大型计算机视觉任务上效果Inception-v3。
>
> * **Xception** 并不是真正意义上的`轻量化模型`，是Google继Inception后提出的对Inception v3的另一种改进，
>
>   主要是采用`depthwise separable convolution`来替代原来的Inception v3中的卷积操作，这种性能的提升是来自于`更有效的使用模型参数`而不是提高容量。
>
>   一个卷积层尝试去学习特征在3维空间–（高、宽、通道），包含了空间的相关性和跨通道的相关性。



**Xception做了一个加强的假设，就是卷积的时候要将通道的卷积与空间的卷积进行分离，这样会不会更合理？**

#### 1) Xception结构演变

既然是在Inception v3上进行改进的，那么Xception是如何一步一步的从Inception v3演变而来。

> Inception v3结构如下图1，当时提出Inception的初衷可以认为是：
>
> **特征的提取和传递可以通过1x1卷积，3x3卷积，5x5卷积，pooling等，到底哪种才是最好的特征提取方式呢？**
>
> **Inception结构将这个疑问留给网络自己训练，也就是将一个输入同时给这几种提取特征方式，然后做concat。** 
>
> Inception v3和Inception v1（GoogLeNet）对比主要是将5x5卷积换成两个3x3卷积层的叠加。



![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210116150630895.png)




**注：1x1卷积的作用:**

1）降维：较少计算量 

2）升维：小型网络，通道越多，效果会更好 

3）1x1是有一个参数学习的卷积层，可以增加跨通道的相关性。



> **图2** 简化了的inception module（就只考虑1x1的那条支路，不包含Avg pool)

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210116150906408.png)



> 对于一个输入的Feature Map，首先通过三组$1\times 1$卷积得到三组Feature Map，它和先使用一组$1\times 1$卷积得到Feature Map,再将这组Feature Map分成三组是完全等价的（**图3**）。假设图2中的$1\times 1$卷积核的个数都是$k_1$,$3\times 3$的卷积核的个数都是$k_2$。输入Feature Map的通道数为$m$。那么这个简单版本的参数个数为
> $$
> m\times k_1+3\times 3\times 3\times \frac{k_1}{3}\times\frac{k_2}{3}=m\times k_1+3\times k_1\times k_2
> $$
> 对比相同通道数，但是没有分组的普通卷积，普通卷积的参数数量为：
> $$
> m\times k_1+3\times 3\times k_1\times k_2
> $$
> 参数数量约为Inception的三倍。

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210116151023865.png)

> 如果Inception是将$3\times 3$卷积分成3组，那么考虑一种`极端`的情况,如果我们将Inception的$1\times 1$卷积得到$k_1$个通道的Feature Map完全分开呢？也就是使用$k_1$个不同的卷积分别在每个通道上进行卷积，它的参数数量是：
> $$
> m\times k_1+k_1\times 3\times 3
> $$
> 更多时候我们希望两组卷积的输出Feature Map相同，这里我们将Inception的$1\times 1$卷积的通道数设为$k_2$,即参数量为
> $$
> m\times k_2+k_2\times 3 \times 3
> $$
> 它的参数数量是普通卷积的$\frac{1}{k_1}$.我们把这种形式的Inception叫做Extreme Inception .
>
> 如**图4** An“extreme” version of Inception module，
>
> 先用$1\times 1$卷积核对各通道之间（cross-channel）进行卷积，之后使用$3\times 3$的卷积对`每个输出通道`进行卷积操作，也就是$3\times 3$卷积的个数和$1\times 1$卷积的输出channel个数相同。

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210116151043436.png)



在Xception中主要采用depthwise separable convolution，和原版的相比有两个不同之处：

1. 原版的Depthwise convolution，先是逐通道卷积，再1x1卷积；而Xception是反过来，先1x1卷积，再逐通道卷积。
2. 原版Depthwise convolution的两个卷积之间是不带激活函数的，而Xception再经过1x1卷积之后会带上一个Relu的非线性激活函数。



#### 2) Xception的网络结构

**完全基于深度可分离卷积的卷积神经网络结构：**

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210116151240898.png)

改进如下

> 1. Entry flow保持不变，但是增加了更多的Middle flow；
> 2. 将步长为2的max-pooling替换为深度可分离卷积，这样也便于随时替换为空洞卷积；
> 3. 在深度可分离卷积之后增加了BN和ReLU。

DeepLab v3+的Xception结构如下图所示。

![image-20220724103735215](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220724103735215.png)

**注意点**

* Keras的`SeparalbeConv`函数是由$3\times 3$的depthwise卷积和$1\times 1$的pointwise卷积组成，因此可用于升维和降维
* 图5中的$\oplus$是add操作，即两个Feature Map进行单位加。



Xception结构由36层卷积层组成网络的特征提取基础，分为Entry flow，Middle flow，Exit flow；被分成了14个模块，除了第一个和最后一个外，其余模块间均有线性残差连接。



#### 3) 代码

~~~python
import math
import os
import torch
import torch.nn as nn
# import torch.utils.model_zoo as model_zoo

bn_mom = 0.0003


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 activate_first=True, inplace=True):
        super(SeparableConv2d, self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first

    def forward(self, x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, strides=1, atrous=None, grow_first=True, activate_first=True,
                 inplace=True):
        super(Block, self).__init__()
        if atrous == None:
            atrous = [1] * 3
        elif isinstance(atrous, int):
            atrous_list = [atrous] * 3
            atrous = atrous_list
        idx = 0
        self.head_relu = True
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters, momentum=bn_mom)
            self.head_relu = False
        else:
            self.skip = None

        self.hook_layer = None
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters
        self.sepconv1 = SeparableConv2d(in_filters, filters, 3, stride=1, padding=1 * atrous[0], dilation=atrous[0],
                                        bias=False, activate_first=activate_first, inplace=self.head_relu)
        self.sepconv2 = SeparableConv2d(filters, out_filters, 3, stride=1, padding=1 * atrous[1], dilation=atrous[1],
                                        bias=False, activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_filters, out_filters, 3, stride=strides, padding=1 * atrous[2],
                                        dilation=atrous[2], bias=False, activate_first=activate_first, inplace=inplace)

    def forward(self, inp):

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, downsample_factor):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        stride_list = None
        if downsample_factor == 8:
            stride_list = [2, 1, 1]
        elif downsample_factor == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('xception.py: output stride=%d is not supported.' % os)
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)
        # do relu here

        self.block1 = Block(64, 128, 2)
        self.block2 = Block(128, 256, stride_list[0], inplace=False)
        self.block3 = Block(256, 728, stride_list[1])

        rate = 16 // downsample_factor
        self.block4 = Block(728, 728, 1, atrous=rate)
        self.block5 = Block(728, 728, 1, atrous=rate)
        self.block6 = Block(728, 728, 1, atrous=rate)
        self.block7 = Block(728, 728, 1, atrous=rate)

        self.block8 = Block(728, 728, 1, atrous=rate)
        self.block9 = Block(728, 728, 1, atrous=rate)
        self.block10 = Block(728, 728, 1, atrous=rate)
        self.block11 = Block(728, 728, 1, atrous=rate)

        self.block12 = Block(728, 728, 1, atrous=rate)
        self.block13 = Block(728, 728, 1, atrous=rate)
        self.block14 = Block(728, 728, 1, atrous=rate)
        self.block15 = Block(728, 728, 1, atrous=rate)

        self.block16 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block17 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block18 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block19 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])

        self.block20 = Block(728, 1024, stride_list[2], atrous=rate, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)

        self.conv4 = SeparableConv2d(1536, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)

        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, 1 * rate, dilation=rate, activate_first=False)
        self.layers = []

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, input):
        self.layers = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        low_featrue_layer = self.block2.hook_layer
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)
        return low_featrue_layer, x



def xception(downsample_factor=16,pretrained=False):
    model = Xception(downsample_factor=downsample_factor)
    return model

if __name__ == '__main__':
    images=torch.randn(size=(1,3,224,224))
    model=xception()
    low_featrue_layer, x=model(images)
    # 做了16倍下采样
    print(low_featrue_layer.shape)
    print(x.shape)
~~~





#### 4) 小结

Xception作为Inception v3的改进，主要是在`Inception v3的基础上引入了depthwise separable convolution`，在基本不增加网络复杂度的前提下提高了模型的效果。有些人会好奇为什么引入depthwise separable convolution没有大大降低网络的复杂度，因为depthwise separable convolution在mobileNet中主要就是为了降低网络的复杂度而设计的。原因是作者加宽了网络，使得参数数量和Inception v3差不多，然后在这前提下比较性能。因此Xception目的`不在于模型压缩`，而是`提高性能`。



### 2.2 ASPP 模块

ASPP（Atrous Spatial Pyramid Pooling），空洞空间卷积池化金字塔。简单理解就是个至尊版池化层，其目的与普通的池化层一致，尽可能地去提取特征。ASPP 的结构如下：

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/7548f8d2dfdc4c34884860e5c6e4cdb9.png)

如图所示，ASPP 本质上

由一个1×1的`卷积`（最左侧绿色） +`池化金字塔`（中间三个蓝色） + ASPP `Pooling`（最右侧三层）组成。

而池化金字塔各层的膨胀因子可自定义，从而实现自由的多尺度特征提取。



#### ASPP Conv

~~~python
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

~~~

空洞卷积层与一般卷积间的差别在于膨胀率，膨胀率控制的是卷积时的 padding 以及 dilation。通过不同的填充以及与膨胀，可以获取不同尺度的感受野，提取多尺度的信息。注意卷积核尺寸始终保持 3×3 不变。



#### ASPP Pooling

~~~python
class ASPPPooling(nn.Sequential):
	def __init__(self, in_channels, out_channels):
	    super(ASPPPooling, self).__init__(
	        nn.AdaptiveAvgPool2d(1),
	        nn.Conv2d(in_channels, out_channels, 1, bias=False),
	        nn.BatchNorm2d(out_channels),
	        nn.ReLU())
	        
	def forward(self, x):
	   size = x.shape[-2:]
	   for mod in self:
	       x = mod(x)
	   return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

~~~

ASPP Polling 首先是一个 `AdaptiveAvgPool2d` 层。所谓自适应均值池化，其自适应的地方在于不需要指定 kernel size 和 stride，只需指定最后的输出尺寸（此处为 1×1）。通过将各通道的特征图分别压缩至 1×1，从而提取各通道的特征，进而获取全局的特征。然后是一个 1×1 的卷积层，对上一步获取的特征进行进一步的提取，并降维。需要注意的是，在 ASPP Polliing 的网络结构部分，只是对特征进行了提取；而在 forward 方法中，除了顺序执行网络的各层外，最终还将特征图从1×1 上采样回原来的尺寸。



#### ASPP

~~~python
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 注释 1
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 注释 2
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 注释 3
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        # 注释 4
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    
    # 注释 5
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

~~~

注释：

1. 最开始是一个 1×1 的卷积层，进行降维；
2. 构建 “池化金字塔”。对于给定的膨胀因子 atrous_rates，叠加相应的空洞卷积层，提取不同尺度下的特征；
3. 添加空洞池化层；
4. 出层，用于对ASPP各层叠加后的输出，进行卷积操作，得到最终结果；
5. forward() 方法，其顺序执行ASPP的各层，将各层的输出按通道叠加，并通过输出层的 conv -> bn -> relu -> dropout 降维至给定通道数，获取最终结果。



#### 完整代码

~~~python
# 空洞卷积
import torch
from torch import nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


if __name__ == '__main__':

    images = torch.randn(size=(1, 2048, 32, 32))
    model = ASPP(in_channels=2048, atrous_rates=[1, 1])
    # 因为使用了batch_size=1
    # 可能会报错 Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])
    # 解决方案https://blog.csdn.net/qq_45365214/article/details/122670591
    if images.shape[0] == 1:
        model.eval()
    out = model(images)
    print(out.shape)

~~~





简化版

~~~python
#without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1)) #(1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


~~~



### 2.3 Decoder

decoder 模块。

xception 一共输出 2个语义信息，一个是低层语义信息，一个是高级语义信息

* 高级语义信息，通过一个aspp模块，进行一个4倍的上采样，
* 低级语义信息，通过一个卷积，与4倍上采样完毕的 高级语义信息进行concat 操作
* 最后通过一个分割头，得到最后的分割结果

![image-20220430105245730](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220430105245730.png)







## 3. 代码

~~~python
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# xception 模块
bn_mom = 0.0003
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 activate_first=True, inplace=True):
        super(SeparableConv2d, self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first

    def forward(self, x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, strides=1, atrous=None, grow_first=True, activate_first=True,
                 inplace=True):
        super(Block, self).__init__()
        if atrous == None:
            atrous = [1] * 3
        elif isinstance(atrous, int):
            atrous_list = [atrous] * 3
            atrous = atrous_list
        idx = 0
        self.head_relu = True
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters, momentum=bn_mom)
            self.head_relu = False
        else:
            self.skip = None

        self.hook_layer = None
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters
        self.sepconv1 = SeparableConv2d(in_filters, filters, 3, stride=1, padding=1 * atrous[0], dilation=atrous[0],
                                        bias=False, activate_first=activate_first, inplace=self.head_relu)
        self.sepconv2 = SeparableConv2d(filters, out_filters, 3, stride=1, padding=1 * atrous[1], dilation=atrous[1],
                                        bias=False, activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_filters, out_filters, 3, stride=strides, padding=1 * atrous[2],
                                        dilation=atrous[2], bias=False, activate_first=activate_first, inplace=inplace)

    def forward(self, inp):

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, downsample_factor):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        stride_list = None
        if downsample_factor == 8:
            stride_list = [2, 1, 1]
        elif downsample_factor == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('xception.py: output stride=%d is not supported.' % os)
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)
        # do relu here

        self.block1 = Block(64, 128, 2)
        self.block2 = Block(128, 256, stride_list[0], inplace=False)
        self.block3 = Block(256, 728, stride_list[1])

        rate = 16 // downsample_factor
        self.block4 = Block(728, 728, 1, atrous=rate)
        self.block5 = Block(728, 728, 1, atrous=rate)
        self.block6 = Block(728, 728, 1, atrous=rate)
        self.block7 = Block(728, 728, 1, atrous=rate)

        self.block8 = Block(728, 728, 1, atrous=rate)
        self.block9 = Block(728, 728, 1, atrous=rate)
        self.block10 = Block(728, 728, 1, atrous=rate)
        self.block11 = Block(728, 728, 1, atrous=rate)

        self.block12 = Block(728, 728, 1, atrous=rate)
        self.block13 = Block(728, 728, 1, atrous=rate)
        self.block14 = Block(728, 728, 1, atrous=rate)
        self.block15 = Block(728, 728, 1, atrous=rate)

        self.block16 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block17 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block18 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block19 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])

        self.block20 = Block(728, 1024, stride_list[2], atrous=rate, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)

        self.conv4 = SeparableConv2d(1536, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)

        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, 1 * rate, dilation=rate, activate_first=False)
        self.layers = []

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, input):
        self.layers = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        low_featrue_layer = self.block2.hook_layer
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)
        return low_featrue_layer, x


def xception(downsample_factor=16,pretrained=False):
    model = Xception(downsample_factor=downsample_factor)
    return model

############################################################################
# aspp 模块
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class aspp(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(aspp, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# if __name__ == '__main__':
#
#     images = torch.randn(size=(1, 2048, 32, 32))
#     model = aspp(in_channels=2048, atrous_rates=[2,2])
#     if images.shape[0] == 1:
#         model.eval()
#     out = model(images)
#     print(out.shape)


#################################################
# deeplabv3+ 模块
class deeplabv3plus(nn.Module):
    def __init__(self, num_classes, backbone="xception", pretrained=False, downsample_factor=16):
        super(deeplabv3plus, self).__init__()

        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            # 输入 torch.Size([batch_size, 3, 512,512])
            #   浅层特征    torch.Size([batch_size, 256, 128, 128])
            #   主干部分    torch.Size([batch_size, 2048, 32, 32])
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = None
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = aspp(in_channels=in_channels, out_channels=256,
                         atrous_rates=[16 // downsample_factor, 16 // downsample_factor])

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    model=deeplabv3plus(num_classes=19)

    from thop import profile
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))

~~~







参考资料

> [DeepLab V3++ 论文笔记_Tianchao龙虾的博客-CSDN博客_deeplabv3+论文](https://blog.csdn.net/wuchaohuo724/article/details/119757041)
>
> [ASPP 详解_晓野豬的博客-CSDN博客_aspp代码](https://blog.csdn.net/qq_41731861/article/details/120967519)

