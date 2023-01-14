# 2015-ResNet CVPR

>论文题目：Deep Residual Learning for Image Recognition
>
>论文链接: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
>
>论文代码：[https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
>
>论文作者: 何恺明、张祥雨、任少卿、孙剑。微软亚洲研究院
>
>发表时间：2015年12月
>
>引用：He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
>
>引用数：129095

## 1. 简介

### 1.1 简介

发表在`2015`年，`2016年CVPR`最佳论文:Deep Residual Learning for Image Recognition。 通过`残差模块`解决深层网络的退化问题，大大提升神经网络深度，各类计算机视觉任务均从深度模型提取出的特征中获益。

ResNet获得2015年ImageNet图像分类、定位、目标检测竞赛冠军，MS COCO目标检测、图像分割冠军。并首次在ImageNet图像分类性能上`超过人类水平`。

### 1.2 存在的问题(深度网络退化问题)

`Resnet`网络是为了解决深度网络中的退化问题，即网络层数越深时，在数据集上表现的性能却越差，如下图所示是论文中给出的深度网络退化现象。

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20200727200621995.png)

从图中我们可以看到，作者在`CIFAR-10`数据集上测试了20层和56层的深度网络，结果就是56层的训练误差和测试误差反而比层数少的20层网络更大，这就是`ResNet`网络要解决的深度网络退化问题。

### 1.3 解决方案(亮点)-残差结构

而采用`ResNet`网络之后，可以解决这种退化问题，如下图所示。

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20200727200631861.png)

从图中作者在ImageNet数据集上的训练结果可以看出，在没有采用`ResNet`结构之前，如左图所示，34层网络plain-34的性能误差要大于18层网络`plain-18`的性能误差。而采用ResNet网络结构的34层网络结构ResNet-34性能误差小于18层网络ResNet-18。因此，采用ResNet网络结构的网络层数越深，则性能越佳。



## 2. 网络

### 2.1 总体架构

#### ResNet网络结构图

了解了上述`BasicBlock`基础块和`BotteNeck`结构后，`ResNet`结构就直接叠加搭建了。5种不同层数的`ResNet`结构图如图所示，

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20200727200750875.png)



#### ResNet34 具体结构图

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20200727200803188.png)

### 2.2 残差结构

接下来介绍`ResNet`网络原理及结构。

假设我们想要网络块学习到的映射$H(x)$,而直接学习$H(x)$是比较困难的。若我们学习另一个残差函数$F(x)=H(x)-x$是可以很容易的。因此此时网络块的训练目标是将$F(x)$逼近与0，而不是某一个特定映射。因此，最后的映射$H(x)$就是将$F(x)$和$x$相加。$H(x)=F(x)+x$。如图所示

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20200727200643141.png)

因此，这个网络块的输出$y$为
$$
y=F(x)+x
$$
由于相加必须保证$x$和$F(x)$是同维度的，因此可以写成通式如下，$W_s$用于匹配维度
$$
y=F(x,\{W_i\})+W_sx
$$
文中提到两种维度匹配的方式（A）用`zero-padding`增加维度； (B）用`1x1`卷积增加维度。

具体的`残差结构代码`，下面会讲解



## 3. 网络实现细节

### 3.1 卷积操作讲解

#### a) 1*1 卷积

~~~python
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=(1, 1),
                     stride=(stride, stride),
                     bias=False)
~~~

1*1卷积,只能`升级通道数`

因为 F=1,S=1,P=0, 
$$
\frac{W-F+2P}{S}+1=int(\frac{W-1+0}{1})+1=W
$$
所以1*1 卷积是不改变宽高的

理由是python是向下取整的。不是四舍五入

~~~python
print(int(5.5))
~~~

结果为

~~~bash
5
~~~

所以
$$
int(\frac{W-1+0}{1})=W-1
$$

#### b) 3*3 卷积

~~~python
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=(3, 3),
                     stride=(stride, stride),
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=(dilation, dilation))
~~~

3*3卷积用来`提取特征`，进行`下采样`

如果步长为1的话，宽高不变。
$$
\frac{W-F+2P}{S}+1=\frac{W-3+2}{1}+1=W
$$
如果步长为2的话，宽高直接变成1/2.类似于`下采样`
$$
\frac{W-F+2P}{S}+1=\frac{W-3+2}{2}+1=\frac{W}{2}
$$




### 3.3 两种残差块

官方实现的ResNet中

* ResNet18,Resnet34 使用的普通的`Basicblock`
* ResNet50,ResNet101,ResNet152使用的都是`Bottleneck`瓶颈结构

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20201112210055437.png)

#### 3.1.1 BasicBlock

~~~python
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
~~~





#### 3.1.2 Bottleneck

这样子设计的理由

> 在resnet50以后，由于层数的增加残差块发生了变化，从原来3x3卷积变为三层卷积，卷积核分别为1x1、3x3、1x1，`减少了网络参数`。主要通过两种方式：1.用zero-padding去增加维度 2.用1x1卷积来增加维度



Bottleneck 还有==两种结构==

* 一种是输入的x进行了卷积后的out和残差identity 相加

  ![image-20220404111950904](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220404111950904.png)

* 一种是输入的x进行了卷积后的out和 对残差identity 进行下采样后,进行相加

  ![image-20220404111928585](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220404111928585.png)



这两种不同的连接结构对应代码位置不同的部分就是==downsample==,这个参数

~~~python
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

~~~



#### 3.1.3 Bottleneck 使用细节

与基础版的不同之处只在于这里是三个卷积，分别是1x1,3x3,1x1,分别用来压缩维度，卷积处理，恢复维度，inplane是输入的通道数，plane是输出的通道数，expansion是对输出通道数的倍乘，在basic中expansion是1，此时完全忽略expansion这个东东，输出的通道数就是plane，然而bottleneck就是不走寻常路，它的任务就是要对通道数进行压缩，再放大，于是，plane不再代表输出的通道数，而是block内部压缩后的通道数，输出通道数变为plane*expansion。接着就是网络主体了。

## 4. 代码

### 4.1 ResNet18 实现手写数字识别

#### 创建模型

~~~python
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.utils import data  # 获取迭代数据
from torch.autograd import Variable  # 获取变量
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import mnist  # 获取数据集
import matplotlib.pyplot as plt
from torch import nn

import os


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=(3, 3),
                     stride=(stride, stride),
                     padding=1, bias=False)


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """
         小模块
        :param ch_in:输入通道
        :param ch_out: 输出通道
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """

        :param x: [batch_size, channel, height, weight]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module:[b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(3, 3), padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks

        # [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = ResBlk(64, 128, stride=2)

        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)

        # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)

        # [b, 512, h, w] => [b, 512, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        print(x)
        # [b, 1, h, w] => [b, 64, h, w]
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 512, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print(x.shape) # [b, 512, 1, 1]
        # 意思就是不管之前的特征图尺寸为多少，只要设置为(1,1)，那么最终特征图大小都为(1,1)
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x

~~~

#### 加载数据

~~~python
path = r"./model"
if not os.path.exists(path):
    os.mkdir(path)


def get_dataloader(mode):
    """
    获取数据集加载
    :param mode:
    :return:
    """
    #准备数据迭代器
    # 这里我已经下载好了，所以是否需要下载写的是false
    #准备数据集，其中0.1307，0.3081为MNIST数据的均值和标准差，这样操作能够对其进行标准化
    #因为MNIST只有一个通道（黑白图片）,所以元组中只有一个值
    dataset = torchvision.datasets.MNIST('../../data/mini', train=mode,
                                         download=False,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))

    return DataLoader(dataset, batch_size=64, shuffle=True)
~~~



#### 进行训练和测试

```python
def train(epoch):
    loss_count = []
    # 获取训练集
    train_loader = get_dataloader(True)
    print("训练集的长度", len(train_loader))
    for i, (x, y) in enumerate(train_loader):
        # 通道数是1 ,28*28的灰度图,batch_size=64
        batch_x = Variable(x)  # torch.Size([batch_size, 1, 28, 28])
        batch_y = Variable(y)  # torch.Size([batch_size])
        # 获取最后输出
        out = model(batch_x)  # torch.Size([batch_size,10])
        # 获取损失
        loss = loss_func(out, batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        opt.step()  # 将参数更新值施加到net的parmeters上
        if i % 200 == 0:
            loss_count.append(loss.item())
            print('训练次数{}---{}:\t--损失值{}'.format(
                epoch,
                i, loss.item()))
            # 保存训练模型，以便下次使用

            torch.save(model.state_dict(), r'./model/resnet_model.pkl')
    # 打印测试诗句
    # print(loss_count)
    plt.figure('PyTorch_CNN_的损失值')
    plt.plot(range(len(loss_count)), loss_count, label='Loss')
    plt.title('PyTorch_CNN_的损失值')
    plt.legend()
    plt.show()


def test():
    # 获取测试集
    accuracy_sum = []
    test_loader = get_dataloader(False)
    for index, (a, b) in enumerate(test_loader):
        test_x = Variable(a)
        test_y = Variable(b)
        out = model(test_x)
        accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
        accuracy_sum.append(accuracy.mean())
        if index % 100 == 0:
            print('测试了100批次准确率为:\t', accuracy.mean())

    print('总准确率：\t', sum(accuracy_sum) / len(accuracy_sum))
    # 精确率图
    plt.figure('Accuracy')
    print(accuracy_sum)
    plt.plot(range(len(accuracy_sum)), accuracy_sum, 'o', label='accuracy')
    plt.title('Pytorch_CNN_准确率')
    plt.legend()
    plt.show()


for epoch in range(3):
    train(epoch)
    test()
```



### 4.2 ResNet50 实现

~~~python
import torch
from thop import profile
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=(3, 3),
                     stride=(stride, stride),
                     padding=dilation,
                     groups=groups,
                     bias=False, dilation=(dilation, dilation))


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=(1, 1),
                     stride=(stride, stride),
                     bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        # 如果下采样的话，步长就变成了2
        stride = 2 if down_sample else 1

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
        mid_channels = out_channels // 4

        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride, groups)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = conv1x1(mid_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:

        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        # 假设输入x=[batch_size,3,224,224]
        # [batch_size,3,224,224] -> [batch_size,64,112,112]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)  # [3,112,112]

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化 # [batch_size,64,112,112] -> [batch_size,64,56,56]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 输入 x=[batch_size,64,56,56] -> [batch_size,256,56,56]
        self.layer1 = self._make_layer(block, 64, 256, layers[0])

        # x=[batch_size,256,56,56] -> [batch_size,512,28,28]
        self.layer2 = self._make_layer(block, 256, 512, layers[1], down_sample=True)

        # x=[batch_size,512,28,28] -> [batch_size,1024,14,14]
        self.layer3 = self._make_layer(block, 512, 1024, layers[2], down_sample=True)

        # x=[batch_size,1024,14,14] -> [batch_size,2048,7,7 ]
        self.layer4 = self._make_layer(block, 1024, 2048, layers[3], down_sample=True)

        # [batch_size,2048,7,7 ] -> [batch_size,2048,1,1 ]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # [batch_size,2048,7,7 ] -> [batch_size,num_class,1,1 ]
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, down_sample=False):
        layers = []
        #  第一个模块，进行下采样
        layers.append(block(in_channels, out_channels, down_sample))

        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
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
        x = self.avgpool(x)
        # --------------------------------------#
        # 按照x的第1个维度拼接（按照列来拼接，横向拼接）
        # 拼接之后,张量的shape为(batch_size,2048)
        # --------------------------------------#
        x = torch.flatten(x, 1)
        # --------------------------------------#
        # 过全连接层来调整特征通道数
        # (batch_size,2048)->(batch_size,1000)
        # --------------------------------------#
        x = self.fc(x)
        return x


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


if __name__ == '__main__':
    from thop import profile

    model = resnet50()
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))

~~~

## 5. 自己常见问题解答



参考资料

> https://blog.csdn.net/weixin_43593330/article/details/107620042