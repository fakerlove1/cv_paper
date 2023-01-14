# 2017-DenseNet CVPR

>论文链接：[https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
>代码的github链接：[https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet)
>
>感谢我的研究生导师！！！
>
>[霹雳吧啦Wz的个人空间_哔哩哔哩_bilibili](https://space.bilibili.com/18161609)
>
>[跟李沐学AI的个人空间_哔哩哔哩_bilibili](https://space.bilibili.com/1567748478)



## 1. 简介

文章是CVPR2017的oral。论文中提出的DenseNet主要还是和ResNet及Inception网络做对比，思想上有借鉴，但却是全新的结构，网络结构并不复杂，却非常有效，在CIFAR指标上全面超越ResNet。

DenseNet`脱离了加深网络层数(ResNet)`和`加宽网络结构(Inception)`来提升网络性能的定式思维,从特征的角度考虑,通过**特征重用和旁路(Bypass)设置**,既大幅度减少了网络的参数量,

DenseNet论文中本身引入太多公式，所以仅对其进行总结。作者则是从feature入手，通过对feature的极致利用达到更好的效果和更少的参数。



**特点**

(1).减轻梯度消失(vanishing-gradient)。

(2).加强feature传递。

(3).鼓励特征重用(encourage feature reuse)。

(4).较少的参数数量。

## 2. 网络

### 2.1 稠密连接

基础结构为Res Block在每一个Dense Block中，任何两层之间都有直接的连接。通过密集连接，缓解梯度消失问题，加强特征传播，鼓励特征复用，极大的减少了参数量。

看下图可以看到，

> * ResNet是每个层与前面的某层（一般是2~3层）短路连接在一起，连接方式是通过元素级相加。
> * 而在DenseNet中，每个层都会与前面所有层在channel维度上连接（concat）在一起（这里各个层的特征图大小是相同的，后面会有说明），并作为下一层的输入。
> * 对于一个$L$层的网络，DenseNet共包含$\frac{L(L+1)}{2}$个连接，相比ResNet，这是一种密集连接。而且DenseNet是直接concat来自不同层的特征图，这可以实现特征重用，提升效率，这一特点是DenseNet与ResNet最主要的区别。

![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/521868268f5a4cb98e506a542d33b0b2.png)

![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/7dd99e2fc39048dcb3069f161d94b0a5.png)

> 图1为ResNet网络的短路连接机制（其中+代表的是元素级相加操作），作为对比，图2为DenseNet网络的密集连接机制（其中c代表的是channel级连接操作）。



如果用公式表示的话，传统的网络在$L$层的输出为

$x_l=H_l(x_{l-1})$

而对于ResNet，增加了来自上一层输入的identity函数：

$x_l=H_l(x_{l-1})+x_{l-1}$

在DenseNet中，会连接前面所有层作为输入：

$x_l=H_l[x_0,x_1,\cdots,x_{l-1}]$

其中，上面的$H_l(\cdot)$代表是非线性转化函数（non-linear transformation），它是一个组合操作，其可能包括一系列的BN(Batch Normalization)，ReLU，Pooling及Conv操作。注意这里的$l$层与$l-1$层之间可能实际上包括多个卷积层。

DenseNet的前向过程如图3所示，可以更直观地理解其密集连接方式，比如$h_3$的输入不仅包括来自$h_2$的$x_2$，还包括前面两层的$x_1$和$x_2$,它们是在channel维度上连接在一起的。

![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/08547e3b4312480aa24843ef0f5ced51.png)



### 2.2 Transition

CNN网络一般要经过Pooling或者stride>1的Conv来降低特征图的大小，而DenseNet的密集连接方式需要特征图大小保持一致。为了解决这个问题，DenseNet网络中使用DenseBlock+Transition的结构，其中DenseBlock是包含很多层的模块，每个层的特征图大小相同，层与层之间采用密集连接方式。而Transition模块是连接两个相邻的DenseBlock，并且通过Pooling使特征图大小降低。图4给出了DenseNet的网路结构，它共包含4个DenseBlock，各个DenseBlock之间通过Transition连接在一起。

![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/a92c458f5de746deb4f224e493421bb9.png)

### 2.3 总体架构

主要包含DenseBlock和transition layer两个组成模块。其中Dense Block为稠密连接的highway的模块，transition layer为相邻2个Dense Block中的那部分。

![在这里插入图片描述](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/e2a2cf482a5f4070b58061400f80cd2d.png)







## 3. 代码



~~~python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    # from torchsummary import summary
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = convnext_tiny(num_classes=5)
    # model.to(device)
    # print(model)
    # x = torch.randn(1,3,224,224,device=device)
    # y = model(x)
    # print(y)

    # summary(model, input_size=(3, 224, 224))
    # from torchvision.models import
    from thop import profile
    model =DenseNet(num_classes=1000)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops/1e9))
    print("params:{:.3f}M".format(params/1e6))
    from torchvision.models import  densenet121
~~~









参考资料

> [DenseNet算法详解-ResNet升级版_alex1801的博客-CSDN博客_densenet改进](https://blog.csdn.net/weixin_34910922/article/details/107435714)
>
> [深度学习入门笔记之DenseNet网络_ysukitty的博客-CSDN博客_densenet网络](https://blog.csdn.net/ysukitty/article/details/123532318)