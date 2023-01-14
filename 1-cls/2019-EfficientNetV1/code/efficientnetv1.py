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
