import torch
import torch.nn as nn
from timm.models.layers import DropPath

"""
代码来自https://github.com/NNDEV1/RepLKNet
没有重参数
"""


class DepthWiseConv2d(nn.Module):

    def __init__(self, c_in, c_out, kernels_per_layer, kernel_size, stride,padding=0):
        super(DepthWiseConv2d, self).__init__()

        self.conv_dw = nn.Conv2d(c_in, c_in * kernels_per_layer, kernel_size=kernel_size, padding=1, groups=c_in,
                                 bias=False,)
        self.conv_pw = nn.Conv2d(c_in * kernels_per_layer, c_out, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        return self.conv_pw(self.conv_dw(x))


class Stem(nn.Module):

    def __init__(self, c_in, c_out):
        super(Stem, self).__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, bias=False)
        self.conv_dw1 = DepthWiseConv2d(c_out, c_out, kernels_per_layer=8, stride=1, kernel_size=3)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv_dw2 = DepthWiseConv2d(c_out, c_out, kernels_per_layer=8, stride=2, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv2(x)
        x = self.conv_dw2(x)

        return x


class Transition(nn.Module):

    def __init__(self, c_in, c_out, kernels_per_layer, kernel_size):
        super(Transition, self).__init__()

        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=1, stride=1, padding="same", bias=False)
        self.dw_conv = DepthWiseConv2d(c_in, c_out, kernels_per_layer, kernel_size, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv(x)

        return x


class RepLKBlock(nn.Module):

    def __init__(self, kernel_size, c_in, c_out, prob):
        super(RepLKBlock, self).__init__()

        # Only works for kernel sizes up to 9x9 5 1 7 2 9 3 11 4 13 5 15 6 17 7 19 8 21 9 23 10 25 11 27 12 29 13 31 14
        if kernel_size <= 9:
            padding = kernel_size // 3
        elif kernel_size % 2 == 1:
            padding = kernel_size // 2 - 1
        # print(kernel_size)
        self.bn = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False)
        self.conv_dw = DepthWiseConv2d(c_out, c_out, kernel_size=kernel_size, stride=1, kernels_per_layer=8)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=padding, bias=False)
        self.drop_path = DropPath(prob)

    def forward(self, x):
        add = x
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv_dw(x)
        x = self.conv2(x)
        return self.drop_path(x) + add


class ConvFFN(nn.Module):

    def __init__(self, c_in, c_out, prob):
        super(ConvFFN, self).__init__()

        self.bn = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding="same", stride=1, bias=False)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding="same", stride=1, bias=False)
        self.drop_path = DropPath(prob)

    def forward(self, x):
        add = x
        x = self.bn(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return self.drop_path(x) + add


# Stem
# Stage 1: (RepLKBlock, ConvFFN, ..., RepLKBlock, ConvFFN)
# Transition 1
# Stage 2: (RepLKBlock, ConvFFN, ..., RepLKBlock, ConvFFN)
# Transition 2
# Stage 3: (RepLKBlock, ConvFFN, ..., RepLKBlock, ConvFFN)
# Transition 3
# Stage 4: (RepLKBlock, ConvFFN, ..., RepLKBlock, ConvFFN)
# Transition 4

class RepLKNet(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 layers=[3, 3, 3, 3],
                 drop_path_rate=0.3,
                 kernel_sizes=[31, 29, 15, 7],
                 c_in=3,
                 channels=[32, 32, 32, 32]):
        super(RepLKNet, self).__init__()
        c_out = channels[0]
        self.stem = Stem(c_in, c_out)

        ################
        c_out = channels[0]
        modules1 = []
        for i in range(layers[0]):
            modules1.append(RepLKBlock(kernel_sizes[0], c_out, c_out, prob=drop_path_rate))
            modules1.append(ConvFFN(c_out, c_out, prob=drop_path_rate))
        #  stage 1
        self.stage1 = nn.Sequential(*modules1)
        self.transition1 = Transition(c_out, channels[1], kernels_per_layer=8, kernel_size=3)

        #####################
        c_out = channels[1]
        modules2 = []
        for i in range(layers[1]):
            modules2.append(RepLKBlock(kernel_sizes[1], c_out, c_out, prob=drop_path_rate))
            modules2.append(ConvFFN(c_out, c_out, prob=drop_path_rate))
        self.stage2 = nn.Sequential(*modules2)
        self.transition2 = Transition(c_out, channels[2], kernels_per_layer=16, kernel_size=3)

        #####################
        c_out = channels[2]
        modules3 = []
        for i in range(layers[2]):
            modules3.append(RepLKBlock(kernel_sizes[2], c_out, c_out, prob=drop_path_rate))
            modules3.append(ConvFFN(c_out, c_out, prob=drop_path_rate))
        self.stage3 = nn.Sequential(*modules3)
        self.transition3 = Transition(c_out, channels[3], kernels_per_layer=32, kernel_size=3)

        #####################
        c_out = channels[3]
        modules4 = []
        for i in range(layers[3]):
            modules4.append(RepLKBlock(kernel_sizes[3], c_out, c_out, prob=drop_path_rate))
            modules4.append(ConvFFN(c_out, c_out, prob=drop_path_rate))
        self.stage4 = nn.Sequential(*modules4)

        # self.transition4 = Transition(c_out, channels[3], kernels_per_layer=64, kernel_size=3)
        c_out = channels[3]

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(*[nn.Linear(c_out, c_out // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(c_out // 2, c_out // 4),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(c_out // 4, num_classes)])

    def forward(self, x):

        x = self.stem(x)  # 1/4 图
        #  [31, 29, 27, 13]
        x = self.stage1(x)
        x = self.transition1(x)  # 1/8 图
        x = self.stage2(x) # torch.Size([1, 256, 28, 28])
        # print(x.shape)
        x = self.transition2(x)  # 1/16 图
        x = self.stage3(x) # torch.Size([1, 512, 14, 14])
        x = self.transition3(x)  # # 1/32 图
        x = self.stage4(x)  # 1/32 图

        x = self.adaptive_pool(x)
        x = x.view((x.size(0), -1))
        x = self.fc(x)

        return x

def create_RepLKNet31Small(drop_path_rate=0.3, num_classes=1000, **kwargs):
    return RepLKNet(kernel_sizes=[31, 29, 15, 7],
                    layers=[2, 2, 18, 2],
                    channels=[128, 256, 512, 1024],
                    drop_path_rate=drop_path_rate,
                    num_classes=num_classes, **kwargs)

def create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000, **kwargs):
    return RepLKNet(kernel_sizes=[31, 29, 27, 13],
                    layers=[2, 2, 18, 2],
                    channels=[128, 256, 512, 1024],
                    drop_path_rate=drop_path_rate,
                    num_classes=num_classes, **kwargs)


def create_RepLKNet31L(drop_path_rate=0.3, num_classes=1000, **kwargs):
    return RepLKNet(kernel_sizes=[31, 29, 27, 13],
                    layers=[2, 2, 18, 2],
                    channels=[192, 384, 768, 1536],
                    drop_path_rate=drop_path_rate,
                    num_classes=num_classes, **kwargs)


def create_RepLKNetXL(drop_path_rate=0.3, num_classes=1000, **kwargs):
    return RepLKNet(kernel_sizes=[27, 27, 27, 13],
                    layers=[2, 2, 18, 2],
                    channels=[256, 512, 1024, 2048],
                    drop_path_rate=drop_path_rate,
                    num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    from thop import profile

    model = create_RepLKNet31Small(num_classes=1000)
    input = torch.randn(1, 3, 224, 224)
    #  使用B的模型时，1/16的特征图必须大于 27的kernel_size。所以 B 以上模型，必须 input = torch.randn(1, 3, 432,432)
    # model = create_RepLKNet31B(num_classes=1000)
    # input = torch.randn(1, 3, 512,512)
    y = model(input)
    print(y.shape)
    # flops, params = profile(model, inputs=(input,))
    # print("flops:{:.3f}G".format(flops / 1e9))
    # print("params:{:.3f}M".format(params / 1e6))
