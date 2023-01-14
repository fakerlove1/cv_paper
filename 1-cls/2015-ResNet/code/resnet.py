import torch
from thop import profile
import torch.nn as nn
from torch import Tensor
from torchvision.models import mobilenetv2

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


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes)


def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3],num_classes)


# if __name__ == '__main__':
#     from thop import profile

#     model = resnet50(num_classes=1000)
#     # print(model)
#     input = torch.randn(1, 3, 224, 224)
#     flops, params = profile(model, inputs=(input,))
#     print("flops:{:.3f}G".format(flops / 1e9))
#     print("params:{:.3f}M".format(params / 1e6))
