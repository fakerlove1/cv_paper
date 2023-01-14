import torch.nn as nn
import torch


class block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(block2, self).__init__()
        # 两个卷积层,只做通道上的融合,
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode="reflect",
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode="reflect",
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class block3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(block3, self).__init__()
        # 两个卷积层,只做通道上的融合,
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode="reflect",
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode="reflect",
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode="reflect",
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class vgg(nn.Module):
    def __init__(self, num_classes=10):
        super(vgg, self).__init__()
        self.conv1 = block2(3, 64)
        # 最大池化只做宽高的下采样
        self.mpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = block2(64, 128)
        self.mpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = block3(128, 256)
        self.mpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = block3(256, 512)
        self.mpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = block3(512, 512)
        self.mpool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.classifier = nn.Sequential(  # 最后的三层全连接层 （分类网络结构）
            nn.Dropout(p=0.5),  # 与全连接层连接之前，先展平为1维，为了减少过拟合进行dropout再与全连接层进行连接（以0.5的比例随机失活神经元）
            nn.Linear(512 * 7 * 7, 2048),  # 原论文中的节点个数是4096，这里简化为2048
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.mpool1(x)
        x = self.conv2(x)
        x = self.mpool2(x)
        x = self.conv3(x)
        x = self.mpool3(x)
        x = self.conv4(x)
        x = self.mpool4(x)
        x = self.mpool5(self.conv5(x))
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)  # 全连接层进行分类
        return x


if __name__ == '__main__':
    """
    这里我们进行一下测试,样式是不是对的
    """
    data = torch.randn(size=(1, 3, 224, 224))
    model = vgg(num_classes=10)
    out = model(data)
    print(out.shape)
