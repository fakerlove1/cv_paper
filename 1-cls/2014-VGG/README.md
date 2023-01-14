# 2014-VGG

> 感谢我的研究生导师！！！
>
> [霹雳吧啦Wz的个人空间_哔哩哔哩_bilibili](https://space.bilibili.com/18161609)
>
> [跟李沐学AI的个人空间_哔哩哔哩_bilibili](https://space.bilibili.com/1567748478)



## 1. 简介

其名称来源于作者所在的`牛津大学视觉几何组`(Visual Geometry Group)的缩写。

VGG模型是2014年ILSVRC竞赛的`第二名`，第一名是GoogLeNet。而且，从图像中提取CNN特征，VGG模型是首选算法。它的缺点在于，参数量有140M之多，需要更大的存储空间。但是这个模型很有研究价值。

## 2. 网络

VGG是2014年被提出的，与之前的state-of-the-art的网络结构，错误率大幅下降，并取得了ILSVRC2014比赛分类项目的第二名和定位项目的第一名。同时，VGG的拓展性很强，迁移到其他图片数据上的泛化性非常好。VGG的结构简洁，整个网络都使用同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。到目前为止，VGG仍然被用来提取图像特征。



![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/aHR0cHM6Ly9zdGF0aWMubLzAzMTQvMDIyOTM5X1BsMTJfODc2MzU0LnBuZw.png)

* 1、结构简洁
  VGG由5层卷积层、3层全连接层、softmax输出层构成，层与层之间使用max-pooling（最大化池）分开，所有隐层的激活单元都采用ReLU函数。

* 2、小卷积核和多卷积子层

  VGG使用多个较小卷积核（3x3）的卷积层代替一个卷积核较大的卷积层，一方面可以减少参数，另一方面相当于进行了更多的非线性映射，可以增加网络的拟合/表达能力。
  小卷积核是VGG的一个重要特点，虽然VGG是在模仿AlexNet的网络结构，但没有采用AlexNet中比较大的卷积核尺寸（如7x7），而是通过降低卷积核的大小（3x3），增加卷积子层数来达到同样的性能（VGG：从1到4卷积子层，AlexNet：1子层）。

  VGG的作者认为两个3x3的卷积堆叠获得的感受野大小，相当一个5x5的卷积；而3个3x3卷积的堆叠获取到的感受野相当于一个7x7的卷积。

* 3、小池化核

  相比AlexNet的3x3的池化核，VGG全部采用2x2的池化核。

* 4、通道数多

  VGG网络第一层的通道数为64，后面每层都进行了翻倍，最多到512个通道，通道数的增加，使得更多的信息可以被提取出来。

* 5、层数更深、特征图更宽

  由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，控制了计算量的增加规模。

* 6、全连接转卷积（测试阶段）

  这也是VGG的一个特点，在网络测试阶段将训练阶段的三个全连接替换为三个卷积，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高为的输入，这在测试阶段很重要。
  如本节第一个图所示，输入图像是224x224x3，如果后面三个层都是全连接，那么在测试阶段就只能将测试的图像全部都要缩放大小到224x224x3，才能符合后面全连接层的输入数量要求，这样就不便于测试工作的开展。

VGG的论文中全部使用3x3的卷积核和2x2的池化核，通过不断加深网络结构来提升性能。下图为VGG各级别的网络结构图。



![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20170810151704458.png)

VGG拥有5段卷积，每段有多个卷积层，同时，每段结束都会连接一个最大池化层，池化层的作用是特征增强，同时缩小Feature Map的尺寸。在VGG网络中，只有C结构设置了1x1的卷积核，其余都是3x3的卷积，这种操作减小了参数量，论文中给出的参数量对比如下

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20170822095312457.png)



1、输入224x224x3的图片，经64个3x3的卷积核作两次卷积+ReLU，卷积后的尺寸变为224x224x64
2、作max pooling（最大化池化），池化单元尺寸为2x2（效果为图像尺寸减半），池化后的尺寸变为112x112x64
3、经128个3x3的卷积核作两次卷积+ReLU，尺寸变为112x112x128
4、作2x2的max pooling池化，尺寸变为56x56x128
5、经256个3x3的卷积核作三次卷积+ReLU，尺寸变为56x56x256
6、作2x2的max pooling池化，尺寸变为28x28x256
7、经512个3x3的卷积核作三次卷积+ReLU，尺寸变为28x28x512
8、作2x2的max pooling池化，尺寸变为14x14x512
9、经512个3x3的卷积核作三次卷积+ReLU，尺寸变为14x14x512
10、作2x2的max pooling池化，尺寸变为7x7x512
11、与两层1x1x4096，一层1x1x1000进行全连接+ReLU（共三层）
12、通过softmax输出1000个预测结果
从上面的过程可以看出VGG网络结构还是挺简洁的，都是由小卷积核、小池化核、ReLU组合而成。其简化图如下（以VGG16为例）：



![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/aHR0cHM6Ly9zdGF0aWMub3NjaGluYS5uZXQvdXBsb2Fkcy9zcGFjZS8yMDE4LzAzMTQvMDIzMTExX0dHOWtfODc2MzU0LnBuZw.png)



## 3. 代码

数据集准备

[CIFAR-10 和 CIFAR-100 数据集 (toronto.edu)](https://www.cs.toronto.edu/~kriz/cifar.html)

下完完毕后，解压数据为

![image-20220409145814908](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220409145814908.png)



### 加载数据

~~~python

import pickle

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    dict_name=unpickle(r"E:\note\cv\data\cifar-10-batches-py\data_batch_1")
    images=dict_name[b'data']
    print(images.shape)

    labels=dict_name[b'labels']
    print(len(labels))
    for i in dict_name:
        print(i)

~~~

结果如下

~~~bash
(10000, 3072)
10000
b'batch_label'
b'labels'
b'data'
b'filenames'
~~~



现在呢？？需要使用pytorch 的dataset。

~~~python
import torch.nn as nn
import torch
import numpy as np

import pickle

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_train_data(batch_size):
    """
    因为一共10000条数据,所以手写一个按批次读取
    Args:
        batch_size:
    Returns:

    """
    dict_name = unpickle(r"E:\note\cv\data\cifar-10-batches-py\data_batch_1")
    data = dict_name[b'data']
    labels = dict_name[b'labels']
    length = len(labels)
    for i in range(int(length / batch_size)):
        # 如果最后一组数据 小于batch_size数目,则返回最后所有剩余数据
        start = batch_size * i
        end = batch_size * (i + 1)
        if batch_size * (i + 1) > length:
            end = length
        images = torch.FloatTensor(np.array([np.array(x).reshape((3, 32, 32)) for x in data[start:end]]))
        targets = torch.LongTensor(labels[start:end])
        yield images, targets


if __name__ == '__main__':
    for images, targets in get_train_data(10):
        print(images.shape)
        print(targets)

~~~

结果为

![image-20220409164841282](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220409164841282.png)

### 创建分类网络

~~~python
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
    data = torch.randn(size=(8, 3, 224, 224))
    model = vgg(num_classes=10)
    out = model(data)
    print(out.shape)

~~~



### 训练

因为cidar的数据集为$[3\times 32\times 32]$. 维数不够进行vgg16 进行拆分的。所以要对vgg16模型进行修改只做。

~~~python
import os.path
import pickle

import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.utils import data  # 获取迭代数据
from torch.autograd import Variable  # 获取变量
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import mnist  # 获取数据集
import matplotlib.pyplot as plt
import math


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(block, self).__init__()
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
        self.conv1 = block(3, 64)
        # 最大池化只做宽高的下采样
        self.mpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = block(64, 128)
        self.mpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = block(128, 256)
        self.mpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = block3(256, 512)
        self.mpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = block3(512, 512)
        self.mpool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        self.classifier = nn.Sequential(  # 最后的三层全连接层 （分类网络结构）
            nn.Dropout(p=0.5),  # 与全连接层连接之前，先展平为1维，为了减少过拟合进行dropout再与全连接层进行连接（以0.5的比例随机失活神经元）
            nn.Linear(512 * 1 * 1, 128),  # 原论文中的节点个数是4096，这里简化为2048
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
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
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)  # 全连接层进行分类
        return x


def get_train_data(batch_size):
    """
    因为一共10000条数据,所以手写一个按批次读取
    Args:
        batch_size:
    Returns:

    """
    dict_name = unpickle(r"E:\note\cv\data\cifar-10-batches-py\data_batch_1")
    data = dict_name[b'data']
    labels = dict_name[b'labels']
    length = len(labels)
    for i in range(int(length / batch_size)):
        # 如果最后一组数据 小于batch_size数目,则返回最后所有剩余数据
        start = batch_size * i
        end = batch_size * (i + 1)
        if batch_size * (i + 1) > length:
            end = length
        images = torch.FloatTensor(np.array([np.array(x).reshape((3, 32, 32)) for x in data[start:end]]))
        targets = torch.LongTensor(labels[start:end])
        yield images, targets


def get_test_data(batch_size):
    """
    因为一共10000条数据,所以手写一个按批次读取
    Args:
        batch_size:
    Returns:

    """
    dict_name = unpickle(r"E:\note\cv\data\cifar-10-batches-py\test_batch")
    data = dict_name[b'data']
    labels = dict_name[b'labels']
    length = len(labels)
    for i in range(int(length / batch_size)):
        # 如果最后一组数据 小于batch_size数目,则返回最后所有剩余数据
        start = batch_size * i
        end = batch_size * (i + 1)
        if batch_size * (i + 1) > length:
            end = length
        images = torch.FloatTensor(np.array([np.array(x).reshape((3, 32, 32)) for x in data[start:end]]))
        targets = torch.FloatTensor(labels[start:end])
        yield images, targets


model = vgg()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
loss_func = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    loss_count = []
    # 获取训练集
    for i, (x, y) in enumerate(get_train_data(100)):
        # 通道数是3 ,32,32
        batch_x = Variable(x)  # torch.Size([batch_size, 3 ,32,32])
        batch_y = Variable(y)  # torch.Size([batch_size])
        # 获取最后输出
        out = model(batch_x)  # torch.Size([batch_size,10])
        # 获取损失
        loss = loss_func(out, batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        opt.step()  # 将参数更新值施加到net的parmeters上
        if i % 20 == 0:
            loss_count.append(loss.item())
            print('训练次数{}---{}:\t--损失值{}'.format(
                epoch,
                i, loss.item()))
            # 保存训练模型，以便下次使用

            torch.save(model.state_dict(), r'./model/vgg_model.pkl')
    # 打印测试诗句
    # print(loss_count)
    plt.figure('PyTorch_CNN_的损失值')
    plt.plot(range(len(loss_count)), loss_count, label='Loss')
    plt.title('PyTorch_CNN_的损失值')
    plt.legend()
    plt.show()


def test():
    model.eval()
    # 获取测试集
    accuracy_sum = []
    for index, (a, b) in enumerate(get_test_data(100)):
        test_x = Variable(a)
        test_y = Variable(b)
        out = model(test_x)
        accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
        accuracy_sum.append(accuracy.mean())
        if index % 10 == 0:
            print('第{}100批次准确率为{}:\t'.format(index, accuracy.mean()))

    print('总准确率：\t', sum(accuracy_sum) / len(accuracy_sum))
    # 精确率图
    plt.figure('Accuracy')
    print(accuracy_sum)
    plt.plot(range(len(accuracy_sum)), accuracy_sum, label='accuracy')
    plt.title('Pytorch_CNN_准确率')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if not os.path.exists("./model"):
        os.makedirs("./model")

    for i in range(3):
        train(i)
        test()

~~~

结果

~~~bash
总准确率：	 0.40440000000000004
总准确率：	 0.3553
总准确率：	 0.40440000000000004
~~~

我只训练了epoch3 ，

## 4. 别人优化后的代码

~~~python
import torch.nn as nn
import torch
 
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):   # features: 由make_features生成的提取特征网络结构
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(                    # 最后的三层全连接层 （分类网络结构）
            nn.Dropout(p=0.5),                              # 与全连接层连接之前，先展平为1维，为了减少过拟合进行dropout再与全连接层进行连接（以0.5的比例随机失活神经元）
            nn.Linear(512*7*7, 2048),                       # 原论文中的节点个数是4096，这里简化为2048
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()
 
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)                 # 进入卷积层提取特征
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)    # 展平（第0个维度是batch，所以从第一个维度展平）
        # N x 512*7*7
        x = self.classifier(x)               # 全连接层进行分类
        return x
 
    def _initialize_weights(self):                  # 初始化权重
        for m in self.modules():                    # 遍历网络的每一层
            if isinstance(m, nn.Conv2d):            # 如果当前层是卷积层
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)   # 初始化卷积核的权重
                if m.bias is not None:              # 如果采用了bias，则将bias初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):          # 当前层是全连接层
                nn.init.xavier_uniform_(m.weight)   # 初始化全连接层的权重
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
 
 
# 生成提取特征网络结构
def make_features(cfg: list):  # 传入含有网络信息的列表
    layers = []
    in_channels = 3   # R G B
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)   # 将列表通过非关键字参数的形式传入
 
 
cfgs = {
 
    # 卷积核大小3*3
    # 数字表示卷积核个数，‘M’表示maxpooling
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
 
 
# 实例化VGG网络
def vgg(model_name="vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model
~~~



参考链接

[VGG16模型_王小波_Libo的博客-CSDN博客_vgg16模型](https://blog.csdn.net/qq_38900441/article/details/104631287)