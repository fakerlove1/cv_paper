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
