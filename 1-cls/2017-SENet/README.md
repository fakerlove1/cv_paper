# 【图像分类】2017-SENet CVPR



> 论文题目：Squeeze-and-Excitation Networks
>
> 论文地址: [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)
>
> 论文代码：[https://github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet)
>
> 发表时间：2017年9月
>
> 引用：Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141.
>
> 引用数：14421

## 1. 简介

### 1.1 简介

WMW以极大的优势获得了最后一届 ImageNet 2017 竞赛 `Image Classification` 任务的冠军，

SENet则主要关注通道上可做点，通过显示的对卷积层特征之间的通道相关性进行建模来提升模型的表征能力；并以此提出了特征重校准机制：通过使用全局信息去选择性的增强可信息化的特征并同时压缩那些无用的特征。

`SE 模块`可以嵌入到现在几乎所有的`网络结构`中。通过在原始网络结构的 building block 单元中嵌入 SE 模块，我们可以获得不同种类的 SENet。如 SE-BN-Inception、SE-ResNet、SE-ReNeXt、SE-Inception-ResNet-v2 等等。



## 2. 网络

### 2.1 SE模块

即插即用的模块

![image-20220606154958715](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220606154958715.png)

如上图所示是 SE 注意力机制模块的网络架构图，

1. 为了获得在通道维度上的注意力，特征图输入后，先通过基于特征图的宽度和高度进行`全局平均池化`，使空间特征降维到 1×1，如公式 1 所示。
   $$
   Z_c=F_{sq}(u_c)=\frac{1}{H\times W}\sum_{i=1}^H\sum_{j=1}^W u_c(i,j)
   $$
   
2. 紧接着使用两个全连接层和非线性激活函数建立通道间的连接，如公式 2 所示。
   $$
   \hat{z}=T_2(ReLU(T_1(z)))
   $$
   
3. 然后经过 Sigmoid 激活函数获得归一化权重，最后通过`乘法逐通道加权到原始特征图的每一个通道上`，完成通道注意力对原始特征的重新标定。如公式如下所示。

$$
\hat{X}=X\cdot \sigma(\hat{z})
$$

经过全局平均池化，可以获得全局的感受野，在第一次全连接时通过减少特征图的维度，大大减少了参数和计算量，之后经过非线性激活函数后再通过一个全连接恢复到原来的通道数，完成了通道间相关性的建立。

## 3. 代码

~~~python
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, mode, channels, ratio):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False),
        )
		self.sigmoid = nn.Sigmoid()
     
    
    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v

if __name__ == "__main__":
    model = SEBlock("max", 54, 9)
    feature_maps = torch.randn((8, 54, 32, 32))
    model(feature_maps)

~~~



参考资料

> https://blog.csdn.net/weixin_43913124/article/details/123113339