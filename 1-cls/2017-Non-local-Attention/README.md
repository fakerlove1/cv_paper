# 2018-Non-local CVPR

基于自相似的非局域注意力机制

> 论文地址: [https://arxiv.org/abs/1711.07971](https://arxiv.org/abs/1711.07971)
>
> 代码地址: [https://github.com/facebookresearch/video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net)

## 1. 简介

### 1.1 简介

Non-Local是王小龙在`2018`年提出的一个`自注意力`模型。

Non-Local Neural Network和Non-Local Means非局部均值去燥滤波有点相似的感觉。普通的滤波都是3×3的卷积核，然后在整个图片上进行移动，处理的是3×3局部的信息。Non-Local Means操作则是结合了一个比较大的搜索范围，并进行加权。

卷积操作`只关注于局部的感受野`，是典型的`local operation`，如果要增大神经元的感受野，一般是`堆叠卷积层和池化层`来实现，但是`这样计算量和复杂度`都会增加，并且feature map的尺寸会较小。为了突破，作者借鉴图像去噪这个领域的`non-local`操作，提出了non-local neural network，`用来捕捉长距离像素之间的信息`，最终实现每个像素的全局感受野。并且通过不同的操作，还可以得到不同空间、时间、通道的像素之间的信息相关性。


创新点

- 通过少的参数，少的层数，捕获远距离的依赖关系；
- 即插即用



### 1.2 背景

以往论文对视频分类往往是基于本地邻域(local neighborhood)进行一系列的操作，这些方法往往将一些虽然时空位置上相距较远但之间具有紧密联系的一些事物忽略掉。于是诞生了一个Non-local 的方法。这个方法高效，简单，通用，可以应用在多种模型结构上。



## 2. 网络

个人感觉 Non-local其实就是self-attention的一种变体，可以理解为Transformer结构的一小部分

### 2.1 operation

按照非局部均值的定义，我们定义在深度神经网络中的non-local操作如下：
$$
y_i=\frac{1}{C(x)}\sum_{\forall j}f(x_i,x_j)g(x_j)
$$
$i$是输出位置的索引（可能是空间，时间，或者时空，例如第8帧的[32,46]位置上）

$j$是所有需要与$i$计算的位置索引

$x$是输入的信号（它可以是图像，序列，视频，通常都是它们的特征）

$y$则是与$x$大小一致的输出信号

$f(x_i,x_j)$是一个二元函数，其函数生成的值为一个标量，此标量用来表示$x_i,x_j$之间的近似关系(比如$f(x_i,x_j)$值很大，则$x_i,x_j$关系非常密切)

$g(x_j)$是对$x_j$位置的一个加权结果。(在此节，它被简单地设置为$g(x_j)=W^T_gx_j$)

$C(x)$是一个归一化函数，用于将结果归一化处理，他的形式与$f(x_i,x_j)$有关。



首先探讨一下对于$f(x_i,x_j)$选取

**1.Gaussian函数**
$$
f(x_i,x_j)=e^{x_i^Tx_j}
$$


在这里简单说明为何使用Gaussian函数。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-9b15ad9c51c4a61e58cc7059e53fe6a6_720w.jpg)

可以看出在距离中心点越近的区域其值越大，利用这个特性，在比较两个特征的时候，其特征越相似，那么这两个特征所产生的输出值越大。也就是说二元高斯函数本质上是计算两个点的距离远近程度，而在这里将距离换为特征值，由此体现两个特征之间的相似性。

**2.Embedded Gaussian**
$$
f(x_i,x_j)=e^{\theta(x_i)^T\phi(x_j)}
$$


其中$\theta(x_i)=W_\theta x_i,\phi(x_j)=W_{\phi}x_j$实际上是在高斯函数的基础上，对两个参数增加一个权重。



**3.Dot-Product**
$$
f(x-i,x_j)=\theta(x_i)^T\phi(x_j)
$$
采用简单的点积相乘的形式，在一定程度上也可以反应两个特征的相似度。且$C(x)=N$



**4.Concatenation**
$$
f(x_i,x_j)=Relu(W^T_f[\theta(x_i),\phi(x_j)])
$$
$[\theta(x_i),\phi(x_j)]$表示两条向量拼接，此处拼接后依然是一维的向量。之后对加权后的向量使用Relu。



### 2.2 Non-local Block

将上式中的non-local操作变形成一个non-local block，以便其可以被插入到已有的结构中。 定义一个non-local block为：
$$
Z_i=W_zy_i+X_i
$$
其中yi已经在(1)式中给出了，+xi则表示的是一个residual connection。residual connection的结构使得我们可以在任意的pretrain模型中插入一个新的non-local block而不需要改变其原有的结构。给出non-local block的示例


![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/35df2d701b32ad221136dc34a23010be.png)

上图是论文中对于non-local的结构图。可以看到，先通过$1\times 1$的卷积，降低一下通道数，然后通过$\theta$和$\phi$分别是query和key，然后这两个卷积得到$(N，N)$的矩阵，然后再与$g(value)$进行矩阵乘法。


输入特征图为$[Batch\ Channel\ Height\ Width]$，我们先把这个输入特征图$x$分别放入：

- query卷积层-，得到$Batch\times Channel//8 \times Height\times Width$
- key卷积层，得到$Batch\times Channel//8 \times HeightxWidth$
- value卷积层，得到$Batch\times Channel\times Height\times Width$



**细节图**

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20200219181248416.png)

feature map size为temporal, h, w, size。 通过三个1x1卷积减少一半的通道后（有助于减少计算量）进行reshape，再进行embedded gaussian操作后得到feature map， 通过1x1的输出卷积层恢复通道数量后，最后进行跳连作为最终的输出。


## 3. 代码

代码都是抄过来的。

### 代码

~~~python
import torch
import torch.nn as nn
import torchvision


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


if __name__=='__main__':
    model = NonLocalBlock(channel=16)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)

~~~



代码2

~~~python
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
 
        out = self.gamma*out + x
        return out,attention

~~~



代码3

如果想要减少计算量可以添加下采样pooling或者减少通道，此外我这里使用了wn的方法，没有加BN，是因为应用在超分(SR)网络中

~~~python
# Non-local module
class Nl(nn.Module):
    def __init__(self, wn, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(Nl, self).__init__()
        # self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels

        self.f_key = wn(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                                  kernel_size=1, stride=1, padding=0))
        self.f_query = self.f_key
        self.f_value = wn(nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                    kernel_size=1, stride=1, padding=0))
        self.W = wn(nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                              kernel_size=1, stride=1, padding=0))
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        # sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        context += x
        return context

~~~



### 缺点

Non-Local模块是一种特别有用的`语义分割`技术，但也因其难以进行计算和占用GPU内存而受到批评。

个人想法

> non-local NN， non-local block = transformer layer (single head) - FFN - positional encoding啊！
>
> 证明了FFN(即MLP)的重要性。这可能解释了为什么几层Non-local layer叠起来提升不大。positional encoding对分割任务是有提升的。另外很多transformer for cv的paper都证明multi head表现的比single head更好。

参考

> [详解Non-local Neural Networks:非本地网络 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/340493529)
>
> [(60 封私信 / 1 条消息) 在计算机视觉任务中，运用Transformer和Non-local有何本质区别？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/427061585/answer/1541669750)
>
> [论文阅读|Non-local Neural Networks非局部操作self-attention - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/363314228)