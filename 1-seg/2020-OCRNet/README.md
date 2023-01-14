# 2020-OCRNet ECCV 

> 论文 题目：Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation
>
> 论文代码：[https://arxiv.org/abs/1909.11065](https://arxiv.org/abs/1909.11065)
>
> 代码链接：[https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)
>
> 发表时间：2019年9月
>
> 引用：Yuan Y, Chen X, Wang J. Object-contextual representations for semantic segmentation[C]//European conference on computer vision. Springer, Cham, 2020: 173-190.
>
> 引用数：557



## 1. 简介

### 1.1 简介

* `OCR`是MSRA和中科院的一篇`语义分割`工作，结合每一类的类别语义信息给`每个像素加权`，再和原始的pixel特征concat组成最终每个像素的特征表示，个人理解其是一个类似coarse-to-fine的语义分割过程。
* 目前cityscape的分割任务中，排名最高的还是HRNetv2+OCR，参考[paperswithcode](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes)
* 一句话概括一下 `HRNet-ocr`就是将`self-attention架构`加入了`HRNet`后。相当于HRNet的`语义分割版本`



### 1.2 前期知识储备

这个文章是以HRNet这个论文加以改进的，所以首先得先研究一下HRNet

HRNet: Deep High-Resolution Representation Learning for Visual Recognition, CVPR 2019

[论文地址](https://arxiv.org/abs/1902.09212)

> 当前的语义分割方法需要高分辨率特征，主流方法是通过一个网络得到 低分辨 feature map，然后通过上采样或反卷积 恢复 到高分辨率。



## 2. 网络

### 2.1 整体架构



![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210207175308322.png)

其中，粉红色虚线框内为形成的软对象区域（Soft Object Regions），紫色虚线框中为物体区域表示（Object Region Representations），橙色虚线框中为对象上下文表示和增强表示。


$$
y_i=\rho(\sum_{k=1}^Kw_{ik}\delta(f_k))
$$



> step1: 计算一个coarse的segmentation结果，即文中说的soft object region
> 实现过程：从backbone（ResNet或HRNet）最后的输出的FM，再接上一组conv操作，然后计算cross-entropy loss
> step2: 结合图像中的所有像素计算每个object region representation，即公式中的fk
> 实现过程：对上一步计算的soft object region求softmax，得到每个像素的类别信息，然后再和原始的pixel representation相乘
> step3: 利用object region representation和原始的pixel representation计算得到pixel-region relation，即得到公式中的wik
> 实现过程：将object region representation和pixel representation矩阵相乘，再求softmax
> step4: 计算最终每个像素的特征表示
> 实现过程：将step3的结果object region representation矩阵相乘，得到带有权重的每个像素的特征表示，并和原始的pixel representation连接到一起





### 2.1 第一步 

将上下文像素划分为`一组软对象区域`，每个soft object regions对应一个类，即从深度网络(backbone)计算得到的`粗软分割`（粗略的语义分割结果）。这种划分是在ground-truth分割的监督下学习的。根据网络中间层的特征表示估测粗略的语义分割结果作为 OCR 方法的一个输入，即结构图中`粉红色框内`的Soft Object Regions

~~~python
self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
 aux_out = self.aux_head(high_level_features) #soft object regions
 #high_level_features为backbone输出的粗略的高层特征结果

~~~

将backbone的输出结果经过`1*1`的卷积后输出`b×k×h×w`的张量作为软对象区域，其中，`k`为粗略分类后对象的类别数（eg：若有17个类别，则网络输出为：b×17×h×w）



### 2.2 第二步

 根据**粗略的语义分割结果**（soft object regions）和**网络最深层输出的像素特征**（Pixel Representations）表示计算出$K$组向量，即**物体区域表示**（Object Region Representations），其中每一个向量对应一个语义类别的特征表示

~~~python
 def forward(self, feats, probs):
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), \
            probs.size(3)

        # each class image now a vector
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)

        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context

~~~

其中，feats为网络最深层输出的像素特征，由backbone输出的粗略的高层特征结果经卷积处理得到。在调用此函数时，将aux-out作为实参传给probs。

~~~python
self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
 feats = self.conv3x3_ocr(high_level_features)

~~~

这段代码的功能如图

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/2021022715390395.png)

经上述处理，得到$b*c*k$的张量ocr-context，其中，b为batch-size，c为类别个数，每个类别可用一个大小为c的向量来描述。这个描述则为下文的注意力机制提供了参照信息（query）



### 2.3 第三步

1----计算**网络最深层输出的像素特征表示**（Pixel Representations）与计算得到的**物体区域特征表示**（Object Region Representation）之间的关系矩阵，然后根据每个像素和物体区域特征表示在关系矩阵中的数值把物体区域特征加权求和，得到最后的物体上下文特征表示 (Object Contextual Representation)，即**OCR**

~~~python
class ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature
                            maps (save memory cost)
    Return:
        N X C X H X W
    '''
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.in_channels),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear',
                                    align_corners=cfg.MODEL.ALIGN_CORNERS)

        return context

~~~

这段代码主要功能如图

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210207223853900.png)



通过对**网络最深层输出的像素特征表示**（Pixel Representations）与**物体区域特征表示**（Object Region Representation）进行卷积处理，获得query，key和value值，按照下述公式计算相似度得分，其中，dk取dk=key_channels。
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
这一步体现了注意力机制的应用，也是ocrnet性能提升的关键步骤

2--当把**物体上下文特征表示 OCR** 与**网络最深层输入的特征**表示拼接之后作为**上下文信息增强的特征表示**（Augmented Representation），可以基于增强后的特征表示预测每个像素的语义类别。



~~~python
class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale=1,
                 dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels,
                                                         key_channels,
                                                         scale)
        if cfg.MODEL.OCR_ASPP:
            self.aspp, aspp_out_ch = get_aspp(
                in_channels, bottleneck_ch=cfg.MODEL.ASPP_BOT_CH,
                output_stride=8)
            _in_channels = 2 * in_channels + aspp_out_ch
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0,
                      bias=False),
            BNReLU(out_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        if cfg.MODEL.OCR_ASPP:
            aspp = self.aspp(feats)
            output = self.conv_bn_dropout(torch.cat([context, aspp, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

~~~

将第三步第一小步中得到的context（物体上下文特征表示 OCR）与feats（网络最深层输出的像素特征）直接用cat拼接，得到上下文信息增强的特征表示。

综上，整个代码结构如下图所示

![](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/20210207225153785.png)



其中，代码中faets对应网络结构图中的**网络最深层输出的像素特征表示（Pixel Representations）**，aut_out对应**粗略的语义分割结果（soft object regions）**，ocr-context对应**物体区域表示（Object Region Representations）**





## 3. 代码







参考资料

> [OCRNet_Jumi爱笑笑的博客-CSDN博客_ocrnet](https://blog.csdn.net/weixin_39326879/article/details/121164489)
>
> [透过Transformer重新看OCRNet - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/391989204)
>
> [语义分割: 一文读懂 OCRNet_大林兄的博客-CSDN博客_ocrnet](https://blog.csdn.net/weixin_46142822/article/details/123887999)
>
> [【小白入门】超详细的OCRnet详解（含代码分析）_Alvarez的博客-CSDN博客_ocrnet](https://blog.csdn.net/alvarez/article/details/113744646)
>
> [ASPP详解 - CSDN](https://www.csdn.net/tags/OtTaYgwsNzY2MzAtYmxvZwO0O0OO0O0O.html)
>
> https://blog.csdn.net/Alvarez/article/details/113744646