# 【图像分类】2020-ViT ICLR

> 论文题目: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
>
> 论文链接: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
>
> 官方的代码: [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
>
> 别人写的代码: [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
>
> 视频讲解:[https://www.bilibili.com/video/BV15P4y137jb](https://www.bilibili.com/video/BV15P4y137jb) 讲的非常好，看了5遍！！！
>
> 发表时间:2020年10月
>
> 引用：Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[J]. arXiv preprint arXiv:2010.11929, 2020.
>
> 引用数：6017



## 1. 简介

### 1.1 简介



* ViT挑战了2012年AlexNet以来，在计算机视觉领域的绝对地位，

* 结论: 如果在`足够多的数据上去做预训练`，那我们也`不需要卷积神经网络`，直接使用标准的Transformer 就能把视觉任务做的很好。

  作者团队证明了脱离神经网络，使用一个`纯的transformer`结构也能在图像分类任务上表现的很好。

  甚至当我们在大规模数据集(google 的JTF300M或者Imagenet21K)上进行预训练然后在小数据集上进行微调时，它的表达效果甚至超过了传统的卷积神经网络。并且随着数据集的扩大vision transformer还`没有出现过饱和的现象。`



### 1.2 解决的问题

**ViT能解决的问题？ViT的功能**

> ViT可以解决CNN难以解决的问题，例如针对一些图片（如遮挡，纹理偏移，对抗贴图，分块排列组合等）

**Transformer用到视觉领域的问题**

> `Transformer是处理的是文本信息，是1维的。我们该如何使用Transformer处理图片信息(2维的)`
>
> Transformer是自己和自己做自注意力，两两做互动，需要的复杂度为$O(N^2)$。目前硬件设备所支持的大概在几百到一千。BERT模型中Transformer需要的参数是`512`。
>
> 想法1: 直接将2D的图片转换为1D的序列。一般图片的输入为$224\times 224$。如果直接把图片中的每个像素点当做单词来看。这里的$N=224\times 224=50176$。远远超过了目前硬件设备所能支持的长度。



**前人的工作：如何才能把自注意操作用到 计算机视觉领域里来呢？？？**

> 这些工作都在干一件事。就是图片的序列长度太长。现在的目标就是 `减少序列长度`,所以有一些前沿工作
>
> * 把网络的特征图当做Transformer的输入。(Non-local neural networks)
>
> * 使用轴注意力，对图片的高度，做一个自注意力操作，再对图片的宽度，做一个自注意力操作
> * 孤立注意力，一些是取局部的小窗口等。
>

**ViT是怎么处理的呢？？**

把图片打成patch,把一个图片的$16\times 16$的当成一个元素。

直接对图片进行1/16的下采样。




## 2. 网络

### 2.1 总体架构

下面这张图就是 ViT的整体架构

![image-20220629164330143](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220629164330143.png)

说一下，运行的流程

- Path-embeddings: 将输入为 [H, W, C] 的图像，依照 [P, P, C] 的大小切成 N 份，再通过linear projection 到 D维，输出尺寸变为 [N, D]。
- Append class token embedding: 像 BERT 一样在第0位添加一个可以学习的 embedding 来作为类别的token，输出为 [N+1, D]。
- Concat position-embeddings: 直接用1D的position embeddings，从0开始到N，文中有实验用2D的也差不多，输出为 [N+1, D+1]。
- Concat position-embeddings: 直接用1D的position embeddings，从0开始到N，文中有实验用2D的也差不多，输出为 [N+1, D+1]。
- 做 classification: 在 class token 那个位置上的输出后接 MLP head 用以做分类classification。



**一句话讲完**

**本文中，将图像使用卷积进行分块（14*14=196），再每一块进行展平处理变成序列，然后将这196个序列添加位置编码和cls token，再输入多层Transformer结构中。最后将cls tooken取出来通过一个MLP（多层感知机）用于分类。**



![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-d4fbd8f1473a9411225d461356ed3634_720w.jpg)





### 2.2 预处理-编码器

**首先看一下预处理模块？我们假设图片输入为224\*224\*3。走一下预处理的流程**

标准的transformer的输入是1维的**token embedding**。

为了处理二维图像，我们将尺寸为$H\times W\times C$的图像reshape为拉平的2维图块，尺寸为$(N\times(P^2\cdot C))$。

其中，$(P,P)$为图块的带下，$N=HW/P^2$。$N$是图块的数量，会影响输入序列的长度。

Transformer在所有图层上使用恒定的隐矢量$D$，因此我们将图块拉平，并使用可训练的线性投影映射到$D$的大小，将此投影的输出称为patch embedding。

> 如果以图像$224\times 224 \times 3$为例。我们**将会得到多少的token呢？？**
>
> $N=\frac{HW}{P^2}=\frac{224\times 224}{16^2}=196$。意味着我们会得到196个token 信息。
>
> **那么每个token 的维度是多少呢？**
>
> $16\times 16\times 3=768$。
>
> 这样子我们就获得了196个token，每个token 拥有768维度的。就变成了$196\times 768$
>
> **具体怎么操作的呢？？看一下下面的这个**
>
> * 首先需要图片分成$16\times 16$的patch。16倍的下采样
>
>   所以分成了$\frac{224}{16}\times \frac{224}{16}=14\times 14$的图像块。
>
>   $224\times 224 \times 3\to 14\times 14\times (16\times 16\times 3)$
>
>   $=14\times 14\times768$
>
> * 然后直接将前两维图片拉平$14\times 14\times768\to 196\times 768$

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-28d25ebe0008126f4b546eb1df57247c_720w.jpg)

* 然后加上特殊字符串cls token。最终的维度会变成$196+1=197$。
* 然后在加上位置编码信息。注意是**加**，是sum。所以呢？最后维度是不变的。维度还是$197\times 768$
* 最终输入到transformer中的维度为$197\times 768$



### 2.3 Transformer 解码器

前面讲过，一个图片$224\times 224\times 3$,经过一个预处理，处理出来的维度为$197\times 768$。把处理出来的token输入进标准的transformer。

接下来就是一个标准的transformer

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-a1cbf19e63f8e7dc59ab968c78ce442f_720w.jpg)

Transformer模块主要有两个部分，一个是Muti-head Attention，另一个是MLP。

#### 1) Muti-head Attention

这里讲一下流程，具体操作可以看李宏毅老师的transformer课。还有代码就可以了

输入的维度$197\times 768$。然后呢？？这样子就变成3分的k,q,v。

使用**多头注意力**，如果是Vit-Base版本，头的数目是12，每个输入的维度就是$\frac{768}{12}=64$。所以**输入**为$197\times 64$。$dim(k,q,v)=197\times 64$

12个头做自注意力操作，每个头出来的维度还是$197\times 64$。然后进行**拼接**。维度再次变成$197\times 768$



#### 2) MLP

MLP就是一个两层感知机。具体看代码



**总结**

transformer block输入是$197\times 768$,出来还是$197\times 768$。维度是不变的，所以是transformer block相加多少就加多少

### 2.4 分类头

分类头很简单，就是取特征层如$197*768$的第一个向量，即$1*768$，再对此进行线性全连接层进行多分类即可。



**到这里呢？整体的流程就结束了。**



## 3. 代码

~~~python
"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

"""
说明：本代码关于维度相关的注释，均以vit的base模型为基础
"""


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
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
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding，二维图像patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)  # 图片尺寸224*224
        patch_size = (patch_size, patch_size)  # 下采样倍数，一个grid cell包含了16*16的图片信息
        self.img_size = img_size
        self.patch_size = patch_size
        # grid_size是经过patchembed后的特征层的尺寸
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # path个数 14*14=196

        # 通过一个卷积，完成patchEmbed
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果使用了norm层，如BatchNorm2d，将通道数传入，以进行归一化，否则进行恒等映射
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape  # batch,channels,heigth,weigth
        # 输入图片的尺寸要满足既定的尺寸
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # proj: [B, C, H, W] -> [B, C, H,W] , [B,3,224,224]-> [B,768,14,14]
        # flatten: [B, C, H, W] -> [B, C, HW] , [B,768,14,14]-> [B,768,196]
        # transpose: [B, C, HW] -> [B, HW, C] , [B,768,196]-> [B,196,768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """
    muti-head attention模块，也是transformer最主要的操作
    """

    def __init__(self,
                 dim,  # 输入token的dim,768
                 num_heads=8,  # muti-head的head个数，实例化时base尺寸的vit默认为12
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 平均每个head的维度
        self.scale = qk_scale or head_dim ** -0.5  # 进行query操作时，缩放因子
        # qkv矩阵相乘操作，dim * 3使得一次性进行qkv操作
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # 一个卷积层
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim] 如 [bactn,197,768]
        B, N, C = x.shape  # N:197 , C:768

        # qkv进行注意力操作，reshape进行muti-head的维度分配，permute维度调换以便后续操作
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim] 如 [b,197,2304]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head] 如 [b,197,3,12,64]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv的维度相同，[batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 矩阵相乘操作
        attn = attn.softmax(dim=-1)  # 每一path进行softmax操作
        attn = self.attn_drop(attn)

        # [b,12,197,197]@[b,12,197,64] -> [b,12,197,64]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 维度交换 transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 经过一层卷积
        x = self.proj_drop(x)  # Dropout
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU,  # GELU是更加平滑的relu
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果out_features不存在，则为in_features
        hidden_features = hidden_features or in_features  # 如果hidden_features不存在，则为in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # fc层1
        self.act = act_layer()  # 激活
        self.fc2 = nn.Linear(hidden_features, out_features)  # fc层2
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    基本的Transformer模块
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)  # norm层
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 代码使用了DropPath，而不是原版的dropout
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # norm层
        mlp_hidden_dim = int(dim * mlp_ratio)  # 隐藏层维度扩张后的通道数
        # 多层感知机
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # attention后残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # mlp后残差连接
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes  # 分类类别数量
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  # distilled在vit中没有使用到
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 层归一化
        act_layer = act_layer or nn.GELU  # 激活函数

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # [1,1,768],以0填充
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 按照block数量等间距设置drop率
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)  # layer_norm

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)，分类头,self.num_features=768
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init，权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # cls_token类别token [1, 1, 768] -> [B, 1, 768]，扩张为batch个cls_token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 196, 768]-> [B, 197, 768]，维度1上的cat
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)  # 添加位置嵌入信息
        x = self.blocks(x)  # 通过attention堆叠模块（12个）
        x = self.norm(x)  # layer_norm
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])  # 返回第一层特征，即为分类值
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        # 分类头
        x = self.forward_features(x)  # 经过att操作，但是没有进行分类头的前传
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):  # fc层初始化
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):  # conv层初始化
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):  # LayerNorm层初始化
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = vit_base_patch16_224(num_classes=1000)
    from thop import profile

    # from torchstat import stat
    # d = stat(model, (3, 1024, 2048))

    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))

~~~







## 4. 结果

### 4.1 ViT更需要预训练

ViT的模型整体参数量是较大的，一个ViT-base的预训练权重就高达400M，相较于MobileNet-v2的13M和ResNet34的85M，超出较多。所以，ViT模型相较于CNN网络更加需要大数据集的预训练。文中做了一个实验，使用不同规模的ImageNet和JFT数据集，进行预训练，比较其与CNN模型的性能。如图7所示。

![image-20220701103721006](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220701103721006.png)

**如果训练的数据不够多，ViTs是不如传统的ResNet。只有在足够多的数据下，ViT的效果是拔群的**



### 4.2 ViT模型更容易泛化到下游任务

我们知道，对于CNN网络，即使有预训练权重，当使用这个网络泛化到其他下游任务时，也需要训练较长时间才能达到较好的结果。但是，对于ViT模型来说，当拥有ViT的预训练权重时，只需要训练几个epoch既可以拥有很好的性能。

我曾做过实验，无论是使用小模型和轻量化模型AlexNet、MobileNetv2，还是使用大模型ResNet50，要达到较好预测，都要训练30-50epoch甚至更高。而使用ViT模型仅需要2-3个epoch便可达到更优秀的性能。这部分实验的文章稍后会写。

在文章关于此部分的实验结果如图8所示，可以看出训练7个epoch时，ViT类的模型相较于CNN模型，效果更好。

![image-20220701104013496](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220701104013496.png)



参考代码

> [ViT论文及代码解读-ICLR2021：Transformer用于视觉分类也有很好的性能 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/510218124)
>
> [[论文笔记\] ViT - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/359071701)
>
> [VIT论文介绍_赵卓不凡的博客-CSDN博客_vit论文](https://blog.csdn.net/sgzqc/article/details/124698205)





