# 【图像分类】2021-CvT 

> 论文题目：CvT: Introducing Convolutions to Vision Transformers
>
> 论文链接：[https://arxiv.org/abs/2103.15808](https://arxiv.org/abs/2103.15808)
>
> 论文代码：[https://github.com/microsoft/CvT](https://github.com/microsoft/CvT)
>
> 发表时间：2021年3月
>
> 引用：Wu H, Xiao B, Codella N, et al. Cvt: Introducing convolutions to vision transformers[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 22-31.
>
> 引用数：393



## 1. 简介

### 1.1 简介

本文提出了一种新的结构，称为卷积视觉变换器（CvT），它通过在 ViT 中引入卷积来提高视觉变换器（ViT）的性能和效率。这是通过两个主要修改来实现的：包含`新卷积令牌嵌入的 Transformer 层次结构，以及利用卷积投影的卷积 Transformer 块`。这些变化将卷积神经网络（CNN）的理想特性引入 ViT 体系结构（即平移、缩放和失真不变性），同时保持 Transformer 的优点（即动态注意、全局上下文和更好的泛化）。

我们通过进行大量实验来验证 CvT，结果表明，与 ImageNet-1k 上的其他 ViT 和 Resnet 相比，该方法实现了最先进的性能，参数更少，触发器更少。此外，在对更大的数据集（例如ImageNet-22k）进行预训练并对下游任务进行微调时，可以保持性能提升。我们的 CvT-W24 在 ImageNet-22k 上进行了预训练，在 ImageNet-1k val 集上获得了 87.7% 的顶级精度。

最后，我们的结果表明，位置编码是现有 ViT 中的一个关键组件，可以在我们的模型中安全地删除，从而简化高分辨率视觉任务的设计。



### 1.2 贡献

本文提出了**使用卷积来生成Token和生成QKV**，将卷积融入到Transformer当中。

既保留Transformer的优点：全局的感知能力，动态的注意力和更好的泛化能力

又引入了卷积的优点：平移、缩放、变形不变性

与此同时，使用卷积可以大大减少参数数量并提升运算速度。

在本文的实验中发现，删掉位置编码并不影响性能，进一步提升了运算速度。

## 2. 网络

### 2.1 整体架构



![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/pipeline.svg)



作者提出了三种大小的网络，CvT13 CvT21和CvTW24

![image-20220804165824607](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220804165824607.png)

### 2.2 Convolutional Token Embedding

通过卷积7*7获得conv embedding。**conv2d并不是深度卷积，可以理解为一个patch embedding操作**，在大部分cv transformer中均会进行。这里是**每个阶段**会使用一次Patch embedding,来增加通道和降低分辨率。



~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x
if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    model=ConvEmbed()
    y=model(x)
    #  patch_size=7,embed_dim=64,stride=4
    #  224*224 的图片做 4倍下采样。变成 56*56
    # 
    print(y.shape)

~~~

可以看到，输入的x经过卷积后，生成特征图f（b,c w,h）并将其压缩成(b hw c)的形状放入LayerNorm中计算，最后在变回去f(b,c,w,h) 用于下一层的计算。

其实这里的f(b hw c)就是Token，其中HW就是Token的个数，c是Token的维度。

相比起ViT，ConvEmbed没有将Patch_size和stride等同，也就是说，卷积会有很多重叠的部分，生成的Token个数要远远多于ViT的Patch_size*Patch_size

官方实现给出的第一层Patch_size=(7,7) stride=4



**原本ViT的patch embedding**

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_, to_2tuple


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=7, channels=3, embed_dim=64, norm_layer=None):
        super(PatchEmbed, self).__init__()
        patch_height = patch_size
        patch_width = patch_height
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, embed_dim),
        )

    def forward(self, x):
        return self.to_patch_embedding(x)

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    model=PatchEmbed()
    y=model(x)
    #  patch_size=7,embed_dim=64,stride=4
    #  224*224 的图片做 4倍下采样。变成 56*56
    #
    print(y.shape)


~~~



### 2.3 Convolutional Projection for Attention(使用卷积生成kqv)

通过**深度卷积**进行conv proj，即将特征转化成query ，value,key向量。这种转化方式可以见下

![image-20220804163150615](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220804163150615.png)

- (a)是ViT中生成QKV的方法
- (b)是卷积生成QKV的方法
- (c)是使用Squeezed Convolution生成QKV的方法

~~~python
def forward_conv(self, x, h, w):
    if self.with_cls_token:
        cls_token, x = torch.split(x, [1, h * w], 1)

    x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    if self.conv_proj_q is not None:
        q = self.conv_proj_q(x)
    else:
        q = rearrange(x, 'b c h w -> b (h w) c')

    if self.conv_proj_k is not None:
        k = self.conv_proj_k(x)
    else:
        k = rearrange(x, 'b c h w -> b (h w) c')

    if self.conv_proj_v is not None:
        v = self.conv_proj_v(x)
    else:
        v = rearrange(x, 'b c h w -> b (h w) c')

    if self.with_cls_token:
        q = torch.cat((cls_token, q), dim=1)
        k = torch.cat((cls_token, k), dim=1)
        v = torch.cat((cls_token, v), dim=1)

    return q, k, v
~~~

传入的x(b hw c)，即上一层产生的Token，首先将其解压缩变形成(b c h w)，然后分别使用三种卷积核conv_proj_q、conv_proj_k、conv_proj_v生成QKV然后加上cls_token。

其中卷积操作是Depth-wise Convolution+BatchNorm实现的

如图b所示，直接使用卷积操作，需要的参数量是$s^2C^2$,复杂度是$O(s^2C^2T)$

- s是kernel size
- C是 Token Channel Dimension
- T是 Token number

为了减少参数量，这里使用Depth-wise separeble convolution。这样参数量就减少到了$s^2C$,复杂度就减少到了$O(s^2CT)$

与此同时，在生成KV的时候，将步长调整到2或者更大，可以减少KV的尺寸，生成Local Sequeeze Key和Local Sequeeze Value

图像中相邻像素会在外观和语义上有信息冗余，这样做会引起微小的性能损失，但是Q补偿了这部分损失。



### 2.4 移除位置信息

在这个框架中，Convolution Projection和Convolutional Token Embedding应用在每一个Transformer Block中。卷积本身就带有位置信息，给予了模型感知位置信息的能力。因此可以将位置编码安全的移除并不影响任何性能损失。

在实验部分，作者做了消融实验，结果如下：

![image-20220804165757034](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220804165757034.png)

## 3. 代码

### 3.1 原版

~~~python
from functools import partial
from itertools import repeat

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_, to_2tuple


# # From PyTorch internals
# def _ntuple(n):
#     def parse(x):
#         if isinstance(x, container_abcs.Iterable):
#             return x
#         return tuple(repeat(x, n))
#
#     return parse

#
# to_1tuple = _ntuple(1)
# to_2tuple = _ntuple(2)
# to_3tuple = _ntuple(3)
# to_4tuple = _ntuple(4)
# to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))

        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T - 1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
                hasattr(module, 'conv_proj_q')
                and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
                hasattr(module, 'conv_proj_k')
                and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
                hasattr(module, 'conv_proj_v')
                and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                                .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                                .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # print(x.shape)
        x = self.head(x)
        return x


def CvT_13(**kwargs):
    spec = {
        "INIT": 'trunc_norm',
        "NUM_STAGES": 3,
        "PATCH_SIZE": [7, 3, 3],
        "PATCH_STRIDE": [4, 2, 2],
        "PATCH_PADDING": [2, 1, 1],
        "DIM_EMBED": [64, 192, 384],
        "NUM_HEADS": [1, 3, 6],
        "DEPTH": [1, 2, 10],
        "MLP_RATIO": [4.0, 4.0, 4.0],
        "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_PATH_RATE": [0.0, 0.0, 0.1],
        "QKV_BIAS": [True, True, True],
        "CLS_TOKEN": [False, False, True],
        "POS_EMBED": [False, False, False],
        "QKV_PROJ_METHOD": ['dw_bn', 'dw_bn', 'dw_bn'],
        "KERNEL_QKV": [3, 3, 3],
        "PADDING_KV": [1, 1, 1],
        "STRIDE_KV": [2, 2, 2],
        "PADDING_Q": [1, 1, 1],
        "STRIDE_Q": [1, 1, 1],
    }
    model = ConvolutionalVisionTransformer(
        **kwargs,
        spec=spec,
    )
    return model


def CvT_21(**kwargs):
    spec = {
        "INIT": 'trunc_norm',
        "NUM_STAGES": 3,
        "PATCH_SIZE": [7, 3, 3],
        "PATCH_STRIDE": [4, 2, 2],
        "PATCH_PADDING": [2, 1, 1],
        "DIM_EMBED": [64, 192, 384],
        "NUM_HEADS": [1, 3, 6],
        "DEPTH": [1, 4, 16],
        "MLP_RATIO": [4.0, 4.0, 4.0],
        "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_PATH_RATE": [0.0, 0.0, 0.1],
        "QKV_BIAS": [True, True, True],
        "CLS_TOKEN": [False, False, True],
        "POS_EMBED": [False, False, False],
        "QKV_PROJ_METHOD": ['dw_bn', 'dw_bn', 'dw_bn'],
        "KERNEL_QKV": [3, 3, 3],
        "PADDING_KV": [1, 1, 1],
        "STRIDE_KV": [2, 2, 2],
        "PADDING_Q": [1, 1, 1],
        "STRIDE_Q": [1, 1, 1],
    }
    model = ConvolutionalVisionTransformer(
        **kwargs,
        spec=spec,
    )
    return model


def CvT_W24(**kwargs):
    spec = {
        "INIT": 'trunc_norm',
        "NUM_STAGES": 3,
        "PATCH_SIZE": [7, 3, 3],
        "PATCH_STRIDE": [4, 2, 2],
        "PATCH_PADDING": [2, 1, 1],
        "DIM_EMBED": [192, 768, 1024],
        "NUM_HEADS": [3, 12, 16],
        "DEPTH": [2, 2, 20],
        "MLP_RATIO": [4.0, 4.0, 4.0],
        "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_PATH_RATE": [0.0, 0.0, 0.3],
        "QKV_BIAS": [True, True, True],
        "CLS_TOKEN": [False, False, True],
        "POS_EMBED": [False, False, False],
        "QKV_PROJ_METHOD": ['dw_bn', 'dw_bn', 'dw_bn'],
        "KERNEL_QKV": [3, 3, 3],
        "PADDING_KV": [1, 1, 1],
        "STRIDE_KV": [2, 2, 2],
        "PADDING_Q": [1, 1, 1],
        "STRIDE_Q": [1, 1, 1],
    }
    model = ConvolutionalVisionTransformer(
        **kwargs,
        spec=spec,
    )
    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = CvT_13(in_chans=3,
                   num_classes=1000,
                   act_layer=QuickGELU,
                   norm_layer=partial(LayerNorm, eps=1e-5))
    y = model(x)
    print(y.shape)

~~~

### 3.2 简略版



~~~python
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CvT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 192,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 3
        layers = []

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
            ))

            dim = config['emb_dim']

        self.layers = nn.Sequential(*layers)

        self.to_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        latents = self.layers(x)
        return self.to_logits(latents)
~~~





参考链接

> [【论文笔记】CvT: Introducing Convolutions to Vision Transformers_来自γ星的赛亚人的博客-CSDN博客](https://blog.csdn.net/m0_58678659/article/details/123751467)