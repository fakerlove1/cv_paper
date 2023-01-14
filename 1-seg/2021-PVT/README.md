# 【语义分割】2021-PVT  ICCV

> 论文题目:Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
>
> 论文链接:[https://arxiv.org/abs/2102.12122](https://arxiv.org/abs/2102.12122)
>
> 论文代码： [https://github.com/whai362/PVT](https://github.com/whai362/PVT)
>
> 论文翻译：[PVT，PVTv2 - 简书 (jianshu.com)](https://www.jianshu.com/p/8b0d7fc91cb6)
>
> 发表时间：2021年2月
>
> 引用：Wang W, Xie E, Li X, et al. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 568-578.
>
> 引用数：688



## 1. 简介

### 1.1 简介

之前的所总结的ViT backbone，本身并没有针对视觉中诸如分割、检测等密集预测型的任务的特定，设计合适结构。后续SERT等论文也只是简单的将VIT作为Encoder，将其提取到的单尺度特征通过一些简单的Decoder的处理，验证了transformer在语义分割任务上的效果。但是，我们知道，在语义分割任务上，多尺度的特征是非常重要的，因此在PVT中提出了一种能够提取多尺度特征的vision transformer backbone。



### 1.2 ViT在语义分割上存在的问题

我们知道，ViT 的设计方案中输出的特征图和输入大小基本保持一致。将其应用到分割、检测等密集预测任务上将会面临着两方面的问题：

**1）计算开销剧增**

分割和检测相对于分类任务而言，往往需要较大的分辨率图片输入。

因此，一方面，我们需要相对于分类任务而言划分更多个patch才能得到相同粒度的特征。如果仍然保持同样的patch数量，那么特征的粒度将会变粗，从而导致性能下降

另一方面，我们知道，Transformer的计算开销与token化后的patch数量正相关， patch数量越大，计算开销越大。所以，如果我们增大patch数量，可能就会让我们本就不富裕的计算资源雪上加霜。

以上是ViT应用于密集预测任务上的第一个缺陷。

**2）缺乏多尺度特征**

ViT 输出的特征图和输入大小基本保持一致。这导致ViT作为Encoder时，只能输出单尺度的特征。

而在CNN中，多尺度的特征已经早就被证实对分割、检测等任务有着重要的作用，一些经典的工作如deeplab系列、PSPNet等都有效的利用多尺度特征取得了性能上的提升。

因此，如何利用vision transformer获取多尺度的特征是另一个问题。



**改进**

计算机视觉中CNN backbone经过多年的发展，沉淀了一些通用的设计模式。最为典型的就是金字塔结构。

简单的概括就是：

1）feature map的**分辨率随着网络加深，逐渐减小**；

2）feature map的**channel数随着网络加深，逐渐增大**。

几乎所有的密集预测（dense prediction）算法都是围绕着特征金字塔设计的

这个结构怎么才能引入到Transformer里面呢？

最终还是发现：**简单地堆叠多个独立的Transformer encoder效果**是最好的。

然后我们就得到了PVT，如下图所示。在每个Stage中**通过Patch Embedding来逐渐降低输入的分辨率**。



## 2. 网络

### 2.1 整体架构

![image-20220702163725998](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220702163725998.png)

模型总体上由4个stage组成，每个stage包含Patch Embedding和若干个Transformer模块（相对于原本的transformer有所改动）组成。



### 2.2  Patch embedding

在每个stage开始，首先像ViT一样对输入图像进行token化，即进行patch embedding，patch大小除第1个stage的是$4\times 4$外，其余均采用$2\times 2$大小。这个思想有些类似于池化或者带步长的卷积操作，减小图像的分辨率，使得模型能够提取到更为抽象的信息。这意味着每个stage（第一个stage除外）最终得到的特征图维度是减半的，tokens数量对应减少4倍。每个patch随后会送入一层Linear中，调整通道数量，然后再reshape以将patch token化。

**这使得PVT总体上与resnet看起来类似，4个stage得到的特征图相比原图大小分别是1/4，1/8，1/16 和 1/32。这也意味着PVT可以产生不同尺度的特征**。

> Note：由于不同的stage的tokens数量不一样，所以每个stage采用不同的position embeddings，在patch embed之后加上各自的position embedding，当输入图像大小变化时，position embeddings也可以通过插值来自适应。



### 2.3 Spatial-reduction attention（SRA）

在Patch embedding之后，需要将token化后的patch输入到若干个transformer 模块中进行处理。为了进一步减少计算量，作者将multi-head attention (MHA)用所提出的spatial-reduction attention (SRA)来替换。从名字上就可以看出这个替换实在做什么操作。把Q，K，V的空间分辨率减小以减小参数量。果不其然，作者在MHA中将K和V的分辨率都降低了R倍。其示意图如下。

![image-20220702165521472](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220702165521472.png)



在实现上，首先将维度为$(HW,C)$的K，V通过 reshape变换到维度为$(H,W, C)$的3-D特征图，然后均分大小为$R * R$的patchs，每个patchs通过线性变换将得到维度为$(H*W / R*R，C)$的patch embeddings（这里实现上其实和patch emb操作类似，等价于一个卷积操作），最后应用一个layer norm层，这样就可以大大降低K和V的数量。

每个stage，经过若干个SRA模块的处理后，将得到的特征，再次reshape成3D特征图的形式输入到下一个Stage中。



### 2.4 总体概览

![image-20220702165721549](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220702165721549.png)

其中P为patch的size，C为特征的维度，R之前已经解释过，N为多头attention的head数量，E为FFN的扩展系数。

## 3. 代码

image的一千类分类

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
    'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])

        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches + 1, embed_dims[3]))
        self.pos_drop4 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        x = x + self.pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        x = x + self.pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed4
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_huge_v2(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[128, 256, 512, 768],
        num_heads=[2, 4, 8, 12],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 10, 60, 3],
        sr_ratios=[8, 4, 2, 1],
        # drop_rate=0.0, drop_path_rate=0.02)
        **kwargs)
    model.default_cfg = _cfg()

    return model

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    model=pvt_small()
    y=model(x)
    print(y.shape)
~~~

