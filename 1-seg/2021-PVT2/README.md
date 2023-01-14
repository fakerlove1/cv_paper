# 2021-PVT2 CVMJ

> 论文题目：PVT v2: Improved Baselines with Pyramid Vision Transformer
>
> 论文链接: [https://arxiv.org/abs/2106.13797](https://arxiv.org/abs/2106.13797)
>
> 论文代码：[https://github.com/whai362/PVT](https://github.com/whai362/PVT)
>
> 论文翻译：[PVT，PVTv2 - 简书 (jianshu.com)](https://www.jianshu.com/p/8b0d7fc91cb6)
>
> 发表时间：2021年6月
>
> 引用：Wang W, Xie E, Li X, et al. Pvt v2: Improved baselines with pyramid vision transformer[J]. Computational Visual Media, 2022, 8(3): 415-424.
>
> 引用数：126

## 1. 简介

计算机视觉中的Transformer最近取得了令人鼓舞的进展。在这项工作中，作者通过添加3个改进设计来改进原始金字塔视觉Transformer（PVTv1），其中包括：

- 具有卷积的局部连续特征；
- 具有zero paddings的位置编码，
- 具有平均汇集。

通过这些简单的修改，PVTv2在分类、检测和分割方面显著优于PVTv1。此外，PVTv2在ImageNet-1K预训练下取得了比近期作品（包括 Swin Transformer）更好的性能。



## 2. 网络

PVTv1[33]的主要局限性有以下三个方面:

(1)与ViT类似，在处理高分辨率输入时(如短边为800像素)，PVTv1的计算复杂度相对较大。

(2) PVTv1将一幅图像视为一组不重叠的patch序列，在一定程度上丧失了图像的局部连续性;

(3) PVTv1中的位置编码是固定大小的，对于处理任意大小的图像是不灵活的。这些问题限制了PVTv1在视觉任务中的性能。

### 2.1 总体架构





### 2.2 Linear Spatial Reduction Attention

用LinearSRA替代SRA。这里需要说明的一个问题，作者在PVTv1中说自己没用到卷积，但是在压缩K、V的时候使用的是Conv2D（参见github中代码）。在PVTv2中使用平均池化替代Conv2D。



### 2.3 Overlapping Patch Embedding(重叠补丁嵌入)





### 2.4  Convolutional FeedForward





## 3. 代码 pvt2-upernet

~~~python
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dwconv = DWConv(hidden_dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, mlp_ratio=4, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=64, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size // 2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


pvtv2_settings = {
    'B1': [2, 2, 2, 2],  # depths
    'B2': [3, 4, 6, 3],
    'B3': [3, 4, 18, 3],
    'B4': [3, 8, 27, 3],
    'B5': [3, 6, 40, 3]
}


class PVTv2(nn.Module):
    def __init__(self, model_name: str = 'B1') -> None:
        super().__init__()
        assert model_name in pvtv2_settings.keys(), f"PVTv2 model name should be in {list(pvtv2_settings.keys())}"
        depths = pvtv2_settings[model_name]
        embed_dims = [64, 128, 320, 512]
        drop_path_rate = 0.1
        self.embed_dims = embed_dims
        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # transformer encoder
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, 8, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, 4, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, 4, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x1, x2, x3, x4


class PPM(nn.ModuleList):
    """
    金字塔池化模型 Pyramid Pooling Module
    https://arxiv.org/abs/1612.01105
    CVPR 2017年 的工作
    使用最大池化，获取
    """

    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], ):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, 4 * self.out_channels,
                      kernel_size=1),
            nn.BatchNorm2d(4 * self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)

        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, out_channels=512, num_classes=19, channels=[64, 128, 320, 512]):
        """

        Args:
            out_channels: 最后一层融合的 通道数，在分类前的通道数
            num_classes:  最后分类数目
            channels: 四层backbone的通道数
        """
        super(FPNHEAD, self).__init__()
        self.num_classes = num_classes
        self.PPMHead = PPMHEAD(in_channels=channels[-1], out_channels=channels[-1] // 4)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels[-2], channels[-2], 1),
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(channels[-2] + channels[-1], channels[-2], 1),
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels[-3], channels[-3], 1),
            nn.BatchNorm2d(channels[-3]),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(channels[-3] + channels[-2], channels[-3], 1),
            nn.BatchNorm2d(channels[-3]),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels[-4], channels[-4], 1),
            nn.BatchNorm2d(channels[-4]),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(channels[-4] + channels[-3], channels[-4], 1),
            nn.BatchNorm2d(channels[-4]),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(sum(channels), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.cls_seg = nn.Sequential(
            nn.Conv2d(out_channels, self.num_classes, kernel_size=3, padding=1),
        )

    def forward(self, input_fpn):
        """
        Args:
            input_fpn: 四个特征图

        Returns:

        """
        ##############################
        # x1 = torch.randn(1, 64, 56, 56)
        # x2 = torch.randn(1, 128, 28, 28)
        # x3 = torch.randn(1, 320, 14, 14)
        # x4 = torch.randn(1, 512, 7, 7)

        #  1/32的特征图 使用PPMHead torch.Size([1, 2048, 7, 7])
        # x1= [1, 512, 7, 7]
        x1 = self.PPMHead(input_fpn[-1])
        # print(x1.shape)

        # [1, 512, 7, 7]-->[1, 512, 14, 14]
        x = F.interpolate(x1,
                          size=(x1.size(2) * 2, x1.size(3) * 2),
                          mode='bilinear',
                          align_corners=True)

        # 融合1/16的图  torch.Size([1, 3072, 14, 14])。仅仅在通道上拼接
        # [1, 512, 14, 14] + [1,320, 14, 14] =[1, 832, 14, 14]
        x = torch.cat([x, self.Conv_fuse1(input_fpn[-2])], dim=1)

        ##############################
        # [1, 832, 14, 14] -->[1, 320, 14, 14] ,进行通道数上的减少
        x2 = self.Conv_fuse1_(x)
        #  [1, 320, 14, 14]->[1, 320, 28,28]
        x = F.interpolate(x2,
                          size=(x2.size(2) * 2, x2.size(3) * 2),
                          mode='bilinear',
                          align_corners=True)

        # 融合1/8的图
        # [1, 320, 28,28] +[1,  128, 28,28] = [1,  448, 28,28]
        x = torch.cat([x, self.Conv_fuse2(input_fpn[-3])], dim=1)
        # print(x.shape)
        ##############################
        # [1,  448, 28,28] -> [1, 128, 28, 28]进行通道上缩减。
        x3 = self.Conv_fuse2_(x)

        #  对1/8---> 1/4
        # [1, 128, 28, 28]-> [1, 128, 56, 56]
        x = F.interpolate(x3,
                          size=(x3.size(2) * 2, x3.size(3) * 2),
                          mode='bilinear',
                          align_corners=True)
        # 融合1/4的图
        # [1, 128, 56, 56]+[1, 64, 56, 56]=[1, 192, 56, 56]
        x = torch.cat([x, self.Conv_fuse3(input_fpn[-4])], dim=1)

        ##############################
        # [1, 192, 56, 56]-> [1, 64, 56, 56]
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))
        # print(x.shape)
        x = F.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)
        # print(x.shape)
        x = self.cls_seg(x)
        return x


class pvt2_upernet(nn.Module):
    def __init__(self, num_classes, channels, size="B1"):
        """
        类别数
        Args:
            num_classes:
        """
        super(pvt2_upernet, self).__init__()
        self.backbone = PVTv2(size)
        self.decoder = FPNHEAD(num_classes=num_classes, channels=channels)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x


def pvt2_B1_upernet(num_classes):
    model = pvt2_upernet(num_classes=num_classes, size="B1", channels=[64, 128, 320, 512])
    return model


def pvt2_B2_upernet(num_classes):
    model = pvt2_upernet(num_classes=num_classes, size="B2", channels=[64, 128, 320, 512])
    return model


def pvt2_B3_upernet(num_classes):
    model = pvt2_upernet(num_classes=num_classes, size="B3", channels=[64, 128, 320, 512])
    return model

def pvt2_B4_upernet(num_classes):
    model = pvt2_upernet(num_classes=num_classes, size="B3", channels=[64, 128, 320, 512])
    return model

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    model=pvt2_B2_upernet(num_classes=19)
    y=model(x)
    print(y.shape)
~~~



参考资料

> [PVTv2: Improved Baselines with Pyramid Vision Transformer——PVT2解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/485391506)
>
> [Transformer主干网络——PVT_V2保姆级解析_只会git clone的程序员的博客-CSDN博客](https://blog.csdn.net/qq_37668436/article/details/122495068)

