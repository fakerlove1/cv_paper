import collections.abc

import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple([x] * 2)


class FeedForward(nn.Module):
    """
    mlp结构
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        """

        Args:
            dim: 输入维度
            num_patch: patch的数目
            token_dim:
            channel_dim:
            dropout: dropout的比例
        """
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):

    def __init__(self,
                 in_channels=3,
                 dim=512,
                 patch_size=16,
                 image_size=224,
                 depth=8,
                 mlp_ratio=(0.5, 4.0),
                 num_classes=1000, ):
        """

        Args:
            in_channels: 输入通道数
            dim: 分类头 中间通道数
            patch_size: patch大小
            image_size: 图片大小
            depth: block重复次数
            mlp_ratio: mix_block 中间维度 缩放数，用来计算token_dim, channel_dim
            num_classes: 分类数
        """
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # patch 的数目
        self.num_patch = (image_size // patch_size) ** 2

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        # 可以换成 linear projection
        # self.to_patch_embedding=nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
        #                                                 p1=patch_size, p2=patch_size),
        #               nn.Linear((patch_size ** 2) * in_channels, dim), )

        # 计算token_dim, channel_dim
        token_dim, channel_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x=[1,3,224,224]
        x = self.to_patch_embedding(x)  # [1,3,224,224] ->[1, 196, 512]
        # 3*224*224 使用patch_size 为 16 的话 。意思为下采样16倍 。变成 [1,dim,224/patch_size,224/patch_size]
        # [1,3,224,224] -> [1,512,14,14]
        # [1,512,14,14] -> [1, 14*14, 512]=[1, 196, 512]

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)  # [1, 196, 512]-> [1, 196, 512]
        x = x.mean(dim=1)  # [1, 196, 512]->[1, 512]

        return self.mlp_head(x)


def mixer_s32(**kwargs):
    """ Mixer-S/32
    """
    model = MLPMixer(patch_size=32,
                     depth=8,
                     dim=512,
                     **kwargs)
    return model


def mixer_s16(**kwargs):
    """ Mixer-S/16
    """
    model = MLPMixer(patch_size=16,
                     depth=8,
                     dim=512,
                     **kwargs)
    return model


def mixer_s4(**kwargs):
    """ Mixer-S/4
    """
    model = MLPMixer(patch_size=4,
                     depth=8,
                     dim=512,
                     **kwargs)
    return model


def mixer_b32(**kwargs):
    """ Mixer-B/32
    """
    model = MLPMixer(patch_size=32,
                     depth=12,
                     dim=768,
                     **kwargs)
    return model


def mixer_b16(pretrained=False, **kwargs):
    """ Mixer-B/16
    """
    model = MLPMixer(patch_size=16,
                     depth=12,
                     dim=768,
                     **kwargs)
    return model


def mixer_l32(**kwargs):
    """ Mixer-L/32
    """
    model = MLPMixer(patch_size=32,
                     depth=24,
                     dim=1024,
                     **kwargs)
    return model


def mixer_l16(**kwargs):
    """ Mixer-L/16 224x224
    """
    model = MLPMixer(patch_size=16,
                     depth=24,
                     dim=1024,
                     **kwargs)
    return model


if __name__ == "__main__":
    model = mixer_s4(num_classes=1000,image_size=32)

    #  计算参数量
    from thop import profile
    input = torch.randn(1, 3, 32,32)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
