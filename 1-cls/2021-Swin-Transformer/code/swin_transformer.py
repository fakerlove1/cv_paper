import torch.nn as nn
import torch
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 普通的线性组合
        self.fc1 = nn.Linear(in_features, hidden_features)
        #  激活层
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #  随机丢弃
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class PatchEmbed(nn.Module):
    """
    这个是模块的第一部分
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=512, embed_dim=96, norm_layer=None):
        """

        Args:
            img_size:(int): 输入图片的尺寸
            patch_size:(int): Patch token的尺度. Default: 4. 一个patch 默认是 4*4的图像
            in_chans: (int): 输入通道的数量
            embed_dim:(int): 线性投影后输出的维度
            norm_layer:(nn.Module, optional): 这可以进行设置，在本层中设置为了None
        """
        super(PatchEmbed, self).__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        # 计算出patch的数量利用Patch token在长宽方向上的数量相乘的结果
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        #  输出通道数，输入通道数
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 判断是否使用norm_layer，在这里我们没有应用
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None


    def forward(self, x):
        # 解析输入的维度
        B, C, H, W = x.shape
        # 判断图像是否与设定图像一致，如果不一致会报错
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 经过一个卷积层来进行一个线性变换，并且在第二个维度上进行一个压平操作，维度为(B, C, Ph*Pw),后在进行一个维与二维的一个转置，维度为：(B Ph*Pw C)
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """
        这里就是计算复杂度的啥用处没有
        Returns:

        """
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


if __name__ == '__main__':
    x=torch.randn(size=(1,3,224,224))
    p=PatchEmbed(in_chans=3)
    out=p(x)
    print(out.shape)
