# 【图像分类】2021-Twins NeurIPS

> 论文题目：Twins: Revisiting the Design of Spatial Attention in Vision Transformers
>
> 论文链接：[https://arxiv.org/abs/2104.13840](https://arxiv.org/abs/2104.13840)
>
> 论文代码：[https://github.com/Meituan-AutoML/Twins](https://github.com/Meituan-AutoML/Twins)
>
> 作者团队：Twins 是美团和阿德莱德大学合作提出的视觉注意力模型
>
> 发表时间：2021年4月
>
> 引用：Chu X, Tian Z, Wang Y, et al. Twins: Revisiting the design of spatial attention in vision transformers[J]. Advances in Neural Information Processing Systems, 2021, 34: 9355-9366.
>
> 引用数：195



## 1. 简介

### 1.1 简介



文章总结了ViT,PVT,Swin-Transformer 等模型，

* ViT 原生的视觉注意力模型做主干网络并不能很好地适配目标检测、语义分割等常用的稠密预测任务。此外，相比于卷积神经网络，ViT 计算量通常要更大，推理速度变慢，不利于在实际业务中应用。因此设计更高效的视觉注意力模型，并更好地适配下游任务成为了当下研究的重点。

* PVT--香港大学、商汤联合提出的金字塔视觉注意力模型 PVT 借鉴了卷积神经网络中的图像金字塔范式来生成多尺度的特征，这种结构可以和用于稠密任务的现有后端直接结合，支持多种下游任务，。但由于 PVT 使用了静态且定长的位置编码，通过插值方式来适应变长输入，不能针对性根据输入特征来编码，因此性能受到了限制。另外，PVT 沿用了 ViT 的全局自注意力机制，计算量依然较大。
* Swin-Transformer -- 微软亚研院提出的 Swin  复用了 PVT 的金字塔结构。在计算自注意力时，使用了对特征进行窗口分组的方法（如图 3 所示），将注意力机制限定在一个个小的窗口（红色格子），而后通过对窗口进行错位使不同组的信息产生交互。这样可以避免计算全局自注意力而减少计算量，其缺点是损失了全局的注意力，同时由于窗口错位产生的信息交互能力相对较弱，一定程度上影响了性能。



**视觉注意力模型设计的难点**

简单总结一下，当前视觉注意力模型设计中需要解决的难点在于：

- **高效率的计算**：缩小和卷积神经网络在运算效率上的差距，促进实际业务应用；
- **灵活的注意力机制**：即能够具备卷积的局部感受野和自注意力的全局感受野能力，兼二者之长；
- **利于下游任务**：支持检测、分割等下游任务，尤其是输入尺度变化的场景。



### 1.2 贡献



Twins 提出了两类结构，分别是 Twins-PCPVT 和 Twins-SVT：

- Twins-PCPVT 将金字塔 Transformer 模型 PVT [2] 中的固定位置编码（Positional Encoding）更改为团队在 CPVT [3] 中提出的条件式位置编码 （Coditional Position Encoding, CPE），从而使得模型具有平移等变性（即输入图像发生平移后，输出同时相应发生变化），可以灵活处理来自不同空间尺度的特征，从而能够广泛应用于图像分割、检测等变长输入的场景。
- Twins-SVT 提出了空间可分离自注意力机制（Spatially Separable Self-Attention，SSSA）来对图像特征的空间维度进行分组，分别计算各局部空间的自注意力，再利用全局自注意力机制对其进行融合。这种机制在计算上更高效，性能更优。

Twins 系列模型实现简单，部署友好，在 ImageNet 分类、ADE20K 语义分割、COCO 目标检测等多个经典视觉任务中均取得了业界领先的结果。



## 2. 网络

### 2.1 整体架构



![image-20220731164945683](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220731164945683.png)







### 2.2 Twins-PCPVT

作者发现PVT中的global sub-sampled attention采用一个合适的位置编码是非常有效的，它的性能可以超过Swin Transformer，在这篇论文中，作者认为PVT的性能偏低的原因是因为它使用的绝对位置编码，Swin Transformer采用了相对位置编码

作者将PVT中的绝对位置编码替换为CPVT中conditional position encoding，将position encoding generator（CPE）放在每一个stage中第一个encoder block的后面

下图展示了团队在 CPVT中提出的条件位置编码器的编码过程。首先将 的输入序列转为 的输入特征，再用 根据输入进行条件式的位置编码，而且输出尺寸和输入特征相同，因此可以转为 序列和输入特征进行逐元素的加法融合。

![e4264dbd6799acea9cadeaa047ae8f70.png](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/e4264dbd6799acea9cadeaa047ae8f70.png)

其中，编码函数 可以由简单的深度可分离卷积实现或者其他模块实现，PEG 部分的简化代码如下。其中输入 feat_token 为形状为 的张量， 为 batch， 为 token 个数， 为编码维度（同图 5 中 ）。将 feat_token 转化为 的张量 cnn_feat 后，经过深度可分离卷积 （PEG）运算，生成和输入 feat_token 相同形状的张量，即条件式的位置编码。


~~~python
class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        # Only for demo use, more complicated functions are effective too.
    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:] # cls token不参与PEG
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat # 产生PE加上自身
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
    return x
~~~



由于条件位置编码 CPE 是根据输入生成，支持可变长输入，使得 Twins 能够灵活处理来自不同空间尺度的特征。另外 PEG 采用卷积实现，因此 Twins 同时保留了其平移等变性，这个性质对于图像任务非常重要，如检测任务中目标发生偏移，检测框需随之偏移。实验表明 Twins-PCPVT 系列模型在分类和下游任务，尤其是在稠密任务上可以直接获得性能提升。该架构说明 PVT 在仅仅通过 CPVT 的条件位置编码增强后就可以获得很不错的性能，由此说明 PVT 使用的位置编码限制了其性能发挥。




### 2.3 Twins-SVT

Twins-SVT （如下图 6 所示）对全局注意力策略进行了优化改进。全局注意力策略的计算量会随着图像的分辨率成二次方增长，因此如何在不显著损失性能的情况下降低计算量也是一个研究热点。

Twins-SVT 提出新的融合了`局部-全局注意力`的机制，可以类比于卷积神经网络中的深度可分离卷积 （Depthwise Separable Convolution），并因此命名为空间可分离自注意力（Spatially Separable Self-Attention，SSSA）。

与深度可分离卷积不同的是，Twins-SVT 提出的空间可分离自注意力（如下图 7 所示）是对特征的空间维度进行分组，并计算各组内的自注意力，再从全局对分组注意力结果进行融合。



![24ecd6d129344d019409c777b7dc85b6.png](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/24ecd6d129344d019409c777b7dc85b6.png)



Twins-SVT-S 模型结构，右侧为两个相邻 Transformer Encoder 的结合方式

![image-20220805103845280](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220805103845280.png)



#### LSA

类似于depthwise convolution，首先将二维特征图平均划分为子窗口，并仅在窗口内部进行Self-Attention计算，计算量会大大减少。

假设特征图被分为$m\times n$个windows（假设$\text{H%m}=0 $且$\text{W%n}=0$,则每组包含$\frac{HW}{mn}$个像素。 因此，这个window中self_attention的计算成本是
$$
\mathcal{O}\left(\frac{H^{2} W^{2}}{m^{2} n^{2}} d\right)
$$
则总成本为$\mathcal{O}\left(\frac{H^{2} W^{2}}{m n} d\right)$。当$k_{1} \ll H \text { and } k_{2} \ll W$时，改进最有效。当$k_1\text{ and } k_2$固定时，计算量随$H,W$变化呈线性增长。

虽然分组减少了计算量，但图像被分成了不重叠的子窗口，这使得感受野变小并且显着降低了性能。因此，我们需要一种机制来在不同的子窗口之间进行通信。

~~~python
class GroupAttention(nn.Module): #LSA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=7):
        super(GroupAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws 
        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3) 

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
~~~





#### GSA

比较简单的一个方法是，在LSA后面再接一个Global Self-Attention Layer，这种方法在实验中被证明也是有效的，但是其计算复杂度会较高:$O(H^2W^2d)$

如果使用一个有代表性的值来代表每个 sub-windows，那么全局 attention 的计算量就为
$$
\mathcal{O}(m n H W d)=\mathcal{O}\left(\frac{H^{2} W^{2} d}{k_{1} k_{2}}\right)
$$
这实质上等同于使用子采样特征映射作为注意操作的关键，因此作者称之为全局子采样注意（GSA）。

综合使用`LSA和GSA`，可以取得类似于`Separable Convolution（Depth-wise+Point-wise）`的效果。

整体的计算复杂度为：$\mathcal{O}\left(\frac{H^{2} W^{2} d}{k_{1} k_{2}}+k_{1} k_{2} H W d\right)$

同时有$\frac{H^{2} W^{2} d}{k_{1} k_{2}}+k_{1} k_{2} H W d \geq 2 H W d \sqrt{H W}$。

当$k_1\cdot k_2=\sqrt{HW}$时取得最小值。这样看来，在每个stage都有适合的$k$值，但为了简便，所有的$k$均设置为7。

~~~python
class Attention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
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
~~~

整个Transformer Block可以被表示为：
$$
\begin{array}{l}
\hat{\mathbf{z}}_{i j}^{l} = \operatorname{LSA}\left(\operatorname{LayerNorm}\left(\mathbf{z}_{i j}^{l-1}\right)\right)+\mathbf{z}_{i j}^{l-1} \\
\mathbf{z}_{i j}^{l} = \operatorname{FFN}\left(\text { LayerNorm }\left(\hat{\mathbf{z}}_{i j}^{l}\right)\right)+\hat{\mathbf{z}}_{i j}^{l} \\
\hat{\mathbf{z}}^{l+1} = \operatorname{GSA}\left(\text { LayerNorm }\left(\mathbf{z}^{l}\right)\right)+\mathbf{z}^{l} \\
\mathbf{z}^{l+1} = \operatorname{FFN}\left(\operatorname{LayerNorm}\left(\hat{\mathbf{z}}^{l+1}\right)\right)+\hat{\mathbf{z}}^{l+1}, \\
i \in\{1,2, \ldots, m\}, j \in\{1,2, \ldots, n\}
\end{array}
$$
![image-20220805111328795](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20220805111328795.png)

同时在每个Stage的第一个Block中会引入CPVT中的的PEG对位置信息进行编码。



### 2.4 结果



Ade20k结果

| Model        | Alias in the paper | mIoU(ss/ms) | FLOPs(G) | #Params (M) | URL                                                          | Log                                                          |
| ------------ | ------------------ | ----------- | -------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| PCPVT-Small  | Twins-PCPVT-S      | 46.2/47.5   | 234      | 54.6        | [pcpvt_small.pth](https://drive.google.com/file/d/1PkkBULZZUhIkFKq_D9db1DXUIHwIPlvp/view?usp=sharing) | [pcpvt_s.txt](https://github.com/Meituan-AutoML/Twins/blob/main/logs/upernet_pcpvt_s.txt) |
| PCPVT-Base   | Twins-PCPVT-B      | 47.1/48.4   | 250      | 74.3        | [pcpvt_base.pth](https://drive.google.com/file/d/16sCd0slLLz6xt3C2ma3TkS9rpMS9eezT/view?usp=sharing) | [pcpvt_b.txt](https://github.com/Meituan-AutoML/Twins/blob/main/logs/upernet_pcpvt_b.txt) |
| PCPVT-Large  | Twins-PCPVT-L      | 48.6/49.8   | 269      | 91.5        | [pcpvt_large.pth](https://drive.google.com/file/d/1wsU9riWBiN22fyfsJCHDFhLyP2c_n8sk/view?usp=sharing) | [pcpvt_l.txt](https://github.com/Meituan-AutoML/Twins/blob/main/logs/upernet_pcpvt_l.txt) |
| ALTGVT-Small | Twins-SVT-S        | 46.2/47.1   | 228      | 54.4        | [alt_gvt_small.pth](https://drive.google.com/file/d/18OhG0sbAJ5okPj0zn-8YTydKG9jS8TUx/view?usp=sharing) | [svt_s.txt](https://github.com/Meituan-AutoML/Twins/blob/main/logs/upernet_svt_s.txt) |
| ALTGVT-Base  | Twins-SVT-B        | 47.4/48.9   | 261      | 88.5        | [alt_gvt_base.pth](https://drive.google.com/file/d/1LNtdvACihmKO6XyBPoJDxbrd6AuHVVvq/view?usp=sharing) | [svt_b.txt](https://github.com/Meituan-AutoML/Twins/blob/main/logs/upernet_svt_b.txt) |
| ALTGVT-Large | Twins-SVT-L        | 48.8/50.2   | 297      | 133         | [alt_gvt_large.pth](https://drive.google.com/file/d/1xS91hytfzuMZ5Rgb-W-cOJ9G7ptjVwlO/view?usp=sharing) | [svt_l.txt](https://github.com/Meituan-AutoML/Twins/blob/main/logs/upernet_svt_l.txt) |



## 3. 代码

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import Attention as TimmAttention


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


class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                     drop_path, act_layer, norm_layer)

    def forward(self, x, H, W):
        return super(SBlock, self).forward(x)


class GroupBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)

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


# borrow from PVT https://github.com/whai362/PVT.git
class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))
            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(embed_dims) - 1 else self.patch_embeds[
                -1].num_patches
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, patch_num, embed_dims[i])))
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k])
                for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.norm = norm_layer(embed_dims[-1])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))

        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for pos_emb in self.pos_embeds:
            trunc_normal_(pos_emb, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
            cur += self.depths[k]

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
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class CPVTV2(PyramidVisionTransformer):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block):
        super(CPVTV2, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads, mlp_ratios,
                                     qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, depths,
                                     sr_ratios, block_cls)
        del self.pos_embeds
        del self.cls_token
        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def no_weight_decay(self):
        return set(['cls_token'] + ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x.mean(dim=1)  # GAP here


class PCPVT(CPVTV2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=SBlock):
        super(PCPVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                    mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                    norm_layer, depths, sr_ratios, block_cls)


class ALTGVT(PCPVT):
    """
    alias Twins-SVT
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=GroupBlock, wss=[7, 7, 7]):
        super(ALTGVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                     mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                     norm_layer, depths, sr_ratios, block_cls)
        del self.blocks
        self.wss = wss
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.apply(self._init_weights)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pcpvt_small_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pcpvt_base_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pcpvt_large_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def alt_gvt_small(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def alt_gvt_base(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    model.default_cfg = _cfg()
    return model


@register_model
def alt_gvt_large(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    model.default_cfg = _cfg()
    return model

~~~





参考资料

> [【NeurIPS2021】Twins: Revisiting the Design of Spatial Attention in Vision Transformers - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/445328841)
>
> [NeurIPS 2021 ｜ Twins：重新思考高效的视觉注意力模型设计_美团技术团队的博客-CSDN博客_视觉注意模型](https://blog.csdn.net/MeituanTech/article/details/123725459)
>
> [NeurIPS 2021 ｜ Twins：重新思考高效的视觉注意力模型设计 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/487435548)