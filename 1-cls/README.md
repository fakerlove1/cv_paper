

# 图像分类backbone

## 1. 常见模型

### 1.1 常规的模型

* 2012-ALexNet
* 2014-VGG
* 2014-GoogLeNet
* 2015-ResNet
* 2018-CBAM
* 2019-HRNet
* 2019-EfficientNetV1
* 2020 ECA-Net
* 2020-VIT
* 2021-Swin-Transformer
* 2022-ConvNext
* 



测试输入

~~~python
input = torch.randn(1, 3, 224, 224)
~~~

分类都是1000类，性能比较



| 网络名称              | backbone名称       | 参数量   | 浮点数  |
| --------------------- | ------------------ | -------- | ------- |
| 2014-VGG              | VGG16              | 138.358M | 15.484G |
|                       | VGG19              | 143.667M | 19.647G |
| 2015-ResNet           | ResNet50           | 25.557M  | 4.112G  |
|                       | ResNet101          | 44.549M  | 7.834G  |
| 2016-ResNext          | ResNext50          |          |         |
| 2017-DenseNet         | DenseNet           | 1.109M   | 14.530G |
|                       | densenet121        | 7.979M   | 2.866G  |
|                       | densenet201        | 20.014M  | 4.341G  |
|                       |                    |          |         |
| 2017-MobleNetV1       | MobleNetV1         |          |         |
|                       | MobleNetV2         | 3.505M   | 0.314G  |
|                       | mobilenet_v3_small | 2.543M   | 0.060G  |
| 2017-ShuffleNet       | ShuffleNetV1       | 1.878M   | 0.146G  |
|                       | ShuffleNetV2_b1    | 2.716M   | 0.277G  |
|                       |                    |          |         |
| 2019-HRNet            | hrnet18            | 9.641M   | 3.557G  |
|                       | hrnet48            | 65.859M  | 17.944G |
| 2019-EfficientNet     | efficientnet_b0    | 5.289M   | 0.402G  |
|                       | efficientnet_b7    | 66.348M  | 5.267G  |
| 2020-ViT              | ViT-B/16           | 86M      | 16.849G |
|                       | ViT-Large          | 307M     |         |
|                       | ViT-Huge           | 632M     |         |
|                       |                    |          |         |
| 2021-Swin-Transformer | swin-transformer   | 28.240M  | 4.351G  |
|                       |                    |          |         |
| 2022-ConvNext         | convnext_tiny      | 28.566M  | 4.457G  |



### 1.2 轻量模型

* 2017-SqueezeNet
* 2017-MobleNetV1





| 网络名称        | backbone名称       | 参数量 | 浮点数 |
| --------------- | ------------------ | ------ | ------ |
|                 |                    |        |        |
| 2017-MobleNetV1 | MobleNetV1         |        |        |
|                 | MobleNetV2         | 3.505M | 0.314G |
|                 | mobilenet_v3_small | 2.543M | 0.060G |
| 2017-ShuffleNet | ShuffleNetV1       | 1.878M | 0.146G |
|                 | ShuffleNetV2_b1    | 2.716M | 0.277G |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
|                 |                    |        |        |
| 2022 EdgeViTs   | S                  | 11.1M  | 1.9G   |
|                 | XS                 | 6.7M   | 1.1G   |
|                 | XXS                | 4.1M   | 0.6G   |
|                 |                    |        |        |
|                 |                    |        |        |





## 2. 创新点

* MobleNetV1使用了深度可分离卷积
* ShuffleNetV1使用 组卷积+通道混排 代替了1*1逐点卷积
* ShuffleNetV2 使用了另一种指标来代替FLOPs，给出了设计网络的4个原则，根据原则重新设计了ShuffleNet Unit的单元

* ResNet使用了x=x+residual这样子的残差结构，实现了深层次网络的训练
* EfficientNetV1从 网络的看宽度，深度，输入的大小 3个方向，重新思考了3个方向对网络的影响。并且加入了SE注意力模块





## 3. 数据集上成绩

### CIFAR-100

| 模型   | 大小 | 准确度 | 是否使用额外数据集 |
| ------ | ---- | ------ | ------------------ |
| EffNet | L2   | 96.08  | 是                 |
| ViT    | Base | 94.55  |                    |
|        |      |        |                    |





### ImageNet10k





|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |

参考资料

