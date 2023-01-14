# FCN论文翻译

[深度学习与TensorFlow:FCN论文翻译 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/38057272)

Abstract

Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixelsto-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build “fully convolutional” networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet [22], the VGG net [34], and GoogLeNet [35]) into fully convolutional networks and transfer their learned representations by fine-tuning [5] to the segmentation task. We then define a skip architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional network achieves stateof-the-art segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes less than one fifth of a second for a typical image.



## 摘要

卷积网络在特征分层领域是非常强大的视觉模型。我们证明了经过`端到端`、`像素到像素`训练的卷积网络可以超过最先进的语义分割技术。我们的关键目标是建立“`全卷积`”网络，可以输入任意大小的数据，经过有效的推理和学习产生相应大小的输出。我们定义并且详细说明了全卷积网络的空间，解释它们在空间范围内dense prediction任务(预测每个像素所属的类别)和获取与先验模型联系的应用。我们`改造当前的分类网络`(AlexNet [22] ,the VGG net [34] , and GoogLeNet [35] )到全卷积网络和通过微调 [5] 传递它们的学习表现到分割任务中。然后我们`定义了一个跳跃式的架构`，结合来自深、粗层的语义信息和来自浅、细层的表征信息来产生准确和精细的分割。我们的完全卷积网络成为了在PASCAL VOC最出色的分割方式（在2012年相对62.2%的平均IU提高了`20%`），NYUDv2，和SIFT Flow,对一个典型图像推理只需要花费不到0.2秒的时间。



Introduction

Convolutional networks are driving advances in recognition. Convnets are not only improving for whole-image classification [22, 34, 35], but also making progress on local tasks with structured output. These include advances in bounding box object detection [32, 12, 19], part and keypoint prediction [42, 26], and local correspondence [26, 10]. The natural next step in the progression from coarse to fine inference is to make a prediction at every pixel. Prior approaches have used convnets for semantic segmentation [30, 3, 9, 31, 17, 15, 11], in which each pixel is labeled with the class of its enclosing object or region, but with shortcomings that this work addresses

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-4feffd0937cc87c39e13e7df4984065a_720w.jpg)

We show that a fully convolutional network (FCN) trained end-to-end, pixels-to-pixels on semantic segmentation exceeds the state-of-the-art without further machinery. To our knowledge, this is the first work to train FCNs end-to-end (1) for pixelwise prediction and (2) from supervised pre-training. Fully convolutional versions of existing networks predict dense outputs from arbitrary-sized inputs. Both learning and inference are performed whole-image-ata-time by dense feedforward computation and backpropagation. In-network upsampling layers enable pixelwise prediction and learning in nets with subsampled pooling. This method is efficient, both asymptotically and absolutely, and precludes the need for the complications in other works. Patchwise training is common [30, 3, 9, 31, 11], but lacks the efficiency of fully convolutional training. Our approach does not make use of pre- and post-processing complications, including superpixels [9, 17], proposals [17, 15], or post-hoc refinement by random fields or local classifiers [9, 17]. Our model transfers recent success in classification [22, 34, 35] to dense prediction by reinterpreting classification nets as fully convolutional and fine-tuning from their learned representations. In contrast, previous works have applied small convnets without supervised pre-training [9, 31, 30]. Semantic segmentation faces an inherent tension between semantics and location: global information resolves what while local information resolves where. Deep feature hierarchies encode location and semantics in a nonlinear



local-to-global pyramid. We define a skip architecture to take advantage of this feature spectrum that combines deep, coarse, semantic information and shallow, fine, appearance information in Section 4.2 (see Figure 3). In the next section, we review related work on deep classification nets, FCNs, and recent approaches to semantic segmentation using convnets. The following sections explain FCN design and dense prediction tradeoffs, introduce our architecture with in-network upsampling and multilayer combinations, and describe our experimental framework. Finally, we demonstrate state-of-the-art results on PASCAL VOC 2011-2, NYUDv2, and SIFT Flow.



## 1. 介绍

卷积网络正在推动识别技术。卷积网络不仅提高了全图分类[22,34,35]，而且在结构化输出的本地局部任务上取得了进展。包括在目标检测边界框 [32,12,19] 、部分和关键点预测 [42,26] 和局部通信 [26,10] 都取得了进步。从粗到细推演的下一步是对每个像素进行预测,早前的方法已经将卷积网络用于语义分割 [30,3,9,31,17,15,11] ,其中每个像素被标记为其封闭对象或区域的类别，但是这项工作也有缺点

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-4feffd0937cc87c39e13e7df4984065a_720w-16492537104002.jpg)

我们研究表明，完全卷积网络（FCN)训练端到端，语义分割上的像素到像素超过了现有技术水平，而无需其他的操作。我们认为，这是第一次训练端到端(1)的FCN在像素级别的预测，而且来自监督式预处理(2)。全卷积在现有的网络基础上从任意尺寸的输入预测密集输出。 学习和推理都是通过密集的前馈计算和反向传播在整个图像上进行的。网内上采样层能在像素级别预测和通过下采样池化学习。

这种方法既快速又绝对有效，并且不需要有像其他工作中的并发问题。atchwise训练是常见的 [30, 3, 9, 31, 11] ，但是缺少了全卷积训练的有效性。我们的方法不是利用预处理或者后期处理解决并发问题，包括超像素 [9,17] ，proposals [17,15] ，或者对通过随机域事后细化或者局部分类 [9,17] 。我们的模型通过重新解释分类网到全卷积网络和微调它们的学习表现将最近在分类上的成功 [22,34,35] 移植到dense prediction。与此相反，先前的工作应用的是小规模、没有超像素预处理的卷积网。

语义分割面临语义和位置之间固有的紧张关系：全局信息解决全局的信息,局部信息解决局部在哪里。 深度特征层次结构对非线性中的位置和语义进行编码.

我们在4.2节(见图3）定义了一种利用集合了深、粗层的语义信息和浅、细层的表征信息的特征谱的跨层架构.在下一节中，我们将回顾有关深度分类网络，FCN和近期使用小网络进行语义分割的方法。 以下部分解释了FCN设计和密集预测折衷方案，将我们的架构与网内上采样和多层组合相结合，并描述了我们的实验框架。 最后，我们在PASCAL VOC 2011-2，NYUDv2和SIFT Flow上展示了最先进的结果.





Related work

Our approach draws on recent successes of deep nets for image classification [22, 34, 35] and transfer learning [5, 41]. Transfer was first demonstrated on various visual recognition tasks [5, 41], then on detection, and on both instance and semantic segmentation in hybrid proposalclassifier models [12, 17, 15]. We now re-architect and finetune classification nets to direct, dense prediction of semantic segmentation. We chart the space of FCNs and situate prior models, both historical and recent, in this framework.

Fully convolutional networks To our knowledge, the idea of extending a convnet to arbitrary-sized inputs first appeared in Matan et al. [28], which extended the classic LeNet [23] to recognize strings of digits. Because their net was limited to one-dimensional input strings, Matan et al. used Viterbi decoding to obtain their outputs. Wolf and Platt [40] expand convnet outputs to 2-dimensional maps of detection scores for the four corners of postal address blocks. Both of these historical works do inference and learning fully convolutionally for detection. Ning et al. [30] define a convnet for coarse multiclass segmentation of C. elegans tissues with fully convolutional inference.

Fully convolutional computation has also been exploited in the present era of many-layered nets. Sliding window detection by Sermanet et al. [32], semantic segmentation by Pinheiro and Collobert [31], and image restoration by Eigen et al. [6] do fully convolutional inference. Fully convolutional training is rare, but used effectively by Tompson et al. [38] to learn an end-to-end part detector and spatial model for pose estimation, although they do not exposit on or analyze this method.

Alternatively, He et al. [19] discard the nonconvolutional portion of classification nets to make a feature extractor. They combine proposals and spatial pyramid pooling to yield a localized, fixed-length feature for classification. While fast and effective, this hybrid model cannot be learned end-to-end.

Dense prediction with convnets Several recent works have applied convnets to dense prediction problems, including semantic segmentation by Ning et al. [30], Farabet et al.[9], and Pinheiro and Collobert [31]; boundary prediction for electron microscopy by Ciresan et al. [3] and for natural images by a hybrid convnet/nearest neighbor model by Ganin and Lempitsky [11]; and image restoration and depth estimation by Eigen et al. [6, 7]. Common elements of these approaches include

• small models restricting capacity and receptive fields;

• patchwise training [30, 3, 9, 31, 11];

• post-processing by superpixel projection, random field regularization, filtering, or local classification [9, 3, 11];

• input shifting and output interlacing for dense output [32, 31, 11];

• multi-scale pyramid processing [9, 31, 11];

• saturating tanh nonlinearities [9, 6, 31]; and

• ensembles [3, 11],

whereas our method does without this machinery. However, we do study patchwise training 3.4 and “shift-and-stitch” dense output 3.2 from the perspective of FCNs. We also discuss in-network upsampling 3.3, of which the fully connected prediction by Eigen et al. [7] is a special case.

Unlike these existing methods, we adapt and extend deep classification architectures, using image classification as supervised pre-training, and fine-tune fully convolutionally to learn simply and efficiently from whole image inputs and whole image ground thruths.

Hariharan et al. [17] and Gupta et al. [15] likewise adapt deep classification nets to semantic segmentation, but do so in hybrid proposal-classifier models. These approaches fine-tune an R-CNN system [12] by sampling bounding boxes and/or region proposals for detection, semantic segmentation, and instance segmentation. Neither method is learned end-to-end. They achieve state-of-the-art segmentation results on PASCAL VOC and NYUDv2 respectively, so we directly compare our standalone, end-to-end FCN to their semantic segmentation results in Section 5.

We fuse features across layers to define a nonlinear localto-global representation that we tune end-to-end. In contemporary work Hariharan et al. [18] also use multiple layers in their hybrid model for semantic segmentation.

## 2. 相关工作

我们的方法利用了深度网络在图像分类[22,34,35]和转移学习方面取得的最新成果[5,41]。转移首先被证明在各种视觉识别任务中[5,41]，然后进行检测，并在混合融合proposal-classification模型中进行实例操作和语义分割操作[12,17,15]。我们现在重新设计和微调分类网络来指导语义分割的密集预测。我们绘制了FCN的空间框架，并在此框架中放置了过去和近期的一些模型。

全卷积网络据我们所知，扩展到任意大小输入的想法首先是由Matan等人提出.它扩展了经典的LeNet网络结构来识别数字字符串(主要是手写体)。由于他们的网络仅限于一维的输入字符串，Matan等人使用维特比解码来获得它们的输出。Wolf和Platt [40]将邮箱地址输出扩展为邮政地址块四个角的检测分数的二维地图。这些历史操作都是为了检测而进行推理和学习的全卷积。Ning等人 [30]用完全卷积推断定义线虫组织的粗糙细胞的分类分割。

全卷积计算在当今许多的多层次网络也被利用。 比如Sermanet等人的滑动窗口检测,Pinheiro和Collobert [31]的语义分割以及Eigen等人的图像恢复都使用了全卷积推理[6]. 全卷积训练是很少见的，但Tompson等人有效地使用了学习一种端到端的局部检测和姿态估计的空间模型方法。 [38]尽管他们不去解释或分析这种方法。

除此之外，He等人 [19使用]丢弃分类网络的非卷积部分来制作特征提取器。他们将proposals和空间金字塔池合并在一起，以产生用于分类的本地化的固定长度特征。尽管快速且有效，但是这种混合模型不能进行端到端的学习。

基于卷积网的dense prediction近期的一些工作已经将卷积网应用于dense prediction问题，其中包括Ning等人的语义分割。,Farabet等人 [9] 以及Pinheiro和Collobert [31] ；Ciresan等人的电子显微镜边界预测 [3] 以及Ganin和Lempitsky [11] 的通过混合卷积网和最邻近模型的处理自然场景图像;还有Eigen等人 [6,7] 的图像修复和深度估计。这些方法的相同点包括如下：

•限制容量和接受范围的小模型;

• patchwise 学习 [30, 3, 9, 31, 11];

•通过超像素投影，随机场正则化，滤波或局部分类进行后处理[9,3,11]

- 输入移位和dense输出的隔行交错输出 [32,31,11]
- 多尺度金字塔处理 [9,31,11]
- 饱和双曲线正切非线性 [9,6,31]
- 集成 [3,11]

而我们的方法没有这个机制。 但是，我们从FCN的角度来研究patchwise训练(3.4节)和“shift-and-stitch”dense输出(3.2节)。我们还讨论了网内上采样(3.3节)，其中Eigen等人完全连接的预测. [7]是一个特例。

与这些现有的方法不同，我们使用图像分类作为监督式预训练来调整和扩展深度分类体系结构，并通过全卷积地进行微调以从整个图像输入和整个图像ground truths学习中简单高效地学习。

Hariharan等人 [17] 和Gupta等人 [15] 也改编深度分类网到语义分割，但是也在混合proposal-classifier模型中这么做了。这些方法通过采样边界框和region proposal进行微调了R-CNN系统 [12] ,用于检测、语义分割和实例分割。这两种办法都不能进行端到端的学习。他们分别在PASCAL VOC和NYUDv2实现了最好的分割效果，所以在第5节中我们直接将我们的独立的、端到端的FCN和他们的语义分割结果进行比较。

我们将各个层的特征融合在一起来定义一个非线性局部到全局的表示，我们可以调整端到端来去协调。 在现在的工作中Hariharan等. [18]在他们的混合模型中也使用多层进行语义分割。



Fully convolutional networks

Each layer of data in a convnet is a three-dimensional array of size h × w × d, where h and w are spatial dimensions, and d is the feature or channel dimension. The first layer is the image, with pixel size h × w, and d color channels. Locations in higher layers correspond to the locations in the image they are path-connected to, which are called their receptive fields.

Convnets are built on translation invariance. Their basic components (convolution, pooling, and activation functions) operate on local input regions, and depend only on relative spatial coordinates. Writing xij for the data vector at location (i, j) in a particular layer, and yij for the following layer, these functions compute outputs yij by

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-bda1d56870a24db67c795737602dda81_720w.jpg)

where k is called the kernel size, s is the stride or subsampling factor, and fks determines the layer type: a matrix multiplication for convolution or average pooling, a spatial max for max pooling, or an elementwise nonlinearity for an activation function, and so on for other types of layers.

This functional form is maintained under composition, with kernel size and stride obeying the transformation rule

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-905836fdd41261d5336a564ccb5f4be0_720w.jpg)

While a general deep net computes a general nonlinear function, a net with only layers of this form computes a nonlinear filter, which we call a deep filter or fully convolutional network. An FCN naturally operates on an input of any size, and produces an output of corresponding (possibly resampled) spatial dimensions.

A real-valued loss function composed with an FCN defines a task. If the loss function is a sum over the spatial dimensions of the final layer, `(x; θ) = P ij ` 0 (xij ; θ), its gradient will be a sum over the gradients of each of its spatial components. Thus stochastic gradient descent on ` computed on whole images will be the same as stochastic gradient descent on ` 0 , taking all of the final layer receptive fields as a minibatch.

When these receptive fields overlap significantly, both feedforward computation and backpropagation are much more efficient when computed layer-by-layer over an entire image instead of independently patch-by-patch.

We next explain how to convert classification nets into fully convolutional nets that produce coarse output maps. For pixelwise prediction, we need to connect these coarse outputs back to the pixels. Section 3.2 describes a trick, fast scanning [13], introduced for this purpose. We gain insight into this trick by reinterpreting it as an equivalent network modification. As an efficient, effective alternative, we introduce deconvolution layers for upsampling in Section 3.3. In Section 3.4 we consider training by patchwise sampling, and give evidence in Section 4.3 that our whole image training is faster and equally effective.

3.1. Adapting classifiers for dense prediction

Typical recognition nets, including LeNet [23], AlexNet [22], and its deeper successors [34, 35], ostensibly take fixed-sized inputs and produce non-spatial outputs. The fully connected layers of these nets have fixed dimensions and throw away spatial coordinates. However, these fully connected layers can also be viewed as convolutions with kernels that cover their entire input regions. Doing so casts them into fully convolutional networks that take input of any size and output classification maps. This transformation is illustrated in Figure 2

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-d9c0509a6dfdd9e07b85056b0914e1d5_720w.jpg)

Furthermore, while the resulting maps are equivalent to the evaluation of the original net on particular input patches, the computation is highly amortized over the overlapping regions of those patches. For example, while AlexNet takes 1.2 ms (on a typical GPU) to infer the classification scores of a 227×227 image, the fully convolutional net takes 22 ms to produce a 10×10 grid of outputs from a 500×500 image, which is more than 5 times faster than the na¨ıve approach1 .

The spatial output maps of these convolutionalized models make them a natural choice for dense problems like semantic segmentation. With ground truth available at every output cell, both the forward and backward passes are straightforward, and both take advantage of the inherent computational efficiency (and aggressive optimization) of convolution. The corresponding backward times for the AlexNet example are 2.4 ms for a single image and 37 ms for a fully convolutional 10 × 10 output map, resulting in a speedup similar to that of the forward pass.

While our reinterpretation of classification nets as fully convolutional yields output maps for inputs of any size, the output dimensions are typically reduced by subsampling. The classification nets subsample to keep filters small and computational requirements reasonable. This coarsens the output of a fully convolutional version of these nets, reducing it from the size of the input by a factor equal to the pixel stride of the receptive fields of the output units.

3.2. Shift-and-stitch is filter rarefaction

Dense predictions can be obtained from coarse outputs by stitching together output from shifted versions of the input. If the output is downsampled by a factor of f, shift the input x pixels to the right and y pixels down, once for every (x, y) s.t. 0 ≤ x, y < f. Process each of these f 2 inputs, and interlace the outputs so that the predictions correspond to the pixels at the centers of their receptive fields.

Although performing this transformation na¨ıvely increases the cost by a factor of f 2 , there is a well-known trick for efficiently producing identical results [13, 32] known to the wavelet community as the a trous algorithm [ ` 27]. Consider a layer (convolution or pooling) with input stride s, and a subsequent convolution layer with filter weights fij (eliding the irrelevant feature dimensions). Setting the lower layer’s input stride to 1 upsamples its output by a factor of s. However, convolving the original filter with the upsampled output does not produce the same result as shift-and-stitch, because the original filter only sees a reduced portion of its (now upsampled) input. To reproduce the trick, rarefy the filter by enlarging it as

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-d63b5285a1d6c079bb75d45cf1395bd6_720w.jpg)

(with i and j zero-based). Reproducing the full net output of the trick involves repeating this filter enlargement layerby-layer until all subsampling is removed. (In practice, this can be done efficiently by processing subsampled versions of the upsampled input.)

Decreasing subsampling within a net is a tradeoff: the filters see finer information, but have smaller receptive fields and take longer to compute. The shift-and-stitch trick is another kind of tradeoff: the output is denser without decreasing the receptive field sizes of the filters, but the filters are prohibited from accessing information at a finer scale than their original design.

Although we have done preliminary experiments with this trick, we do not use it in our model. We find learning through upsampling, as described in the next section, to be more effective and efficient, especially when combined with the skip layer fusion described later on.

3.3. Upsampling is backwards strided convolution

Another way to connect coarse outputs to dense pixels is interpolation. For instance, simple bilinear interpolation computes each output yij from the nearest four inputs by a linear map that depends only on the relative positions of the input and output cells.

In a sense, upsampling with factor f is convolution with a fractional input stride of 1/f. So long as f is integral, a natural way to upsample is therefore backwards convolution (sometimes called deconvolution) with an output stride of f. Such an operation is trivial to implement, since it simply reverses the forward and backward passes of convolution. Thus upsampling is performed in-network for end-to-end learning by backpropagation from the pixelwise loss.

Note that the deconvolution filter in such a layer need not be fixed (e.g., to bilinear upsampling), but can be learned. A stack of deconvolution layers and activation functions can even learn a nonlinear upsampling.

In our experiments, we find that in-network upsampling is fast and effective for learning dense prediction. Our best segmentation architecture uses these layers to learn to upsample for refined prediction in Section 4.2.

3.4. Patchwise training is loss sampling

In stochastic optimization, gradient computation is driven by the training distribution. Both patchwise training and fully convolutional training can be made to produce any distribution, although their relative computational efficiency depends on overlap and minibatch size. Whole image fully convolutional training is identical to patchwise training where each batch consists of all the receptive fields of the units below the loss for an image (or collection of images). While this is more efficient than uniform sampling of patches, it reduces the number of possible batches. However, random selection of patches within an image may be recovered simply. Restricting the loss to a randomly sampled subset of its spatial terms (or, equivalently applying a DropConnect mask [39] between the output and the loss) excludes patches from the gradient computation.

If the kept patches still have significant overlap, fully convolutional computation will still speed up training. If gradients are accumulated over multiple backward passes, batches can include patches from several images.2

Sampling in patchwise training can correct class imbalance [30, 9, 3] and mitigate the spatial correlation of dense patches [31, 17]. In fully convolutional training, class balance can also be achieved by weighting the loss, and loss sampling can be used to address spatial correlation.

We explore training with sampling in Section 4.3, and do not find that it yields faster or better convergence for dense prediction. Whole image training is effective and efficient.



## 3. 全卷积网络

卷积网络中的每一层数据都是尺寸为h×w×d的三维数组，其中h和w是空间维度，d是特征或通道维度。第一层是图像，像素大小为h×w，以及d个颜色通道。 较高层中的位置对应于它们路径连接的图像中的位置，这些位置称为它们的接受域。

卷积网络建立在平移不变性的基础上。 它们的基本组成部分（卷积，池化和激活函数）在局部输入区域上运行，并且仅依赖于相对空间坐标。

在特定层记$X_{ij}$为在坐标(i,j)的数据向量，在following layer有$Y_{ij}$，$Y_{ij}$的计算公式如下:

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-bee06ca39691277708cb545881ff71bb_720w.jpg)

其中k称为卷积核大小，s是步长或二次采样因子，f_ks决定图层类型：一个卷积的矩阵乘或者是平均池化，用于最大池的最大空间值或者是一个激励函数的一个非线性elementwise，亦或是层的其他种类等等。当卷积核尺寸和步长遵从转换规则，这个函数形式被表述为如下形式：

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-1a5e973f8667ad3ed821db84a48e0a74_720w.jpg)

虽然一般深网络计算一般非线性函数，但只有这种形式的层的网络计算非线性滤波器，我们称之为深度滤波器或全卷积网络。 FCN自然地对任何大小的输入进行操作，并产生相应的（可能重新采样的）空间维度的输出。

一个实值损失函数有FCN定义了task。如果损失函数是一个最后一层的空间维度总和,

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-9cc43b0b354371237a9523b8bf22ec16_720w.jpg)

，它的梯度将是它的每层空间组成梯度总和。所以在全部图像上的基于l的随机梯度下降计算将和基于l'的梯度下降结果一样，将最后一层的所有接收域作为minibatch（分批处理）。在这些接收域重叠很大的情况下，前反馈计算和反向传播计算整图的叠层都比独立的patch-by-patch有效的多。

接下来我们将解释如何将分类网转换为生成粗略输出图的全卷积网。 对于像素级预测，我们需要将这些粗略输出连接回像素。 第3.2节描述了一个技巧，快速扫描[13]，为此目的而引入。 我们通过将其重新解释为等效的网络修改来深入了解这一技巧。 作为一种有效的替代方法，我们在3.3节介绍了用于上采样的去卷积层。 在第3.4节中，我们考虑采用patchwise抽样进行训练，并在第4.3节中给出证据，证明我们的整个图像训练速度更快，同样有效.

### 3.1 适用分类器用于dense prediction

典型的识别网络，包括**LeNet** [**23**]，**AlexNet** [**22**]及其更深的继承者[**34**,**35**]，表面上采用固定大小的输入并产生非空间输出。 这些网全连接的层具有固定的尺寸并丢弃空间坐标。 然而，这些完全连接的层也可以被视为与覆盖整个输入区域的内核的卷积。 这样做将它们转换为完全卷积网络，可以输入任意大小和输出分类图。 图**2**说明了这种转换

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-ec669d47fcd6eaf16cd659597fbf578d_720w.jpg)

此外，当作为结果的图在特殊的输入patches上等同于原始网络的估计，计算是高度摊销的在那些patches的重叠域上。例如，当AlexNet花费了1.2ms（在标准的GPU上)推算一个$227*227$图像的分类得分，全卷积网络花费22ms从一张$500*500$的图像上产生一个$10*10$的输出网格，比朴素法快了5倍多。

这些卷积化模式的空间输出图可以作为一个很自然的选择对于dense问题，比如语义分割。每个输出单元ground truth可用，正推法和逆推法都是直截了当的，都利用了卷积的固有的计算效率(和可极大优化性)。对于AlexNet例子相应的逆推法的时间为单张图像时间2.4ms，全卷积的10*10输出图为37ms，结果是相对于顺推法速度加快了。

当我们将分类网络重新解释为任意输出尺寸的全卷积域输出图，输出维数也通过下采样显著的减少了。分类网络下采样使filter保持小规模同时计算要求合理。这使全卷积式网络的输出结果变得粗糙，通过输入尺寸因为一个和输出单元的接收域的像素步长等同的因素来降低它。

### 3.2 Shift-and stitch是滤波稀疏

dense prediction能从粗糙输出中通过从输入的平移版本中将输出拼接起来获得。如果输出是因为一个因子f降低采样，平移输入的x像素到左边，y像素到下面，一旦对于每个(x,y)满足0<=x,y<=f.处理f^2个输入，并将输出交错以便预测和它们接收域的中心像素一致。

尽管单纯地执行这种转换增加了f^2的这个因素的代价，有一个非常有名的技巧用来高效的产生完全相同的结果 [13,32] ，这个在小波领域被称为多孔算法 [27] 。考虑一个层（卷积或者池化）中的输入步长s,和后面的滤波权重为f_ij的卷积层（忽略不相关的特征维数）。设置更低层的输入步长到l上采样它的输出影响因子为s。然而，将原始的滤波和上采样的输出卷积并没有产生和shift-and-stitch相同的结果，因为原始的滤波只看得到（已经上采样）输入的简化的部分。为了重现这种技巧，通过扩大来稀疏滤波，如下:

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-ecfaa3e38c424b6ab83c427b05e56e4d_720w.jpg)

如果s能除以i和j，除非i和j都是0。重现该技巧的全网输出需要重复一层一层放大这个filter知道所有的下采样被移除。（在练习中，处理上采样输入的下采样版本可能会更高效。）

在网内减少二次采样是一种折衷的做法：filter能看到更细节的信息，但是接受域更小而且需要花费很长时间计算。Shift-and -stitch技巧是另外一种折衷做法：输出更加密集且没有减小filter的接受域范围，但是相对于原始的设计filter不能感受更精细的信息。

尽管我们已经利用这个技巧做了初步的实验，但是我们没有在我们的模型中使用它。正如在下一节中描述的，我们发现从上采样中学习更有效和高效，特别是接下来要描述的结合了跨层融合。

### 3.3 上采样是向后向卷积

将粗输出连接到密集像素的另一种方法是内插。例如，简单的双线性插值通过线性映射来计算来自最近四个输入的每个输出yij，线性映射仅依赖于输入单元和输出单元的相对位置。

从某种意义上讲，伴随因子f的上采样是对步长为1/f的分数式输入的卷积操作。.只要f是整数，上采样的一种自然方法就是向后卷积（有时称为反卷积），其输出步幅为f。这样的操作实现起来微不足道，因为它简单地反转了卷积的前进和后退过程。因此，上采样是在网络中进行的，通过从像素方向的损失向后传播进行端到端学习。

注意，这种层中的去卷积滤波器不需要是固定的（例如，对于双线性上采样），但是可以被学习。一堆去卷积层和激活函数甚至可以学习非线性上采样。

在我们的实验中，我们发现网络上采样对于学习密集预测是快速有效的。我们最好的分段体系结构使用这些层来学习在4.2节中进行精确预测的上采样。

### 3.4 patchwise训练是一种损失采样

在随机优化中，梯度计算是由训练分布支配的。patchwise 训练和全卷积训练能被用来产生任意分布，尽管他们相对的计算效率依赖于重叠域和minibatch的大小。在每一个由所有的单元接受域组成的批次在图像的损失之下（或图像的集合）整张图像的全卷积训练等同于patchwise训练。当这种方式比patches的均匀取样更加高效的同时，它减少了可能的批次数量。然而在一张图片中随机选择patches可能更容易被重新找到。限制基于它的空间位置随机取样子集产生的损失（或者可以说应用输入和输出之间的DropConnect mask [39] ）排除来自梯度计算的patches。

如果保存下来的patches依然有重要的重叠，全卷积计算依然将加速训练。如果梯度在多重逆推法中被积累，batches能包含几张图的patches。patcheswise训练中的采样能纠正分类失调 [30,9,3] 和减轻密集空间相关性的影响[31,17]。在全卷积训练中，分类平衡也能通过给损失赋权重实现，对损失采样能被用来标识空间相关。

我们研究了4.3节中的伴有采样的训练，没有发现对于dense prediction它有更快或是更好的收敛效果。全图式训练是有效且高效的。





Segmentation Architecture We cast ILSVRC classifiers into FCNs and augment them for dense

prediction with in-network upsampling and a pixelwise loss. We train for segmentation by fine-tuning. Next, we add skips between layers to fuse coarse, semantic and local, appearance information. This skip architecture is learned end-to-end to refine the semantics and spatial precision of the output. For this investigation, we train and validate on the PASCAL VOC 2011 segmentation challenge [8]. We train with a per-pixel multinomial logistic loss and validate with the standard metric of mean pixel intersection over union, with the mean taken over all classes, including background. The training ignores pixels that are masked out (as ambiguous or difficult) in the ground truth.

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-ddef0bffa814d81d31b50fcd9809e613_720w.jpg)

4.1. From classifier to dense FCN

We begin by convolutionalizing proven classification architectures as in Section 3. We consider the AlexNet3 architecture [22] that won ILSVRC12, as well as the VGG nets [34] and the GoogLeNet4 [35] which did exceptionally well in ILSVRC14. We pick the VGG 16-layer net5 , which we found to be equivalent to the 19-layer net on this task. For GoogLeNet, we use only the final loss layer, and improve performance by discarding the final average pooling layer. We decapitate each net by discarding the final classifier layer, and convert all fully connected layers to convolutions. We append a 1 × 1 convolution with channel dimension 21 to predict scores for each of the PASCAL classes (including background) at each of the coarse output locations, followed by a deconvolution layer to bilinearly upsample the coarse outputs to pixel-dense outputs as described in Section 3.3. Table 1 compares the preliminary validation results along with the basic characteristics of each net. We report the best results achieved after convergence at a fixed learning rate (at least 175 epochs).

Fine-tuning from classification to segmentation gave reasonable predictions for each net. Even the worst model achieved ∼ 75% of state-of-the-art performance. The segmentation-equipped VGG net (FCN-VGG16) alreadyTable 1. We adapt and extend three classification convnets. We compare performance by mean intersection over union on the validation set of PASCAL VOC 2011 and by inference time (averaged over 20 trials for a 500 × 500 input on an NVIDIA Tesla K40c). We detail the architecture of the adapted nets with regard to dense prediction: number of parameter layers, receptive field size of output units, and the coarsest stride within the net. (These numbers give the best performance obtained at a fixed learning rate, not best performance possible.)

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-1b6b03aca30e8bec446999b52fefc88e_720w.jpg)

appears to be state-of-the-art at 56.0 mean IU on val, compared to 52.6 on test [17]. Training on extra data raises FCN-VGG16 to 59.4 mean IU and FCN-AlexNet to 48.0 mean IU on a subset of val7 . Despite similar classification accuracy, our implementation of GoogLeNet did not match the VGG16 segmentation result.

4.2. Combining what and where

We define a new fully convolutional net (FCN) for segmentation that combines layers of the feature hierarchy and refines the spatial precision of the output. See Figure 3.

While fully convolutionalized classifiers can be fine- tuned to segmentation as shown in 4.1, and even score highly on the standard metric, their output is dissatisfyingly coarse (see Figure 4). The 32 pixel stride at the final prediction layer limits the scale of detail in the upsampled output.

We address this by adding skips [1] that combine the final prediction layer with lower layers with finer strides. This turns a line topology into a DAG, with edges that skip ahead from lower layers to higher ones (Figure 3). As they see fewer pixels, the finer scale predictions should need fewer layers, so it makes sense to make them from shallower net outputs. Combining fine layers and coarse layers lets the model make local predictions that respect global structure. By analogy to the jet of Koenderick and van Doorn [21], we call our nonlinear feature hierarchy the deep jet.

We first divide the output stride in half by predicting from a 16 pixel stride layer. We add a 1 × 1 convolution layer on top of pool4 to produce additional class predictions. We fuse this output with the predictions computed on top of conv7 (convolutionalized fc7) at stride 32 by adding a 2× upsampling layer and summing6 both predictions (see Figure 3). We initialize the 2× upsampling to bilinear interpolation, but allow the parameters to be learned as described in Section 3.3. Finally, the stride 16 predictions are upsampled back to the image. We call this net FCN-16s. FCN-16s is learned end-to-end, initialized with the parameters of the last, coarser net, which we now call FCN-32s. The new parameters acting on pool4 are zeroinitialized so that the net starts with unmodified predictions. The learning rate is decreased by a factor of 100.

Learning this skip net improves performance on the validation set by 3.0 mean IU to 62.4. Figure 4 shows improvement in the fine structure of the output. We compared this fusion with learning only from the pool4 layer, which resulted in poor performance, and simply decreasing the learning rate without adding the skip, which resulted in an insignificant performance improvement without improving the quality of the output.

We continue in this fashion by fusing predictions from pool3 with a 2× upsampling of predictions fused from pool4 and conv7, building the net FCN-8s. We obtain a minor additional improvement to 62.7 mean IU, and find a slight improvement in the smoothness and detail of our output. At this point our fusion improvements have met diminishing returns, both with respect to the IU metric which emphasizes large-scale correctness, and also in terms of the improvement visible e.g. in Figure 4, so we do not continue fusing even lower layers.

Refinement by other means Decreasing the stride of pooling layers is the most straightforward way to obtain finer predictions. However, doing so is problematic for our VGG16-based net. Setting the pool5 stride to 1 requires our convolutionalized fc6 to have kernel size 14 × 14 to

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-a362861a0f0641e1a3cf3ca7c2f10286_720w.jpg)

maintain its receptive field size. In addition to their computational cost, we had difficulty learning such large filters. We attempted to re-architect the layers above pool5 with smaller filters, but did not achieve comparable performance; one possible explanation is that the ILSVRC initialization of the upper layers is important.

Another way to obtain finer predictions is to use the shiftand-stitch trick described in Section 3.2. In limited experiments, we found the cost to improvement ratio from this method to be worse than layer fusion.

4.3. Experimental framework

Optimization We train by SGD with momentum. We use a minibatch size of 20 images and fixed learning rates of 10−3 , 10−4 , and 5 −5 for FCN-AlexNet, FCN-VGG16, and FCN-GoogLeNet, respectively, chosen by line search. We use momentum 0.9, weight decay of 5 −4 or 2 −4 , and doubled learning rate for biases, although we found training to be sensitive to the learning rate alone. We zero-initialize the class scoring layer, as random initialization yielded neither better performance nor faster convergence. Dropout was included where used in the original classifier nets.

Fine-tuning We fine-tune all layers by backpropagation through the whole net. Fine-tuning the output classifier alone yields only 70% of the full finetuning performance as compared in Table 2. Training from scratch is not feasible considering the time required to learn the base classification nets. (Note that the VGG net is trained in stages, while we initialize from the full 16-layer

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-0b052693710cb2133523de514f6d5866_720w.jpg)

version.) Fine-tuning takes three days on a single GPU for the coarse FCN-32s version, and about one day each to upgrade to the FCN-16s and FCN-8s versions

More Training Data The PASCAL VOC 2011 segmentation training set labels 1112 images. Hariharan et al. [16] collected labels for a larger set of 8498 PASCAL training images, which was used to train the previous state-of-theart system, SDS [17]. This training data improves the FCNVGG16 validation score7 by 3.4 points to 59.4 mean IU.

Patch Sampling As explained in Section 3.4, our full image training effectively batches each image into a regular grid of large, overlapping patches. By contrast, prior work randomly samples patches over a full dataset [30, 3, 9, 31, 11], potentially resulting in higher variance batches that may accelerate convergence [24]. We study this tradeoff by spatially sampling the loss in the manner described earlier, making an independent choice to ignore each final layer cell with some probability 1−p. To avoid changing the effective batch size, we simultaneously increase the number of images per batch by a factor 1/p. Note that due to the efficiency of convolution, this form of rejection sampling is still faster than patchwise training for large enough values of p (e.g., at least for p > 0.2 according to the numbers in Section 3.1). Figure 5 shows the effect of this form of sampling on convergence. We find that sampling does not have a significant effect on convergence rate compared to whole image training, but takes significantly more time due to the larger number of images that need to be considered per batch. We therefore choose unsampled, whole image training in our other experiments.

Class Balancing Fully convolutional training can balance classes by weighting or sampling the loss. Although our labels are mildly unbalanced (about 3/4 are background), we find class balancing unnecessary.

Dense Prediction The scores are upsampled to the inputdimensions by deconvolution layers within the net. Final layer deconvolutional filters are fixed to bilinear interpolation, while intermediate upsampling layers are initialized to bilinear upsampling, and then learned.

Augmentation We tried augmenting the training data by randomly mirroring and “jittering” the images by translating them up to 32 pixels (the coarsest scale of prediction) in each direction. This yielded no noticeable improvement.

Implementation All models are trained and tested with Caffe [20] on a single NVIDIA Tesla K40c. Our models and code are publicly available at [http://fcn.berkeleyvision.org](https://link.zhihu.com/?target=http%3A//fcn.berkeleyvision.org).

## 4.分割架构

我们将ILSVRC分类投射到FCN中，并将它们用于网络上采样和像素损失的密集预测。 我们通过微调分割进行训练。 接下来，我们在图层之间添加跨层来融合粗略，语义和局部的外观信息。 这种跨越式的结构可以端到端地学习来改进输出的语义和空间精度。

为了这项调查，我们为PASCAL VOC 2011分割挑战赛来进行训练和验证。 我们用逐像素多项式逻辑损失进行训练，并用联合的平均像素交叉点的标准度量来验证，其中包括背景在内的所有类别的均值。 该训练忽略在groud truth实况中被掩盖（模棱两可或很难辨认）的像素。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-1384dd3692b34420431b1ffb2590635d_720w.jpg)

### 4.1 从分类器到密集的FCN

我们首先对第三部分中经过验证的分类体系结构进行卷积处理。我们认为赢得ILSVRC12的AlexNet3体系结构[22]，以及在ILSVRC14中的VGG网络[34]和GoogLeNet4 [35]做的很不错。我们选择了VGG的16层net5，我们发现它等同于19层网络的分类效果。对于GoogLeNet，我们只使用最终的损失层，并通过丢弃最后的平均池化层来提高性能。我们通过丢弃最终的分类器层来斩断每个网络的开始，并将所有的全连接层转换为卷积。我们在信道维数21上附加1×1卷积来预测每个粗略输出位置的每个PASCAL类别（包括背景）的分数，然后是一个去卷积层，将粗略输出双线性上采样为像所描述的像素密集输出在3.3节中。表1比较了初步验证结果和每个网络的基本特征。我们发现以固定学习率（至少175个epochs）收敛后取得的最佳成果。

从分类到分割的微调给每个网络提供了合理的预测。 即使是最糟糕的模型也达到了75％的表现。 配备分段的VGG网络（FCN-VGG16）已经在表1中。我们修改和扩展了三个分类网格。 我们通过PASCAL VOC 2011验证集上的均值交叉点平均交叉比和推理时间（在NVIDIA Tesla K40c上对500×500输入进行20次试验的平均值）比较性能。 我们在密集预测方面详细介绍了适应网络的结构：参数层的数量，输出单元的接受场大小和网内最粗糙的步幅。 （这些数字能够以固定的学习速度获得最佳性能，而不是最佳性能。）

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-8a832b16baae97c463422d5743e544f9_720w.jpg)

### 4.2: 结合什么和在哪里

我们为分割定义了一个新的全卷积网络（FCN），它结合了特征层次结构的层次并提高了输出的空间精度。 参见图3。

虽然全卷积化的分类器可以像4.1中所示的那样进行细化分割，甚至在标准度量上得分很高，但它们的输出却非常粗糙（见图4）。 最终预测层的32像素跨度限制了上采样输出的尺寸的细节范围。

我们提出增加结合了最后预测层和有更细小步长的更低层的跨层信息[1]，将一个线划拓扑结构转变成DAG(有向无环图)，并且边界将从更底层向前跳跃到更高（图3）。因为它们只能获取更少的像素点，更精细的尺寸预测应该需要更少的层，所以从更浅的网中将它们输出是有道理的。结合了精细层和粗糙层让模型能做出遵从全局结构的局部预测。与Koenderick 和an Doorn [21]的jet类似，我们把这种非线性特征层称之为deep jet。

我们首先通过预测16像素跨度层来将输出跨度减半。 我们在pool4的顶部添加一个$1×1$的卷积层来产生额外的类别预测。 我们将这个输出以步长32和conv7（卷积化的fc7）顶部计算预测相加，通过添加$2×2$上采样层并对两个预测进行求和（参见图3）。 我们将2倍上采样初始化为双线性插值，但允许按3.3节所述学习参数。 最后，将步长为16预测被上采样回图像。 我们称之为FCN-16s。 FCN-16是端到端学习的，可以被（我们现在称为FCN-32）的参数进行初始化。 作用于pool4的新参数是初始化为0的，因此网络以未变性修改的预测开始。 学习率降低了100倍的。

学习这种跨层网络能在3.0平均IU的有效集合上提高到62.4。图4展示了在精细结构输出上的提高。我们将这种融合学习和仅仅从pool4层上学习进行比较，结果表现糟糕，而且仅仅降低了学习速率而没有增加跨层，导致了没有提高输出质量的没有显著提高表现。

我们继续融合pool3和一个融合了pool4和conv7的2×上采样预测，建立了FCN-8s的网络结构。在平均IU上我们获得了一个较小的附加提升到62.7，然后发现了一个在平滑度和输出细节上的轻微提高。这时我们的融合提高已经得到了一个衰减回馈，既在强调了大规模正确的IU度量的层面上，也在提升显著度上得到反映，如图4所示，所以即使是更低层我们也不需要继续融合。

其他方式精炼化减少池层的步长是最直接的一种得到精细预测的方法。然而这么做对我们的基于VGG16的网络带来问题。设置pool5的步长到1，要求我们的卷积fc6核大小为14*14来维持它的接收域大小。另外它们的计算代价，通过如此大的滤波器学习非常困难。我们尝试用更小的滤波器重建pool5之上的层，但是并没有得到有可比性的结果；一个可能的解释是ILSVRC在更上层的初始化时非常重要的。

另一种获得精细预测的方法就是利用3.2节中描述的shift-and-stitch技巧。在有限的实验中，我们发现从这种方法的提升速率比融合层的方法花费的代价更高。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-d4b205bba84d9eb087ed88fb80cab02d_720w.jpg)

### 4.3: 实验框架

优化我们利用momentum训练了GSD。 对于FCN-AlexNet，FCN-VGG16和FCN-GoogLeNet，我们使用20个图像的小批量大小和10-3,10-4和5-5的固定学习速率，分别通过各自的线性选择。我们利用了0.9momentum，权重衰减为5 -4或2 -4，并将偏差的学习率加倍，尽管我们发现训练仅仅对学习率敏感。 我们对类评分层进行初始化为0的操作，因为随机初始化既没有更好的性能，也没有更快的收敛性。Dropout被包含在用于原始分类的网络中.

微调我们通过反向传播通过整个网络对所有层进行微调。 考虑到学习基本分类网络所需的时间，单独对输出分类器单独进行微调只能获得完整微调性能的70％，因此从头开始进行培训是不可行的。 （请注意，VGG网络是分阶段训练的，而我们从完整的16层初始化后进行训练)对于粗糙的FCN-32s，在单GPU上，微调要花费三天的时间，而且大约每隔一天就要更新到FCN-16s和FCN-8s版本。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-0f31486c056065cae5489570ea5328be_720w.jpg)

更多的训练数据PASCAL VOC 2011分割训练设置1112张图片的标签。Hariharan等人 [16] 为一个更大的8498的PASCAL训练图片集合收集标签，被用于训练先前的先进系统,SDS [17] 。训练数据将FCV-VGG16得分提高了3.4个百分点到59.4。

patch取样正如3.4节中解释的，我们的全图有效地训练每张图片batches到常规的、大的、重叠的patches网格。相反的，先前工作随机样本patches在一整个数据集 [30,3,9,31,11] ，可能导致更高的方差batches，可能加速收敛 [24] 。我们通过空间采样之前方式描述的损失研究这种折中，以1-p的概率做出独立选择来忽略每个最后层单元。为了避免改变有效的批次尺寸，我们同时以因子1/p增加每批次图像的数量。注意的是因为卷积的效率，在足够大的p值下，这种拒绝采样的形式依旧比patchwose训练要快（比如，根据3.1节的数量，最起码p>0.2）图5展示了这种收敛的采样的效果。我们发现采样在收敛速率上没有很显著的效果相对于全图式训练，但是由于每个每个批次都需要大量的图像，很明显的需要花费更多的时间。

密集预测通过网络内的解卷积层将分数上采样到输入尺寸。 最终层去卷积滤波器被固定为双线性插值，而中间上采样层被初始化为双线性上采样，然后被学习。

通过我们试图通过随机镜像和“抖动”图像，通过将图像翻译为每个方向上的32像素（最粗糙的预测尺度）来增强训练数据。 这没有得到明显的改善。

实施所有型号都经过Caffe [20]训练和测试，使用单个NVIDIA Tesla K40c。 我们的模型和代码可在[http://fcn.berkeleyvision.org](https://link.zhihu.com/?target=http%3A//fcn.berkeleyvision.org)上公开获取。

5. Results

We test our FCN on semantic segmentation and scene parsing, exploring PASCAL VOC, NYUDv2, and SIFT Flow. Although these tasks have historically distinguished between objects and regions, we treat both uniformly as pixel prediction. We evaluate our FCN skip architecture on each of these datasets, and then extend it to multi-modal input for NYUDv2 and multi-task prediction for the semantic and geometric labels of SIFT Flow.

Metrics We report four metrics from common semantic segmentation and scene parsing evaluations that are variations on pixel accuracy and region intersection over union (IU). Let nij be the number of pixels of class i predicted to belong to class j, where there are ncl different classes, and let ti = P j nij be the total number of pixels of class i. We compute:

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-b495ec07903919154b829bf6540c51f2_720w.jpg)

PASCAL VOC Table 3 gives the performance of our FCN-8s on the test sets of PASCAL VOC 2011 and 2012, and compares it to the previous state-of-the-art, SDS [17], and the well-known R-CNN [12]. We achieve the best results on mean IU8 by a relative margin of 20%. Inference time is reduced 114× (convnet only, ignoring proposals and refinement) or 286× (overall).

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-987c51a82c79108515af24073cddcf8a_720w.jpg)

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-bee358b2be3537f08bd7e5690c88af25_720w.jpg)

NYUDv2 [33] is an RGB-D dataset collected using the Microsoft Kinect. It has 1449 RGB-D images, with pixelwise labels that have been coalesced into a 40 class semantic segmentation task by Gupta et al. [14]. We report results on the standard split of 795 training images and 654 testing images. (Note: all model selection is performed on PASCAL 2011 val.) Table 4 gives the performance of our model in several variations. First we train our unmodified coarse model (FCN-32s) on RGB images. To add depth information, we train on a model upgraded to take four-channel RGB-D input (early fusion). This provides little benefit, perhaps due to the difficultly of propagating meaningful gradients all the way through the model. Following the success of Gupta et al. [15], we try the three-dimensional HHA encoding of depth, training nets on just this information, as well as a “late fusion” of RGB and HHA where the predictions from both nets are summed at the final layer, and the resulting two-stream net is learned end-to-end. Finally we upgrade this late fusion net to a 16-stride version.

SIFT Flow is a dataset of 2,688 images with pixel labels for 33 semantic categories (“bridge”, “mountain”, “sun”), as well as three geometric categories (“horizontal”, “vertical”, and “sky”). An FCN can naturally learn a joint representation that simultaneously predicts both types of labels. We learn a two-headed version of FCN-16s with semantic and geometric prediction layers and losses. The learned model performs as well on both tasks as two independently trained models, while learning and inference are essentially as fast as each independent model by itself. The results in Table 5, computed on the standard split into 2,488 training and 200 test images,9 show state-of-the-art performance on both tasks.

## 5. 结果

我们测试我们的FCN语义分割和场景分析，研究了PASCAL VOC，NYUDv2和SIFT Flow。 尽管以前这些任务主要用在物体和区域，但我们将这两种任务统一视为像素预测。 我们在每个数据集上评估FCN跨层式架构，然后将其扩展到NYUDv2的多模式输入以及SIFT Flow的语义和几何标签的多任务预测。

度量 我们从常见的语义分割和场景解析评估中提出四种度量，它们在像素准确率和在联合的区域交叉上是不同的。令n_ij为类别i的被预测为类别j的像素数量，有n_ij个不同的类别，令

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-b495ec07903919154b829bf6540c51f2_720w.jpg)

PASCAL VOC表3给出了我们的FCN-8在PASCAL VOC 2011和2012测试集上的性能，并将其与以前最先进的SDS [17]和众所周知的R-CNN[12]进行比较。 我们在平均IU8上获得了最好的结果，相对提升了为20％。 推断时间减少114×（只有卷积网，没有proposals和微调）或286×（总体）。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-987c51a82c79108515af24073cddcf8a_720w.jpg)

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-bee358b2be3537f08bd7e5690c88af25_720w.jpg)

**NVUDv2** [33]是一种通过利用Microsoft Kinect收集到的RGB-D数据集，含有已经被合并进Gupt等人[14]的40类别的语义分割任务的pixelwise标签。我们报告结果基于标准分离的795张图片和654张测试图片。（注意：所有的模型选择将展示在PASCAL 2011 val上)。表4给出了我们模型在一些变化上的表现。首先我们在RGB图片上训练我们的未经修改的粗糙模型（FCN-32s）。为了添加深度信息，我们训练模型升级到能采用4通道RGB-Ds的输入（早期融合）。这提供了一点便利，也许是由于模型一直要传播有意义的梯度的困难。紧随Gupta等人[15]的成功，我们尝试3维的HHA编码深度，只在这个信息上（即深度）训练网络，和RGB与HHA的“后期融合”一样来自这两个网络中的预测将在最后一层进行总结，结果的双流网络将进行端到端的学习。最后我们将这种后期融合网络升级到16步长的版本。

SIFT Flow是包含33个语义类别（“桥”，“山”，“太阳”）以及三个几何类别（“水平”，“垂直”和“天空”）的像素标签的2,688幅图像的数据集。 FCN可以自然地学习共同的权重，同时预测两种类型的标签。 我们学习了带有语义和几何预测层次和损失的FCN-16的双向版本。 学习模型在两个任务上的表现都与两个独立训练的模型一样好，而学习和推理本身与每个独立模型本质上一样快。 表5中的结果是根据标准划分为2,488个训练和200个测试图像计算得出的，显示了两项任务的优越性能。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-76941a0252f94d806436f678de903337_720w.jpg)

6. Conclusion

Fully convolutional networks are a rich class of models, of which modern classification convnets are a special case. Recognizing this, extending these classification nets to segmentation, and improving the architecture with multi-resolution layer combinations dramatically improves the state-of-the-art, while simultaneously simplifying and speeding up learning and inference.

Acknowledgements This work was supported in part by DARPA’s MSEE and SMISC programs, NSF awards IIS1427425, IIS-1212798, IIS-1116411, and the NSF GRFP, Toyota, and the Berkeley Vision and Learning Center. We gratefully acknowledge NVIDIA for GPU donation. We thank Bharath Hariharan and Saurabh Gupta for their advice and dataset tools. We thank Sergio Guadarrama for reproducing GoogLeNet in Caffe. We thank Jitendra Malik for his helpful comments. Thanks to Wei Liu for pointing out an issue wth our SIFT Flow mean IU computation and an error in our frequency weighted mean IU formula.

A. Upper Bounds on IU

In this paper, we have achieved good performance on the mean IU segmentation metric even with coarse semantic prediction. To better understand this metric and the limits of this approach with respect to it, we compute approximate upper bounds on performance with prediction at various scales. We do this by downsampling ground truth images and then upsampling them again to simulate the best results obtainable with a particular downsampling factor. The following table gives the mean IU on a subset of PASCAL 2011 val for various downsampling factors

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-0fc012f829bcaa465461cf1437be5bc6_720w.jpg)

Pixel-perfect prediction is clearly not necessary to achieve mean IU well above state-of-the-art, and, conversely, mean IU is a not a good measure of fine-scale accuracy.

B. More Results

We further evaluate our FCN for semantic segmentation. PASCAL-Context [29] provides whole scene annotations of PASCAL VOC 2010. While there are over 400 distinct classes, we follow the 59 class task defined by [29] that picks the most frequent classes. We train and evaluate on the training and val sets respectively. In Table 6, we compare to the joint object + stuff variation of Convolutional Feature Masking [4] which is the previous state-of-the-art on this task. FCN-8s scores 37.8 mean IU for a 20% relative improvement.

Changelog

The arXiv version of this paper is kept up-to-date with corrections and additional relevant material. The following gives a brief history of changes.

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-3633825f5432478f3674956c86cf99d2_720w.jpg)

v2 Add Appendix A giving upper bounds on mean IU and Appendix B with PASCAL-Context results. Correct PASCAL validation numbers (previously, some val images were included in train), SIFT Flow mean IU (which used an inappropriately strict metric), and an error in the frequency weighted mean IU formula. Add link to models and update timing numbers to reflect improved implementation (which is publicly available)





## 6. 结论

全卷积网络是丰富的模型类别，其中现代分类网格是一种特殊情况。 认识到这一点，将这些分类网络扩展到分割，并通过多分辨率层组合改进体系结构，极大地改进了最新技术，同时简化和加快了学习和推理。

鸣谢 这项工作有以下部分支持DARPA's MSEE和SMISC项目，NSF awards IIS-1427425, IIS-1212798, IIS-1116411, 还有NSF GRFP,Toyota, 还有 Berkeley Vision和Learning Center。我们非常感谢NVIDIA捐赠的GPU。我们感谢Bharath Hariharan 和Saurabh Gupta的建议和数据集工具;我们感谢Sergio Guadarrama 重构了Caffe里的GoogLeNet;我们感谢Jitendra Malik的有帮助性评论;感谢Wei Liu指出了我们SIFT Flow平均IU计算上的一个问题和频率权重平均IU公式的错误。

A. IU的上限

在本文中，即使使用粗略的语义预测，我们在均值IU分割度量上也取得了很好的性能。 为了更好地理解这个度量和这个方法对它的限制，我们用不同尺度的预测来计算性能的近似上界。 我们通过下采样地面实况图像然后再次对其进行上采样来模拟通过特定下采样因子可获得的最佳结果。 下表给出了各种下采样因子在PASCAL 2011 val子集上的平均IU.pixel-perfect预测很显然在取得最最好效果上不是必须的，而且，相反的，平均IU不是一个好的精细准确度的测量标准。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-0fc012f829bcaa465461cf1437be5bc6_720w.jpg)

**B** 更多的结果

我们将我们的FCN用于语义分割进行了更进一步的评估。PASCAL-Context [29] 提供了PASCAL VOC 2011的全部场景注释。有超过400中不同的类别，我们遵循了 [29] 定义的被引用最频繁的59种类任务。我们分别训练和评估了训练集和val集。在表6中，我们将联合对象和Convolutional Feature Masking [4] 的stuff variation进行比较，后者是之前这项任务中最好的方法。FCN-8s在平均IU上得分为37.8，相对提高了20%

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-3633825f5432478f3674956c86cf99d2_720w.jpg)

更新日志

本文的arXiv版本保持最新，并附有更正和其他相关材料。 以下给出了变化的简要历史。v2 添加了附录A和附录B。修正了PASCAL的有效数量（之前一些val图像被包含在训练中），SIFT Flow平均IU（用的不是很规范的度量），还有频率权重平均IU公式的一个错误。添加了模型和更新时间数字来反映改进的实现的链接（公开可用的）。