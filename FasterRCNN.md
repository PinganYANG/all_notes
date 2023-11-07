<img src="https://pic3.zhimg.com/80/v2-c0172be282021a1029f7b72b51079ffe_1440w.webp" alt="img" style="zoom:50%;" />

#### 1. 四部分

Conv Layers: 使用conv+relu+pooling提取feature maps

RPN Region Proposal Networks: 通过softmax判断anchors+还是-。并利用bbox regression修正**anchor**获得好的proposals

Roi pooling: 综合proposals和feature maps的信息

Classification: 利用proposal feature maps 计算proposals类别，最后利用 **bbox regression**来确定精确位置



目前有两个疑问: 

​	anchors是怎么得到的

​	BBox regression是什么





#### 2. 网络结构

![image-20231017095315882](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231017095315882.png)

#### 3. Conv Layers

结构图见上图feature maps前面的部分，是一个VGG16结构

其中

1. 所有的conv层都是：kernel_size=3，pad=1，stride=1
2. 所有的pooling层都是：kernel_size=2，pad=0，stride=2

conv的pad=1，kernel_size=3 使得$M\times N$ 的维度在经过conv层后不变。pooling的kernel_size=2使得$M\times N$的维度在经过pooling层后维度减半

这也使得卷积后的feature map中feature的位置和原图片中对应的位置相同。

#### 4. Region Proposal Network RPN

![image-20231017125315872](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231017125315872.png)

区别于OpenCV adaboost

R-CNN selective search方法

上图显示出了RPN的结构

网络分为两部分，上方是第一部分，通过softmax分类anchors的+或-。下方第二部分用来计算对于**anchors**的bounding box regression偏移量。最后的Proposal部分负责综合positive anchor和bbox regression的偏移量，同时剔除一些太小或出界的Proposal。

这里用到的anchor即可以解答1中的一个疑问。

##### 4.1 anchors用途和用法

##### <img src="C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231017140937441.png" alt="image-20231017140937441" style="zoom:50%;" />

anchor本身是一些通过经验生成的不同尺寸的anchor box。怎么用呢？将这九个anchor box分别用于每个feature map点。其中最后的的conv层输出如果为256张特征图，则每一个特征相当于有256 dim。而每个特征的每个anchor要分+和-，并且要回归（xywh），一共有$256\times featuremap\times9\times6$个参数，过多了。因此训练中会在**合适的anchor**中**随机**选择128个positive anchor和128个negative anchor训练。

**其实RPN最终就是在原图尺度上，设置了密密麻麻的候选Anchor。然后用cnn去判断哪些Anchor是里面有目标的positive anchor，哪些是没目标的negative anchor。所以，仅仅是个二分类而已！**

接下来放进softmax里二分类

<img src="C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231017142845333.png" alt="image-20231017142845333" style="zoom:50%;" />

reshape不用管，是caffe利用blob的格式问题。其中softmax由于只进行了二分类，因此可以用一个sigmoid代替

##### 4.2 Bounding Box Regression 原理

首先对于每一个anchor box，要预测的量就是$t_x t_y t_w t_h$,这些是anchor的偏移量，用来调整anchor中心和长宽来获得精确的Proposal。



![image-20231017143721246](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231017143721246.png)

##### 4.3 Proposal layer

1. 生成anchors，利用$[d_x(A),d_y(A),d_w(A),d_h(A)]$对所有的anchors做bbox regression回归（这里的anchors生成和训练时完全一致）
2. 按照输入的positive softmax scores由大到小排序anchors，提取前pre_nms_topN(e.g. 6000)个anchors，即提取修正位置后的positive anchors
3. 限定超出图像边界的positive anchors为图像边界，防止后续roi pooling时proposal超出图像边界（见文章底部QA部分图21）
4. 剔除尺寸非常小的positive anchors
5. 对剩余的positive anchors进行**NMS**（nonmaximum suppression）
6. Proposal Layer有3个输入：positive和negative anchors分类器结果rpn_cls_prob_reshape，对应的bbox reg的(e.g. 300)结果作为proposal输出



RPN网络结构就介绍到这里，总结起来就是：
**生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals**



#### 5. ROI Pooling

##### 5.1 为什么需要ROI pooling

我们已经知道，Proposal layer会生成很多Proposals，但问题是这些Proposals的形状和大小都不相同。而要将其放到传统CNN进行分类时，其**输入尺寸被要求是相同的**。传统的wrap和crop方法会破坏信息，这样就使用了ROI pooling方法。

##### 5.2 ROI pooling 原理

- 由于proposal是对应MxN尺度的，所以首先使用spatial_scale参数将其映射回(M/16)x(N/16)大小的feature map尺度；
- 再将每个proposal对应的feature map区域水平分为 pool_w*pool_h 的网格；
- 对网格的每一份都进行max pooling处理。

这样处理后，即使大小不同的proposal输出结果都是 pool_w*pool_h 固定大小，实现了固定长度输出。

![img](https://pic1.zhimg.com/80/v2-e3108dc5cdd76b871e21a4cb64001b5c_1440w.webp)



#### 6. Classification

Classification部分利用已经获得的proposal feature maps，通过full connect层与softmax计算每个proposal具体属于那个类别（如人，车，电视等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个proposal的位置偏移量bbox_pred，用于回归更加精确的目标检测框。Classification部分网络结构如图

![img](https://pic2.zhimg.com/80/v2-9377a45dc8393d546b7b52a491414ded_1440w.webp)

***参考 https://zhuanlan.zhihu.com/p/31426458***

