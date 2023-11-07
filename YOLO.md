#### YOLOv1

滑动窗口 回归$(c,x,y,w,h)$ 红色的是onehot 用来区分大目标和小目标

![img](https://pic2.zhimg.com/80/v2-efb58a2c9d3e881df099789f640471ad_720w.webp)

![img](https://pic4.zhimg.com/80/v2-3af308f7096bda4c621c077302b90533_720w.webp)

![img](https://pic3.zhimg.com/80/v2-ce26d13cfd3b7145f4594524435a9b92_720w.webp)

利用卷积，相当于将图片分割为$7\times7$的feature map。预测目标中心点都各自是否在$7\times7$中。蓝色和绿色分别对应于大目标和小目标

![这里写图片描述](https://img-blog.csdn.net/20171011213236071?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTk3NDYzOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### YOLOv2

与RCNN类似采用了Anchor Box的结构

**anchor是从数据集中统计得到的(Faster-RCNN中的Anchor的宽高和大小是手动挑选的)。**

![img](https://pic4.zhimg.com/80/v2-4883b178ed0e2bb95f1d504dc6bed6a7_720w.webp)



偏移量。另一个重要的原因是：直接预测位置会导致神经网络在一开始训练时不稳定，使用偏移量会使得训练过程更加稳定，性能指标提升了5%左右。

预测的位置上不使用Anchor框，宽高上使用Anchor框。以上就是YOLO v2的一个改进。

YOLOv2将最后的$7\times7$增加到了$13\times13$。并且每个位置对应着5个Anchor，20个类别，则是$13\times13\times125$

其中预测的是$t_x,t_y,t_w,t_h$，为偏移量，这样预测偏移量就可以更准确



#### YOLOv3

改进了检测头

![img](https://pic1.zhimg.com/80/v2-4cf1b6f6afec393122305ca2bb2725a4_720w.webp)

三个检测头分别预测大中小目标，提升YOLO预测小目标的能力

但每一个Anchor仅与一个ground truth框相对应，对应的方法如下：

- 对于每一个ground truth边界框，计算其与所有锚框之间的IoU（Intersection over Union，交并比）。
- 选择IoU值最大的那个锚框与这个ground truth边界框进行匹配。

![img](https://pic4.zhimg.com/80/v2-1714579e2a7f9ca88335bdaeae9e1c4f_720w.webp)

#### YOLOv4

改进：

**1.Using multi-anchors for single ground truth**

YOLOv3负责用1个Anchor负责一个GT，v4中利用多个Anchor负责一个GT，利用$IoU(anchor_i,GT)>threshol$来确定，这就相当于你anchor框的数量没变，但是选择的**正样本**的比例增加了，就**缓解了正负样本不均衡的问题**

**2.Eliminate_grid sensitivity**

![image-20231019084938008](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231019084938008.png)

**3.CIoU-loss**

这里采用了改进的IoU Loss

$$L_{CIoU}=1-IoU+\frac{\rho^2(b,b^gt)}{c^2}+\alpha v$$

中间的分式衡量了目标框中心点和GT中心点的距离，最后的则惩罚了彻底包含的问题

#### YOLOv5

加入了自适应anchor选择的聚类方法