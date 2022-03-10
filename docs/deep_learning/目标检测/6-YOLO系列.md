# YOLO系列对比整理

## YOLOv1
### 论文核心思路

1) 将一幅图像分成SxS个网格(grid cell), 如果某个object的中心落在这个网格 中，则这个网格就负责预测这个object

![](https://gitee.com/coronapolvo/images/raw/master/20220309220454.png)

2) 每个网格要预测B个bounding box, 每个bounding box 除了要预测位置之外，还要附带预测一个confidence值，每个网格还要预测C个类别的分数。 confidence表示该位置有物体的概率

![](https://gitee.com/coronapolvo/images/raw/master/20220309220632.png)

3) 网络后处理，使用nms对多余的box去除；

![](https://gitee.com/coronapolvo/images/raw/master/20220309224307.png)


### YOLOV1网络结构



![](https://gitee.com/coronapolvo/images/raw/master/20220309225308.png)


YOLOv1的网络结构可以说是比较简单了，网络结构主要由卷积、池化和全连接三部分组成。最终的输出结构为7x7x30。其中7x7表示一个图片会被划分为7x7的网格，下图更加详细的说明了30的组成部分：

![](https://gitee.com/coronapolvo/images/raw/master/20220309230236.png)

30分别由两个预测框信息，两个置信度信息和20个类别置信度组成（Pascal VOC数据集）

### 损失函数

![](https://gitee.com/coronapolvo/images/raw/master/20220309230500.png)

YOLOv1的损失函数主要采用了平方距离的方式。需要注意的是对于w和h的误差，yolov1采用了根号差平方的形式。

### 一些小问号

:::tip
为什么yolov1的损失函数中在求box的w和h的误差时要先开根号？
:::

因为如果不开根号对于大物体和小物体相同的w和h就会产生相同的误差，如下图所示：

![](https://gitee.com/coronapolvo/images/raw/master/20220309231426.png)

很明显上图的大目标的预测结果要比小目标的预测结果好，但是却具有相同的w。开根号能够使用具有相同w的大物体产生更小的误差，具有相同w的小物体产生更大的误差。

![](https://gitee.com/coronapolvo/images/raw/master/20220309231822.png)



:::tip
YOLOv1的损失函数由几部分组成？
:::

1.  bounding box 损失：box坐标的x和y坐标分别作差求平方后求和
2.  confidence 损失：confidence 作差求平方和
3.  class 损失：类别作差平方求和


:::tip
Yolov1的问题有哪些？
:::

1. 对于群体性的小目标检测结果很差：在YOLOv1的思想中每一个cell只预测两个bounding box而且这两个boxes都是同一个类别的。对于没有每一个cell都只预测一组同一个类别的目标。所以当小目标聚集在一起的时候就会检测效果很差；
2. 对于不同尺寸的同一个目标检测结果并不是很理想
3. 主要错误的原因都是源于定位不准确的问题：这是因为YOLOv1采用直接预测坐标的一个方式，而不是类似Faster RCNN一样采用一个anchor base的方法。



