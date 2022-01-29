---
title: 「YOLOv3」从头实现YOLOv3目标检测（四）置信度阈值与NMS
date: 2021-08-19 15:14:54
tags: [yolo,深度学习基础知识]
categories: [深度学习基础知识,yolo]
cover: https://gitee.com/coronapolvo/images/raw/master/20210818200139maxresdefault-2.jpg
katex: true
---

本教程转载于：[https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/), 在原教程上加入了自己的理解，我的理解将用 `这样的格式写出`.  （ 原博客中有错误的地方在本文中也进行了修正 ）

这是从头开始实现 YOLO v3检测器的教程的第4部分。在上一部分，我们实现了网络的前向传递。在这一部分中，我们通过一个目标置信度和一个非最大抑制度来筛选我们的检测结果。

在前面的部分中，我们建立了一个模型，输出给定一个输入图像的多个目标检测。准确地说，我们的输出是一个形状为 b x 22743 x 85的张量。B 是一批图像的数量，22743是每张图像预测的box的数量，85是包围盒属性的数量。

然而，正如第1部分所描述的，我们必须将我们的输出经过置信度阈值和非极大值抑制的过滤，以获得真实的检测结果。为此，我们将在`utils.py`中创建一个名为`write_result`的函数。

```python
def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
```

函数以`prediction`、`confidence`(objectness score threshold)、 `num_classes` (在我们的例子中是80)和 NMS _ conf (NMS IoU 阈值)作为输入。

我们还需要将置信度低于阈值的项至为0:

```python
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
```

# Object Confidence Thresholding

我们的预测张量包含了关于 B x 22743个box的信息。对于每个得分低于阈值的box盒，我们将其每个属性(表示box的整行)的值设置为零。

# 执行非最大值抑制

> 注意: 我假设您理解 IoU (Intersection over union)是什么，以及非最大抑制是什么。如果事实并非如此，请参考文章末尾的链接)。

>还有一点需要知道的就是YOLOv3中NMS在训练的过程中不执行的，在本套课程里面我们只写了推理的过程，并没有写计算loss的过程。所以就连带NMS进行了编写。正式因为不需要训练，所以也就不需要保留梯度。所以你可以在代码中看到将tensor一会加载到cpu一会加载到gpu上的操作。这种操作在训练过程中都是不允许出现的。

我们现在拥有的box属性是由中心坐标以及边界框的高度和宽度来描述的。然而，使用每个box的一对对角点的坐标来计算两个盒子的IOU更容易。因此，我们将框的(center x，center y，height，width)属性转换为(left-top x，left-top y，right-bottom x，right-bottom y)。

```python
    # xywh => xyxy
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]
```

每幅图像中真实值的数量可能不同。例如，一批图像1、2和3分别有5、2和4个真实检测值。因此，一次只能对一幅图像进行置信阈值分割和 NMS 处理。这意味着，我们不能对所涉及的操作进行矢量化，而必须通过一个for循环来对每一个图片进行处理。

```python
    write = False
    for ind in range(batch_size):
        image_pred = prediction[ind]
        # 置信度过滤
        # NMS
```

`write` 标志用于表示我们有没有初始化输出。

一旦进入循环，让我们捋清楚一下思路。注意每个box行有85个属性，其中80个是类分数。在这一点上，我们只关心具有最大值的类分数。因此，我们从每一行中删除80个类分数，然后添加具有最大值的类的索引以及该类的类分数。

```python
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
```

还记得我们已经将对象置信度小于阈值的box行设置为零吗？下面我们将去除这些box

```python
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue
```

try-except 块用于处理我们没有检测到物体的情况，在这种情况下，我们使用 continue 。

现在，让我们在一个图像中检测出类。

```python
        image_classes = unique(image_pred_[:, -1])
				

def unique(tensor):
    unique_tensor = torch.unique(tensor.clone())
    return unique_tensor
```

然后，我们执行 NMS .

```python
        for cls in img_classes:		
```

一旦进入循环，我们要做的第一件事就是提取特定类的检测(由变量 cls 表示)。我们接着写`write_results`这个函数。

```python
            # 获取某一个特定类别的检测结果
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
            # 根据objectness score 进行排序
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # 检测结果的数量
```

下面我们开始写NMS

```python
            for i in range(idx):
                # 计算所有box的IOU
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # 去除所有IOU小于阈值的检测框
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
```

这里，我们使用一个函数 bbox_iou去计算box的iou

bbox_iou的第二个参数是由多个box组成的张量。`我们将objectness最高的box作为预测值，计算其他box和他的iou值。`

![bbox-3](https://gitee.com/coronapolvo/images/raw/master/20210819183952bbox-3.png)



如果我们有两个同一类的边界框，其中一个IOU大于一个阈值，那么低置信度的那个就被剔除了。我们已经整理好了box，其中包括那些高置信度的box。

在循环的主体中，下面的行给出了 image_pred_class[i+1:] 里面所有box的IOU：

```
ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
```

每次迭代，如果任何一个索引大于 i 的box有一个大于阈值 nms_thresh 的 IoU (包含一个被 i 索引的box) ，那么这个box就被消除。

## 计算IoU

下面是计算IoU的代码：

```python
def bbox_iou(box1, box2):
    """
    返回两个boxes的IoU
    """

    # 获取边界框的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 得到相交矩形的坐标
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.max(b1_x2, b2_x2)
    inter_rect_y2 = torch.max(b1_y2, b2_y2)

    # 交叉面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)
    # union 面积
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou
```

# 写预测代码

`write_results`的输出是一个纬度为D x 8的张量。这里 D 是所有图像中的真实预测值，每个图像由一行表示。每个检测结果具有8个属性，即检测所属的批中图像的索引、4个角坐标、objectness score、最大的类置信度和该类的索引。

正如以前一样，我们不初始化输出张量，除非我们有一个检测分配给它。一旦它被初始化，我们就其与后续的检测结果连接。

```python
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
```

在函数的末尾，我们检查输出是否已经初始化。如果没有意味着在这个batch的图片里面没有任何一张图片。在这种情况下，我们返回0。

```python
    try:
        return output
    except:
        return 0
```

这就是这篇文章的内容。在这篇文章的最后，我们终于有了一个预测的形式张量列出每个预测的结果。现在唯一剩下的，就是创建一个输入管道，从磁盘读取图像，计算预测，在图像上绘制边界框，然后显示/写入这些图像。这就是我们在下一部分将要做的。

# 相关学习链接

1. [PyTorch tutorial](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
2. [IoU](https://www.youtube.com/watch?v=DNEm4fJ-rto)
3. [Non maximum suppresion](https://www.youtube.com/watch?v=A46HZGR5fMw)
4. [Non-maximum Suppression](https://www.google.co.in/search?q=NMS+python&oq=NMS+python+&aqs=chrome..69i57j35i39.2657j0j7&sourceid=chrome&ie=UTF-8)

















