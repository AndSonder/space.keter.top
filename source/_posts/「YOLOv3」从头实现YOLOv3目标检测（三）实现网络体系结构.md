---
title: 「YOLOv3」从头实现YOLOv3目标检测（三）实现网络体系结构
date: 2021-08-19 08:28:13
tags: [yolo,深度学习基础知识]
categories: [深度学习基础知识,yolo]
cover: https://gitee.com/coronapolvo/images/raw/master/20210818200139maxresdefault-2.jpg
katex: true
---

本教程转载于：[https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/), 在原教程上加入了自己的理解，我的理解将用 `这样的格式写出`.  （ 原博客中有错误的地方在本文中也进行了修正 ）

这是从头开始实现 YOLO v3检测器的教程的第3部分。在上一部分中，我们实现了 YOLO 体系结构中使用的层，在这一部分中，我们将使用 PyTorch 搭建 YOLO 的网络体系结构，这样我们就可以生成给定图像的输出。

**我们的目标是设计网络的前向传递。**

# 定义网络

正如之前所说的，我们使用`nn.Module`去构建自定义模块。接下来我们将定义检测器的网络结构。在`darknet.py` 中添加如下的内容：

```python
class Darknet(nn.Module):
    def __init__(self, cfg_file_path):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file_path)
        self.net_info, self.module_list = create_modules(self.blocks)
```

现在我们有了Darknet类，现在他有三个初始化变量，block, net_info 和 module_list

# 实现网络的前向传播

`forward`有两个目的。第一计算输出，第二点将输出的feature maps转化为可以被更容易处理的形式。例如通过转换让多个尺度的feature maps可以拼接起来，如果不进行转化，这就是不可能的，因为它们是不同的维度。

```python
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # 我们为路由层缓存输出
```

`forward` 函数有三个参数，self, x 和 CUDA。我们将使用GPU去计算前向传播。

这里我们使用了self.blocks[1:]，因为blocks的第一项是网络一些基本参数。

由于路由层和残差层需要前一层的feature map，因此我们将每一层的输出feature map缓存到 dict 中。

我们现在遍历`module_list`, 它里面包含了网络的结构。 这里需要注意的是，模块的附加顺序与它们在配置文件中的顺序相同。这意味着，我们可以简单地通过每个模块运行来获得输出。

```python
        write = 0  # 这个过会解释
        for i, module in enumerate(modules):
            module_type = (module['type'])
```

## Convolutional and Upsample Layers

如果模块是卷积模块或上采样模块，处理代码如下：

```python
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
```

## Route Layer / Shortcut Layer

根据路由层的代码，我们需要考虑两种情况(如第2部分所述)。对于必须连接两个feature maps的情况，我们使用`torch.cat`方法，第二个参数设为1. 这是因为我们想沿深度连接feature maps。 

```python
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                # layers 长度为1时不需要进行feature map的拼接
                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                # 需要将两个feature maps进行拼接
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]
```

## YOLO (Detection Layer)

YOLO 的输出是一个卷积feature map，包含沿着feature map深度的box属性。网格预测的box属性是一个接一个地堆放在一起的。 如果你必须访问第二个单元格的边界在(5,6) ，那么你必须通过映射[5,6，(5 + c) : 2 * (5 + c)]来索引它。这种形式对于目标置信阈值化、向添加中心网格偏移量、应用锚点等输出处理非常不方便。

另一个问题是，由于检测发生在三个尺度，预测图的维度将会不同。虽然这三个特征映射的尺寸不同，但是对它们进行的输出处理操作是相似的。如果可以对单个张量而不是对三个单独的张量进行操作就好了。

为了解决这些问题，我们引入了函数 `predict_transform`. 

# 转化输出

函数`predict_transform` 在`util.py`中。

在`util.py`的头部添加如下代码：

```python
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
```

`predict_transform` 一共有五个参数：

```python
def predict_transform(prediction, inp_dim, anchors, num_class, CUDA=True):
    
```

函数获取一个检测feature map，并将其转化为一个二维张量，其中每一行对应于一个box的属性，顺序如下

![bbox_-2](https://gitee.com/coronapolvo/images/raw/master/20210819092910bbox_-2.png)

下面是执行上述转换的代码。

```python
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
```

锚的尺寸与net的`height`和`width`属性一致。这些属性描述了输入图像的尺寸，它比检测图大(大一个步幅因子)。因此，必须根据检测feature map的步长来划分锚点。

```python
    anchors = [(a[0]/stride,a[1]/stride) for a in anchors]
```

现在，我们需要根据第1部分中讨论的公式转换输出

将 x，y 坐标和objection score输出到sigmoid中：

```python
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
```

然后，将网格偏移量添加到中心坐标中协调预测：

```python
    # 添加网格偏移量
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset
```

接下来，在边界框的尺寸上应用anchors：

```python
    # 在边界框的尺寸上应用锚点
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
```

对分类score使用sigmoid激活函数：

```python
prediction[:, :, 5:5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
```

我们在这里要做的最后一件事，是调整detection map到输入图像的大小。这里的box属性是根据feature map(例如，13 x 13)来确定大小的。如果输入的图像是416 x 416，我们将属性乘以32，或者stride

```python
prediction[:, :, :4] *= stride
```



# 检测层再探讨

现在我们已经转换了输出张量，现在可以将三个不同比例的检测映射连接成一个大张量。请注意，在我们转换之前这是不可能的，因为我们不能连接具有不同空间维度的feature maps。但是从现在开始，我们的输出张量仅仅作为一个行的box盒的表，串联是非常可能的。

我们前进道路上的一个障碍是我们不能初始化一个空张量，然后将一个非空(不同形状)张量连接到它。因此，我们推迟collector的初始化(保存检测的张量) ，直到我们得到第一个feature map，然后在我们得到后续检测时连接到映射。

注意函数 forward 中循环之前的 `write = 0`。write 标志用于表示我们是否遇到了第一个检测。如果 write 为 0，则表示collector尚未初始化。如果是1，则意味着collector已经初始化，我们可以将检测映射连接到它。

现在，我们已经有`predict_transform` 这个强大的武器来，下面我们在 forward 函数中编写了处理检测feature map的代码。

在 darknet.py 文件的顶部，添加以下导入。

```python
from utils import * 
```

然后，在`forward`函数中。

```python
            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                # 获取输入纬度
                inp_dim = int(self.net_info['height'])

                # 获取类别数量
                num_classes = int(module['classes'])

                # 转化
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x
```

# 测试forward

这是一个创建测试输入的函数。我们将把这个输入传递给我们的网络。在我们编写这个函数之前，把这个图片保存到你的工作目录中。如果你使用的是 linux，那么输入。

```shell
wget https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png
```

在 darknet.py 文件的顶部定义如下函数:

```python
def get_test_input(img_path):
    img = cv2.imread(img_path)
    # 这个大小需要和你cfg里面的图片大小相对应
    img = cv2.resize(img, (608, 608))
    # BGR => RGB
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    return img_
```

我们运行如下的测试代码：

```python
model = Darknet('cfg/yolov3.cfg').cuda()
inp = get_test_input('dog-cycle-car.png').cuda()
pred = model(inp, torch.cuda.is_available())
print(pred)
```

可以得到如下的结果：

```
tensor([[[1.5188e+01, 1.6643e+01, 8.9002e+01,  ..., 5.2412e-01,
          5.7080e-01, 5.2281e-01],
         [1.4089e+01, 1.8250e+01, 1.0033e+02,  ..., 5.5339e-01,
          4.4268e-01, 5.8961e-01],
         [1.6033e+01, 1.4834e+01, 2.0343e+02,  ..., 4.9131e-01,
          4.3267e-01, 5.1848e-01],
         ...,
         [6.0408e+02, 6.0499e+02, 1.0931e+01,  ..., 3.9232e-01,
          5.2076e-01, 5.1312e-01],
         [6.0494e+02, 6.0365e+02, 1.9087e+01,  ..., 3.5823e-01,
          5.2685e-01, 4.0312e-01],
         [6.0389e+02, 6.0437e+02, 2.9261e+01,  ..., 4.2379e-01,
          4.4921e-01, 5.2107e-01]]], device='cuda:0')
```

这个张量的形状是[1, 22743, 85]。第一个维度是batch size，这只是1，因为我们使用了一个单一的图像。对于批处理中的每个图像，我们有一个22743 x 85的表。每个表的行代表一个边界框。(4个 bbox 属性，1个 objectness 得分，80个class得分) 

在这一点上，我们的网络有随机权重，并不会产生正确的输出。我们需要在网络中加载一个权重文件。为此，我们将使用官方的权重文件。

# 下载与训练模型

将权重文件下载到目录当中：

`wget https://pjreddie.com/media/files/yolov3.weights`

weights很大，你要忍一下。速度实在太慢就爬梯子吧。

# 理解权重文件

官方的权重文件是二进制文件，其中包含以串行方式存储的权重。

读取权重时必须格外小心。权重只是存储为浮点数，没有任何东西指引我们它们属于哪一层。如果你搞砸了，没有什么可以阻止你，比如说，把BN的权重加载到那些卷积层中。因为你只读浮点数，所以没有办法区分哪个权重属于哪一层。因此，我们必须了解权重是如何存储的。

首先，权值只属于两种类型的层，一种是BN层，另一种是卷积层。

这些层的权重完全按照它们在配置文件中出现的顺序存储。当BN层出现在卷积块中时就没有bias。然而，当没有BN层时，bias “权重”必须从文件中读取。

下面的图表总结了权重文件如何存储的权重的：

![wts-1](https://gitee.com/coronapolvo/images/raw/master/20210819111304wts-1.png)



# 加载权重文件

让我们写一个函数来加载权重。它将是 Darknet 类的成员函数。它会采用除了 self 之外的一个参数，即权重文件的路径. 

权重文件的前160个字节存储5个int32值，这些值构成了文件的头部。

```python
    def load_weights(self, weight_file):
        fp = open(weight_file, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
```

其余位现在按照上面描述的顺序表示权重。权重以浮动32位或32位浮点的形式存储。让我们把剩下的重量加载到 np.ndarray 中。

```python
weights = np.fromfile(fp, dtype = np.float32)
```

现在，我们迭代权重文件，并将权重加载到网络的模块中。

```python
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # 如果module_type时convolutional则加载权重
            # 否则就跳过
```

在循环中，我们首先检查卷积块是否具有批量归一化。在此基础上，我们加载权重。

```python
        if module_type == 'convolutional':
            model = self.module_list[i]
            try:
                batch_norm = int(self.blocks[i + 1]['batch_normalize'])
            except:
                batch_norm = 0
            conv = model[0]
```

我们使用一个名为 ptr 的变量来跟踪我们在 weights 数组中的位置。现在，如果 batch_normalize 为正，我们按照以下方式加载权重。

```python
            if batch_norm:
                bn = model[1]

                # 获取BN层的权重参数数量
                num_bn_biases = bn.bias.numel()
                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases

                # 获取权重参数
                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                # Cast the loaded weights into dims of model weights.
                bn_biases = bn_biases.view_as(bn.bias.data)
                bn_weights = bn_weights.view_as(bn.weight.data)
                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                bn_running_var = bn_running_var.view_as(bn.running_var)

                # Copy the data to model
                bn.bias.data.copy_(bn_biases)
                bn.weight.data.copy_(bn_weights)
                bn.running_mean.copy_(bn_running_mean)
                bn.running_var.copy_(bn_running_var)
```

如果 batch_norm 为False，只需加载卷积层的偏差即可。

```python
            else:
                # Number of biases
                num_biases = conv.bias.numel()

                # Load the weights
                conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                ptr = ptr + num_biases

                # reshape the loaded weights according to the dims of the model weights
                conv_biases = conv_biases.view_as(conv.bias.data)

                # Finally copy the data
                conv.bias.data.copy_(conv_biases)
```

最后，加载卷积层的权值。

```python
            # 为卷积层加载权重
            num_weights = conv.weight.numel()

            # Do the same as above for weights
            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(conv.weight.data)
            conv.weight.data.copy_(conv_weights)
```

我们已经完成了这个函数，现在您可以通过调用 Darknet 对象上的 load_weights 函数来加载权重参数

```python
model = Darknet('cfg/yolov3.cfg').cuda()
model.load_weights('yolov3.weights')
```

这就是这一部分的全部内容，随着我们模型的建立，加载了权重，我们终于可以开始检测物体了。在接下来的部分，我们将讨论使用objectness score阈值和非极大值抑制来产生我们的最终检测结果。

























