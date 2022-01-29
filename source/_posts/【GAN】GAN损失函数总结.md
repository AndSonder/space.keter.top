---
title: 「GAN」GAN损失函数总结
date: 2021-08-10 14:01:02
tags: 深度学习基础知识
categories: 深度学习基础知识
katex: true
---



# 朴素GAN

朴素GAN的思想比较单纯，生成器负责生成假的数据。然后判别器负责鉴别这些数据。在计算LOSS的时候就是计算BCE LOSS，看一看代码就非常的清楚了。朴素GAN在计算损失函数的时候的计算依据是真的数据label就是1，假的数据label就是0。

```python
# 首先训练鉴别器
for d_index in range(d_steps):
  #  1A: Train D on real
  d_real_data = Variable(d_sampler(d_input_size))
  # 让判别器判断真的数据
  d_real_decision = D(preprocess(d_real_data))
  # 告诉判别器这些数据是真的
  d_real_error = criterion(d_real_decision, Variable(torch.ones([1])))  # ones = true
  d_real_error.backward() # compute/store gradients, but don't change params

  #  1B: Train D on fake
  d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))          
  # 生成假的数据				
  d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
  # 让判别器判断这些假数据
  d_fake_decision = D(preprocess(d_fake_data.t()))
  # 告诉判别器这些数据是假的
  d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1])))  # zeros = fake
  d_fake_error.backward()
  d_optimizer.step()  

# 然后训练生成器
for g_index in range(g_steps):
  # 基于D的结果训练G，但是要注意不要训练到D了
  G.zero_grad()
  gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
  g_fake_data = G(gen_input)
  dg_fake_decision = D(preprocess(g_fake_data.t()))
  # 欺骗鉴别器这些数据都是真的
  g_error = criterion(dg_fake_decision, Variable(torch.ones([1])))  
  g_error.backward()
  g_optimizer.step()  # 只更新生成器的参数
  ge = extract(g_error)[0]
```

# WGAN

WGAN有以下的优点，只能说太妙了！

- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
- 基本解决了collapse mode的问题，确保了生成样本的多样性
- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高（如题图所示）
- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

WGAN的改进只有如下几点：

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

前两点的其实就是不再使用JS散度，而第三点用于工程实训上保证判别器的目标函数平滑。最后一点其实是在实验中发现的，当使用Adam之类涉及动量的梯度下降算法时，判别器的损失可能会出现大幅度抖动的现象，而使用RMSProb或SGD算法后，这个问题就不会出现。   ·

<img src="https://gitee.com/coronapolvo/images/raw/master/20210728202345image-20210726141055299.png" alt="image-20210726141055299" style="zoom:50%;" />

更多的内容可以参考：[令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)

## WGAN损失函数部分代码

Code from: [https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py](https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py)

```python
while i < len(dataloader):
  ############################
  # (1) 更新鉴别器
  ###########################
  for p in netD.parameters(): # reset requires_grad
    # 此项在更新生成器网络的时候应该设置为Flase
    p.requires_grad = True 

  # 训练鉴别器Diter次
  if gen_iterations < 25 or gen_iterations % 500 == 0:
    Diters = 100
  else:
    Diters = opt.Diters
  j = 0
  while j < Diters and i < len(dataloader):
    j += 1
    data = data_iter.next()
    i += 1
    # train with real
    real_cpu, _ = data
    netD.zero_grad()
    batch_size = real_cpu.size(0)

    if opt.cuda:
      real_cpu = real_cpu.cuda()
      input.resize_as_(real_cpu).copy_(real_cpu)
      inputv = Variable(input)

      errD_real = netD(inputv)
      errD_real.backward(one)

      # train with fake
      noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
      noisev = Variable(noise, volatile = True) # totally freeze netG
      fake = Variable(netG(noisev).data)
      inputv = fake
      errD_fake = netD(inputv)
      errD_fake.backward(mone)
      # 计算EM距离
      errD = errD_real - errD_fake
      optimizerD.step()
    # 将判别器的梯度限制在一个范围之内【WGAN核心之处】
    for p in netD.parameters():
      p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

  ############################
  # (2) 更新生成器
  ###########################
  for p in netD.parameters():
    p.requires_grad = False
  netG.zero_grad()
  # in case our last batch was the tail batch of the dataloader,
  # make sure we feed a full batch of noise
  noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
  noisev = Variable(noise)
  fake = netG(noisev)
  errG = netD(fake)
  errG.backward(one)
  optimizerG.step()
  gen_iterations += 1
```

## WGAN-GP损失函数部分代码

Code from：[https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py](https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py)

计算gradient_penalty部分的代码

```python
def calc_gradient_penalty(netD, real_data, fake_data):
    # 从生成数据和真实数据之间的空间分布中抽取样本用来计算
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
	
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
		
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
```



```python
############################
# (1) 更新鉴别器
###########################
for p in netD.parameters():  
  # 此项在更新生成器网络的时候应该设置为Flase
  p.requires_grad = True  
  for iter_d in xrange(CRITIC_ITERS):
    _data = data.next()
    real_data = torch.Tensor(_data)
    if use_cuda:
      real_data = real_data.cuda(gpu)
      real_data_v = autograd.Variable(real_data)

      netD.zero_grad()

      # train with real
      D_real = netD(real_data_v)
      D_real = D_real.mean()
      # print D_real
      D_real.backward(mone)

      # train with fake
      noise = torch.randn(BATCH_SIZE, 128)
      if use_cuda:
        noise = noise.cuda(gpu)
      noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
      fake = autograd.Variable(netG(noisev).data)
      inputv = fake
      D_fake = netD(inputv)
      D_fake = D_fake.mean()
      D_fake.backward(one)

      # 计算 gradient penalty
      gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
      gradient_penalty.backward()

      D_cost = D_fake - D_real + gradient_penalty
      
      Wasserstein_D = D_real - D_fake
      optimizerD.step()
############################
# (2) Update G network
###########################
for p in netD.parameters():
  # 避免计算鉴别器
  p.requires_grad = False
netG.zero_grad()
noise = torch.randn(BATCH_SIZE, 128)
if use_cuda:
  noise = noise.cuda(gpu)
  noisev = autograd.Variable(noise)
  fake = netG(noisev)
  G = netD(fake)
  G = G.mean()
  G.backward(mone)
  G_cost = -G
  optimizerG.step()
```

# SN-GAN

WGAN-GP使用graident penalty的方法来限制判别器，但是这种方法只能对生成数据分布与真实数据分布之间的分布空间的数据做梯度惩罚，无法对整个空间的数据都做惩罚，这会导致随着训练的进行，生成数据分布与真实数据分布之间的空间会逐渐变化，从而导致graident penalty正则化方法不稳定，在实验中，当我们使用一个比较大的学习率去训练WGAN-GP的时候，WGAN-GP的表现并不稳定。而且因为WGAN-GP涉及比较多的运算，所以训练的过程也比较耗时。

SN-GAN提出用Spectral Normalization方法来让判别器D满足Lipschitz约束。简单的说，SN-GAN只需要改变判别器权值的最大奇异值。

奇异值是矩阵里面的概念，一般通过奇异值分解定理求得。设A为mxn阶矩阵，q=min(m,n)，AxA的q个非负特征值的算术平方根叫做A的奇异值。

## L约束

所以，大多数时候我们都希望模型对输入扰动是不敏感的，这通常能提高模型的泛化性能。也就是说，我们希望<img src="https://www.zhihu.com/equation?tex=\left\|x_{1}-x_{2}\right\|" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">很小时：
<img src="https://www.zhihu.com/equation?tex=\left\|f_{w}\left(x_{1}\right)-f_{w}\left(x_{2}\right)\right\|" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">
也尽可能地小。当然，“尽可能”究竟是怎样，谁也说不准。于是Lipschitz提出了一个更具体的约束，那就是存在某个常数CC（它只与参数有关，与输入无关），使得下式恒成立
<img src="https://www.zhihu.com/equation?tex=\left\|f_{w}\left(x_{1}\right)-f_{w}\left(x_{2}\right)\right\| \leq C(w) \cdot\left\|x_{1}-x_{2}\right\|" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">
换言之，在这里我们认为满足L约束的模型才是一个好模型并且对于具体的模型，我们希望估算出C(w)的表达式，并且希望C(w)越小越好，越小意味着它对输入扰动越不敏感，泛化性越好。

经过一番数学推导我们可以得到F函数是C的一个具体值：

![image-20210728150614326](https://gitee.com/coronapolvo/images/raw/master/20210728150617image-20210728150614326.png)

为了让一个模型更好的服从Lipschitz约束，即让模型更加的平滑，就应当最小化参数C。我们可以将<img src="https://www.zhihu.com/equation?tex=C^2" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">作为一个惩罚项带入普通监督模型的损失函数中，以此来让模型更加平滑。
<img src="https://www.zhihu.com/equation?tex=loss = loss(y,f_w(x)) + \lambda C^2" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">
将<img src="https://www.zhihu.com/equation?tex=||W||_F" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">代入上式。
<img src="https://www.zhihu.com/equation?tex=loss = loss(y,f_w(x)) + \lambda||W||_F^2 " alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">
这其实就是一个<img src="https://www.zhihu.com/equation?tex=l_2" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">正则项。从而可以的出来一个结论，即一个神经网络模型添加了<img src="https://www.zhihu.com/equation?tex=l_2" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">正则项之后，模型的泛华能力以及抗干扰能力会更强，这样符合常识，前面的内容就是从数据的角度证明了这个常识背后的机理。

## SN-GAN loss

SN-GAN中使用了一个叫做Spectral Normalization的方法非常简单。就是判别器的所有权重都进行除以谱范数的操作即<img src="https://www.zhihu.com/equation?tex=\frac{W}{||W||_2}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">, 这样做之所以有效的原因和F函数是一样的。

我们知道传统的GAN如果不加上Lipschitz约束，判别器就会被无线优化，导致判别器与生成器能力之间失衡，造成GAN难以训练，而WGAN，WGAN-GP都通过不同的方式让GAN的判别器服从Lipschitz约束，但是都有各自的问题。其中WGAN-GP梯度惩罚的方式可以满足比较多的情况，但是训练比较慢，随着训练的进行，梯度会出现波动。还有一个值得关注的问题，就是对于类别数据训练，WGAN-GP得不到比较理想的效果，这是因为梯度惩罚方式只针对生成数据分布于真实数据分布之间的空间分布中的数据进行梯度政法，无视其他空间。这种方式使得 它难以处理多类别数据，多类别数据在空间分布中是多样的，因此WGAN0-GP就不知道到底把哪里作为惩罚空间。从而得不到比较好的效果。

对SN-GAN而言，它将谱正则化的思想运用到GAN中，从而提出了谱归一化，通过谱归一化的方式让GAN满足1-Lipschitz约束。

你可以通过公式证明<img src="https://www.zhihu.com/equation?tex=W/||W||_2" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">是严格服从1-Lipschitz约束的。

在训练的过程中，因为直接计算谱范数<img src="https://www.zhihu.com/equation?tex=||W||_2" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">是比较耗时的，为了让模型训练的时候速度比较快，就需要使用一个技巧。power iteration方法通过迭代计算的思想可以比较快速地计算出谱范数的近似值。

所谓的power iteration就是通过下面的迭代格式进行迭代计算。
<img src="https://www.zhihu.com/equation?tex=v = \frac{W^Tu}{||W^Tu||},u=\frac{Wv}{||Wv||}" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">
若干次迭代后，就可以得到谱范数的近似值。
<img src="https://www.zhihu.com/equation?tex=||W||_2 \approx u^TWv" alt="" style="margin: 0 auto;" class="ee_img tr_noresize" eeimg="1">

## 部分关键代码

```python
# l2 正则化
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    # 迭代近似计算
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u
```

```python
class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
      """
      在计算卷积的时候对权重添加惩罚项
      """
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
```











