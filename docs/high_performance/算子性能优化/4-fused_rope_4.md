# 二维位置的旋转式位置编码


在NLP中，语言的位置信息是一维的，换句话说，我们需要告诉模型这个词是句子的第几个词；但是在CV中，图像的位置信息是二维的，即我们需要告诉模型这个特征是在第几行、第几列。这里的二维指的是完整描述位置信息需要两个数字，并不是指位置向量的维数。

如果直接使用Transformer的位置编码，会导致模型无法区分不同位置的特征，因此需要引入二维位置编码。

## 1. 一维位置编码

假设 q 的纬度是 4，一维的位置编码是这样的：

$$
f(q, m) = R_mq = \left(\begin{array}{cc:cc}
\cos m \theta & -\sin m \theta & 0 & 0 \\
\sin m \theta & \cos m \theta & 0 & 0 \\
\hdashline 0 & 0 & \cos m \theta & -\sin m \theta \\
0 & 0 & \sin m \theta & \cos m \theta
\end{array}\right) \left(\begin{array}{c} q_0 \\ q_1 \\ q_2 \\ q_3 \end{array}\right)
$$

扩展到多维后的公式如下：

$$
\left(\begin{array}{ccccccc}
\cos m \theta_0 & -\sin m \theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m \theta_0 & \cos m \theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m \theta_1 & -\sin m \theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m \theta_1 & \cos m \theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d / 2-1} & -\sin m \theta_{d / 2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d / 2-1} & \cos m \theta_{d / 2-1}
\end{array}\right)\left(\begin{array}{c}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right)
$$

上面的稀疏矩阵可以简化为：


$$
\left(\begin{array}{c}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\cos m \theta_0 \\
\cos m \theta_0 \\
\cos m \theta_1 \\
\cos m \theta_1 \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\cos m \theta_{d / 2-1}
\end{array}\right)+\left(\begin{array}{c}
-q_1 \\
q_0 \\
-q_3 \\
q_2 \\
\vdots \\
-q_{d-1} \\
q_{d-2}
\end{array}\right) \otimes\left(\begin{array}{c}
\sin m \theta_0 \\
\sin m \theta_0 \\
\sin m \theta_1 \\
\sin m \theta_1 \\
\vdots \\
\sin m \theta_{d / 2-1} \\
\sin m \theta_{d / 2-1}
\end{array}\right)
$$

## 2. 二维位置编码

作者在介绍二维位置编码时，进行了很繁琐的推导，但是其实可以简单的理解为，将一维的位置编码在两个方向上分别计算，然后拼接在一起。最后得出的二维位置编码。我们也是先看 q 是 4 维的情况：

$$
f(q, x, y) = R_{xy}q = \left(\begin{array}{cc:cc}
\cos x \theta & -\sin x \theta & 0 & 0 \\
\sin x \theta & \cos x \theta & 0 & 0 \\
\hdashline 0 & 0 & \cos y \theta & -\sin y \theta \\
0 & 0 & \sin y \theta & \cos y \theta
\end{array}\right) \left(\begin{array}{c} q_{0,0} \\ q_{0,1} \\ q_{1,0} \\ q_{1,1} \end{array}\right)
$$

扩展到多维后的公式如下：

$$
\left(\begin{array}{cccccccccccc}
\cos x \theta_0 & -\sin x \theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 & \cdots & 0 & 0  \\
\sin x \theta_0 & \cos x \theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 & \cdots & 0 & 0  \\
0 & 0 & \cos x \theta_1 & -\sin x \theta_1 & \cdots & 0 & 0 & 0 & 0 & \cdots & 0 & 0  \\
0 & 0 & \sin x \theta_1 & \cos x \theta_1 & \cdots & 0 & 0 & 0 & 0 & \cdots & 0 & 0  \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots  \\
0 & 0 & 0 & 0 & \cdots & \cos x \theta_{d / 4-1} & -\sin x \theta_{d / 4-1} & 0 & 0 & \cdots & 0 & 0  \\
0 & 0 & 0 & 0 & \cdots & \sin x \theta_{d / 4-1} & \cos x \theta_{d / 4-1} & 0 & 0 & \cdots & 0 & 0\\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \cos y \theta_0 & -\sin y \theta_0 & \cdots & 0 & 0\\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \sin y \theta_0 & \cos y \theta_0 & \cdots & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots  \\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 & \cdots & \cos y \theta_{d / 4-1} & -\sin y \theta_{d / 4-1}\\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 & \cdots & \sin y \theta_{d / 4-1} & \cos y \theta_{d / 4-1}
\end{array}\right)\left(\begin{array}{c}
q_{0,0} \\
q_{0,1} \\
q_{0,2} \\
q_{0,3} \\
\vdots \\
q_{d/2-1,d/2-4} \\
q_{d/2-1,d/2-3} \\
q_{d/2-1,d/2-2} \\
q_{d/2-1,d/2-1}
\end{array}\right)
$$

当然了，这个稀疏矩阵也可以简化为：

$$
\left(\begin{array}{c}
q_{0,0} \\
q_{0,1} \\
q_{0,2} \\
q_{0,3} \\
\vdots \\
q_{d/4-1,d/4-2} \\
q_{d/4-1,d/4-1} \\
q_{d/4-1,d/4} \\
q_{d/4-1,d/4+1} \\
\vdots \\
q_{d/2-1,d/2-2} \\
q_{d/2-1,d/2-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\cos x \theta_0 \\
\cos x \theta_0 \\
\cos x \theta_1 \\
\cos x \theta_1 \\
\vdots \\
\cos x \theta_{d / 4-1} \\
\cos x \theta_{d / 4-1} \\
\cos y \theta_0 \\
\cos y \theta_0 \\
\vdots \\
\cos y \theta_{d / 4-1} \\
\cos y \theta_{d / 4-1}
\end{array}\right)+\left(\begin{array}{c}
-q_{0,1} \\
q_{0,0} \\
-q_{0,3} \\
q_{0,2} \\
\vdots \\
-q_{d/4-1,d/4-2} \\
q_{d/4-1,d/4-1} \\
-q_{d/4-1,d/4} \\
q_{d/4-1,d/4+1} \\
\vdots \\
-q_{d/2-1,d/2-2} \\
q_{d/2-1,d/2-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\sin x \theta_0 \\
\sin x \theta_0 \\
\sin x \theta_1 \\
\sin x \theta_1 \\
\vdots \\
\sin x \theta_{d / 4-1} \\
\sin x \theta_{d / 4-1} \\
\sin y \theta_0 \\
\sin y \theta_0 \\
\vdots \\
\sin y \theta_{d / 4-1} \\
\sin y \theta_{d / 4-1}
\end{array}\right)
$$


可以看到，二维位置编码的计算方式和一维的计算方式是一样的，只是在两个方向上分别计算，然后拼接在一起。

## 3. 代码实现

### 3.1 复数角度实现

我们知道，复数可以用极坐标表示，极坐标下一个复数乘上一个模长为 1 的复数，相当于将这个复数旋转一个角度。因此，我们可以使用复数的极坐标表示来实现位置编码。

首先我们先来实现一个一维的旋转式位置编码，然后我们再实现二维。

```python
import torch

# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度 \theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    """
    xq: q 矩阵
    xk: k 矩阵
    freqs_cis: 位置编码
    """
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 测试代码
freqs = precompute_freqs_cis(128, 14 * 14, 10.0)
xq = torch.randn(2, 14 * 14, 128)
xk = torch.randn(2, 14 * 14, 128)
xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)
```

我们使用 `precompute_freqs_cis` 函数生成位置编码，然后使用 `apply_rotary_emb` 函数应用位置编码。这里的 `freqs_cis` 是一个复数向量，我们使用 `torch.polar` 函数将其转换为极坐标形式的复数。然后，我们将 `xq` 和 `xk` 转换为复数域，应用旋转操作，最后将结果转回实数域。

对于 2D 位置编码，我们只需要修改 `precompute_freqs_cis` 函数即可, `apply_rotary_emb` 函数不需要修改。

```python
def init_t_xy(seq_x_len: int, seq_y_len: int):
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_x_len * seq_y_len, dtype=torch.float32)
    # t_x = [0, 1, 2, ..., seq_x_len-1, 0, 1, 2, ..., seq_x_len-1, ...]
    t_x = (t % seq_x_len).float()
    # t_y = [0, 0, 0, ..., 0, 1, 1, 1, ..., 1, ..., seq_y_len-1, ...]
    t_y = torch.div(t, seq_x_len, rounding_mode="floor").float()
    return t_x, t_y

# 生成 2d 旋转矩阵
def precompute_freqs_cis_2d(dim: int, seq_x_len: int, seq_y_len: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(seq_x_len, seq_y_len)
    # 计算 x 和 y 方向的位置编码
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    # 将 x 和 y 方向的位置编码拼接在一起
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)
```

## 4. 总结

本文介绍了二维位置的旋转式位置编码的计算方式，可以看到二维位置的旋转位置编码和一维的计算方式基本是一样的，我们是需要略微修改一下 freqs 的计算方式，然后将 x 和 y 方向的位置编码拼接在一起。

如果像公式那样展开，会发现这个稀疏矩阵的计算方式和一维的计算方式是一样的，只是 x 和 y 方向分别计算，然后拼接在一起。

如果是在复数域下计算，我们要要求的也是只有计算 freqs 的时候，将 x 和 y 方向分别计算，然后拼接在一起。


## 参考文章

1. https://kexue.fm/archives/8397


