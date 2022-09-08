# 基于Flow的异常检测算法对比

:::tip

对比CS Flow, CFlow 和 FastFlow 三种基于Flow的异常检测算法

:::

## 效果对比

目前Fast Flow是速度最快，效果最好的基于Flow的模型，CSFlow和Cflow要差一些。CSFlow和Cflow的结构都要比Fast Flow复杂，但是不知道为啥Fast Flow的效果是最好的

![图 1](images/4aaaaa96419a32e7216baa9314cc93afa4d62a8c34e653a7ffb21310a9ded9e3.png)  


## 代码实现

首先从代码上来看看这几个Flow模型的差别：

###  Normalize Block 

#### 1、Fast flow

```python 
def create_fast_flow_block(
        input_dimensions: List[int],
        conv3x3_only: bool,
        hidden_ratio: float,
        flow_steps: int,
        clamp: float = 2.0,
) -> SequenceINN:
    """Create NF Fast Flow Block.

    This is to create Normalizing Flow (NF) Fast Flow model block based on
    Figure 2 and Section 3.3 in the paper.

    Args:
        input_dimensions (List[int]): Input dimensions (Channel, Height, Width)
        conv3x3_only (bool): Boolean whether to use conv3x3 only or conv3x3 and conv1x1.
        hidden_ratio (float): Ratio for the hidden layer channels.
        flow_steps (int): Flow steps.
        clamp (float, optional): Clamp. Defaults to 2.0.

    Returns:
        SequenceINN: FastFlow Block.
    """
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes
```

#### 2、CFlow

```python 
def cflow_head(
    condition_vector: int, coupling_blocks: int, clamp_alpha: float, n_features: int, permute_soft: bool = False
) -> SequenceINN:
    """Create invertible decoder network.

    Args:
        condition_vector (int): length of the condition vector
        coupling_blocks (int): number of coupling blocks to build the decoder
        clamp_alpha (float): clamping value to avoid exploding values
        n_features (int): number of decoder features
        permute_soft (bool): Whether to sample the permutation matrix :math:`R` from :math:`SO(N)`,
            or to use hard permutations instead. Note, ``permute_soft=True`` is very slow
            when working with >512 dimensions.

    Returns:
        SequenceINN: decoder network block
    """
    coder = SequenceINN(n_features)
    logger.info("CNF coder: %d", n_features)
    for _ in range(coupling_blocks):
        coder.append(
            AllInOneBlock,
            cond=0,
            cond_shape=(condition_vector,),
            subnet_constructor=subnet_fc,
            affine_clamping=clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=permute_soft,
        )
    return coder
```

#### 3、CS-Flow 

