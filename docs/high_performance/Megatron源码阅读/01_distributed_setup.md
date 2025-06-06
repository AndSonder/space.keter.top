# 分布式环境配置

在上一篇文章中，我们从整体上介绍了 Megatron-LM 的预训练流程，从初始化、模型构建到数据加载，再到训练的核心步骤，带你对这个复杂的框架有了初步了解。

要让大规模模型高效运行，我们首先需要正确配置分布式环境。在 Megatron-LM 中，分布式训练依赖 `torch.distributed` 进行进程管理，同时结合 `mpu（Model Parallel Unit）` 组织不同的并行方式。那么，一个分布式训练任务如何正确启动？ `world_size`、`rank`、`local_rank`` 这些参数如何设置？`initialize_megatron` 这个核心初始化函数都做了哪些关键工作？在这篇文章中，我们将从 分布式环境的基本概念 开始，一步步剖析 Megatron-LM 如何完成分布式训练的初始化。

## 1. 分布式训练的基本概念

在分布式训练中，我们可以把 GPU 看作是参与工作的团队成员，而每个 GPU 的角色和职责都需要明确定义。这时，就需要理解三个核心概念：`world_size`、`rank` 和 `local_rank`。

想象你在一个有多个房间的大楼里工作。每个房间里有几个人，每个人都有自己的工作区域。现在，我们用这个比喻来讲解这些概念。

### 1.1. world_size：团队的总规模

假设这栋大楼是一个训练任务，房间里的每个人代表一块 GPU。那么，`world_size` 就是大楼里总共参与工作的 GPU 数量。无论这些 GPU 分布在几台机器上，只要它们共同完成同一个任务，它们就构成了一个整体团队。

比如：如果你有 2 台机器，每台机器上有 4 块 GPU，那么整个团队的 `world_size` 就是 8。这个数字代表了并行训练中一共有多少个“工人”在同时工作。

### 1.2. rank：每个人的全球身份

在一个团队中，分工是很重要的。为了确保每个人知道自己负责什么，团队需要为每个成员分配一个独一无二的身份编号，这就是 `rank`。

rank 从 0 开始，依次编号。例如，在上面的 2 台机器（8 个 GPU）中，rank 就会从 0 到 7 分配。这是全局视角下的编号，无论哪个 GPU 在哪台机器上，它的 rank 都是唯一的。

通过 rank，我们可以让某些 GPU 执行特定任务，比如负责模型的不同部分，或者处理不同的数据。

### 1.3. local_rank：每个人在自己房间里的身份

如果 `rank` 是全局视角下的编号，那么 `local_rank` 就是每个人在自己所在房间里的编号。

假设每个房间对应一台机器，那么每台机器里的 GPU 会从 0 开始依次编号为它们的 `local_rank`。例如，在一台有 4 块 GPU 的机器中，它们的 `local_rank` 是 0、1、2、3，而这 4 个 GPU 的全局 rank 可能是 4、5、6、7。

### 1.4. 单机多卡和多机多卡的区别

单机多卡时，你只需要关心一台机器里的 GPU 编号。所有 GPU 的 `local_rank` 和 `rank` 是一致的，比如 `local_rank` 是 0，那么它的 `rank` 也一定是 0。

多机多卡时，`rank` 就成了跨机器编号的工具，而 `local_rank` 仍然是局部编号。举个例子，在两台机器上训练，rank=0 的 GPU 可能在第一台机器上，而 rank=4 的 GPU 则在第二台机器上，但它们的 `local_rank` 都可能是 0。

## 2. Megatron-LM 的初始化

在使用 Megatron-LM 进行大规模模型训练之前，第一步就是正确初始化分布式环境。否则，无论你的 GPU 有多少，计算能力有多强，训练都无法高效进行，甚至可能直接卡住。

但是，分布式训练的初始化真的只是简单地运行 `torch.distributed.init_process_group()` 吗？如果你看过 `initialize_megatron()` 这个函数，就会发现它做了远比你想象中复杂的事情。

### 2.1. 理解 `torch.distributed`

在 Megatron-LM 里，每个 GPU 其实是一个独立的进程，它们之间需要互相通信来协调计算。为了管理这些进程，PyTorch 提供了 `torch.distributed`，它的核心任务是把所有 GPU 组织成一个可以协同工作的集体。

但你可能会问：“我们真的需要手动初始化吗？PyTorch 不是已经提供了 `torchrun` 这样的工具吗？”

是的，`torchrun` 确实会自动初始化 `torch.distributed`，但这只是第一步。 Megatron-LM 需要在此基础上做更多的配置，比如：

1. 设定 `world_size`，确保所有 GPU 进程正确连接
2. 绑定 `local_rank`，让每个进程知道自己在哪张 GPU 上
3. 进一步划分数据并行、张量并行、流水线并行的进程组

如果这部分没处理好，模型可能会卡住，梯度不会同步，甚至 GPU 直接闲置。

那么，Megatron-LM 到底是怎么做的？让我们深入 `initialize_megatron()` 这个函数。

### 2.2. `initialize_megatron()`：初始化到底发生了什么

所有 Megatron-LM 的训练脚本都会先调用这个函数：

```python
initialize_megatron(
    extra_args_provider=extra_args_provider,
    args_defaults=args_defaults,
    get_embedding_ranks=get_embedding_ranks,
    get_position_embedding_ranks=get_position_embedding_ranks
)
```

但你有没有想过，这个函数具体做了什么？我们不妨拆开来看。

**1、解析参数，确定训练规则**

Megatron-LM 依赖大量的超参数来控制训练的各个细节，比如 `tensor_model_parallel_size`（张量并行度）、`pipeline_model_parallel_size`（流水线并行度）、`distributed_backend`（通信后端）等。

**2、设置全局变量，建立基础环境**

在训练前，确保多个 GPU 进程共享相同的环境，比如随机种子、日志系统、TensorBoard 记录等。它通过 `set_global_variables(args)` 这个函数来完成。

**3、初始化分布式环境**

到了这一步，我们的 args 变量已经准备好了，日志系统也已经就绪。现在，是时候让不同 GPU 进程真正连起来了。

分布式训练的第一步，就是让多个 GPU 确认彼此的身份，并建立通信连接。这一步由 `_initialize_distributed()` 负责：

我们重点看看 `_initialize_distributed()` 里做了什么：

```python
    args = get_args()
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(args.local_rank)
            device_id = torch.device(f'cuda:{args.local_rank}')
        else:
            device_id = None

        # Call the init process
        init_process_group_kwargs = {
            'backend' : args.distributed_backend,
            'world_size': args.world_size,
            'rank': args.rank,
            'timeout': timedelta(minutes=args.distributed_timeout_minutes),
        }
        torch.distributed.init_process_group(**init_process_group_kwargs)
```

这部分代码的作用是：如果 `torch.distributed`` 进程已经初始化了，就直接获取当前进程的`rank`和`world_size`。

如果还没有初始化，则会手动调用 `torch.distributed.init_process_group()` 启动分布式环境。

这里有几个关键点：

- **通信后端：** 默认使用 NCCL，它是 NVIDIA 专门优化过的 GPU 通信库，适合大规模训练。也可以用 Gloo（支持 CPU 和 GPU）或者 MPI（适用于 HPC 场景）。
- **`rank` 与 `world_size`**：world_size 是总共有多少个训练进程（即 GPU 数），rank 是当前进程的编号。


**4、设定并行策略**

Megatron-LM 不只是简单的数据并行（Data Parallelism），它还支持张量并行（Tensor Parallelism）、流水线并行（Pipeline Parallelism），甚至是多层次的专家模型并行（MoE Parallelism）。

在 `_initialize_distributed()` 里，它会调用 `mpu.initialize_model_parallel()` 来设定并行策略：

```python
mpu.initialize_model_parallel(
    args.tensor_model_parallel_size,
    args.pipeline_model_parallel_size,
    args.virtual_pipeline_model_parallel_size,
    args.pipeline_model_parallel_split_rank,
    ...
)
```

你可以把 `initialize_megatron()` 看作是 Megatron-LM 训练的“启动仪式”。如果你在训练时遇到了分布式相关的错误（比如 rank mismatch 或者 timeout），大概率问题就出在这里。所以，深入理解 `initialize_megatron()` 的执行流程，能够帮助你更好地 debug Megatron-LM，并掌控分布式训练的核心机制。

## 3. Megatron-LM 的 mpu（Model Parallel Unit）  

在深入 mpu 之前，我们先回顾一下一个问题：Megatron-LM 是如何组织并行训练的？

大规模训练任务通常不会简单地采用单一的并行策略，而是结合**数据并行（DP）、张量并行（TP）、流水线并行（PP）**来实现高效训练。然而，这也带来了一个挑战——如何管理多个并行组，确保每个 GPU 在正确的进程组中通信？

这正是 mpu（Model Parallel Unit）的核心职责：它充当 Megatron-LM 分布式训练的调度员，负责组织和管理不同并行方式的 GPU 进程组。代码中看到 `mpu.initialize_model_parallel()`，就是在创建并行训练的基本结构。

让我们拆解 mpu 的工作流程，看看它在 `initialize_model_parallel()` 中到底做了什么。

### 3.1. 为什么需要 mpu

在 PyTorch `torch.distributed` 里，默认情况下，每个 GPU 进程都是平等的。但在 Megatron-LM 里，GPU 进程的角色可以不同：

- **数据并行组（Data Parallel Group）**：每个 GPU 计算自己的一部分数据，并在梯度更新时进行 AllReduce
- **张量并行组（Tensor Parallel Group）**：不同 GPU 共享一个计算层，每个 GPU 只计算一部分张量操作
- **流水线并行组（Pipeline Parallel Group）**：模型被拆分成多个阶段，不同 GPU 负责不同的层，前向/反向传播交错进行。

Megatron-LM 需要手动创建这些并行组，让不同的 GPU 负责不同的工作。而 `mpu` 就是 Megatron-LM 专门设计的模块，用来管理这些并行组的初始化、查询和使用。

### 3.2. 如何构建并行训练结构

当 `initialize_model_parallel()` 被调用时，它会完成以下几件事情：

1. 获取 world_size 和 rank
2. 计算数据并行、张量并行、流水线并行的规模
3. 创建并行进程组
4. 为不同并行方式分配 GPU 进程
5. 存储这些分组，供训练时使用

Megatron-LM 依赖 PyTorch `torch.distributed`，所以在初始化时必须确保 `torch.distributed` 已经启动。然后它会获取全局 GPU 进程总数（world_size）和当前 GPU 进程的 ID（rank）。

```python
assert torch.distributed.is_initialized()
world_size: int = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()
```

下一步 Megatron-LM 需要明确数据并行、张量并行和流水线并行各占多少 GPU。代码是这样计算的：

```python
total_model_size = (
    tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
)
data_parallel_size: int = world_size // total_model_size
```

如果 world_size=16，你设置了：

- `tensor_model_parallel_size=2`
- `pipeline_model_parallel_size=4`

那么 `data_parallel_size` 就是 16 // (2 * 4) = 2。这意味着每个数据并行组有 2 个 GPU。

接下来，`mpu` 会创建数据并行组、张量并行组和流水线并行组。每个组都有自己的 `group` 和 `rank`，用来区分不同的 GPU 进程。

```python
# 遍历数据并行（Data Parallel, DP）的 rank 组
for ranks in generator_wrapper('dp'):
    # 创建数据并行的 NCCL 通信组，所有属于同一数据并行组的 GPU 会被分配到同一组
    group = torch.distributed.new_group(
        ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
    )
    # 额外创建一个 Gloo 后端的通信组，主要用于非 GPU 设备或跨网络同步
    group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
    
    # 如果当前 GPU 进程属于该数据并行组，则存储相应的通信组信息
    if rank in ranks:
        _DATA_PARALLEL_GROUP = group  # 记录 NCCL 版本的数据并行通信组
        _DATA_PARALLEL_GROUP_GLOO = group_gloo  # 记录 Gloo 版本的数据并行通信组
        _DATA_PARALLEL_GLOBAL_RANKS = ranks  # 记录该数据并行组包含的 rank 列表

# 遍历张量并行（Tensor Parallel, TP）的 rank 组
for ranks in generator_wrapper('tp'):
    # 创建张量并行的 NCCL 通信组，负责将模型参数拆分到不同 GPU 进行计算
    group = torch.distributed.new_group(
        ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
    )
    
    # 如果当前 GPU 进程属于该张量并行组，则存储相应的通信组信息
    if rank in ranks:
        _TENSOR_MODEL_PARALLEL_GROUP = group  # 记录张量并行通信组
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks  # 记录该张量并行组的 rank 列表

# 遍历流水线并行（Pipeline Parallel, PP）的 rank 组
for ranks in generator_wrapper('pp'):
    # 创建流水线并行的 NCCL 通信组，负责将模型层拆分到不同 GPU 上进行流水线执行
    group = torch.distributed.new_group(
        ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
    )
    
    # 如果当前 GPU 进程属于该流水线并行组，则存储相应的通信组信息
    if rank in ranks:
        if _PIPELINE_MODEL_PARALLEL_GROUP is None:
            # 如果是第一次初始化流水线并行组，直接赋值
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        elif isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
            # 如果已经存在多个流水线并行组，则追加新的并行组
            _PIPELINE_MODEL_PARALLEL_GROUP.append(group)
            _PIPELINE_GLOBAL_RANKS.append(ranks)
        else:
            # 如果当前存储的是单个流水线并行组，则转换为列表格式存储多个组
            _PIPELINE_MODEL_PARALLEL_GROUP = [_PIPELINE_MODEL_PARALLEL_GROUP, group]
            _PIPELINE_GLOBAL_RANKS = [_PIPELINE_GLOBAL_RANKS, ranks]

    # 获取当前 rank 组中负责 embedding 计算的 ranks
    embedding_ranks = get_embedding_ranks(ranks)
    # 创建 embedding 层的通信组，保证 embedding 在多个 GPU 之间共享
    group = torch.distributed.new_group(
        embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
    )
    if rank in embedding_ranks:
        _EMBEDDING_GROUP = group  # 记录 embedding 计算通信组
        _EMBEDDING_GLOBAL_RANKS = embedding_ranks  # 记录 embedding 组的 rank 列表

    # 获取当前 rank 组中负责位置编码（Position Embedding）计算的 ranks
    position_embedding_ranks = get_position_embedding_ranks(ranks)
    # 创建位置编码（Position Embedding）的通信组
    group = torch.distributed.new_group(
        position_embedding_ranks,
        timeout=timeout,
        pg_options=get_nccl_options('embd', nccl_comm_cfgs),
    )
    if rank in position_embedding_ranks:
        _POSITION_EMBEDDING_GROUP = group  # 记录位置编码计算的通信组
        _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks  # 记录位置编码的 rank 列表
```

除了 `dp`、`tp`、`pp` 三种并行组，`mpu` 还会处理混合并行的情况，你在代码里面还会看到类似下面的代码：

```python
for ranks in generator_wrapper('tp-pp'):
    group = torch.distributed.new_group(
        ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
    )
    if rank in ranks:
        _MODEL_PARALLEL_GROUP = group
        _MODEL_PARALLEL_GLOBAL_RANKS = ranks
```

这段代码负责创建混合并行组，即同时使用张量并行和流水线并行的情况。

上面代码里面最重要的是 RankGenerator 和 `generator_wrapper()`，它会根据不同的并行方式生成不同的 rank 组。

**1、RankGenerator：并行计算的“坐标系”**

RankGenerator 这个类的作用，是根据 张量并行（TP）、数据并行（DP）、流水线并行（PP）、专家并行（EP）、上下文并行（CP） 这些并行方式，来计算 GPU 进程在不同分组中的排列方式。

在 `decoder-only` 模型中，我们主要关心 TP-DP-PP 这三种并行方式的划分。

来看 `RankGenerator` 这个类的初始化部分：

```python
class RankGenerator(object):
    """用于生成不同并行方式下的 rank 分组"""

    def __init__(
        self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0
    ) -> None:
        assert (
            ep == 1 or cp == 1
        ), "EP 和 CP 不能同时大于 1，它们分别属于不同的并行策略"

        # 记录并行方式的规模
        self.tp = tp  # 张量并行大小
        self.ep = ep  # 专家并行大小（MoE）
        self.dp = dp  # 数据并行大小
        self.pp = pp  # 流水线并行大小
        self.cp = cp  # 上下文并行大小（Megatron-LM 1.1 之后的新特性）
        self.rank_offset = rank_offset  # 进程偏移量

        # 计算世界大小（world_size），即总的 GPU 数
        self.world_size = tp * dp * pp * cp * ep

        # 并行方式与对应大小的映射
        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }

        # 解析 order，例如 "tp-dp-pp"
        order = order.lower()
        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"{name} 并行大小为 {self.name_to_size[name]}，但未指定其顺序"
                )
            elif name not in order:
                order = order + '-' + name  # 默认加到末尾

        self.order = order
        self.ordered_size = [self.name_to_size[token] for token in order.split('-')]
```

这里的 `order`，它决定了 GPU 进程的排列方式。 例如，如果 `order="tp-dp-pp"`，表示 先按照张量并行分组，再按照数据并行，再按照流水线并行。

你可能还在疑惑，这些 rank 分组 究竟是怎么计算出来的？`generator_wrapper()` 只是调用了 `RankGenerator.get_ranks()`，它内部究竟做了什么？


这就涉及到 `generate_masked_orthogonal_rank_groups()` 这个函数，它的作用就是 从数学上求解这些 rank 组，并确保它们满足 正交分布（orthogonal decomposition）。换句话说，这个函数的本质就是在一个多维网格中计算索引关系，确保每个 GPU 进程正确分配到不同的并行组里。

在 Megatron-LM 的分布式训练中，每张 GPU 的 全局索引（global rank）计算方式可以抽象为一个高维坐标系：

$$
R = T + D \cdot S_T + P \cdot (S_T \cdot S_D)
$$

其中：
- **$R$** ：全局 rank（global rank）
- **$T$** ：张量并行（Tensor Parallel, TP）索引
- **$D$** ：数据并行（Data Parallel, DP）索引
- **$P$** ：流水线并行（Pipeline Parallel, PP）索引
- **$S_T, S_D, S_P$** ：分别是张量、数据、流水线并行的 **进程数量（size）**

:::tip

这个公式本质上是一个 **高维坐标到 1D 索引的映射**，类似于我们在行优先存储的矩阵中计算线性索引的方法。

:::

下面我们看一下例子：

假设：

- $S_T = 2$（张量并行 2）
- $S_D = 3$（数据并行 3）
- $S_P = 4$（流水线并行 4）

总 **GPU 数量** $N = S_T \times S_D \times S_P = 2 \times 3 \times 4 = 24$

那么，GPU **全局 rank 计算规则** 如下：

$$
R = T + D \cdot 2 + P \cdot (2 \times 3)
$$

:::tip

这里的索引计算方式 **与 Numpy 的 `np.ravel_multi_index` 非常类似**，它将多维索引展平成 1D 数组索引。

:::

**具体 GPU Rank 示例：**

| $T$ | $D$ | $P$ | `global_rank` 计算 |
|------------|------------|------------|----------------|
| 0 | 0 | 0 | $0 + 0 \times 2 + 0 \times 6 = 0$ |
| 1 | 0 | 0 | $1 + 0 \times 2 + 0 \times 6 = 1$ |
| 0 | 1 | 0 | $0 + 1 \times 2 + 0 \times 6 = 2$ |
| 1 | 1 | 0 | $1 + 1 \times 2 + 0 \times 6 = 3$ |
| 0 | 0 | 1 | $0 + 0 \times 2 + 1 \times 6 = 6$ |
| 1 | 0 | 1 | $1 + 0 \times 2 + 1 \times 6 = 7$ |


Megatron-LM 还需要在 **不同的并行模式** 下划分 GPU 进程组，例如：

- **数据并行（DP）**
- **张量并行（TP）**
- **流水线并行（PP）**
- **张量+数据并行（TP-DP）**
- ……


我们可以利用上面的 **全局索引公式**，**固定某些维度，枚举另一些维度**，从而得到进程组。


假设 `mask = [False, True, False]`（忽略 TP、PP 维度，仅关注 DP），  
我们要找到 **所有 rank 具有相同 `D`（数据并行索引）** 的 GPU。


$$
D_{\text{group index}} = T + P \cdot S_T
$$

$$
\text{DP 进程组} = \{ D_{\text{group index}} + D \cdot (S_T \cdot S_P) \mid D \in [0, S_D) \}
$$


继续使用 `S_T=2, S_D=3, S_P=4`：

- `T + P * 2` 生成 **8 组 DP 进程**
- 每组有 `S_D=3` 个 GPU，`global_rank` 按照 `D` 维度变化

| DP Group Index | GPU 进程组 (`DP`) |
|------------------|-------------------------|
| 0  | [0, 2, 4]  |
| 1  | [1, 3, 5]  |
| 2  | [6, 8, 10] |
| 3  | [7, 9, 11] |
| 4  | [12, 14, 16] |
| 5  | [13, 15, 17] |
| 6  | [18, 20, 22] |
| 7  | [19, 21, 23] |


如果 `mask = [True, False, False]`（忽略 DP、PP，仅关注 TP），  
我们要找到 **所有 rank 具有相同 `T`（张量并行索引）** 的 GPU。

$$
T_{\text{group index}} = D + P \cdot S_D
$$

$$
\text{TP 进程组} = \{ T_{\text{group index}} + T \cdot 1 \mid T \in [0, S_T) \}
$$

假设：

- `D + P * 3` 生成 **6 组 TP 进程**
- 每组有 `S_T=2` 个 GPU，`global_rank` 按照 `T` 维度变化

| TP Group Index | GPU 进程组 (`TP`) |
|------------------|-------------------------|
| 0  | [0, 1]  |
| 1  | [2, 3]  |
| 2  | [4, 5]  |
| 3  | [6, 7]  |
| 4  | [8, 9]  |
| 5  | [10, 11] |

---

如果 `mask = [False, False, True]`（忽略 TP、DP，仅关注 PP），  
我们要找到 **所有 rank 具有相同 `P`（流水线并行索引）** 的 GPU。

$$
P_{\text{group index}} = T + D \cdot S_T
$$

$$
\text{PP 进程组} = \{ P_{\text{group index}} + P \cdot (S_T \times S_D) \mid P \in [0, S_P) \}
$$

假设：

- `T + D * 2` 生成 **6 组 PP 进程**
- 每组有 `S_P=4` 个 GPU，`global_rank` 按照 `P` 维度变化

| PP Group Index | GPU 进程组 (`PP`) |
|------------------|-------------------------|
| 0  | [0, 6, 12, 18]  |
| 1  | [1, 7, 13, 19]  |
| 2  | [2, 8, 14, 20] |
| 3  | [3, 9, 15, 21] |
| 4  | [4, 10, 16, 22] |
| 5  | [5, 11, 17, 23] |


:::note

直接这样看可能有点抽象，但实际上就是在 **多维坐标系** 中计算 **不同维度的索引**，然后根据这些索引生成 **不同的 GPU 进程组**。如果还不理解可以看看这篇文章中关于 Mesh 的描述：[主流框架如何定义和管理分布式张量](https://space.keter.host/docs/high_performance/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/%E5%88%86%E5%B8%83%E5%BC%8F%E5%8E%9F%E7%94%9F%E5%B1%9E%E6%80%A7)

:::


你可以把 mpu 想象成 Megatron-LM 训练过程中的分布式调度中心，它的核心作用包括：

1. 管理并行策略：让 GPU 知道自己属于哪个并行组
2. 创建并行进程组：让 PyTorch `torch.distributed` 了解 GPU 之间的通信拓扑
3. 提供查询接口：训练过程中随时可以获取当前 GPU 在不同并行方式下的 rank

## 4. 总结

在本篇文章中，我们从 分布式环境的基本概念 出发，深入剖析了 Megatron-LM 是如何利用 `torch.distributed` 进行进程管理，并通过 `mpu（Model Parallel Unit）` 组织不同的并行方式。

希望你通过这篇文章可以了解到：

1. world_size、rank、local_rank：理解了它们在分布式环境中的角色，尤其是如何在多机多卡的情况下正确映射进程到 GPU
2. `initialize_megatron()` 的完整流程，知道如何初始化分布式进程，并确保训练的正确性
3. mpu 模块的作用，理解它如何将不同的 GPU 分配到 数据并行（DP）、张量并行（TP）和流水线并行（PP） 组中
4. 分布式进程组的计算方法，能够理解如何计算不同并行方式的进程组

如果你能通过这篇文章去引申出下列问题，那么你一定是一个具有深度思考能力的人才：

1. 本文主要讨论了进程组的创建，但 Megatron-LM 是如何在训练过程中让这些并行方式协同工作的呢？它们的通信是如何组织的？
2. 数据是如何被加载并分配到不同的 GPU 上的？
3. 前向传播和后向传播如何在多个 GPU 之间高效执行？

这些问题都是在后续文章中会逐步展开讨论的，希望你能继续关注我的文章，一起探索 Megatron-LM 的深奥之处。

## 参考资料

1. https://space.keter.host/docs/high_performance/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/%E5%88%86%E5%B8%83%E5%BC%8F%E5%8E%9F%E7%94%9F%E5%B1%9E%E6%80%A7
2. https://zhuanlan.zhihu.com/p/650383289
3. https://zhuanlan.zhihu.com/p/650237820



