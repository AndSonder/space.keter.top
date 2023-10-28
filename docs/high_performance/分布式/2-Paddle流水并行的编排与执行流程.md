# Paddle流水并行编排与执行流程

## 简介

流水线并行是一种模型并行技术,它沿着前向计算和反向传播的方向,将深度学习模型划分成多个线性的阶段,使不同阶段可以并行运行在不同的设备上。这种并行方式可以有效提升计算效率,缩短训练时间。

在流水线并行中,一个训练步骤通常被划分为三个子阶段:

- Forward: 前向计算,每个阶段计算并输出中间结果给下一个阶段;
- Backward: 反向传播,每个阶段根据上一个阶段的梯度计算并传递当前阶段的梯度;
- Update: 参数更新,收集所有阶段的梯度并更新模型参数。

通过并行运行,不同阶段的计算可以高效重叠,极大提升吞吐量。

Paddle 通过 Strategy 和 Engine 等模块支持流水线并行。用户只需简单配置,即可对模型自动进行流水线切分,简化训练流程。

Engine 是 Paddle 中用于支持流水线并行的高层API (`python/paddle/distributed/auto_parallel/static/engine.py`)

Engine提供了高层封装,整合了自动并行转换、执行调度等关键流程。通过 Strategy 配置流水线策略, Engine 可以自动对模型进行流水线切分,生成分段Program; 然后组织分布式执行, 大幅降低使用门槛。

后面我们将详细介绍Engine中流水线并行的自动编排与执行流程的核心代码实现。

## 流水线并行策略配置

在Engine中,流水线并行策略是按照传入的strategy配置的。比如你想使用 `FThenB` 的流水线并行策略, 可以这样配置:

```python
# From test/auto_parallel/pipeline_scheduler_unittest.py
def apply_pass(schedule_mode="FThenB", enable_send_recv_overlap=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    pipeline = strategy.pipeline
    pipeline.enable = True
    pipeline.schedule_mode = schedule_mode
    pipeline.accumulate_steps = 4
    pipeline.enable_send_recv_overlap = enable_send_recv_overlap

    return strategy

def main():
    strategy = apply_pass(schedule_mode="FThenB", enable_send_recv_overlap=False)
    model = xxx
    loss = xxx
    opt = xxx
    engine = auto.Engine(model, loss, opt, strategy=strategy)
```

`auto.Strategy` 是一个配置类，主要用于存储用户配置的策略信息。流水线并行策略主要包含 `FThenB`、`1F1B` 两种模式。 

下面我们用两张图来说明两种编排模式。下图说明了 `FThenB` 的编排模式，等待所有的设备都执行完前向计算后，再执行反向传播。

![picture 0](images/3c550ede284b7c40d4eaf54f44af682a04e02ff2f7ca27767d7f32fc88d7bfa1.png)  

在 `1F1B` 的编排模式下，每个设备先执行前向计算，然后再执行反向传播。不等待所有设备都执行完前向计算，就开始执行反向传播。

![picture 1](images/b658885619688b369504f4de8057ed2a17f5d79e60d550ebf23f73cf6cd848fe.png)  

## 自动编排阶段

Engine在接收到用户定义的流水线并行策略后,会自动完成训练 Program 的并行编排。下面我们以 Engine 中的 `_prepare_program` 方法作为基线, 来看看具体的实现逻辑。

`_prepare_program()` 方法主要包含模型的构建、规划、并行化和初始化等过程。它的代码如下:

```cpp
def _prepare_program(self, mode, init_parameters=True):
    # Do the build process
    self._build(mode)
    # Do the planning process
    self._plan(mode)
    # Do the parallel process
    self._parallel(mode)
    # Init comm
    self._init_comm()
    if init_parameters:
        # startup program
        self._initialize(mode)
    self._has_prepared[mode] = True
```

具体来说:

1. _build() 方法会构建计算图。在动态图模式下,会调用to_static方法将动态图转为静态图;在静态图模式下, 会直接构建静态计算图。
2. _plan() 方法会进行并行策略的规划,生成并行方案。
3. _parallel() 方法会根据并行方案对模型进行并行化改造。 
4. _init_comm() 方法会实例化 Program 中的通信操作。
5. _initialize() 方法会进行参数初始化等操作,完成并行环境的准备。

其中 `_build()`、`_plan()` 和 `_parallel()` 是核心的模型构建和并行化过程。`_init_comm()` 和 `_initialize()` 则主要是进行并行环境的初始化。

`_prepare_program()` 方法被 `fit()`、`evaluate()`、`predict()` 等调用, 目的是在训练/验证/预测前构建并准备好并行执行的环境。这样后续的训练循环等就可以直接在这个环境下高效地运行了。

### 串行 program 的构建

:::tip

PaddlePaddle 的 Program 是一个计算图，它描述了一个深度学习模型的结构和计算过程。 ([Program 介绍](https://space.keter.top/docs/high_performance/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91/%E9%9D%99%E6%80%81%E5%9B%BE%E7%BB%84%E7%BD%91%E8%BF%87%E7%A8%8B#%E4%BB%80%E4%B9%88%E6%98%AF-program))

:::

_build()函数是 AutoParallel 中的模型构建过程, 它会构建计算图来表示模型。 `_build` 函数首先检测当前的执行模式，以确定是动态图还是静态图模式。如果当前处于动态图模式，函数会进入这个分支。在动态图模式下，模型的构建是动态的，因此需要创建一个 `ProgramHelper` 实例，该实例帮助构建动态图。它包括创建前向计算图、计算损失和度量指标等。在静态图模式下，模型的计算图是静态的，因此函数会克隆原始的静态计算图，并创建占位符（placeholder）用于输入数据和标签。

```python
# From python/paddle/distributed/auto_parallel/static/engine.py
def _build(self, mode):
    # 检查当前是否在动态图模式，或者已经处于动态图模式
    if in_dynamic_mode() or self._dygraph_mode:
        # 进入动态图模式
        paddle.disable_static()  # 关闭静态图模式
        self._dygraph_mode = True  # 标记当前为动态图模式
        self._logger.info("Building model with 'to_static' method.")

        # 创建 ProgramHelper 对象，用于帮助构建计算图
        self.program_helper = ProgramHelper(
            self._model,
            self._loss,
            self._metrics,
            self._inputs_spec,
            self._labels_spec,
        )
        # 构建前向计算主Program
        with utils.unique_name.guard():
            self.program_helper.build_program(mode)

        # 获取具体的计算图（Concrete Program）以及静态图模式下的主Program和启动Program
        self.concrete_program = self.program_helper.concrete_program
        serial_main_prog = self.program_helper.main_program
        serial_startup_prog = self.program_helper.startup_program

        # 获取输入数据和标签，以及模型的输出、损失和度量指标
        self._inputs = self.program_helper.input_vars
        self._labels = self.program_helper.label_vars
        outputs = self.program_helper.output_vars
        self._losses = self.program_helper.loss_vars
        metrics = self.program_helper.metric_vars

        # 恢复静态图模式
        paddle.enable_static()
    else:
        # 进入静态图模式
        # 检查是否已经构建了静态图模式下的计算上下文
        dist_context = self._dist_contexts.get(mode, None)
        if dist_context is not None:
            return

        outputs = []
        metrics = []
        self._losses = []
        # 克隆原始静态主Program
        serial_main_prog = self._orig_main_prog.clone()
        # 克隆原始静态启动Program
        serial_startup_prog = self._orig_startup_prog.clone()  
        if not self._skip_build:
            with static.program_guard(
                serial_main_prog, serial_startup_prog
            ), utils.unique_name.guard():
                self._inputs = [
                    s._create_feed_layer() for s in self._inputs_spec
                ]
                self._labels = [
                    s._create_feed_layer() for s in self._labels_spec
                ]

                outputs = auto_utils.to_list(self._model(*self._inputs))

                # 如果不是预测模式，并且定义了损失函数（loss）
                if mode != "predict" and self._loss:
                    # 确保损失函数是 paddle.nn.Layer 类的实例或可调用函数
                    assert isinstance(
                        self._loss, paddle.nn.Layer
                    ) or callable(
                        self._loss
                    ), "the type of `loss` of the Engine arguments should be sub classes of `paddle.nn.Layer` or any callable function."
                    self._losses = auto_utils.to_list(
                        self._loss(*(outputs + self._labels))
                    )

                # 如果不是预测模式，并且存在输出或标签数据
                if mode != "predict" and (outputs or self._labels):
                    # 计算度量指标
                    for metric in self._metrics:
                        metrics.append(
                            auto_utils.to_list(
                                metric.compute(*(outputs + self._labels))
                            )
                        )
        # 如果是训练模式，确保损失是一个 Variable
        elif mode == "train":
            assert isinstance(
                self._loss, Variable
            ), "the type of `loss` of the Engine arguments should be Variable."
            self._losses = auto_utils.to_list(self._loss)
```

论是动态图还是静态图模式，函数都会准备分布式训练所需的上下文。这包括前向计算图（serial_main_prog）、启动计算图（serial_startup_prog）、损失函数（self._loss）、输入占位符（self._inputs）、输出变量（outputs）、度量指标（metrics）等信息。

如果启用了分布式数据并行，函数会将输入数据和标签的占位符根据数据并行策略进行分割，以支持数据的并行处理。

```python 
# From python/paddle/distributed/auto_parallel/static/engine.py
    # 获取默认的分布式上下文，该上下文包含了关于分布式训练的配置信息
    default_ctx = get_default_distributed_context()

    # 检查默认上下文是否具有分布式注解
    if not default_ctx.has_annotation:
        # 如果默认上下文没有分布式注解，说明当前模型没有明确的分布式策略

        # 创建一个新的全局进程组（world process group）
        # 这是因为数据并行通常需要所有的进程参与，以便正确地在不同设备上分发数据
        new_process_group(list(range(self._nranks)))

        # 将默认上下文的 data_parallel 属性设置为 True，表示当前模型使用了数据并行
        default_ctx.data_parallel = True

        # 对输入占位符（input placeholders）进行处理，确保它们符合数据并行的要求
        # 对于每个输入变量 var，使用 auto_utils.set_data_parallel(var) 来将其标记为数据并行
        # 这意味着输入数据将根据并行策略在不同设备上分发
        self._inputs = [
            auto_utils.set_data_parallel(var) for var in self._inputs
        ]

        # 对标签占位符（label placeholders）进行类似的处理，以支持数据并行
        self._labels = [
            auto_utils.set_data_parallel(var) for var in self._labels
        ]
```

函数构建完成后，返回的是一个包含所有构建信息的 `DistributedContext` 对象。这个对象包括前向计算图、启动计算图、优化器、损失函数、输入占位符、输出变量、度量指标等，以支持分布式训练和数据并行。

```python 
# From python/paddle/distributed/auto_parallel/static/engine.py
    # 定义 feed_vars 字典，用于存储输入和标签的占位符
    feed_vars = {"inputs": self._inputs, "labels": self._labels}

    # 定义 fetch_vars 字典，用于存储需要从计算图中获取的变量
    fetch_vars = {
        "outputs": paddle.utils.flatten(outputs),  # 将输出变量展平以便获取
        "loss": self._losses,  # 损失函数的变量
        "metrics": metrics,  # 度量指标的变量
    }

    # 如果当前模式不是训练模式，将 serial_main_prog 设置为测试模式的克隆
    if mode != "train":
        serial_main_prog = serial_main_prog.clone(for_test=True)

    # 使用 auto_utils.set_recompute_segments 函数设置需要重新计算的分段（segments）
    # 这通常与模型的某些部分相关，用于控制哪些部分需要重新计算以支持梯度合并等功能
    auto_utils.set_recompute_segments(
        self._model,  # 当前模型
        self._losses,  # 损失函数的变量
        self._strategy,  # 分布式策略配置
        serial_main_prog  # 主Program
    )

    # 创建分布式上下文对象，并将其存储在 self._dist_contexts[mode] 中
    # 该上下文包括主Program、启动Program、优化器、损失函数、输入占位符、输出变量、集群配置、策略配置以及 JSON 配置
    self._dist_contexts[mode] = DistributedContext(
        serial_main_prog,  # 主Program
        serial_startup_prog,  # 启动Program
        self._optimizer,  # 优化器
        self._losses,  # 损失函数的变量
        feed_vars,  # 输入占位符
        fetch_vars,  # 输出变量
        self._cluster,  # 集群配置
        self._strategy,  # 分布式策略配置
        self._json_config  # JSON 配置
    )

    # 创建另一个分布式上下文对象，并将其存储在 self._fwd_dist_contexts[mode] 中
    # 这个上下文对象与前一个对象类似，用于前向计算
    self._fwd_dist_contexts[mode] = DistributedContext(
        serial_main_prog,  # 主Program
        serial_startup_prog,  # 启动Program
        self._optimizer,  # 优化器
        self._losses,  # 损失函数的变量
        feed_vars,  # 输入占位符
        fetch_vars,  # 输出变量
        self._cluster,  # 集群配置
        self._strategy,  # 分布式策略配置
        self._json_config  # JSON 配置
    )

    # 设置当前模式的梯度缩放因子，根据分布式策略配置中的 gradient_scale
    self._dist_contexts[mode].gradient_scale = self._strategy.gradient_scale

    # 创建当前模式的前向 program 的克隆，以备后续使用
    self._fwd_main_progs[mode] = serial_main_prog.clone()
```

我们可以注意到 `auto_utils.set_recompute_segments` 用于设置需要重新计算的分段。这个函数的目的是配置计算图中的重新计算分段，以便在分布式训练中可以选择性地重新计算这些分段，从而提高性能和降低通信开销。这在大规模深度学习模型的分布式训练中非常有用。这里不做详细介绍，感兴趣的小伙伴可以自行阅读源码。

### 生成并行策略

_plan() 函数是 AutoParallel 中的规划过程,它会生成并行执行的详细方案。下面是 `_plan()` 函数的代码:

```python
# From python/paddle/distributed/auto_parallel/static/planner_v2.py
def _plan(self, mode):
    if self._planned_mode is None:
        # 如果之前没有规划模式，将当前模式设置为已规划的模式
        self._planned_mode = mode  
    elif self._strategy.auto_mode != "semi":
        # 如果策略的自动模式不是 "semi"，则初始化分布上下文
        # "semi" 表示半自动模式，即用户可以自定义一些规划信息
        self._init_dist_context(mode)  

    # 创建一个名为 Planner 的对象，用于规划分布式计算任务
    self._planners[mode] = Planner(mode, self._dist_contexts[mode])
    self._planners[mode].plan()

    # 推断数据并行的相关信息
    inputs_var = self._dist_contexts[mode].serial_feed_vars["inputs"]
    labels_var = self._dist_contexts[mode].serial_feed_vars["labels"]
    block = self._dist_contexts[mode].serial_main_program.global_block()
    feed_list = []

    # 获取用于数据并行的 feed 列表
    for var in inputs_var + labels_var:
        if var.name in block.vars:
            feed_list.append(block.vars[var.name])

    self._dp_world_sizes = []  # 存储数据并行的设备数量
    self._dp_ranks = []  # 存储数据并行的设备id

    # 遍历 feed 列表，获取数据并行的设备数量和设备id
    for feed_var in feed_list:
        dp_world_size, dp_rank = auto_utils.get_input_split_info(
            self._cur_rank, feed_var, self._dist_contexts[mode]
        )
        self._dp_world_sizes.append(dp_world_size)
        self._dp_ranks.append(dp_rank)
```


`Planner` 类是 `AutoParallel` 中实现自动并行规划的核心类,它的主要作用是生成并行方案并完成计算图的分布式标注。plan() 是 Planner 最关键的方法。

```python 
# From python/paddle/distributed/auto_parallel/static/planner_v2.py
def plan(self):
    # 获取日志记录器，用于记录日志信息
    logger = get_logger(logging.INFO)
    
    # 初始化变量 path
    path = None

    # 如果存在 JSON 配置，尝试获取 "tuner_load_path" 键对应的值作为 path
    if self._dist_context._json_config:
        try:
            path = self._dist_context._json_config["tuner_load_path"]
        except:
            path = None

    # 检查 path 是否存在以及是否是有效的文件路径
    if path and os.path.exists(path):
        # 加载已有的并行方案相关逻辑
        ...

    # 如果没有加载分布式属性
    if not self._load:
        if self._strategy.auto_mode != "semi":
            self._parallel_tuner.tune()
        else:
            # 完成前向注释
            self._completer.complete_forward_annotation()

    # 根据环境变量 "PADDLE_AUTO_PARALLEL_STAGE" 来判断是否执行
    if os.getenv("PADDLE_AUTO_PARALLEL_STAGE", "run") != "run":
        sys.exit()

    # 解析前向子块
    self._dist_context.block_state.parse_forward_blocks(
        self._dist_context.serial_main_program
    )
```

`Planner` 的 `plan()` 方法首先会检查是否存在已有的并行方案。如果存在，它会加载已有的并行方案，否则根据自动并行策略进行自动规划。我们可以看到，自动规划的核心是 `self._parallel_tuner.tune()` 和 `self._completer.complete_forward_annotation()`。

`self._parallel_tuner.tune()` 的作用是自动搜寻最优的并行策略。它会构建搜索空间,然后通过评估不同的并行方案,选择总体执行时间最短的方案作为最终的并行策略。

`self._completer.complete_forward_annotation()` 的作用是补全计算图中的并行属性,完成并行化。如果在semi模式下,直接使用用户定义的并行属性进行补全。否则使用使用tune函数优化得到的最优策略进行补全。在 `tune()` 方法中也会调用 `complete_forward_annotation()`。

:::tip

分布式标注(Distributed Annotation)指在分布式训练的计算图上,给操作符(Operators)和张量(Tensors)添加分布式属性的过程。这些分布式属性用于指导程序如何将运算分配到不同的设备上执行。Paddle Complete 类的作用就是完成计算图的分布式标注,给图中的操作符和张量添加分布式属性。比如 `complete_forward_annotation()` 中主要包括一下分布式标记：

- process_mesh: 进程网格,描述了操作符将运行在哪些进程上。
- dims_mapping: 维度映射表,描述了张量在不同进程间如何分割。
- impl_type: 实现类型,表示具体的分布式实现算法。
- impl_idx: 实现索引,用于区分不同的实现。

:::

### 并行化改造










## 执行流程



