# 如何使用 Megatron 训练 GPT3

在阅读源码之前，我们首先看看 Megatron 是如何使用的，首先我们先要会用，否则看源码也是白看。这篇文章将介绍如何使用 Megatron 训练 GPT3 模型，在我们搞明白 Megatron 怎么用了之后，我们再逐步深入到训练代码当中。

:::tip
阅读本文需要了解分布式深度学习训练的基本概念，如数据并行、模型并行、混合精度训练等。
:::


## 1. 启动脚本

Megatron 提供了一个 `/examples/gpt3/train_gpt3_175b_distributed.sh` 脚本，用于训练 GPT3 模型。我们可以通过修改这个脚本来训练不同的模型。

:::tip
Megatron 版本：v0.10.0
:::

```bash
#!/bin/bash

# 运行 "175B" 参数规模的 GPT 模型

# 设置 CUDA 最大连接数，以防止 GPU 之间的过多连接导致性能问题
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 每个节点上的 GPU 数量
GPUS_PER_NODE=8

# 多节点配置参数
MASTER_ADDR=localhost  # 主节点地址（如果多节点部署，需要更改）
MASTER_PORT=6000       # 主节点通信端口
NUM_NODES=1            # 总计算节点数（单节点训练）
NODE_RANK=0            # 当前节点的排名（单节点训练时为 0）
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))  # 全局 GPU 数量

# 从脚本输入参数中读取路径信息
CHECKPOINT_PATH=$1 # 指定模型检查点路径
TENSORBOARD_LOGS_PATH=$2 # 指定 TensorBoard 日志存储路径
VOCAB_FILE=$3 # 指定 GPT-2 词汇表 JSON 文件路径
MERGE_FILE=$4 # 指定 GPT-2 分词合并规则文件路径
DATA_PATH=$5 # 指定训练数据文件的路径前缀

# 分布式训练参数
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE  # 每个节点上的进程数（等于 GPU 数量）
    --nnodes $NUM_NODES  # 总节点数
    --master_addr $MASTER_ADDR  # 主节点地址
    --master_port $MASTER_PORT  # 主节点端口
)

# GPT 模型超参数配置
GPT_MODEL_ARGS=(
    --num-layers 96  # Transformer 层数
    --hidden-size 12288  # 隐藏层维度大小
    --num-attention-heads 96  # 自注意力机制的头数
    --seq-length 2048  # 序列长度（最大 token 数）
    --max-position-embeddings 2048  # 最大位置嵌入数
)

# 训练超参数配置
TRAINING_ARGS=(
    --micro-batch-size 1  # 微批次大小（每个 GPU 上的批次大小）
    --global-batch-size 1536  # 全局批次大小（所有 GPU 总和）
    --rampup-batch-size 16 16 5859375  # 逐步增加批次大小的计划
    --train-iters 500000  # 训练的总迭代次数
    --weight-decay 0.1  # 权重衰减率（L2 正则化）
    --adam-beta1 0.9  # Adam 优化器 beta1 参数
    --adam-beta2 0.95  # Adam 优化器 beta2 参数
    --init-method-std 0.006  # 权重初始化标准差
    --clip-grad 1.0  # 梯度裁剪上限
    --fp16  # 启用混合精度训练（FP16）
    --lr 6.0e-5  # 初始学习率
    --lr-decay-style cosine  # 学习率衰减方式（余弦衰减）
    --min-lr 6.0e-6  # 最小学习率
    --lr-warmup-fraction .001  # 预热学习率比例
    --lr-decay-iters 430000  # 学习率衰减的总步数
)

# 模型并行参数（张量并行与流水线并行）
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8  # 张量模型并行度
    --pipeline-model-parallel-size 16  # 流水线模型并行度
)

# 数据相关参数
DATA_ARGS=(
    --data-path $DATA_PATH  # 训练数据路径
    --vocab-file $VOCAB_FILE  # 词汇表文件
    --merge-file $MERGE_FILE  # 分词合并规则文件
    --split 949,50,1  # 数据集划分比例（训练/验证/测试）
)

# 评估与日志参数
EVAL_AND_LOGGING_ARGS=(
    --log-interval 100  # 日志打印间隔
    --save-interval 10000  # 检查点保存间隔
    --eval-interval 1000  # 评估间隔
    --save $CHECKPOINT_PATH  # 保存检查点路径
    --load $CHECKPOINT_PATH  # 加载检查点路径
    --eval-iters 10  # 评估时的迭代次数
    --tensorboard-dir $TENSORBOARD_LOGS_PATH  # TensorBoard 日志存储路径
)

# 启动分布式训练，执行 GPT 预训练脚本
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
```

`GPUS_PER_NODE=8` 表示在**每个计算节点上可用的 GPU 数量**。在分布式深度学习训练中，每个节点通常由多个 GPU 组成，而该变量定义了在单个计算节点上将使用的 GPU 数量。

`MODEL_PARALLEL_ARGS` 中的参数用于控制不同的并行模式，比如上面的 `--tensor-model-parallel-size 8` 表示张量模型并行度为 8，`--pipeline-model-parallel-size 16` 表示流水线模型并行度为 16。总的并行度等于张量模型并行度乘以流水线模型并行度乘以数据并行度。这里数据并行度默认为 1，总共的并行度为 8 * 16 * 1 = 128，也就是说需要 128 个 GPU 来做训练，每个机器上 8 个 GPU，也就是需要 16 台机器。

当然了，你也可以把并行度太小来在单机上跑，比如 `--tensor-model-parallel-size 4`，`--pipeline-model-parallel-size 2`，这样就只需要 8 个 GPU 来训练了，不过 8 个 GPU 应该是装不下这么大的模型的，我们自己做实验的时候可以适当调小模型规模，比如 `--num-layers 24`，`--hidden-size 3072`。

`torchrun` 是 PyTorch 提供的一个命令行工具，专门用于分布式训练。它替代了早期的 `python -m torch.distributed.launch`，提供更好的功能和易用性。

## 2. 预训练入口

Megatron 提供了一个 `pretrain_gpt.py` 脚本，用于执行 GPT 模型的预训练任务。

### 2.1. 结构概览

`pretrain_gpt.py` 主要由以下几个核心部分组成：

1. 模型构建：定义 `model_provider` 函数来构建 GPT 模型。
2. 数据加载：定义数据集提供函数 `train_valid_test_datasets_provider`。
3. 训练步骤：实现前向传播 `forward_step` 和损失计算 `loss_func`。
4. 主函数入口：通过 `pretrain()` 启动训练流程。

### 2.2. 模型构建

`model_provider` 函数负责初始化 GPT 模型。其参数包括 `pre_process` 和 `post_process`，用于决定是否执行嵌入层的处理和输出层的计算。

```python
def model_provider(pre_process=True, post_process=True):
    # 获取训练的全局参数
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    # 如果启用内存历史记录，记录 CUDA 内存的分配情况
    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True, trace_alloc_max_entries=100000, trace_alloc_record_context=True
        )

    print_rank_0('building GPT model ...')

    # 从 YAML 配置文件或命令行参数加载配置
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)
    
    # 构建 GPT 模型
    ...
    with nullcontext():
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
        )
```

### 2.3. 数据加载

在训练过程中，需要加载训练、验证和测试数据集。该函数负责提供数据集实例。

```python
def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    # 如果启用 mock 数据，则使用 MockGPTDataset 进行测试，否则使用真实数据集
    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    return train_ds, valid_ds, test_ds

```

### 2.4. 前向传播

该函数定义了 `GPT` 训练过程中的前向传播步骤。在 `forward_step` 函数中，首先记录数据加载时间，随后调用 `get_batch` 获取批次数据，并将其输入模型进行前向传播计算，最终返回输出张量及损失函数的部分调用。

```python
def forward_step(data_iterator, model: GPTModel):
    args = get_args()
    timers = get_timers()

    # 计时数据加载时间
    timers('batch-generator', log_level=2).start()

    # 从数据迭代器中获取批次数据
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    timers('batch-generator').stop()

    # 执行前向传播计算
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)
```

### 2.5. 损失计算

在训练过程中，计算损失是关键步骤，该函数用于处理损失计算及异常情况处理。这个没什么特殊的，就是计算损失，然后返回损失值，唯一需要注意的地方就是这里在数据并行组中做了 `all_reduce` 操作，这是因为每个数据并行组都会计算损失，需要将这些损失值合并。

```python
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.sum(losses.view(-1) * loss_mask).view(1)

    # 在数据并行分组中进行 all_reduce 操作
    torch.distributed.all_reduce(loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return loss, local_num_tokens
```

### 2.6. 主函数入口

在主函数入口，调用 `pretrain()` 函数启动完整的 GPT 训练流程。

```python
if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
```

**这一部分大部分都是对其他封装好逻辑的调用，我们在后续的文章中会逐步深入到这些函数的实现细节中。**

## 4. 预训练流程

Megatron-LM 预训练的核心入口是 `pretrain` 函数，它负责从头到尾组织训练过程，包括**初始化、数据加载、训练循环、评估与保存**。可以将整个流程想象成一个精密的流水线，依次执行一系列关键任务，最终推动模型的不断优化。

预训练的第一步是初始化 Megatron，它的作用类似于搭建训练的基础框架。`initialize_megatron` 函数在这一阶段扮演着至关重要的角色，它不仅解析了命令行参数，还完成了分布式环境的配置。想象一下，如果没有这一步，训练过程中多个 GPU 将无法协同工作，也无法正确地分配计算任务。这个函数内部调用了 PyTorch 的 `torch.distributed.init_process_group`，为接下来的模型并行和数据并行奠定了基础。

当 Megatron 完成初始化后，接下来的重点就是模型和优化器的设置。这里，我们要感谢 `setup_model_and_optimizer` 函数，它接管了**构建 GPT 模型、配置优化器以及设定学习率调度器的任务**。在这一步中，`model_provider` 函数被调用来实例化模型，而 Megatron 通过其独特的并行技术，比如张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism），将模型拆分到多个 GPU 上，以充分利用计算资源。与此同时，优化器（如 Adam）也会在这里被配置好，确保模型在训练过程中能够稳定地进行参数更新。

模型准备就绪后，训练自然离不开数据。在 Megatron-LM 中，数据的加载和预处理同样是一个至关重要的环节。`build_train_valid_test_data_iterators` 函数会负责从磁盘中读取训练、验证和测试数据集，并将其转换成适合 GPU 处理的格式。为了确保分布式训练的高效性，Megatron 采用了 `DistributedSampler`，让每个 GPU 处理不同的数据，从而避免不必要的重复计算。

随着数据加载的完成，真正的训练循环正式拉开帷幕。Megatron 通过 `train_step` 函数逐步推进训练过程。这个阶段包括前向传播、反向传播以及参数更新三个核心步骤。前向传播阶段，模型会接收输入数据，并依次通过多个 Transformer 层，最终生成输出结果；反向传播则负责计算梯度，指导模型如何调整自身参数。最关键的是，Megatron 结合了 PyTorch DDP（DistributedDataParallel）技术，在多个 GPU 之间高效地同步梯度，确保模型在每个节点上的参数保持一致。

训练过程中，Megatron 还会定期执行模型评估和保存检查点。这一环节通常由 `evaluate_and_print_results` 和 `save_checkpoint` 函数完成。评估阶段主要在验证集上进行，以检查模型的泛化能力。而检查点的保存则是为了**防止训练中断带来的损失，让我们可以随时恢复训练进度，继续从中断的位置继续优化**。特别是在大规模模型训练中，训练过程中的中断是常见的，比如 Llama3 的训练一次就有几百次的中断，这时候检查点的保存就显得尤为重要。

具体代码实现如下：

```python
def pretrain(
    train_valid_test_dataset_provider,  # 提供训练/验证/测试数据集的函数
    model_provider,  # 提供模型定义的函数
    model_type,  # 指定训练模型的类型
    forward_step_func,  # 执行前向传播的函数
    process_non_loss_data_func=None,  # 处理非损失数据的函数，通常用于TensorBoard等
    extra_args_provider=None,  # 额外的参数提供函数
    args_defaults={},  # 参数默认值
    get_embedding_ranks=None,  # 嵌入层分配的 GPU 排名
    get_position_embedding_ranks=None,  # 位置嵌入层分配的 GPU 排名
    non_loss_data_func=None,  # 自定义评估过程中处理非损失数据的函数
):
    """主训练程序。

    该函数按以下顺序执行：
        1) 初始化 Megatron。
        2) 使用 `model_provider` 设置模型、优化器和学习率调度。
        3) 调用 `train_val_test_data_provider` 获取训练/验证/测试数据集。
        4) 使用 `forward_step_func` 进行模型训练。
    """

    # 初始化 Megatron，解析命令行参数、设置分布式环境、日志工具等
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks
    )

    args = get_args()  # 获取训练参数
    timers = get_timers()  # 获取定时器对象

    if args.log_progress:
        append_to_progress_log("Starting job")  # 记录训练启动日志

    # 设置 PyTorch JIT 层融合选项，并预热 JIT 函数以优化计算性能
    set_jit_fusion_options()

    # 调整启动时间，以确保获取最小启动时间，利于调度分析
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.double,
                                     device='cuda')
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()

    app_metrics = {}
    app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
    app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

    print_rank_0('初始化 Megatron 完成，耗时 (秒): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('初始化 Megatron 之后')
    app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # 记录训练开始的时间点
    one_logger_utils.on_pretrain_start()

    # 处理检查点上下文，如果启用了本地非持久性检查点，抛出异常
    if args.non_persistent_ckpt_type == 'local':
        raise RuntimeError('本地检查点管理器尚未集成')
        checkpointing_context = {
            'local_checkpoint_manager': BasicLocalCheckpointManager(
                args.non_persistent_local_ckpt_dir
            )
        }
    else:
        checkpointing_context = {}

    # 初始化模型、优化器和学习率调度
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, checkpointing_context=checkpointing_context)
    timers('model-and-optimizer-setup').stop()
    print_datetime('模型、优化器和学习率调度器构建完成')
    app_metrics['app_build_optimizer_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    config = get_model_config(model[0])  # 获取模型配置

    # 数据加载流程
    app_metrics['app_build_dataiters_start_time'] = one_logger_utils.get_timestamp_in_ms()
    timers('train/valid/test-data-iterators-setup', log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('数据加载完成')
    app_metrics['app_build_dataiters_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # 记录训练、验证、测试的启用状态
    one_logger_utils.track_config_flags(args.train_iters, args.skip_train, args.do_train,
                                        args.do_valid, args.do_test, args.dataloader_type,
                                        args.retro_project_dir, args.retro_cyclic_train_iters)

    # 启用容错监控
    if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
        ft_integration.get_rank_monitor_client().init_workload_monitoring()
        ft_timeouts = ft_integration.get_rank_monitor_client().timeouts
        print_rank_0(f"容错客户端已初始化，超时时间: {ft_timeouts}")

    # 输出初始化步骤耗时
    print_rank_0('初始化完成，开始训练...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'], barrier=True)

    one_logger = get_one_logger()
    one_logger and one_logger.log_metrics(app_metrics)

    # 训练过程
    if not args.skip_train:
        print_rank_0('开始训练...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0(f"复古循环训练轮数: {args.train_iters}")

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func, model, optimizer, opt_param_scheduler,
                train_data_iterator, valid_data_iterator,
                process_non_loss_data_func, config, checkpointing_context,
                non_loss_data_func)

        print_datetime('训练完成')

        # 保存检查点
        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context,
                            train_data_iterator=train_data_iterator,
                            ft_client=ft_integration.get_rank_monitor_client(
                                ft_integration.StateMachineActions.SAVE_CHECKPOINT),
                            preprocess_common_state_dict_fn=preprocess_common_state_dict)

        one_logger and one_logger.log_metrics({
            'app_train_loop_finish_time': one_logger_utils.get_timestamp_in_ms()
        })

    else:
        print_rank_0('跳过训练 (--skip-train 已启用)...')
        iteration = args.iteration

    # 进行验证
    if args.do_valid:
        prefix = f'第 {iteration} 轮验证集评估'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train,
                                   non_loss_data_func=non_loss_data_func)

    # 进行测试
    if args.do_test:
        prefix = f'第 {iteration} 轮测试集评估'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train,
                                   non_loss_data_func=non_loss_data_func)

    # 完成训练并关闭 WandB 记录
    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()
    maybe_finalize_async_save(blocking=True)

    one_logger and one_logger.log_metrics({
        'app_finish_time': one_logger_utils.get_timestamp_in_ms()
    })
    one_logger_utils.finish()
```

## 5. 总结

通过今天的介绍，我们大致了解了 Megatron-LM 预训练的整体流程：从初始化环境、设置模型和优化器，到数据加载，再到训练循环，最后是评估与保存。虽然这只是冰山一角，但足以帮助我们建立一个清晰的框架。

你看完这篇文章可能会有很多疑问，可能的问题可能会有下面几类：

1. **并行训练的三种策略**
    - 数据并行（DP）：如何划分数据？各个 GPU 如何独立计算梯度并同步？
    - 张量并行（TP）：如何将单个矩阵操作分解到多个 GPU 上进行计算？
    - 流水线并行（PP）：如何划分模型层？前向和反向传播如何穿插执行？
    - 这三种并行方法如何协同工作？Megatron-LM 是否支持混合并行？
2. **训练过程中的 GPU 通信**
    - GPU 之间如何进行数据交换？使用了哪些通信方式（如 AllReduce、Send/Recv）？
    - Megatron-LM 使用哪种通信后端（NCCL、Gloo、MPI）？如何选择合适的后端？
    - 数据并行如何保证各个 GPU 在训练时同步权重？
    - 为什么需要进行梯度规约（Gradient All-Reduce），它如何影响训练性能？
3. **分布式环境的配置**
    - 什么是 `world_size`，它如何决定分布式训练的规模？
    - 什么是 `rank`，它在训练中的作用是什么？
    - 什么是 `local_rank`，它如何与 `global rank` 关联？
4. **流水线并行的深层次问题**
    - Megatron-LM 中的流水线并行是如何实现的？
    - 为什么流水线并行需要 "micro-batching"？
    - 什么时候应该启用虚拟流水线并行（Virtual Pipeline Parallelism）？
5. **模型并行的细节**
    - 如何确定模型应该拆分到多少个 GPU 上？是否有最佳实践？
    - 当使用张量并行时，如何确保矩阵分割后仍能正确执行计算？
    - 在模型并行中，如何处理跨 GPU 的层（如 MLP 和 Attention）？

上面的问题又引出了更多的问题，后续我们会逐步深入到这些问题中，深入探索 Megatron-LM 的源码细节。

