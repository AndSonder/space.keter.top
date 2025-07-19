# veRL 源码解析（四）：ActorRolloutRefWorker 的实现剖析

## 1. 引言

ActorRolloutRefWorker 这个变量名，第一眼看上去可能让人困惑，它究竟是 Actor？是 Rollout？还是 Reference Policy？答案是：它三者皆是，也可以是三者的任意组合。

这种“多重人格”的设计，是出于对 RLHF 流程中一个核心工程问题的深刻洞察——**数据局部性（Data Locality）**。

让我们思考一下 RLHF 的核心循环：

1. 生成 (Rollout)：使用当前 Actor 模型的权重，生成一批新的经验数据（prompt -> response）
2. 训练 (Actor Update)：使用这批新数据，计算策略梯度，更新 Actor 模型的权重
3. KL 散度计算 (Reference)：在训练或奖励计算时，需要用一个固定不变的参考模型（Reference Policy）来计算 KL 散度，以防止策略跑得太偏。这个参考模型，通常就是训练开始时的 Actor 模型

可以发现，这三个角色 Actor、Rollout 和 Reference 在绝大多数情况下，都围绕着 **同一套或同源的模型结构与权重** 在工作。

veRL 的 ActorRolloutRefWorker 正是为了消除这些不必要的开销而设计的。它将这三个逻辑上不同、但物理上强相关的角色，聚合到了同一个 Worker 类中。

在接下来的章节中，我们将深入这个“多面手”的内部，分别剖析它的“训练态”和“生成态”是如何被实现的，并最终探究 veRL 是如何管理和同步这两个核心功能的。

## 2. 训练逻辑剖析：update_actor 方法

当 ActorRolloutRefWorker 接收到训练指令时，训练的核心入口是 `update_actor` 方法，它通过调用 MegatronPPOActor 类的 `update_policy` 方法，来完成一次完整的模型参数更新。

### 2.1. PPO 迭代器实现

训练的第一步，是准备好如何“喂”数据。PPO 算法要求使用同一批数据进行多轮（epoch）的随机小批量（mini-batch）更新。`update_policy` 方法首先会调用 `make_minibatch_iterator` 来创建这样一个数据加载器。


```python
# verl/workers/actor/megatron_actor.py -> MegatronPPOActor.make_minibatch_iterator
def make_minibatch_iterator(self, data: DataProto) -> Iterator:
    # ...
    for _ in range(self.config.ppo_epochs):
        # 1. 在每个 epoch 开始时，重新生成一个随机的排列
        indices = torch.randperm(batch_size, device=get_device_name())
        for start in range(0, batch_size, self.config.ppo_mini_batch_size):
            # 2. 根据随机排列，切分出 mini-batch 的索引
            end = start + self.config.ppo_mini_batch_size
            batch_indices = indices[start:end]
            # 3. 从完整的 DataProto 中，根据索引切片出 mini-batch 数据
            yield data[batch_indices]
```

这个迭代器通过 `torch.randperm` 在每个 epoch 开始时都打乱数据顺序，然后按 `ppo_mini_batch_size` 切分数据，确保了后续训练的随机性。

这里的 `ppo_mini_batch_size` 构成了 veRL 数据处理层级中的一环：它定义了**一次模型参数更新**所用的数据量。而在它之上，还有一个更大的 `data.train_batch_size` ，负责定义**一次经验收集采样的总样本数**；在它之下，还有一个更小的 `ppo_micro_batch_size_per_gpu`，用于在显存不足时，将 mini-batch 进一步拆分以实现梯度累积。

### 2.2. 定义优化目标

对于每一个 mini-batch，我们都需要计算一个损失，来指导模型应该朝哪个方向优化。在 `update_policy` 方法中，将执行具体的损失计算逻辑的逻辑委托给了 Megatron 引擎，而 优化目标 `loss_func` 的逻辑则在 `forward_backward_batch` 中定义：

```python
# verl/workers/actor/megatron_actor.py -> MegatronPPOActor.forward_backward_batch
def forward_backward_batch(self, data: DataProto, ...):
    # ... (数据预处理，切分 micro-batch)

    # 1. 从 Megatron-Core 获取“全权代理”函数
    forward_backward_func = get_forward_backward_func()

    # 2. 定义核心的损失计算逻辑 (这才是真正的 loss_func)
    def loss_func(output, data, meta_info):
        # 从模型输出中获取新的 log_prob
        log_prob = output["log_probs"][:, -response_length - 1 : -1].contiguous()

        # 从输入数据中获取旧的 log_prob 和 advantages
        old_log_prob = data["old_log_probs"]
        advantages = data["advantages"]
        response_mask = data["attention_mask"][..., -response_length:].to(bool)

        # a. 计算策略损失 (Policy Loss) - 调用 core_algos 中的辅助函数
        pg_loss, _, _, _ = core_algos.compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            # ... 其他配置
        )
        policy_loss = pg_loss

        # b. 计算熵损失 (Entropy Loss)
        if calculate_entropy:
            entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
            entropy_loss = core_algos.agg_loss(loss_mat=entropy, loss_mask=response_mask, ...)
            policy_loss = policy_loss - self.config.entropy_coeff * entropy_loss

        # c. 计算 KL 损失 (KL Loss)
        if self.config.use_kl_loss:
            ref_log_prob = data["ref_log_prob"]
            kld = core_algos.kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, ...)
            kl_loss = core_algos.agg_loss(loss_mat=kld, loss_mask=response_mask, ...)
            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

        return policy_loss, { ... metrics ... }

    # 3. 定义单步的前向传播逻辑
    def forward_step(batch_iter, model):
        # ... (准备模型输入)
        output = forward_fn(model, ...) # 调用 Megatron 模型的前向传播
        # 将损失函数与当前批次的数据绑定后返回
        return output, partial(loss_func, data=batch, meta_info={...})

    # 4. 将 forward_step 和数据，委托给 Megatron 的调度器执行
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=batch_generator,
        model=self.actor_module,
        # ...
    )
    # ...
    return losses_reduced
```

这个函数封装了 PPO 算法的精髓：它将核心的 策略损失、用于鼓励探索的 熵损失 以及用于稳定训练的 KL 损失 组合在一起，形成一个最终的、用于反向传播的标量 loss。

值得注意的是，计算策略损失时的一个关键输入是 `old_log_probs`。这并非直接来自 Rollout 阶段，而是由 RayPPOTrainer 在训练前通过调用 `actor_rollout_wg.compute_log_prob` **重新计算**得来的。

这样做是为了矫正 PPO 在多轮 mini-batch 更新中产生的“策略漂移”问题。由于模型在处理完第一个 mini-batch 后权重就已改变，后续 mini-batch 实际上是在进行 Off-Policy 学习。因此，必须在训练循环开始前，获取一个当前策略下精准的 `log_prob` 作为基准，才能通过重要性采样正确地计算损失，保证算法的稳定性。

### 2.3. 将优化目标委托给分布式引擎

到目前为止，我们还只是在用 PyTorch 定义数据处理和损失计算，这些都是单机逻辑。veRL 通过 `forward_backward_func` 实现了一种“控制反转”，将上层逻辑“委托”给了底层的 Megatron 引擎。

```python
# verl/workers/actor/megatron_actor.py -> MegatronPPOActor.update_policy
def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
    # ...
    for data in dataloader:
        # ... (省略了 zero_grad 等准备工作)

        # 调用一个封装了所有分布式计算的函数
        metric_micro_batch = self.forward_backward_batch(
            data,
            calculate_entropy=True,
            # ... 其他参数
        )

        # ... (执行 optimizer.step)
    # ...
```

这个 `forward_backward_func` 是 Megatron 提供的一个高阶函数，它接管了所有的分布式计算细节。我们只需要提供一个损失函数，Megatron 就会负责把它分发到所有的 GPU 上，进行并行计算和梯度同步。

**这是一种极致的解耦，它让做 RL 的人可以专注于算法本身，而做分布式的人则可以专注于优化底层的执行效率，两者互不干扰，但又能完美协作。**

## 3. 生成逻辑剖析：generate_sequences 方法

当 ActorRolloutRefWorker 需要生成经验数据时，它会切换到 “生成” 模式。与“训练态”将计算委托给 Megatron-Core 类似，“生成态”的核心思想也是 **委托**，它将所有与高性能推理相关的复杂工作，全部委托给后端推理引擎，如 SGLang。

在这个过程中，veRL 扮演了一个 “智能前端” 的角色，负责数据的预处理、任务的提交以及结果的后处理。整个流程可以分解为三个主要步骤。

### 3.1. 输入预处理：为推理引擎准备“干净”的输入

从 RayPPOTrainer 传递过来的 DataProto 对象，其 `input_ids` 通常是经过左填充（left-padded）以对齐成一个矩形 Tensor 的。然而，推理引擎为了效率，需要的是一个不包含任何 padding 的、紧凑的 token 序列。

SGLangRollout 首先通过 `_pre_process_inputs` 等逻辑，对输入进行预处理：

```python
# verl/workers/rollout/sglang_rollout/sglang_rollout.py -> _pre_process_inputs
def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> torch.Tensor:
    # 找到第一个非 padding token 的位置
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    # 从该位置切片，移除左边的所有 padding
    return prompt_token_ids[non_pad_index:]

# 在 _batch_level_generate_sequences 中被调用
# ...
non_tensor_batch["raw_prompt_ids"] = np.array(
    [_pre_process_inputs(self.pad_token_id, idx[i]).tolist() for i in range(batch_size)],
    dtype=object,
)
idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
```

这一步的核心工作，就是将一个 batch 的、带有 padding 的 `input_ids` Tensor，转换为一个 Python list，其中每个元素都是一个去除了左边 padding 的、纯粹的 prompt token 列表。这为后续提交给 SGLang 准备了干净的输入。

### 3.2. 异步委托：向 SGLang 提交生成任务

准备好输入后，veRL 就会调用 SGLang 引擎的 `async_generate` 方法来执行推理。这是一个异步操作，体现了“前端+后端服务”的设计模式。

```python
# verl/workers/rollout/sglang_rollout/sglang_rollout.py -> _batch_level_generate_sequences
# ...
# 1. 准备采样参数
request_sampling_params = self.sampling_params.copy()
# ... (根据 do_sample 等条件进行修改)

if self._tp_rank == 0:
    # 2. 获取异步事件循环
    loop = asyncio.get_event_loop()
    # 3. 异步调用 SGLang 引擎的生成方法
    output = loop.run_until_complete(
        self._engine.async_generate(
            sampling_params=request_sampling_params,
            return_logprob=True,
            input_ids=idx_list, # 传入预处理好的输入
            # ...
        )
    )
else:
    output = None
```

通过 asyncio 事件循环，veRL 向 SGLang 引擎提交生成任务。这种异步模式允许 veRL 在等待推理结果的同时，理论上可以去处理其他任务（尽管在当前实现中是阻塞等待的），这是构建高吞吐量服务的基础。

**为什么采用“TP Rank 0 主导”的设计？**

这是一种常见且高效的分布式设计模式，其核心原因在于 **避免任务的重复提交和结果的冲突**。对于一个生成请求，我们只需要一个进程来处理它。如果 TP 组内的所有进程（rank 0, 1, 2, 3）都向 SGLang 提交这个任务，SGLang 就会收到 4 个一模一样的请求，这是完全没有必要的，并且会造成混乱。TP 本质上是几个 GPU 一起工作生成一份结果。

### 3.3. 结果同步与后处理

当 tp rank 0 的进程拿到 SGLang 的推理结果后，需要将其同步给组内的其他进程，并转换回 veRL 所需的、带有 padding 的 Tensor 格式。

```python
# verl/workers/rollout/sglang_rollout/sglang_rollout.py -> _batch_level_generate_sequences
# ...
# 1. 将 rank 0 的结果广播给所有 TP 组内的其他 rank
[output] = broadcast_pyobj(
    data=[output],
    # ...
)

# 2. 对结果进行后处理
out = _post_process_outputs(self.processing_class, output)
response = out[0].to(idx.device)
# ...
# 3. 将可变长度的 response list 重新填充为固定长度的 Tensor
if response.shape[1] < self.config.response_length:
    response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
# ...
```

这里通过自定义的 `broadcast_pyobj` 工具，将 Python 对象形式的复杂结果，高效地从主进程广播给所有其他进程，确保数据的一致性。

## 4. 训练与生成的引擎管理

我们已经分别剖析了 ActorRolloutRefWorker 的“训练态”和“生成态”，这里面省略了一个非常重要的环节：

**训练使用的是 Megatron-Core 的分布式模型（actor_module），而生成则依赖于 SGLang 的推理引擎。这两套系统是如何共享和同步模型权重的？**

答案，就藏在 veRL 精心设计的 `ShardingManager` 和它提供的 上下文管理器（`with` 作用域） 之中。

### 4.1. 引擎的初始化与持有

首先，在 ActorRolloutRefWorker 的 `init_model` 方法中，我们可以看到它确实初始化并持有了两个独立的后端实例，连接这两个独立引擎的桥梁，是一个名为 ShardingManager 的关键组件。

BaseShardingManager 定义了 `__enter__` 和 `__exit__` 方法，这表明它被设计成一个标准的 Python 上下文管理器。在 MegatronSGLangShardingManager 中，这两个方法被重写，并直接调用了 `wake_up` 和 `sleep`。

```python
# verl/workers/sharding_manager/megatron_sglang.py
class MegatronSGLangShardingManager(BaseShardingManager):
    # ...
    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            loop = asyncio.get_event_loop()
            # 进入 with 代码块时，调用 wake_up
            loop.run_until_complete(self.wake_up())

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        # 退出 with 代码块时，调用 sleep
        loop.run_until_complete(self.sleep())
```

### 4.2. 状态切换的核心逻辑

`wake_up` 和 `sleep` 才是执行具体状态切换的地方。

当 with 语句块开始，`wake_up()` 被调用时，它的核心任务是：**将 `actor_module` 最新的权重，“翻译”并“注入”到 SGLang 推理引擎中**。这个过程分为两步：

**第一步：权重的“提取”与“翻译”**

由于 Megatron-Core 和 SGLang 对模型权重的存储与切分方式（即 Sharding 格式）可能完全不同，直接进行内存拷贝是行不通的。因此，需要一个“翻译官”来完成格式转换。这个角色，就由 `per_tensor_generator` 扮演。

```python
# verl/workers/sharding_manager/megatron_sglang.py -> wake_up
async def wake_up(self):
    # ...
    # 1. 从 actor_module 中提取、转换并逐个生成权重
    per_tensor_param = per_tensor_generator(
        self.actor_module,
        self.model_config,
        self.weight_converter,
        self.transformer_config,
        self.layer_name_mapping,
    )
    # 2. 将转换后的权重更新到推理引擎
    await self.update_weights(per_tensor_param)
    # ...
```

在张量并行场景下，权重会被分到多个 GPU 上。`per_tensor_generator` 会先将权重通信回完整的状态，然后再将其转换为 SGLang 所需的格式。转换的时候。它的 格式 可能仍然与 SGLang 的要求不符。比如 Megatron 中会将 Q，K，V 三个张量合并成一个，而 SGLang 需要它们分开。`weight_converter` 就是负责这种转换的工具。

`per_tensor_generator` 将 “组装” 和 “翻译” 封装在一个 Python 生成器中。它处理完一个权重（例如 wte），就立刻通过 `yield` 将其 `(name, tensor)` 对产出，然后 `update_weights` 方法就立刻消费它，但它并**不直接通过网络传输庞大的张量数据**。这正是 veRL “零拷贝”魔法的核心所在。

在 `update_weights` 内部，每个张量都会经过 `MultiprocessingSerializer.serialize(tensor)` 的处理。这是 sglang 提供的一个高效序列化工具，它内部封装了一个 ForkingPickler，会将 CUDA 张量序列化为一个轻量级的 **“句柄元组” (Handle Tuple)**。 这个元组不包含任何实际的权重数据，而是包含了在另一个进程中“重建”此张量所需的一切元信息：**类型、形状、步长，以及最关键的 CUDA IPC 句柄**。SGLang 引擎在接收端通过反序列化，利用这些 IPC 句柄直接创建出指向同一块物理显存的新张量对象。


veRL 的论文中提到了一种利用训练并行组（PP/DP）来优化权重同步的通信方案。然而，在当前的代码中，这套逻辑已被移除（具体见
  PR #1444 (https://github.com/volcengine/verl/pull/1444)）。

这是一个主动的架构优化，原因很简单：**将训练和推理的并行策略强行绑定，会牺牲灵活性**。 推理只应该关心自己：`generate_sequences` 的过程，只和推理引擎自身的并行配置有关，完全不应该去“迁就”训练引擎（Megatron）的并行方式。

但是我目前就不清楚该优化去掉了之后对性能的影响，论文中提到这个通信优化对性能有显著提升。也许在未来的版本中，veRL 会重新引入类似的优化，但会以更灵活的方式实现。

## 5. 总结

ActorRolloutRefWorker 的核心，是一种优雅的 **“委托模式”**。它自身专注于实现上层的 RL 算法逻辑，而将分布式训练和高性能推理这两件“重活”，分别委托给了 Megatron 和 SGLang 这两个高度优化的专业后端。

ShardingManager 在其中扮演了关键的“粘合剂”角色，通过 `with` 上下文管理器，无缝地完成了训练与推理之间的权重同步和状态切换。

这引发了我们更深层次的思考：**一个高性能框架的价值，究竟是“自己实现一切”，还是“更好地组织与调度”？** veRL 显然选择了后者。

后续的文章中，我们将继续深入 veRL 的 Worker 体系，看看 CriticWorker 和 RewardModelWorker 是如何被实现的。