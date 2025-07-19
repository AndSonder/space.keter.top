# veRL 源码解析（五）：剖析 Critic 与 Reward Model 的实现

## 1. 引言

在 RLHF 的训练流程中，为了让模型学会人类的偏好，我们需要对它生成的回复进行“评价”。但这种评价并非单一维度，而是包含了两个层面：**“即时偏好”** 和 **“长远价值”**。

1. **即时偏好**：这个回答本身有多好？它是否符合人类的期望？这个问题，由 `Reward Model` 来回答。对一个 `(prompt, response)` 对进行打分，给出一个直接的、标量的分数。这个分数，将作为 PPO 算法中 reward 信号的核心来源，直接告诉 Actor 当前这一步做得“对”或“不对”。
2. **长远价值**：在生成了这个回答后，对话进入了一个新的状态，这个状态对于达成最终的、长远的目标是否有利？这个问题，则由 `Critic` 来回答。它负责预测在当前状态 s 下，未来可能获得的总回报期望值，即价值 `V(s)`。

在 veRL 中，CriticWorker 和 RewardModelWorker 分别负责这两个角色。本篇文章中，我们将深入它们的源码，看看 veRL 实现他们的。

## 2. CriticWorker 的实现

Critic 的核心任务，简单来说，就是看 **“棋谱”学“棋局”**。它需要学会预测：在当前的对话“局面”（状态 s）下，最终能拿到多少“分数”（未来的总回报）。

虽然 Critic 身处复杂的 RL 循环之中，但它的学习方式，却和我们熟悉的监督学习没什么两样。它就是一个学生，需要一个“标准答案”来告诉它，它对棋局的判断到底准不准。这个过程，在 veRL 中被实现为一个清晰的 **价值回归任务**。

### 2.1. 模型结构：与 Actor 的同与不同

要学会评估棋局，Critic 首先需要一个“大脑”——也就是它的模型。这个模型的设计，遵循了“同源而不同功”的哲学。

Critic 模型的主体部分（Transformer 网络）与 Actor 模型是完全一样的。这很好理解：能学会“下棋”（生成回复）的那个大脑，也应该具备“看懂棋局”（评估价值）的能力。它们都源自同一个强大的预训练模型。

在 veRL 的模型构建流程中，Critic 和 Actor 都会进入 `BaseModelInitializer` 的 `initialize` 方法来构建：

```python
# verl/models/mcore/model_initializer.py -> BaseModelInitializer.initialize
def initialize(
    self,
    # ...
    value: bool = False,
    **extra_kwargs,
) -> GPTModel:
    # ...
    # 1. 首先，构建一个标准的、带有默认 LM Head 的 GPTModel
    model = GPTModel(
        config=self.tfconfig,
        # ...
        transformer_layer_spec=transformer_layer_spec,
        post_process=post_process, # post_process=True 意味着包含 output_layer
    )

    # 2. 如果 value 参数为 True，则执行“换头”
    if post_process and value:
        from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

        # 3. 创建一个新的、输出维度为 1 的线性层作为 Value Head
        model.output_layer = LinearForLastLayer(
            input_size=self.tfconfig.hidden_size,
            output_size=1, # <--- 输出维度为 1
            config=self.tfconfig
        )

    return model
```

虽然这里用的是 GPTModel，但是可以根据传入的 `transformer_layer_spec` 来构建不同的 Transformer 变体。veRL 中有一套注册表机制，当 CriticWorker 构建模型时，框架首先会从模型的 Hugging Face 配置中读取其 `architectures` 字段（例如 ["LlamaForCausalLM"]）。

通过 `verl/models/mcore/registry.py` 中的注册表机制，"LlamaForCausalLM" 这个名字会被映射到 DenseModel 这个初始化器类。DenseModel 正是 veRL 为 Llama、Qwen2 等稠密模型提供的。DenseModel 通过覆盖 `get_transformer_layer_spec` 方法，来指定 Llama 解码器层的具体实现：

```python
# verl/models/mcore/model_initializer.py
class DenseModel(BaseModelInitializer):
    def get_transformer_layer_spec(self):
        # 指定使用 RMSNorm 和 Transformer Engine 优化
        return get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True)
```

这样在进行 `initialize` 的时候就可以通过其返回的 `transformer_layer_spec` 来构建 Llama 的 Transformer 层。不过做个题外话，veRL 中这类简单的类继承实现多态的设计真的是信手拈来，各种合理的架构设计使得 veRL 的代码非常的好读。

好了，现在我们已经清楚了 Critic 的模型结构，它和 Actor 是同源的，只不过**在输出层上多了一个 Value Head，用于输出一个标量的价值预测**。

### 2.2. 价值预测函数

在 Critic 模型构建完成之后，它的首要任务就是对外提供价值预测服务。这个功能由 CriticWorker 的 `compute_values` 方法实现：

```python
# verl/workers/megatron_workers.py -> CriticWorker.compute_values
@register(...)
@DistProfiler.annotate(color="cyan")
def compute_values(self, data: DataProto):
    # ...
    # 调用内部 MegatronPPOCritic 实例的 compute_values
    values = self.critic.compute_values(data=data)
    output = DataProto.from_dict(tensors={"values": values})
    # ...
    return output
```

CriticWorker 作为一个封装层，将请求转发给了内部的 MegatronPPOCritic 实例。真正的计算逻辑在 `MegatronPPOCritic.compute_values` 中：

```python
# verl/workers/critic/megatron_critic.py -> MegatronPPOCritic.compute_values
@GPUMemoryLogger("megatron critic", logger=logger)
def compute_values(self, data: DataProto) -> DataProto:
    # ...
    with torch.no_grad():
        # 2. 调用通用的 forward_backward_batch 方法，但只执行前向传播
        output = self.forward_backward_batch(
            data=data,
            forward_only=True,
            # ...
        )
        # 3. 从分布式计算结果中提取 'vpreds' (value predictions)
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            values = [o["vpreds"] for o in output["output"]]
            values = torch.cat(values, dim=0).to(torch.float32)
            # ... (处理分布式结果的拼接和排序)
        else:
            values = torch.empty_like(...)

    # 4. 在流水并行维度上广播结果，确保所有进程拿到一致的 values
    torch.distributed.broadcast(
        tensor=values,
        src=mpu.get_pipeline_model_parallel_last_rank(),
        group=mpu.get_pipeline_model_parallel_group(),
    )
    # ...
    return values
```


代码会从输出中提取这些 `vpreds（value predictions`，并将它们拼接起来。最后，通过一次 broadcast 操作，将最终的 values 张量从最后一个 PP Stage 发送给所有其他 Stage。

最终，这个 values 张量（形状通常为 `[batch_size, sequence_length]`）会被返回给 Driver 端的 RayPPOTrainer。它将作为计算优势函数 `A(s, a)` 所需的 `V(s)` 和 `V(s')` 的基础。

### 2.3. 更新 Critic 的权重

Critic 的学习目标，是让自己的预测越来越准。在 PPO 算法中，这个“准确”的参照物，或者说“标准答案”，就是 `returns` (累积回报)。returns 是 RayPPOTrainer 在 Driver 端，根据 Reward Model 给出的即时奖励 r，通过 GAE 算法计算出的、对未来奖励的估计。

`update_critic` 方法的核心，就是用 returns 作为监督信号（label），来指导 Critic 模型的学习。

CriticWorker 接收到 `update_critic` 的远程调用后，会将其转发给内部的 MegatronPPOCritic。`MegatronPPOCritic.update_critic` 方法会：

1. 创建一个 PPO mini-batch 迭代器
2. 循环遍历每个 mini-batch，调用 `forward_backward_batch` 方法，启动一次完整的“前向-计算损失-反向-更新”的训练流程

`forward_backward_batch` 的实现，与我们之前在 Actor 中看到的类似：

```python
# verl/workers/critic/megatron_critic.py -> MegatronPPOCritic.forward_backward_batch
def forward_backward_batch(self, data: DataProto, forward_only=False, ...):
    # ... (数据预处理，切分 micro-batch)

    # 1. 从 Megatron-Core 获取“全权代理”函数
    forward_backward_func = get_forward_backward_func()

    # 2. 定义核心的损失计算逻辑
    def loss_func(output, data, meta_info):
        # 如果是纯前向，直接返回
        if forward_only:
            return torch.tensor(1.0, device=output.device), {"vpreds": output}

        # 从模型输出中获取价值预测值
        vpreds = output
        # 从输入数据中获取“标准答案”
        returns = data["returns"]

        # 调用核心算法，计算 value loss (核心是 MSE)
        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
            vpreds=vpreds,
            returns=returns,
            # ...
        )
        stats = {"critic/vf_loss": vf_loss.detach().item(), ...}
        return vf_loss, stats

    # 3. 定义单步的前向传播逻辑
    def forward_step(batch_iter, model):
        batch = next(batch_iter)
        # ... (准备 input_ids, attention_mask 等)

        # 调用 Megatron 模型的前向传播
        output = forward_fn(model, input_ids, ..., value_model=True)

        # 将损失函数与当前批次的数据绑定后返回
        return output, partial(loss_func, data=batch, meta_info={})

    # 4. 将 forward_step 和数据，委托给 Megatron 的调度器执行
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=batch_generator,
        model=self.critic_module,
        num_microbatches=n_micro_batch,
        forward_only=forward_only,
    )
    # ...
    return losses_reduced
```

通过这种方式，veRL 将 Critic 的训练，清晰地实现为一个标准的、以 returns 为监督信号的价值回归任务，并将其无缝地委托给了 Megatron 的分布式执行引擎。

## 3. RewardModelWorker 的实现


与 CriticWorker 预测长远价值不同，RewardModelWorker 的职责更为直接：

**为当前的 `(prompt, response)` 对给出一个即时的偏好分数。**

RewardModelWorker 同样是一个封装层，它接收请求后，会调用内部持有的 MegatronRewardModel 实例的 `compute_reward` 方法来执行核心的打分逻辑。

`compute_reward` 的实现与 Critic 的 `compute_values` 非常相似，都是一次无梯度的纯前向传播：

```python
def compute_reward(...):
    # ...
    # token_level_rewards 的形状是 [batch_size, sequence_length]
    token_level_rewards = logits

    # 1. 找到每个序列中，最后一个有效 token 的位置索引
    ends = attention_mask.cumsum(dim=-1).argmax(dim=-1).view(-1, 1)

    # 2. 使用 gather 操作，根据索引提取出每个序列的最后一个有效 token 的分数
    rewards = torch.gather(token_level_rewards, dim=1, index=ends) # 形状变为 [batch_size, 1]

    # 3. 创建一个只在最后一个 token 位置为 1 的 eos_mask
    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
    eos_mask = torch.zeros_like(attention_mask)
    eos_mask[torch.arange(batch_size), eos_mask_idx] = 1.0

    # 4. 将这个“序列最终得分”，广播回 response 的所有 token 位置上
    #    (为了与后续 KL 惩罚等 token-level 的计算保持形状一致)
    token_level_rewards = rewards.expand_as(attention_mask) * eos_mask
    token_level_rewards = token_level_rewards[:, -response_length:]

    batch = TensorDict({"rm_scores": token_level_rewards}, ...)
    return DataProto(batch=batch)
```

这里的 logits，实际上就是 Reward Model 的 Value Head 输出的标量分数。因为 Reward Model 的结构与 Critic 模型几乎完全相同（共享骨干网络 + 一个输出维度为 1 的 Value Head），所以它们的前向传播流程也如出一辙。

在拿到模型为每个 token 都计算出的分数 logits (形状为 `[batch_size, sequence_length]`) 之后，Reward Model 需要决定到底用哪个分数作为最终的“答案”。

在 RLHF 的普遍实践中，通常使用 **整个序列最后一个有效 token 对应的 hidden_state 所计算出的分数**。

```python
# verl/workers/reward_model/megatron/reward_model.py -> MegatronRewardModel.compute_reward
def compute_reward(...):
    # ...
    # token_level_rewards 的形状是 [batch_size, sequence_length]
    token_level_rewards = logits

    # 1. 找到每个序列中，最后一个有效 token 的位置索引
    # attention_mask.cumsum(...).argmax(...) 是一个巧妙的技巧，可以高效地找到最后一个 '1' 的位置
    ends = attention_mask.cumsum(dim=-1).argmax(dim=-1).view(-1, 1)

    # 2. 使用 gather 操作，根据索引提取出每个序列的最后一个有效 token 的分数
    rewards = torch.gather(token_level_rewards, dim=1, index=ends) # 形状变为 [batch_size, 1]

    # 3. 将这个“序列最终得分”，广播回 response 的所有 token 位置上
    #    (为了与后续 KL 惩罚等 token-level 的计算保持形状一致)
    eos_mask = torch.zeros_like(attention_mask)
    eos_mask[torch.arange(batch_size), eos_mask_idx] = 1.0
    token_level_rewards = rewards.expand(...) * eos_mask
    token_level_rewards = token_level_rewards[:, -response_length:]

    batch = TensorDict({"rm_scores": token_level_rewards}, ...)
    return DataProto(batch=batch)
```

为什么“最后一个 token”能代表整段输出的 reward 呢？不是因为“最后那个 token 特别重要”，而是因为 Reward Model 接收的是**整段 prompt + response**，最后输出一个整体的分数。这个分数实际上是对整段 response 的评价，只是技术上它被建模成了一个 `token-level score`。但这些 scores **不是 per-token 的“局部 reward”，而是共享上下文的“累计打分轨迹”**。

最后一个 token 的 score 其实就是模型在看完整个 response 后输出的 summary judgment，它不是仅关于最后一个 token 的，而是聚合了上下文语义。在实现的时候直接取最后一个 token 的 score 简单有效和人类偏好数据训练方式一致（监督最后一步）。

不过我们上面得到的这个分数，并不是 PPO 算法最终使用的 reward。

在原始分数和最终奖励之间，还有一个 `reward_fn`。这是一个在 Driver 端的 `RayPPOTrainer` 中执行的函数，它负责将 Worker 提供的基础分数转换为 PPO 所需的 reward 信号。

在 RayPPOTrainer 的 fit 方法中，我们可以看到这个清晰的调用流程：

```python
# verl/trainer/ppo/ray_trainer.py -> RayPPOTrainer.fit
# ...
# 1. Worker 负责打分
if self.use_rm:
    reward_tensor = self.rm_wg.compute_rm_score(new_batch)
    new_batch = new_batch.union(reward_tensor) # 将 rm_scores 并入 DataProto

# 2. Driver 负责计算最终奖励
#    self.reward_fn 是一个可配置的函数，指向了 verl/trainer/ppo/reward.py 中的逻辑
reward_result = self.reward_fn(new_batch, return_dict=True)
reward_tensor = reward_result["reward_tensor"]
# ...
new_batch.batch["token_level_scores"] = reward_tensor
# ...
```

做算法的人都知道，训练 RLHF 模型时，reward 策略经常需要调整。今天你可能想试试 plain reward，明天就想加个 KL 惩罚，后天又想搞点 token-level 不确定性进去。如果每次都得去改 PPOTrainer 里的逻辑，不仅麻烦，而且容易把一堆细节耦死，复用和调试都很痛苦。

veRL 里的 `reward_fn` 就是为了解决这个问题而设计的。它提供了一个清晰的扩展点，让各种奖励策略都可以通过插件化的方式接入，而不用动底层的训练框架。

在 `verl/trainer/ppo/reward.py` 里，`compute_reward` 和 `load_reward_manager` 是整个体系的核心。`load_reward_manager` 会根据配置加载对应的 RewardManager（如 NaiveRewardManager, DAPORewardManager 等）。

这种设计带来的最大好处是：算法逻辑和训练流程彻底解耦了。做研究的时候，只需要改一个 `RewardManager`，就能快速迭代不同的 reward 策略，底层 Worker 和 Trainer 代码全都不用碰。

## 4. 总结

从对 CriticWorker 和 RewardModelWorker 的分析可以看出，veRL 把它们明确地定位成两个职责单一的“打分服务”。Critic 通过一次标准的价值回归，学会评估未来的长期回报；而 Reward Model 则通过一次前向传播，输出对当前输出的偏好打分。

这样的设计体现出 veRL 对 职责分离 原则的深刻把握。Worker 层被设计得尽可能“纯粹”且“无状态”，只负责执行那些紧贴模型和硬件、可并行化的计算任务——比如打分、预测等。而所有涉及算法逻辑的部分，比如 advantage 计算、奖励塑造、PPO 更新流程等，都被整体上提到了 RayPPOTrainer 这一层。