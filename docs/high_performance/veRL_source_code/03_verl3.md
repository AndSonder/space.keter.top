# veRL 源码解析（三）：“静态融合”与“分离部署”

在当前的 RLHF 实践中，一个典型的训练流程通常涉及多个模型，如 Actor、Critic、Reward Model 和 Reference Policy。这些模型在 PPO 等算法的数据流图中，其计算存在时间上的先后依赖关系，并非时刻都在被使用。

这就带来了两个实际的工程挑战：

1. **GPU 利用率问题**：如果为每个模型角色分配独立的 GPU 资源，那么在数据流的任一时刻，都不可避免地存在部分 GPU 闲置，导致计算资源利用率低下。一个自然的问题是：**框架能否让多个在时间上互斥的计算任务，共享同一组物理硬件？**
2. **资源异构性问题**：参与 RLHF 的模型，其规模往往差异巨大，例如，使用一个 70B 的 Actor 模型，同时使用一个 7B 的 Critic 模型。在这种场景下，如果强行让它们共享资源，必然会导致小模型占用过多资源（浪费），或大模型无法获得足够资源（低效）。这就引出了第二个问题：**框架是否支持为不同规模的模型，分配和隔离不同规模的、专属的物理资源？**

这两个问题是所有分布式 RLHF 框架在设计资源调度时必须面对的核心。veRL 通过其设计的两种核心模式——“静态融合”与“分离部署”——给出了它的答案。本文将深入 veRL 的源码，剖析其如何实现这两种模式。

## 1. “静态融合”，GPU 的分时复用

在资源有限的场景下，最大化 GPU 利用率是所有 RLHF 框架面临的核心挑战。veRL 为此提供了一种极其巧妙的解决方案 —— **“静态融合”**。其核心思想是，在系统初始化时，就将多个逻辑上独立的 Worker（如 Actor, Critic, Reward Model）“融合”并部署到同一组物理 GPU。这使得这些角色虽然在不同的时间点活跃，却能共享同一块物理显存和计算单元，从而实现了“分时复用”。

这种“物理融合，逻辑解耦”的魔法，是通过 RayPPOTrainer、RayWorkerGroup 和一个名为 `create_colocated_worker_cls` 的关键工具函数协同完成的。

**第一步：在 `RayPPOTrainer` 中定义“如何融合”**

```python
# verl/trainer/ppo/ray_trainer.py -> init_workers
def init_workers(self):
    # ...
    # 1. self.resource_pool_to_cls 是一个字典，它的 key 是 ResourcePool 对象。
    #    在默认配置下，只有一个 "global_pool"。
    self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    # 2. 将所有 Worker 角色（如 actor_rollout_cls, critic_cls）的定义，
    #    都添加到同一个 resource_pool 对应的字典条目中。
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
    self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    if self.use_critic:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
        self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
    # ...
```

**第二步：`WorkerDict` 的“代理”模式与动态绑定**

在 RayPPOTrainer 中定义好需要融合的 Worker 后，veRL 并不会手动编写一个复杂的、包含所有功能的巨型 Worker 类。相反，它采用了一种极为精巧的“代理”模式，在运行时动态地将多个 Worker 的功能“注入”到一个统一的代理类中。

`create_colocated_worker_cls` 函数就是实现这一功能的核心：

```python
# verl/single_controller/ray/base.py -> create_colocated_worker_cls

def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    # ... (一些准备工作)

    # 1. 定义一个临时的代理类 WorkerDict
    class WorkerDict(worker_cls): # worker_cls 通常是 verl.single_controller.base.worker.Worker
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                # ...
                # 2. 在代理类内部，实例化每一个真正的 Worker
                #    注意：此时 Worker 的初始化被临时屏蔽，避免了不必要的资源加载
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    self.worker_dict[key] = user_defined_cls(
                        *init_args_dict[key].get("args", ()),
                        **init_args_dict[key].get("kwargs", {})
                    )

    # 3. 关键：通过 Monkey patch 将内部 Worker 的方法绑定到外层代理类上
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    # ...
    return remote_cls
```

函数首先定义了一个名为 WorkerDict 的内部类。在 WorkerDict 的 `__init__` 方法中，它会遍历传入的 `cls_dict`（包含了 Actor, Critic 等所有待融合的 Worker），**并在内部逐个实例化它们**。这些实例被存储在一个名为 `self.worker_dict` 的字典中。这就好比一个“总代理”手下管理着多个“真实员工”。

代码通过 `_bind_workers_method_to_parent` 函数，动态地将 Actor 的 `update_policy`、Critic 的 `compute_values` 等所有公开方法，都“复制”并“绑定”到了外层的 WorkerDict 类上。这样一来，当我们调用代理类 WorkerDict 的 update_policy 方法时，它会自动转发给内部真正的 Actor 实例去执行。

通过这种“组合优于继承”的代理模式，`create_colocated_worker_cls` 成功地将多个独立 Worker 的功能无缝地聚合到了一个单一的类上。这个最终由 `ray.remote(WorkerDict)` 包装成的 Ray Actor，从外部看，它就是一个万能的 Worker，能够响应所有角色的指令，完美地实现了“物理融合，逻辑解耦”的目标。

**第三步：`RayWorkerGroup` 完成最终部署**

现在我们已经有了融合多种功能的 `WorkerDict` 类，但把它部署到哪里呢？`RayWorkerGroup` 在执行部署前，必须先向 Ray 集群申请一块专属的、绑定的物理资源。这个“圈地”的任务，正是通过 `RayResourcePool` 来完成的。

`RayResourcePool` 的核心职责，就是通过 Ray 的 `PlacementGroup` 功能，在物理集群中 **预留** 出满足条件的计算资源。

这个“圈地”的过程，主要由 RayResourcePool 的 `get_placement_groups` 方法完成。

```python
# verl/single_controller/ray/base.py -> RayResourcePool
class RayResourcePool(ResourcePool):
    def __init__(
        self,
        process_on_nodes: Optional[list[int]] = None, # e.g., [8, 8] for 2 nodes, 8 GPUs each
        use_gpu: bool = True,
        max_colocate_count: int = 10, # 允许在一个 GPU 资源上融合多少个 Worker
        ...
    ):
        # ...

    def get_placement_groups(self, strategy="STRICT_PACK", ...):
        # ...
        # 1. 定义每个 GPU 资源包 (bundle) 的需求
        #    除了 1 个 GPU，还需要 CPU 资源来支持多个 Worker 实例的共存
        bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            bundle["GPU"] = 1

        # 2. 根据 process_on_nodes 配置，生成 placement group 的方案
        #    e.g., [[bundle]*8, [bundle]*8]
        pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

        # 3. 调用 Ray 的 placement_group API，向 Ray Scheduler 申请资源
        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=...)
            for idx, bundles in enumerate(pg_scheme)
        ]

        # 4. 阻塞等待，直到 Ray 确认资源已成功预留
        ray.get([pg.ready() for pg in pgs])

        self.pgs = pgs
        return pgs
```

RayWorkerGroup 是 veRL 中负责管理一组 Ray Actor 的核心组件。在初始化时，它会将 WorkerDict 这个融合类，精确地“安放”到 RayResourcePool 中预留的资源上。

```python
# verl/single_controller/ray/base.py -> RayWorkerGroup._init_with_resource_pool
class RayWorkerGroup(WorkerGroup):
    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, ...):
        # ...
        # 1. 从资源池获取预留好的 Placement Groups
        pgs = resource_pool.get_placement_groups(...)

        # ...
        # 2. 遍历每一个预留的资源节点 (Placement Group)
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            # ...
            # 3. 在每个节点的每个 GPU 资源包 (bundle) 上创建一个 Worker
            for local_rank in range(local_world_size):
                # ...
                # 4. 关键：将融合后的 Worker 类 (ray_cls_with_init.cls)
                #    通过 .options() 指定部署到当前的 placement_group 和 bundle
                worker = ray_cls_with_init.cls.options(
                    num_gpus=num_gpus,
                    placement_group=pg,
                    placement_group_bundle_index=local_rank,
                    # ... 其他选项
                ).remote(*ray_cls_with_init.args, **ray_cls_with_init.kwargs)

                self._workers.append(worker)
```

RayWorkerGroup 首先从 RayResourcePool 获取已经由 Ray Scheduler 确认并预留的 PlacementGroup 列表。它会遍历这些 PlacementGroup，对应到集群中的每一个物理节点。在循环的内部，它执行了最关键的一步。通过调用 `ray_cls_with_init.cls.options(...)`，向 Ray 发出指令。

`cls` 就是我们之前定义的 WorkerDict 类，`options` 明确指定这个 WorkerDict 实例必须创建在当前遍历到的 `placement_group` 的特定 bundle 上。这个 bundle 正是 RayResourcePool 定义的、包含一个 GPU 和若干 CPU 的资源包。

在不启用 Offload 的情况下，**“分时复用”指的是“计算单元”而非“显存”**。 当 Actor 和 Critic 的模型规模差异巨大时（例如，一个 70B 的 Actor 和一个 7B 的 Critic），将它们强行融合在同一组 GPU 上，可能会导致显存资源的浪费。

所以说如何进行资源配置也是一个很大的学问，我们就来看看 veRL 是如何处理这个问题的。

## 2. “分离部署”，应对异构需求的利器

我们在前文深入探讨了 veRL 的“静态融合”模式，它可以将多个 Worker 聚合到 GPU 上，实现了计算单元的高效“分时复用”。对于许多中小型模型，或多个角色共享相似模型结构的场景（如 Actor 和 Reference Police），这是一种简单、高效且能有效提升 GPU 利用率的策略。

然而，“静态融合”并非万能的灵丹妙药。它的核心思想是 **资源共享**，这也带来了它的内在局限性：**所有被融合的 `Worker` 必须共享同一个资源池的配置**。当不同角色的资源需求出现巨大差异时，这种“一刀切”的资源分配方式就会导致严重的性能瓶颈和资源浪费。

为了解决这种 **资源需求异构** 的核心矛盾，veRL 提供了更为强大和灵活的 **“分离部署”（Placement）** 模式。它允许我们将不同的 Worker 角色部署到各自独立的、规模和配置都可以完全不同的专属资源池中，从而实现真正的、精细化的资源管理。

下面让我们结合一个官方示例代码来看看 veRL 是如何实现这种分离部署的。

```python
# examples/split_placement/main_ppo_split.py -> main_task

# ...
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

role_worker_mapping = {
    Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    Role.Critic: ray.remote(CriticWorker),
}

# 1. 初始化两个独立的资源池 ID
actor_rollout_ref_pool_id = "actor_rollout_ref_pool"
critic_pool_id = "critic_pool"

# 2. 定义资源规格 (resource_pool_spec)，将集群资源切分
#    这里的逻辑是：如果节点数是偶数，则每个角色各占一半节点；
#    如果是单节点，则每个角色各占一半的 GPU。
if config.trainer.nnodes // 2 == 0 and config.trainer.n_gpus_per_node // 2 > 0:
    resource_pool_spec = {
        actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node // 2] * config.trainer.nnodes,
        critic_pool_id: [config.trainer.n_gpus_per_node // 2] * config.trainer.nnodes,
    }
else:
    resource_pool_spec = {
        actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node] * (config.trainer.nnodes // 2),
        critic_pool_id: [config.trainer.n_gpus_per_node] * (config.trainer.nnodes // 2),
    }
print(f"resource_pool_spec: {resource_pool_spec}")

# 3. 定义角色到资源池的映射 (mapping)
mapping = {
    Role.ActorRollout: actor_rollout_ref_pool_id,
    Role.Critic: critic_pool_id,
}

# 4. 将其他角色也映射到相应的资源池
#    RefPolicy (与 Actor 结构相同) -> actor_rollout_ref_pool
if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
    role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    mapping[Role.RefPolicy] = actor_rollout_ref_pool_id

#    RewardModel (与 Critic 类似，用于打分) -> critic_pool
if config.reward_model.enable:
    # ...
    role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
    mapping[Role.RewardModel] = critic_pool_id

# 5. 创建资源管理器
resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

# ... 后续代码使用此 resource_pool_manager 初始化 trainer
```

这个官方示例非常直观地展示了如何实现 “分离部署”，步骤如下：

1. 定义两个资源池 ID
2. 切分物理资源：划分集群资源到两个资源池中
3. 精准分配 Worker：给每个 Worker 分配对应的资源池 ID
4. 绑定相关组件，物以类聚：不仅是核心角色，其他配套模块也做了合理绑定，比如和 Actor 强相关的 `RefPolicy` 也放进 Actor 的资源池 s

## 3. 终极蓝图：Auto-Mapping 算法

我们已经看到了 veRL 提供的“融合”与“分离”两种手动的部署模式，但这自然引出了一个更高阶的问题：在复杂的场景下，我们如何知道何时该融合、何时该分离？对于一个分离部署的 8-GPU Actor，它的最优并行策略（TP/PP/DP 组合）又是什么？

veRL 论文中提出的 Auto-Mapping 算法，正是为这个目标设计的终极蓝图。但是很遗憾，截止我写文章的时候，veRL 并没有开源出来论文里面算法的代码。可能是因为这个工具需要依赖于大量的真实性能数据来优化成本模型，而这些数据可能是字节内部的专有资源。

但是我们可以详细看看论文里面的策略是怎么做的，这个就类似目前基本上各家都有的预训练 autotune 工具，目前应该是作为字节内部的脚本工具没有公开。

veRL 论文中提出的 Auto-Mapping 算法，是一个典型的、分层解决复杂优化问题的范例。它将一个巨大的、难以处理的“部署方案”搜索空间，巧妙地分解为了三个可以迭代求解的子问题： **放置（Placement）**、**资源分配（Allocation）** 和 **并行策略（Parallelism）**。

**第一层：枚举“放置”方案 (Placement)**

算法的第一步，是回答“谁跟谁在一起？”的问题。它会枚举出所有可能的“融合”方案。例如，对于一个包含 Actor, Critic, Reward Model (RM) 的标准 RLHF 流程，可能的放置方案就包括：

1. 完全分离: `{{Actor}, {Critic}, {RM}}`
2. 完全融合: `{{Actor, Critic, RM}}`
3. 部分融合: `{{Actor, RM}, {Critic}}，{{Actor, Critic}, {RM}}` 等等

**第二层：搜索“资源分配”方案 (Allocation)**

对于上一步生成的 **每一种** 放置方案，算法会接着为这个方案中的每一个“物理分组”分配具体的 GPU 数量。

例如，对于 `plm = {{Actor}, {Critic}}` 这个分离部署方案，在一个拥有 8 个 GPU 的集群上，算法会遍历所有可能的资源分配 A：

1. A = `{"Actor": 2, "Critic": 6}`
2. A = `{"Actor": 3, "Critic": 5}`

当然，这个遍历会受到模型最小显存需求的约束（例如，70B 的 Actor 至少需要 2 张卡才能启动）。

**第三层：搜索“并行策略”**

当一种“放置”方案和一种“资源分配”方案被确定后，算法就进入了最精细的优化层面：为每个模型（或融合后的模型组）找到在其被分配的 GPU 上的最优并行策略 (TP,
  PP, DP)。

在三层搜索的每一步，算法都需要一个“裁判”来评估当前方案的优劣。这个裁判就是论文中提到的 `simu()` 和 `d_cost()` 函数，即 成本模型 (Cost Model)。

这个成本模型的准确性，直接决定了整个 Auto-Mapping 算法的成败。而要构建一个准确的成本模型，就必须依赖于大量的真实性能数据。这恰恰解释了 veRL 框架为什么要内置一个如此强大的 DistProfiler——它正是为成本模型提供 “养料” 的关键工具。

未来如何自动搜索出来最优的放置方案，可能会是 RL 性能优化的一个重要方向。

## 4. 总结

veRL 提供的两种部署模式。第一种是 “静态融合”，它通过巧妙的代理机制，将多个逻辑 Worker 聚合到同一物理资源上，实现了计算单元的分时复用。

veRL 的真正实力，体现在处理复杂、异构工作负载的 **“分离部署”** 模式上。 我们看到，框架将拓扑定义的复杂度，优雅地抽象为 `resource_pool_spec` 和 `mapping`，用户只需修改配置，就能将不同规模、不同用途的 Worker 精确地部署到独立的 GPU 池中。

虽然目前 veRL 还没有开源自动放置算法的代码，但论文中提出的 Auto-Mapping 思路，已经为未来的 RLHF 框架提供了一个清晰的优化路径。






