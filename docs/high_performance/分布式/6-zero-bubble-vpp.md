# ZB-VPP 编排代码解读

在上一篇博客中，我们介绍了 Zero Bubble VPP 的编排策略，本篇我们将深入代码，解读 Zero Bubble VPP 的编排实现。其中核心的代码都在 [v_schedule.py](https://github.com/AndSonder/zero-bubble-pipeline-parallelism/blob/2efcc0951fb16155cd88d3e7ed69305d1c22962d/megatron/core/pipeline_parallel/v_schedule.py#L46-L516) 中。但是官方的实现，代码实在是有些难懂，给的超链接里面包含了我加的详细注释，希望能帮助大家理解。

![picture 0](images/ee591cd8cf1acc81e97feb94023f2b73444e963c98ac9f0479998c769ab829e8.png)  

编排的核心代码包含俩个主要函数，一个是 `try_v_schedule`，一个是 `get_v_schedule`。`try_v_schedule` 函数是 Zero Bubble VPP 的核心编排函数，它会尝试生成一个调度计划，`get_v_schedule` 函数则是调用 `try_v_schedule` 函数，尝试生成一个调度计划，并且会打印一些调度的统计信息。

## 1. try_v_schedule 函数

首先我们来看 `try_v_schedule` 函数，这个函数是 Zero Bubble VPP 的核心编排函数，它会尝试生成一个调度计划。

也就是生成一个类似图中的调度计划，其中包含了前向传播（F）、反向传播（B）和权重更新（W）任务，以及通信操作。这个函数的输入参数包括 `fill_f` 和 `fill_b`，分别表示是否填充前向传播和反向传播的泡沫。`approved_bubble` 是一个数组，表示每个stage的最大泡沫时间。

try_v_schedule 是类 PipelineGraph 的一个方法，这个类包含一些比较重要的参数：n_stage（stage数量）、n_micro（微批处理数量）、fbw_cost（前向、反向和权重更新任务的成本）、c_cost（通信成本）和 max_mem（最大内存）。zp-vpp 默认 fbw_cost 是 [1000, 1000,1000]，c_cost 是 1。f_mem, b_mem, w_mem 分别是前向、反向和权重更新任务的内存占用。这三个参数是根据模型的 hidden_size, num_attention_heads 和 seq_length 估算出来的。

下面我们来具体看一下 `try_v_schedule` 函数的实现。 首先 `try_v_schedule` 函数会初始化一些变量，包括计数器 count、结束时间 end_time、当前时间 cur_time、内存 mem、stage泡沫时间 stage_bubble、待处理的 W 任务队列 pending_w、调度方案 schedule 和输出字符串 stage_str。

```python
def try_v_schedule(self, fill_f=True, fill_b=True, approved_bubble=None):
    # 初始化计数器count,对于每个stage都有6个计数器(对应F/B/W,每个有两个chunk)
    count = []
    for i in range(self.n_stage):
        count.append([0] * 6)

    # 初始化结束时间数组end_time,节点编号从0到n_node-1
    end_time = [-1] * self.n_node
    # 初始化当前时间cur_time为全0
    cur_time = [0] * self.n_stage
    # 初始化内存占用mem为全0
    mem = [0] * self.n_stage
    # 初始化每个stage的泡沫时间为全0
    stage_bubble = [0] * self.n_stage
    # 初始化待处理的W任务队列pending_w,每个stage一个
    pending_w = [deque() for _ in range(self.n_stage)]
    # 初始化调度方案schedule为n_stage个空列表
    schedule = [[] for _ in range(self.n_stage)]
    # 生成n_stage个用于输出的前缀字符串
    stage_str = ["    " * i for i in range(self.n_stage)]

    # 如果approved_bubble为None,则初始化为n_stage个-1
    if approved_bubble is None:
        approved_bubble = [-1] * self.n_stage
    # 计算approved_bubble中的最大值
    max_approved_bubble = max(approved_bubble)
```

接着 `try_v_schedule` 函数定义了一个内部函数 `get_max_stage_bubble`，用于获取最大stage泡沫时间。

```python
    # 定义获取最大stage泡沫时间的函数
    def get_max_stage_bubble(stage=-1):
        max_stage_bubble = 0
        for bb in stage_bubble:
            max_stage_bubble = max(max_stage_bubble, bb)
        # 如果给定了stage,则还需要考虑该stage的approved_bubble
        if stage >= 0:
            max_stage_bubble = max(
                max_stage_bubble, max_approved_bubble - approved_bubble[stage])
        return max_stage_bubble
```

为了方便的加入不同类型的任务，`try_v_schedule` 函数定义了一个内部函数 `put`，这个函数用于插入 F/B/W 任务。是一个比较核心的函数，下面我们来仔细看一下。

首先是函数的输入参数，`cat` 表示任务类型，0 为 F，1 为 B，2 为 W；`chunk` 表示任务块，0 为 chunk 0，1 为 chunk 1；`stage` 表示stage编号；`assert_cnt` 表示是否检查计数器。

```python
    # 定义插入F/B/W任务的函数
    def put(cat, chunk, stage, assert_cnt=True):
        """
        @param cat: 任务类型,0为F,1为B,2为W
        @param chunk: 任务块,0为chunk 0,1为chunk 1
        @param stage: stage编号
        @param assert_cnt: 是否检查计数器
        """
        task_end_time = _no_bubble = cur_time[stage] + self.fbw_cost[cat]
        # Note: 为什么是 cat * 2 + chunk?
        # 0 -> F0, 1 -> F1, 2 -> B0, 3 -> B1, 4 -> W0, 5 -> W1
        # 默认 vpp degree 是 2，这里看起来需要后续需要修改为 cat * vpp_degree + chunk
        _cnt = count[stage][cat * 2 + chunk]
        # assert _cnt < self.n_micro
        if _cnt >= self.n_micro:
            if not assert_cnt:
                stage_str[stage] += "    "
                cur_time[stage] = task_end_time  # TODO
                return
        assert mem[stage] + self.fbw_mem[cat] <= self.max_mem
        # 更新输出字符串, FfBbWw 分别代表 F/B/W 任务,后面的数字代表任务编号
        stage_str[stage] += "FfBbWw"[cat * 2 + chunk] + \
            str(_cnt + 1) + " " * (3 - len(str(_cnt + 1)))
```

count 是一个二维数组，用于记录每个stage的 F/B/W 任务的数量。end_time 是一个一维数组，用于记录每个任务的结束时间。cur_time 是一个一维数组，用于记录每个stage的当前时间。mem 是一个一维数组，用于记录每个stage的内存占用。stage_bubble 是一个一维数组，用于记录每个stage的泡沫时间。pending_w 是一个二维数组，用于记录每个stage待处理的 W 任务。schedule 是一个二维数组，用于记录调度方案。stage_str 是一个一维数组，用于记录输出字符串。

:::note

megatron 在做编排的时候是多卡的视角，对所有卡的任务进行编排。

:::

这里需要注意一个问题就是为什么任务的索引是 `cat * 2 + chunk`，这是因为 Zero Bubble VPP 中每个任务有俩个 chunk，所以需要乘以 2。如果后续需要适配 vpp_degree > 2 的逻辑，需要修改为 `cat * vpp_degree + chunk`。

下一步就是更新当前任务的结束时间，这里需要考虑依赖任务是否完成。如果任务不是前向传播的第一个 chunk（即不是 F0_1），则需要检查依赖任务是否完成。计算依赖任务的标识，如果是前向或反向传播任务，确认其依赖的前一个任务已完成。对于权重更新任务，确认其依赖的反向传播任务已完成。

```python
        if cat > 0 or chunk > 0:
            # 如果任务不是前向传播的第一个chunk（即不是F0_1），则需要检查依赖任务是否完成
            last_id = cat * 2 + chunk - 1
            # 计算依赖任务的标识
            if cat < 2:
                # 如果是前向或反向传播任务，确认其依赖的前一个任务已完成
                assert end_time[self.get_id(
                    last_id // 2, last_id % 2, stage, _cnt)] >= 0
            else:
                # 对于权重更新任务，确认其依赖的反向传播任务已完成
                assert end_time[self.get_id(1, chunk, stage, _cnt)] >= 0
```

如果如果是前向或反向传播的第一个chunk，需要等待前一个stage相应的任务完成。

```python
        if chunk == 0 and cat < 2:
            # 如果是前向或反向传播的第一个chunk，需要等待前一个stage相应的任务完成
            if stage > 0:
                # 获取前一个stage的相应任务ID
                _fa_id = self.get_id(cat, chunk, stage - 1, _cnt)
                # 确保前一个stage的任务已完成
                assert end_time[_fa_id] >= 0, f"{cat}, {chunk}, {stage}, {_cnt}"
                # 更新当前任务的预计结束时间，考虑通信成本和任务本身的执行时间
                task_end_time = max(task_end_time, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
```

最后需要更新当前任务的结束时间，更新当前stage的时间，更新内存使用情况，将任务加入到当前stage的调度计划中，如果是反向传播任务，将对应的权重更新任务加入待处理队列，更新当前stage内指定类型和 chunk 的任务计数。

```python
        _id = self.get_id(cat, chunk, stage, _cnt)
        # 为当前任务生成唯一ID
        if count[stage][0] > 0:
            # 如果在当前stage已经有任务被安排，则计算stage内的空闲时间（泡沫）
            stage_bubble[stage] += task_end_time - _no_bubble
        # 更新当前任务的结束时间
        end_time[_id] = task_end_time
        # 更新当前stage的时间，以反映新任务的安排
        cur_time[stage] = task_end_time
        # 更新内存使用情况
        mem[stage] += self.fbw_mem[cat]
        # 将任务加入到当前stage的调度计划中
        schedule[stage].append((cat, chunk, _cnt))
        if cat == 1:
            # 如果是反向传播任务，将对应的权重更新任务加入待处理队列
            pending_w[stage].append((2, chunk, _cnt))
        # 更新当前stage内指定类型和chunk的任务计数
        count[stage][cat * 2 + chunk] += 1
```

可以看到，`put` 函数主要是用于插入 F/B/W 任务，插入了一个任务后，会更新当前任务的结束时间，更新当前stage的时间，更新内存使用情况方便后续的任务插入。


`put_w` 函数用于插入权重更新任务，主要就是调用了 `put` 函数，插入权重更新任务后，会更新当前任务的结束时间，更新当前stage的时间，更新内存使用情况，将任务加入到当前stage的调度计划中，更新当前stage内指定类型和 chunk 的任务计数。

```python
    def put_w(stage):
        assert len(pending_w[stage]) > 0
        _, chunk_, _ = pending_w[stage].popleft()
        put(2, chunk_, stage)
```

接下来就是具体的编排逻辑了，首先是插入前向传播任务，然后是插入反向传播任务，最后是插入权重更新任务。

```python
    # ------------------------------------------------
    # 插入F任务的chunk 0
    # ------------------------------------------------
    for i in range(self.n_stage):
        put(FORWARD, 0, i)
        
    # ------------------------------------------------
    # 从最后一个卡开始,插入F任务的chunk 1
    # 结合 VPP 的图来理解，形状是一个 V 形状
    # ------------------------------------------------
    for i in range(self.n_stage - 1, -1, -1):
        if i == self.n_stage - 1: # 最后一个卡直接插入F任务的chunk 1
            put(FORWARD, 1, i)  # 插入F任务的chunk 1
            continue
        tmp = end_time[self.get_id(0, 1, i + 1, 0)] + self.c_cost
        # 如果 i 卡内存 mem[i] 加上 F 任务的内存占用小于最大内存,且当前时间 cur_time[i] 加上 F 任务的时间小于 tmp,且 F 任务的 chunk 0 数量小于 n_micro
        # 则插入 F 任务的 chunk 0
        while mem[i] + self.fbw_mem[FORWARD] * (2 + i * 2) <= self.max_mem and cur_time[i] + self.fbw_cost[FORWARD] <= tmp and count[i][0] < self.n_micro:
            for j in range(i + 1):
                put(FORWARD, 0, j)  # 插入F任务的chunk 0
        put(FORWARD, 1, i)  # 插入F任务的chunk 1
```

首先是插入前向传播任务，首先插入 F 任务的 chunk 0，然后从最后一个卡开始，插入 F 任务的 chunk 1。

:::note

一定要结合图去看，要不很难懂这里的代码逻辑

:::

下面是插入第一个backward之前的 F 任务

```python
    # ------------------------------------------------
    # 插入第一个backward之前剩下的 F
    # 形成 V 形
    # ------------------------------------------------
    iter_chunk_ = 0
    end_tmp = 0
    for i in range(self.n_stage):
        if i == 0:
            end_tmp = cur_time[0] + self.fbw_cost[1]
            continue
        tmp = end_tmp + self.c_cost
        while count[i][0] + count[i][1] < count[i - 1][0] + count[i - 1][1] or count[i][1] <= count[i - 1][1] < self.n_micro:
            for j in range(self.n_stage - 1, i - 1, -1):
                if count[j][iter_chunk_] < self.n_micro:
                    put(FORWARD, iter_chunk_, j)
            iter_chunk_ = 1 - iter_chunk_
```

逐步插入B和W任务,尽量填充泡沫。为什么这里是 2 * self.n_micro 呢？因为每个stage有两个 chunk，每个 chunk 最多有 n_micro 个任务。这里后续如果需要适配 vpp_degree > 2 的逻辑，就会变成 vpp_degree * self.n_micro

```
    for _ in range(2 * self.n_micro):
        ...
```

接下来是用来编排稳定stage的编排逻辑，非常的绕，大概可以分为6步。

第一步是检查内存，如果不够就先处理 pending_w 队列。

```python
        # 1. 检查内存,如果不够就先处理 pending_w 队列
        for i in range(self.n_stage):
            while mem[i] + self.fbw_mem[BACKWARD] > self.max_mem:
                assert len(pending_w[i]) > 0
                put_w(i)
```

第二步是根据条件分别将每个stage插入 b0 或 b1 列表。

```python
        # Note(sonder): 这里默认也是 vpp_degree = 2，需要后续适配 vpp_degree > 2 的逻辑
        b0_ranks, b1_ranks = [], []
        # 2. 根据条件分别将每个stage插入b0或b1列表
        for i in range(self.n_stage):
            # 如果 B 任务的 chunk 1 数量大于等于 chunk 0 数量,则插入 b0_ranks
            if count[i][3] >= count[i][2]:
                b0_ranks.append(i)
            elif i == self.n_stage - 1: # 如果是最后一个卡,则插入 b1_ranks
                b1_ranks.append(i)
            else:
                fa_id = self.get_id(1, 1, i + 1, count[i][3])
                if end_time[fa_id] >= 0 or count[i][2] >= self.n_micro:
                    b1_ranks.append(i)
                else:
                    b0_ranks.append(i)
```

b0 和 b1 列表的插入顺序是有讲究的，先插入 b1_ranks 中的 B 任务，再插入 b0_ranks 中的 B 任务。

```python
        b_ranks = [] # B任务列表
        # Node(sonder): 为什么要先加入 b1_ranks 再加入 b0_ranks?
        # 因为 backward 依赖关系和 forward 是相反的，backward 的 chunk 0 依赖 chunk 1
        # 3. 先插入b1_ranks中的B任务
        # Note(sonder): 这里是倒序插入，再结合图看一下为啥是倒序插入
        for i in reversed(b1_ranks):
            b_ranks.append((i, 1)) # (stage编号, chunk编号)
        # 4. 再插入b0_ranks中的B任务
        for i in b0_ranks:
            b_ranks.append((i, 0)) # (stage编号, chunk编号)
```

这里有一个值得注意的地方，为什么要先加入 b1_ranks 再加入 b0_ranks？因为 backward 依赖关系和 forward 是相反的，backward 的 chunk 0 依赖 chunk 1。

接下来是插入 B 任务，尽量填充泡沫。

```python
        # 5. 插入B任务,尽量填充泡沫
        # Note(sonder): 单卡视角下，一次只会插入一个 B 任务 b0/b1
        for i, _chunk_ in b_ranks: 
            fa_id = -1
            if _chunk_ == 1 and i < self.n_stage - 1:
                fa_id = self.get_id(1, 1, i + 1, count[i][3])
            if _chunk_ == 0 and i > 0:
                fa_id = self.get_id(1, 0, i - 1, count[i][2])
            # 检查内存,如果不够就先处理pending_w队列
            while len(pending_w[i]) > 0 and fa_id >= 0 and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]:
                # 填充泡沫
                put_w(i)
            if len(pending_w[i]) > 0 and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]:
                # 如果泡沫时间大于0, 则尽量填充泡沫
                if _chunk_ == 1: # 如果是 chunk 1, 则尽量填充泡沫
                    put_w(i)
                elif fill_b: # 如果是 chunk 0, 则根据 fill_b 来决定是否填充泡沫
                    put_w(i)
            put(BACKWARD, _chunk_, i) # 插入B任务
```

需要注意的是，单卡视角下，一次只会插入一个 B 任务 b0/b1。这里就是对应图里面插入 B_0_0 和 B_0_1 等任务的逻辑。

第6步是插入剩下的F任务，尽量填充泡沫。

```python
        # 6. 插入F任务,尽量填充泡沫
        # Note(sonder): 单卡视角下，一次只会插入一个 F 任务
        for i in range(self.n_stage):
            # 该卡的 F1 都已经插入, 跳过
            if count[i][1] >= self.n_micro:
                continue
            put_item = None
            # 如果 F1 的数量大于等于 F0 的数量,则插入 F0
            if count[i][1] >= count[i][0]:
                put_item = 0
            # 如果是最后一个卡,则插入 F1
            elif i == self.n_stage - 1:
                put_item = 1
            else:
                if end_time[self.get_id(0, 1, i + 1, count[i][1])] >= 0:
                    put_item = 1
                elif count[i][0] < self.n_micro:
                    if i == 0:
                        put_item = 0
                    elif end_time[self.get_id(0, 0, i - 1, count[i][0])] >= 0:
                        put_item = 0
            if put_item is None:
                continue
            # 检查内存,如果不够就先处理pending_w队列
            while mem[i] + self.fbw_mem[FORWARD] > self.max_mem:
                assert len(pending_w[i]) > 0
                put_w(i)
            fa_id = -1
            if put_item == 0 and i > 0:
                fa_id = self.get_id(0, 0, i - 1, count[i][0])
            if put_item == 1 and i < self.n_stage - 1:
                fa_id = self.get_id(0, 1, i + 1, count[i][1])
            while len(pending_w[i]) > 0 and fa_id >= 0 and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]:
                # 用 w 来填充泡沫
                put_w
```

但是源码这里用了一个比较别扭的写法，就是用了一个 `put_item` 变量来表示插入 F0 还是 F1。如果不照着图有可能会比较难理解。加上注释应该好理解多了。

在最后需要把没有 pop 出来的 W 任务处理掉。

```python
    # 处理没有 pop 出来的 W 任务
    for i in range(self.n_stage):
        while len(pending_w[i]) > 0:
            put_w(i)
```

最后返回调度计划。

```python
    return schedule, end_time, get_max_stage_bubble()
```

这就是 Zero Bubble VPP 的核心编排函数 `try_v_schedule` 的实现。而 `get_v_schedule` 函数则是调用 `try_v_schedule` 函数，尝试生成一个调度计划。

## 2. get_v_schedule 函数

`get_v_schedule` 函数是调用 `try_v_schedule` 函数，尝试生成一个调度计划，它会尝试不同的填充策略，找到一个最小的泡沫时间。

```python
def get_v_schedule(self, only_run_time=False):
    # 初始化调度（计划执行顺序）、结束时间和最大空闲时间（泡沫）变量
    schedule, end_time, max_bubble = None, None, None
    # 根据前向和反向传播的成本以及微批处理数量计算预期时间
    expected_time = sum(self.fbw_cost) * self.n_micro * 2
    # 遍历前向填充和反向填充的所有组合
    for fill_b in [True, False]:
        for fill_f in [True, False]:
            # 尝试生成一个调度计划
            _schedule, _end_time, _max_bubble = self.try_v_schedule(
                fill_b=fill_b, fill_f=fill_f
            )
            # 如果这是第一个调度或者找到了一个更小的泡沫，则更新调度计划
            if max_bubble is None or _max_bubble < max_bubble:
                max_bubble = _max_bubble
                schedule = _schedule
                end_time = _end_time
```

如果只需要运行时间，则返回总预期时间加上最大空闲时间。

```python
    # 如果只需要运行时间，则返回总预期时间加上最大空闲时间
    if only_run_time:
        return max_bubble + expected_time
```

接下来会计算泡沫率，了解调度的效率。

```python
    # 计算泡沫率，了解调度的效率
    bubble_rate = max_bubble / (expected_time + max_bubble)
```

然后打印一些调度的统计信息。

```python
    # 打印一些调度的统计信息
    print("%2d %3d, [%5d %5d %5d %5d], %6d -> %6.4f" %
          (self.n_stage, self.n_micro, *self.fbw_cost, self.c_cost, self.max_mem // self.f_mem, bubble_rate))
```

接下来是为每个stage构建详细的执行顺序。

```python
    # 为每个stage构建详细的执行顺序
    local_order = [[] for _ in range(self.n_stage)]
    # 通信ID字典和计数器，用于管理通信操作的唯一性
    comm_id = {}
    comm_id_counter = 0
    # 初始化后验证时间
    post_validation_time = 0
    # 从最后一个stage开始反向遍历每个stage
    for i in range(self.n_stage - 1, -1, -1):
        # 计算后验证ID
        pv_id = min(2 * (self.n_stage - 1 - i), self.n_micro - 1)
        # 更新后验证时间
        post_validation_time = max(post_validation_time, end_time[self.get_id(
            0, 0, i, pv_id)] - self.fbw_cost[0] - self.c_cost)
        # 遍历发送、接收和无操作，为每个stage添加后验证节点
        for it in ["RECV_", "SEND_", ""]:
            # 跳过特定stage的不必要操作
            if i == 0 and it == "SEND_":
                continue
            if i == self.n_stage - 1 and it == "RECV_":
                continue
            # 为当前stage添加后验证节点
            stage_ = i
            local_order[stage_].append(ScheduledNode(
                type=it + "POST_VALIDATION",
                chunk=0,
                stage=stage_,
                minibatch=0,
                start_time=post_validation_time,
                completion_time=post_validation_time,
            ))
            # 更新通信ID
            comm_id[local_order[stage_][-1]] = comm_id_counter
            comm_id_counter += 1
```

主要功能是反向遍历每个stage，计算后验证时间，并根据条件跳过某些操作，为每个stage添加后验证节点，并更新通信ID。

接下来是遍历每个stage，根据调度添加计算节点。

```python
    # 遍历每个stage，根据调度添加计算节点
    for i in range(self.n_stage):
        for _cat_, _chunk_, _micro_ in schedule[i]:
            # 计算完成时间
            complete_time = end_time[self.get_id(
                _cat_, _chunk_, i, _micro_)]
            # 添加计算节点
            local_order[i].append(ScheduledNode(
                type="FBW"[_cat_],
                chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                stage=i,
                minibatch=_micro_,
                start_time=complete_time - self.fbw_cost[_cat_],
                completion_time=complete_time,
            ))
            # 如果是权重更新（W）则不需要通信
            if _cat_ == 2:  # 没有通信的情况
                continue
            # 定义前向或反向的通信操作
            cat_str = "FORWARD" if _cat_ == 0 else "BACKWARD"

            def communicate(send_recv, stage_):
                # 添加通信节点
                local_order[stage_].append(ScheduledNode(
                    type=send_recv + cat_str,
                    chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                    stage=stage_,
                    minibatch=_micro_,
                    start_time=complete_time,
                    completion_time=complete_time,
                ))
                comm_id[local_order[stage_][-1]] = comm_id_counter

            # 根据块的位置和stage管理发送和接收操作
            if _chunk_ == 1 and i > 0:
                communicate("SEND_", i)
                communicate("RECV_", i - 1)
            if _chunk_ == 0 and i < self.n_stage - 1:
                communicate("SEND_", i)
                communicate("RECV_", i + 1)
            comm_id_counter += 1
```

这段代码的主要功能是遍历每个stage，根据调度添加计算节点和通信节点。

首先，它遍历每个stage，并对每个stage的调度进行遍历。对于每个调度，它计算完成时间，并添加一个计算节点到 `local_order` 列表中。节点的类型是 `"FBW"` 中的一个字符，取决于 `_cat_` 的值。节点的 `chunk` 值取决于 `_cat_` 的值，如果 `_cat_`为 0，则chunk为 `_chunk_`，否则为 `1 - _chunk_`。

然后，如果 `_cat_` 为 2，表示这是一个权重更新操作，不需要通信，所以跳过当前循环。

接着，定义了一个函数communicate，用于添加通信节点。这个函数接受一个表示发送或接收的字符串和一个stage编号，然后添加一个通信节点到local_order列表中。

最后，根据 `_chunk_` 的值和stage编号，决定是否需要添加发送和接收操作。如果 `_chunk_` 为 1 且stage编号大于 0，那么添加发送操作和接收操作。如果 `_chunk_` 为 0 且stage编号小于 `self.n_stage - 1`，那么也添加发送操作和接收操作。每添加一次操作，`comm_id_counter` 就增加 1。

在分布式计算中，数据需要在不同的计算节点之间进行传输。这里的 `_chunk_` 可以理解为数据块的位置，i 是当前的计算stage。当 `_chunk_` 为 1 且stage编号大于 0 时，表示数据块在当前stage的计算节点，需要将数据发送到下一个stage的计算节点，并从上一个stage的计算节点接收数据。当 `_chunk_` 为 0 且stage编号小于 `self.n_stage - 1` 时，表示数据块不在当前stage的计算节点，需要将数据发送到下一个stage的计算节点，并从上一个stage的计算节点接收数据。

接下来需要对每个stage的节点进行排序，优先处理通信节点。

```python
    # 对每个stage的节点进行排序，优先处理通信节点
    for rank in range(self.n_stage):
        def even_breaker(x: ScheduledNode):
            # 计算节点总是延迟
            if x.type in ['F', 'B', 'W']:
                return comm_id_counter
            # 通信节点按它们的唯一通信ID排序
            return comm_id[x]
        local_order[rank] = list(sorted(
            local_order[rank],
            key=lambda x: (x.start_time, even_breaker(x))
        ))
        # 如果接收操作与前一个计算节点重叠，则重新排序以优先执行接收，允许重叠
        for i in range(len(local_order[rank])):
            if i > 0 and local_order[rank][i - 1].type in {'F', 'B', 'W'} and \
                local_order[rank][i].type.startswith('RECV') and \
                "POST_VALIDATION" not in local_order[rank][i].type and \
                    local_order[rank][i].start_time <= local_order[rank][i - 1].completion_time:
                local_order[rank][i], local_order[rank][i -
                                                        1] = local_order[rank][i - 1], local_order[rank][i]
```

这样处理的原因是因为计算节点总是延迟，通信节点按它们的唯一通信ID排序，如果接收操作与前一个计算节点重叠，则重新排序以优先执行接收，允许重叠。首先，定义了一个函数 `even_breaker`，该函数根据节点类型返回一个值。如果节点类型是'F'、'B'或'W'（代表计算节点），则返回 `comm_id_counter`（一个计数器）。如果节点是通信节点，它返回该节点的唯一通信ID。

然后，它对每个stage的节点进行排序。排序的关键是一个元组，包含节点的开始时间和 `even_breaker` 函数的返回值。这样，通信节点（具有较小的唯一通信ID）将优先于计算节点（具有较大的comm_id_counter）。

最后，它检查是否有接收操作与前一个计算节点重叠。如果有，它会重新排序这两个节点，使接收操作优先于计算节点。这是通过交换这两个节点在local_order列表中的位置来实现的。这样做的目的是允许接收操作与计算操作重叠，以提高效率。

由于 Zero Bubble VPP 中引入了 rollback 的机制，所以需要需要回滚的通信进行处理。

```python
    # 对需要回滚的通信进行处理
    local_order_with_rollback = [[] for _ in range(self.n_stage)]
    for rank in range(self.n_stage):
        rollback_comm = set()
        if rank > 0:
            for node in local_order[rank - 1]:
                if node.type == "POST_VALIDATION":
                    break
                if node.type == "SEND_FORWARD":
                    assert node.chunk == 0
                    rollback_comm.add(node.minibatch)
        for node in local_order[rank]:
            if node.type == "RECV_FORWARD" and node.chunk == 0 and node.minibatch in rollback_comm:
                rollback = True
                rollback_comm.remove(node.minibatch)
            else:
                rollback = False
            local_order_with_rollback[rank].append(ScheduledNode(
                type=node.type,
                chunk=node.chunk,
                stage=node.stage,
                minibatch=node.minibatch,
                start_time=node.start_time,
                completion_time=node.completion_time,
                rollback=rollback,
            ))
```


这段代码的主要功能是处理需要回滚的通信。它首先创建一个新的列表 `local_order_with_rollback` 来存储处理后的结果。然后，它遍历每个stage（由rank表示），并创建一个集合 `rollback_comm` 来存储需要回滚的通信。

如果当前stage不是第一个stage，它会遍历前一个stage的所有节点。如果节点类型是 "POST_VALIDATION"，则跳出循环。如果节点类型是"SEND_FORWARD" 并且chunk为0，那么将 minibatch 添加到 rollback_comm 集合中。

这段代码的目的是确定哪些通信操作需要回滚。在分布式系统中，如果一个stage（stage）发送了一个消息，但是在后续的处理中发现了错误，可能需要回滚这个发送操作。这就是为什么需要检查 "SEND_FORWARD" 类型的节点，并且只有当 chunk 为0（也就是说，这是一个新的消息，而不是一个已经分块的消息的一部分）时，才将 minibatch 添加到 rollback_comm 集合中。

另一方面，"POST_VALIDATION" 类型的节点表示一个stage已经完成了其工作并且已经验证了结果，所以如果遇到这种类型的节点，就没有必要继续检查前一个stage的其他节点，因此跳出循环。

## 3. 总结

Zero Bubble VPP 是一个用于分布式训练的调度器，它可以根据不同的填充策略生成一个调度计划。Zero Bubble VPP 的核心编排函数是 `try_v_schedule`，它会尝试生成一个调度计划。`get_v_schedule` 函数是调用 `try_v_schedule` 函数，尝试生成一个调度计划，它会尝试不同的填充策略，找到一个最小的泡沫时间。最后，它会对每个stage的节点进行排序，优先处理通信节点。由于 Zero Bubble VPP 中引入了 rollback 的机制，所以需要需要回滚的通信进行处理。
