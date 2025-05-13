import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import time
from typing import List
import numpy as np  # 用于计算平均值
import datetime  # 用于设置超时


# ---- 分布式环境设置 ----
def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ["MASTER_ADDR"] = "localhost"
    # 设置超时以防止无限挂起
    timeout_seconds = 60
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: Process group initialized.", flush=True)  # 确认初始化完成


def cleanup():
    """销毁分布式环境"""
    dist.destroy_process_group()


# ---- 分布式 Ring Attention 核心逻辑 ----
def ring_attention_distributed_fwd(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    rank: int,
    world_size: int,
    prev_rank: int,
    next_rank: int,
    attn_mask_chunk: torch.Tensor = None,
) -> torch.Tensor:
    """
    分布式 Ring Attention 前向传播 (简化版，使用阻塞 send/recv 并避免死锁)。
    """
    batch_size, n_heads, chunk_size, head_dim = q_local.shape
    device = q_local.device

    o_local = torch.zeros_like(q_local)
    l_local = torch.zeros(
        batch_size, n_heads, chunk_size, 1, device=device, dtype=torch.float32
    )
    m_local = torch.full(
        (batch_size, n_heads, chunk_size, 1),
        -float("inf"),
        device=device,
        dtype=torch.float32,
    )

    k_curr = k_local.clone()
    v_curr = v_local.clone()
    k_recv_buffer = torch.empty_like(k_local)
    v_recv_buffer = torch.empty_like(v_local)

    # --- Ring Attention 核心循环 (阻塞版本) ---
    for step in range(world_size):
        if rank == 0:
            print(f"[Rank 0] Entering Step {step}", flush=True)

        # 1. 本地计算: 使用 k_curr, v_curr (来自上一步或初始值)
        if rank == 0:
            print(f"[Rank 0] Step {step}: Starting computation", flush=True)
        # --- 计算逻辑开始 ---
        scores_ij = torch.matmul(q_local, k_curr.transpose(-2, -1)) / (head_dim**0.5)
        if attn_mask_chunk is not None:
            pass  # 简化处理

        m_ij = torch.max(scores_ij, dim=-1, keepdim=True)[0]
        p_ij = torch.exp(scores_ij - m_ij)
        l_ij = torch.sum(p_ij, dim=-1, keepdim=True)
        o_ij = torch.matmul(p_ij, v_curr)

        m_prev = m_local
        l_prev = l_local
        o_prev = o_local

        m_new = torch.maximum(m_prev, m_ij)
        exp_diff_m_prev = torch.exp(m_prev - m_new)
        exp_diff_m_ij = torch.exp(m_ij - m_new)
        l_new = exp_diff_m_prev * l_prev + exp_diff_m_ij * l_ij
        o_new = exp_diff_m_prev * o_prev + exp_diff_m_ij * o_ij

        o_local = o_new
        l_local = l_new
        m_local = m_new
        # --- 计算逻辑结束 ---
        if rank == 0:
            print(f"[Rank 0] Step {step}: Finished computation", flush=True)

        # --- 通信 (使用 send/recv 并处理死锁) ---
        # k_to_send: 当前计算刚用完的 k_curr
        # v_to_send: 当前计算刚用完的 v_curr
        k_to_send = k_curr
        v_to_send = v_curr

        if rank % 2 == 0:
            # 偶数 Rank: 先 Send 再 Recv
            if rank == 0:
                print(
                    f"[Rank 0] Step {step}: Even rank - Sending K to {next_rank}",
                    flush=True,
                )
            dist.send(tensor=k_to_send, dst=next_rank)
            if rank == 0:
                print(
                    f"[Rank 0] Step {step}: Even rank - Sending V to {next_rank}",
                    flush=True,
                )
            dist.send(tensor=v_to_send, dst=next_rank)

            if rank == 0:
                print(
                    f"[Rank 0] Step {step}: Even rank - Receiving K from {prev_rank}",
                    flush=True,
                )
            dist.recv(tensor=k_recv_buffer, src=prev_rank)
            if rank == 0:
                print(
                    f"[Rank 0] Step {step}: Even rank - Receiving V from {prev_rank}",
                    flush=True,
                )
            dist.recv(tensor=v_recv_buffer, src=prev_rank)
        else:
            # 奇数 Rank: 先 Recv 再 Send
            # if rank == 1: print(f"[Rank 1] Step {step}: Odd rank - Receiving K from {prev_rank}", flush=True) # 调试其他 rank
            dist.recv(tensor=k_recv_buffer, src=prev_rank)
            # if rank == 1: print(f"[Rank 1] Step {step}: Odd rank - Receiving V from {prev_rank}", flush=True)
            dist.recv(tensor=v_recv_buffer, src=prev_rank)

            # if rank == 1: print(f"[Rank 1] Step {step}: Odd rank - Sending K to {next_rank}", flush=True)
            dist.send(tensor=k_to_send, dst=next_rank)
            # if rank == 1: print(f"[Rank 1] Step {step}: Odd rank - Sending V to {next_rank}", flush=True)
            dist.send(tensor=v_to_send, dst=next_rank)

        if rank == 0:
            print(f"[Rank 0] Step {step}: Finished communication", flush=True)

        # 更新 k_curr 和 v_curr 为下一轮计算准备
        k_curr = k_recv_buffer.clone()  # 需要 clone
        v_curr = v_recv_buffer.clone()
        # 为下一次接收准备新的空缓冲区 (或者不清空，recv 会覆盖)
        # k_recv_buffer = torch.empty_like(k_local)
        # v_recv_buffer = torch.empty_like(v_local)

        if rank == 0:
            print(f"[Rank 0] Step {step}: Finished updating K/V state", flush=True)
        dist.barrier()  # 每步结束同步一次，确保所有进程一起进入下一步

    # --- 最终归一化 ---
    l_local_safe = torch.where(l_local == 0, torch.ones_like(l_local), l_local)
    final_o_local = o_local / l_local_safe

    return final_o_local


# ---- 主执行函数 ----
def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != 4:
        if rank == 0:
            print(
                "错误：此脚本设计为在 4 个 GPU 上运行。请使用 --nproc_per_node=4 启动。",
                flush=True,
            )
        return
    if torch.cuda.device_count() < world_size:
        if rank == 0:
            print(
                f"错误：检测到 {torch.cuda.device_count()} 个 GPU，但需要 {world_size} 个。",
                flush=True,
            )
        return

    setup(rank, world_size)

    prev_rank = (rank - 1 + world_size) % world_size
    next_rank = (rank + 1) % world_size

    if rank == 0:
        print(
            f"已启动 {world_size} 个 GPU 进程 (Rank {rank}/{world_size})...", flush=True
        )

    # --- 参数定义 ---
    batch_size = 8
    seq_len = 1024 * 2
    n_heads = 8
    head_dim = 64
    dtype = torch.float32
    num_warmup_iters = 5
    num_timed_iters = 20

    assert (
        seq_len % world_size == 0
    ), f"序列长度 {seq_len} 必须能被 GPU 数量 {world_size} 整除"
    chunk_size = seq_len // world_size

    # --- 准备接收张量 (在目标 GPU 上, 初始不带梯度) ---
    q_local_nograd = torch.empty(
        (batch_size, n_heads, chunk_size, head_dim), dtype=dtype, device=rank
    )
    k_local_nograd = torch.empty(
        (batch_size, n_heads, chunk_size, head_dim), dtype=dtype, device=rank
    )
    v_local_nograd = torch.empty(
        (batch_size, n_heads, chunk_size, head_dim), dtype=dtype, device=rank
    )

    if rank == 0:
        print(
            f"Rank 0: Local tensors allocated (no grad) on device {q_local_nograd.device}",
            flush=True,
        )

    # --- 生成模拟数据 (仅 Rank 0) ---
    q_full, k_full, v_full = None, None, None
    scatter_q_list, scatter_k_list, scatter_v_list = None, None, None

    if rank == 0:
        print("Rank 0: 正在生成模拟数据...", flush=True)
        q_full_cpu = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype)
        k_full_cpu = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype)
        v_full_cpu = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype)
        print("Rank 0: 数据生成完毕。", flush=True)

        # 准备 Scatter 需要的列表，并将每个块移动到 Rank 0 的 GPU
        scatter_q_list = [
            chunk.to(rank) for chunk in q_full_cpu.split(chunk_size, dim=2)
        ]
        scatter_k_list = [
            chunk.to(rank) for chunk in k_full_cpu.split(chunk_size, dim=2)
        ]
        scatter_v_list = [
            chunk.to(rank) for chunk in v_full_cpu.split(chunk_size, dim=2)
        ]
        print(
            f"Rank 0: Scatter list chunks moved to device {scatter_q_list[0].device}",
            flush=True,
        )

        # 将完整数据移至 rank 0 的 GPU，用于标准 Attention 计算
        q_full = q_full_cpu.to(rank)
        k_full = k_full_cpu.to(rank)
        v_full = v_full_cpu.to(rank)

    # --- 数据分发 ---
    if rank == 0:
        print("Rank 0: 正在分发数据块 (Scatter)...", flush=True)

    dist.barrier()  # 确保所有 rank 都准备好了接收张量

    # 使用 Scatter 将 rank 0 上 GPU 的数据块分发到每个 rank 对应的 GPU 上的接收张量 (nograd 版本)
    dist.scatter(
        q_local_nograd, scatter_list=scatter_q_list if rank == 0 else None, src=0
    )
    dist.scatter(
        k_local_nograd, scatter_list=scatter_k_list if rank == 0 else None, src=0
    )
    dist.scatter(
        v_local_nograd, scatter_list=scatter_v_list if rank == 0 else None, src=0
    )

    dist.barrier()  # 确保所有 rank 都收到了数据

    if rank == 0:
        print("Rank 0: 数据已加载到各个 GPU。", flush=True)

    # --- 在 Scatter 之后设置 requires_grad ---
    q_local = q_local_nograd.requires_grad_(True)
    k_local = k_local_nograd.requires_grad_(True)
    v_local = v_local_nograd.requires_grad_(True)

    if rank == 0:
        print(f"Rank 0: requires_grad set for local tensors.", flush=True)

    attn_mask_chunk = None  # 暂不使用 mask

    # --- 预热 Warm-up ---
    if rank == 0:
        print(f"执行 {num_warmup_iters} 次预热迭代 (简化版)...", flush=True)
    for iter_idx in range(num_warmup_iters):
        if rank == 0:
            print(f"[Rank 0] Warmup Iter {iter_idx} starting...", flush=True)

        _ = ring_attention_distributed_fwd(
            q_local,
            k_local,
            v_local,
            rank,
            world_size,
            prev_rank,
            next_rank,
            attn_mask_chunk,
        )

        if rank == 0:
            print(f"[Rank 0] Warmup Iter {iter_idx} finished.", flush=True)

    dist.barrier()
    if rank == 0:
        print("预热完成。", flush=True)

    # --- 性能测试 Performance Timing ---
    if rank == 0:
        print(f"执行 {num_timed_iters} 次计时迭代 (简化版)...", flush=True)

    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_timed_iters)
    ]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_timed_iters)]

    dist.barrier()
    total_start_time = time.perf_counter()

    for i in range(num_timed_iters):
        if rank == 0:
            print(f"[Rank 0] Timed Iter {i} starting...", flush=True)
        start_events[i].record()
        output_local = ring_attention_distributed_fwd(  # 使用简化版函数
            q_local,
            k_local,
            v_local,
            rank,
            world_size,
            prev_rank,
            next_rank,
            attn_mask_chunk,
        )
        end_events[i].record()
        if rank == 0:
            print(f"[Rank 0] Timed Iter {i} finished.", flush=True)

    dist.barrier()
    total_end_time = time.perf_counter()
    if rank == 0:
        print("计时完成。", flush=True)

    # --- 计算并打印计时结果 ---
    if rank == 0:
        torch.cuda.synchronize()  # 确保 GPU 操作完成
        gpu_times = [
            start.elapsed_time(end) for start, end in zip(start_events, end_events)
        ]
        avg_gpu_time_ms = np.mean(gpu_times) if gpu_times else 0
        std_gpu_time_ms = np.std(gpu_times) if gpu_times else 0
        total_cpu_time_s = total_end_time - total_start_time
        print(f"\n--- 性能测试结果 (简化版) (基于 Rank 0) ---", flush=True)
        print(
            f"平均 GPU 执行时间: {avg_gpu_time_ms:.3f} ms (标准差: {std_gpu_time_ms:.3f} ms)",
            flush=True,
        )
        print(
            f"总 CPU 执行时间 ({num_timed_iters} 次迭代): {total_cpu_time_s:.3f} s",
            flush=True,
        )
        print(f"------------------------------------", flush=True)

    # --- 数值正确性验证 Correctness Verification ---
    if rank == 0:
        print("\n开始数值正确性验证...", flush=True)

    # 1. 收集 Ring Attention 结果
    if rank == 0:
        print(f"[Rank 0] Starting all_gather...", flush=True)
    output_local_contiguous = output_local.contiguous()
    all_gathered_outputs = [
        torch.empty_like(output_local_contiguous) for _ in range(world_size)
    ]
    dist.all_gather(all_gathered_outputs, output_local_contiguous)
    dist.barrier()  # 确保 all_gather 完成
    if rank == 0:
        print(f"[Rank 0] all_gather finished.", flush=True)

    if rank == 0:
        try:
            # 2. 在 Rank 0 上拼接 Ring Attention 结果
            ring_output_full = torch.cat(all_gathered_outputs, dim=2)
            print(
                f"Rank 0: Ring Attention 聚合后形状: {ring_output_full.shape}",
                flush=True,
            )

            # 3. 在 Rank 0 上计算标准 Attention
            print(
                "Rank 0: 正在计算标准 Scaled Dot Product Attention 作为基准...",
                flush=True,
            )
            with torch.no_grad():
                q_ref = q_full.to(rank)  # 确保在 GPU 0
                k_ref = k_full.to(rank)
                v_ref = v_full.to(rank)
                standard_output_full = F.scaled_dot_product_attention(
                    q_ref, k_ref, v_ref, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            print(
                f"Rank 0: 标准 Attention 输出形状: {standard_output_full.shape}",
                flush=True,
            )

            # 4. 比较结果
            are_close = torch.allclose(
                ring_output_full, standard_output_full, rtol=1e-3, atol=1e-4
            )
            if are_close:
                print("\n✅ Rank 0: 数值正确性验证通过！", flush=True)
            else:
                print("\n❌ Rank 0: 数值正确性验证失败！", flush=True)
                diff = torch.abs(ring_output_full - standard_output_full).mean()
                print(f"  平均绝对差值: {diff.item()}", flush=True)

        except Exception as e:
            print(f"\n❌ Rank 0 进行数值验证时出错: {e}", flush=True)
            import traceback

            traceback.print_exc()

    # --- 清理分布式环境 ---
    if rank == 0:
        print("正在清理分布式环境...", flush=True)

    dist.barrier()  # 清理前最后同步一次
    cleanup()
    if rank == 0:
        print("分布式环境已清理。", flush=True)


if __name__ == "__main__":
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("错误：请使用 torchrun 启动此脚本。")
        print("例如: torchrun --nproc_per_node=4 your_script_name.py")
    else:
        main()
