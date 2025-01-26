# Megatron 基础架构

Megatron 是一个基于 PyTorch 的深度学习框架，旨在通过多种并行化策略来优化 Transformer 模型的训练效率。Megatron-LM 通过模型并行、数据并行、流水线并行等策略，充分发挥 GPU 计算能力。本文将介绍 Megatron 源码的基础架构，帮助读者更好地理解 Megatron 的设计思路。

