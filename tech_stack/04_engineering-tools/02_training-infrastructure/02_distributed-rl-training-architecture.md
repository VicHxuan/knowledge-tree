# 分布式强化学习训练

## 1. 概述

同步分布式 PPO 将环境采样分配到多个 Rank，每个 Rank 保留网络副本和本地数据，再通过梯度同步执行共同更新。系统设计同时确定样本权重、参数状态和通信顺序，使并行计算对应明确的全局优化目标。

## 2. 基础与定义

### 2.1 并行结构

环境并行批量推进多个环境；数据并行在不同设备计算同一模型的梯度；模型并行拆分网络；Actor–Learner 分离将采样和优化交给不同进程。以下主线采用每 GPU 一个 Rank 的同步数据并行。

### 2.2 样本规模

第 $r$ 个 Rank 有 $N_r$ 个环境，采样 $T_r$ 步，则 $B_r=N_rT_r$，全局 rollout 大小为 $B=\sum_rB_r$。一次优化更新使用的 mini-batch 大小与整轮 rollout 分别记录。

## 3. 核心原理

### 3.1 采样与价值目标

每轮使用同一策略版本采样，本地保存观测、动作、旧 log-prob、价值、奖励、末观测和结束标志。真实终止掩码与轨迹连续掩码分别用于自举与 GAE，具体定义见[任务设计](/knowledge-tree/learning-based-planning/engineering-practice/rl-engineering-pitfalls/)。GAE 在连续的本地轨迹上计算。

### 3.2 全局目标与梯度同步

设一次更新中第 $r$ 个 Rank 有 $b_r$ 个有效样本，本地损失 $L_r$ 为这些样本的均值，则全局样本等权目标为

$$
L=\frac{\sum_rb_rL_r}{\sum_rb_r},\qquad
\nabla L=\frac{\sum_rb_r\nabla L_r}{\sum_rb_r}.
$$

各 Rank 样本数相同时，普通梯度平均与全局样本均值等价。DDP 在反向传播中组织梯度通信，优化器在各 Rank 使用同步后的梯度更新。[PyTorch DDP](https://docs.pytorch.org/docs/2.9/notes/ddp.html)

<details class="kt-derivation" id="weighted-ddp">
<summary>不等样本数的梯度权重</summary>

设 DDP 对 $G$ 个 Rank 的梯度取算术平均。将局部均值损失缩放为

$$
\widetilde L_r=\frac{G b_r}{B_{\mathrm{step}}}L_r,
\qquad B_{\mathrm{step}}=\sum_rb_r,
$$

则同步后梯度为

$$
\frac1G\sum_r\nabla\widetilde L_r
=\frac1{B_{\mathrm{step}}}\sum_rb_r\nabla L_r.
$$

权重对应当前同步更新的有效样本数。梯度累积、补齐样本和重复采样会改变有效权重，应按实际批次重算。空批次需要保持集体通信一致并明确零贡献，不能在单个 Rank 任意跳过同步。

</details>

### 3.3 参数与统计状态

初始化和恢复后保持模型、优化器及调度器状态一致。环境随机流按 Rank 派生，以覆盖不同场景。优势全局归一化汇总有效样本统计；局部归一化改变各 Rank 的相对尺度，作为独立设计选择记录。

KL 若控制提前停止或学习率，先全局聚合再统一决策。课程等级与全局步数在共同轮次边界更新。梯度同步完成后再裁剪；Actor、Critic 分别裁剪或联合裁剪取决于所定义的参数分组，共享参数只处理一次。

### 3.4 采样调度

固定长度采样的等待由最慢 Rank 决定。DD-PPO 采用提前结束慢 Worker 采样的机制，随后仍共同优化；可变采样长度需要处理末状态自举、样本权重和一致的更新次数。[DD-PPO](https://arxiv.org/abs/1911.00357)

VER 等方法进一步研究可变经验贡献与采样利用率。异步 Actor–Learner 还需记录策略版本和数据滞后，并按照具体算法处理分布差异。[VER](https://arxiv.org/abs/2210.05064)

## 4. 实现与应用

### 4.1 一轮同步更新

```text
同步初始化与恢复状态
    → 各 Rank 独立采样
    → 本地价值目标与优势
    → 全局统计与统一批次计划
    → 前向、损失、反向、梯度同步
    → 梯度裁剪、优化器更新
    → 全局诊断、课程决策与检查点
```

各 Rank 保持集体通信次数和顺序一致。使用 DDP 自动同步时，不再为同一梯度叠加重复的手动同步。

<details class="kt-derivation" id="distributed-example">
<summary>采样规模与更新示例</summary>

8 个 Rank、每 Rank 64 个环境、每环境 256 步，得到 $8\times64\times256=131072$ 条转移。若每次同步更新每 Rank 使用 2048 条样本，则每轮数据遍历有 8 次同步更新；3 个 epoch 共 24 次。此例假设固定长度、无补齐且每个样本每 epoch 使用一次。

动作包含多个物理子步时，分别记录策略步与物理步。按地图均衡重采样会改变经验权重，需与全局样本等权目标区分，并用固定批次核对实际梯度。

</details>

### 4.2 正确性与性能

用冻结的小批次比较单进程全局损失梯度与分布式聚合梯度；检查更新后的参数、优化器步数和归一化状态。计时分别覆盖采样、推理、更新、通信、等待和写盘，注明通信与计算重叠。

固定全局工作量的强扩展使用 $S(G)=T(1)/T(G)$、$E(G)=S(G)/G$。固定每卡工作量的弱扩展比较总吞吐和每卡效率，同时记录全局 batch 的变化。

## 5. 分析与总结

并行训练改变资源分配，也可能改变样本分布与每次更新规模。比较训练效果时同时控制或报告总交互步、全局 batch、epoch 和更新次数。跨节点训练额外验证网络、进程发现与故障恢复；多台服务器独立实验分别保留运行 ID 和输出目录。

## 6. 关联与参考

- [训练部署与恢复](./01_multi-gpu-training-deployment-guide.md)。
- [PPO 工程走读](/knowledge-tree/learning-based-planning/engineering-practice/ppo-walkthrough/)。
- [DD-PPO](https://arxiv.org/abs/1911.00357)。
- [VER](https://arxiv.org/abs/2210.05064)。
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。
