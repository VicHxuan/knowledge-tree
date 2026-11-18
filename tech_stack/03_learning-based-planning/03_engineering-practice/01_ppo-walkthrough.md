# PPO 全流程走读与调参

## 1. 概述

一轮 PPO 训练可以沿着六个环节检查：**接口定义 → 采样 → 优势与价值目标 → 参数更新 → 日志解释 → checkpoint 与复现**。每个环节都应能说清楚输入来自哪里、输出交给谁，以及哪些量在本轮内保持固定。

[PPO 算法页](../02_rl-algorithms/02_model-free-rl/03_ppo.md)解释目标函数、GAE 和梯度如何构成算法闭环；本文把这些机制对应到训练脚本、缓冲区、批量参数和实验记录。读完后，应能走读一份 PPO 实现，核对采样与更新次数，并用多种证据判断一次训练是否有效。

## 2. 基础与定义

### 2.1 环境与网络接口

| 接口 | 至少明确什么 | 直接检查什么 |
|---|---|---|
| Observation | 每个分量的含义、形状、单位、归一化；是否包含历史信息 | `reset` 与 `step` 输出的维度、有限值、范围是否一致 |
| Action | 离散选择或连续控制；分布参数、实际执行值与变换关系 | 采样动作、送入环境的动作、计算 log-prob 的动作是否对应 |
| Reward | 每项奖励的触发条件、量纲、权重与累计方式 | 总奖励能否分解；关键行为是否确实改变奖励 |
| Episode 边界 | `terminated`、`truncated`、自动 reset 后的最终观测 | bootstrap 是否使用结束前的观测；GAE 是否跨入下一回合 |
| Actor / Critic | 各自输入、输出、参数共享关系与优化器 | Actor 输出动作分布；Critic 输出标量价值；共享参数是否重复 step |

本文以无循环状态的 Actor-Critic 为主。循环策略还要保存 hidden state、序列边界和 padding mask，minibatch 应保持训练所需的时间顺序，不能直接套用逐 transition 任意打乱的方法。

### 2.2 训练计数

| 记号 | 常见配置名 | 含义 |
|---|---|---|
| $E$ | `num_envs` | 当前采样进程中的并行环境数 |
| $T$ | `rollout` / `n_steps` | 每次更新前，每个环境收集的步数 |
| $N=ET$ | rollout batch size | 本轮采集的 transition 总数 |
| $M$ | `num_minibatches` / `mini_batches` | 每个 epoch 的 minibatch 数 |
| $B$ | `minibatch_size` / `batch_size` | 每个 minibatch 的样本数 |
| $K$ | `epochs` / `learning_epochs` | 同一批数据计划复用的轮数 |
| $\gamma,\lambda$ | discount / GAE 参数 | 回报折扣与优势估计的权重参数 |

一个 vectorized `env.step()` 为 $E$ 个环境各推进一步，所以环境 transition 计数增加 $E$；它不等于一次 optimizer step。日志中应区分环境步数、PPO iteration、epoch 和实际梯度更新次数。

## 3. 核心原理

<span id="1全文逻辑梳理"></span>

### 3.1 采样缓冲区

固定本轮行为策略，以其产生动作并记录采样时的结果。一条 transition 至少需要表达：

$$
(o_t,a_t,r_t,o_{t+1}^{\mathrm{transition}},
\mathrm{terminated}_t,\mathrm{truncated}_t,
\log\pi_{\mathrm{old}}(a_t\mid o_t),V_{\mathrm{old}}(o_t)).
$$

这里 $o_{t+1}^{\mathrm{transition}}$ 是这次转移的下一观测；自动 reset 的接口可能另行返回下一回合的初始观测，需要从环境提供的最终观测字段恢复前者。保存的旧 log-prob、旧价值及动作应脱离计算图，避免把整段采样过程接入更新图。

真正的任务终止不 bootstrap；外部时间截断通常仍要从截断前的最终观测 bootstrap，同时在回合边界切断优势递推。Gymnasium 对这两个信号的区分见[时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。

### 3.2 优势估计与优化流程

采样结束后，利用奖励、旧价值、下一观测价值和边界 mask，计算未归一化的优势 $\hat A_t$ 与价值回归目标：

$$
\hat R_t=\hat A_t+V_{\mathrm{old}}(o_t).
$$

随后可对 Actor 使用的优势副本做归一化；Critic 的 $\hat R_t$ 应来自归一化前的优势。本文采用同一轮 rollout 内固定 old log-prob、优势与 target 的流程。这样，逐 minibatch 重算的是当前策略的 log-prob 和当前价值预测，比较基准保持稳定。GAE 的完整递推与两个边界 mask 见 [PPO 算法页](../02_rl-algorithms/02_model-free-rl/03_ppo.md)。

进入第一个优化 step 前，用同一份观测、动作和预处理重新计算 log-prob，检查 $\exp(\log\pi-\log\pi_{\mathrm{old}})$ 是否接近 1。若已经明显偏离，先排查模型模式、观测归一化、动作变换、hidden state 和旧数据来源。

### 3.3 批量配置与更新次数

整批数据均匀分组且没有丢弃尾批时：

$$
N=E T=M B.
$$

每个 epoch 一般重新排列样本并依次处理 $M$ 个 minibatch；若每批执行一次联合 Actor-Critic 更新，且没有 KL 提前停止，一轮 PPO 的计划 optimizer step 数为：

$$
U_{\mathrm{planned}}=K M.
$$

若 $N$ 不能整除 $B$，要确认实现是保留较小尾批、丢弃样本还是要求整除；此时不能直接使用 $N=MB$。若 Actor 和 Critic 分别 step，也应分别计数。一个明确带 `optimizer.step()` 的代码位置，才对应一次实际更新。

<details class="kt-derivation" id="ppo-batch-accounting">
<summary>批量与更新计数算例</summary>

以下仅用于核对计数：设 $E=64$、$T=128$，则本轮 $N=8192$。若 $M=8$，每批 $B=1024$；复用 $K=4$ 个 epoch，计划执行 $U=32$ 次联合优化。

将环境数减半、rollout 长度加倍，仍可保持 $N=8192$，但采样的时间连续性和并行性改变，不能据此断言学习效果相同。提高 epoch 数也会增加同一批数据的使用次数与策略偏移，应观察实际 KL 和提前停止次数。

在 $W$ 个同步采样进程中，若每个进程各有 $E$ 个环境，则全局样本量为 $WET$。每次 DDP 同步 step 合并各进程的梯度；全局 optimizer step 次数不会仅因有 $W$ 个进程就再乘 $W$。具体采用本地还是全局 batch 配置，需要对照该实现的采样器与同步位置。

</details>

### 3.4 日志与检查点

一次 update 返回实际处理的 minibatch 数、各项损失和策略变化指标；训练器再把这些值与本轮采样统计、环境步数关联起来。日志应同时保留“计划更新次数”和“实际更新次数”，避免 KL 早停后仍按计划次数解释数据。

更新后的策略用于下一轮新采样。定期从明确的训练边界保存 checkpoint，并通过独立评估任务判断策略表现。训练中的奖励滑动平均、单次 minibatch loss 和固定场景上的评估成功率，分别回答不同问题。

## 4. 实现与应用

### 4.1 训练脚本结构

下表可直接作为阅读顺序。具体函数名取决于框架，关键是找到这些操作实际发生的位置。

| 阶段 | 找到的代码位置 | 验证结果 |
|---|---|---|
| 初始化 | 环境、网络、预处理、optimizer、seed 创建 | 网络接口与环境空间一致；版本和最终解析配置有记录 |
| 采样 | 动作分布采样、`env.step`、缓冲区写入 | 写入条数为 $ET$；旧 log-prob 与实际采样策略对应 |
| 计算目标 | 最后状态估值、GAE、returns、优势归一化 | 正确区分终止/截断；target 不依赖更新后的 Critic |
| 优化 | epoch/minibatch 循环、backward、step、KL 早停 | 计数可核对；旧量固定；梯度与参数确实变化 |
| 记录 | 指标聚合、记录步数、评估入口 | 明确均值分母、损失符号、统计窗口和评估条件 |
| 保存/恢复 | state dict、optimizer、预处理器、配置读取 | 恢复同一观测的输出一致，继续训练所需状态齐全 |

最小可用验证是“小规模采样一轮 → 算 target → 更新一轮 → 保存/加载 → 再评估”。它能暴露接口错误，比直接运行长训练更容易定位问题。

### 4.2 训练监控指标

| 问题 | 配合查看的量 |
|---|---|
| 任务是否学会 | 固定评估协议下的回报、成功率、失败类型、回合长度 |
| 策略是否在合理变化 | ratio 分布、KL、clip fraction、熵或动作标准差 |
| 优化器是否真的工作 | 实际 step 数、learning rate、梯度范数、更新前后参数差异 |
| Critic 是否提供可用估值 | target / value 的分布、value loss、explained variance、边界样本 |
| 数据是否异常 | reward 分项、终止/截断比例、观测与动作范围、NaN/Inf |

这些是推荐观测量，不要求所有框架采用同一 tag。记录 value loss 时说明是否乘了系数、是否对 value/return 归一化；target 方差接近零时，explained variance 也难以提供稳定解释。全站命名参考 [TensorBoard 指标模板](./03_training-metrics-and-diagnosis.md)。

### 4.3 训练状态保存与恢复

推理所需的 Actor 参数与完整续训所需状态应分别明确。至少保存 Actor/Critic、optimizer、预处理统计、学习率调度器（若使用）、训练计数、最终配置、代码版本和随机状态；混合精度训练还要保存 scaler。PyTorch 官方[模型保存与恢复教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)说明了模型与 optimizer 状态的保存方式；skrl 的实际可保存模块和入口见[保存、加载与日志文档](https://skrl.readthedocs.io/en/latest/intro/data.html)。

环境或仿真器是否支持状态快照，决定能否从同一物理状态继续采样；只恢复网络和 seed 不能据此声称逐步重现中断前轨迹。恢复后先比较固定输入的输出、预处理统计和计数，再执行一轮采样与更新。该清单是一种复现组织方法，实际 checkpoint 字段要以所使用版本的保存实现为准。

## 5. 分析与总结

<span id="调参经验"></span>

### 5.1 采样时域、折扣因子与 GAE 参数

原有经验把三者概括为“当前动作受到多远未来的评价”。这个直觉可以保留，但三者控制的位置不同：

| 参数 | 直接改变什么 | 调整时检查什么 |
|---|---|---|
| $T$：rollout 长度 | 一次更新前每个环境采集的真实时间片段，以及采样/更新的频率 | 回合边界、bootstrap 误差、内存与采样耗时 |
| $\gamma$：折扣因子 | 任务目标对未来奖励的权重，也进入 TD 误差 | 控制周期、奖励尺度、任务所需的信用分配跨度 |
| $\lambda$：GAE 参数 | 多步 TD 误差的混合权重，改变估计的偏差与方差 | 优势波动、Critic 准确度、长程信号能否传播 |

$T$ 较短不等于只优化未来 $T$ 步，因为轨迹末端仍可 bootstrap；增大 $\lambda$ 也不会凭空增加实际采样长度。若单步控制周期为 $\Delta t$，rollout 的物理时长是 $T\Delta t$，跨控制频率比较配置时应同时记录它。

<details class="kt-derivation" id="ppo-effective-horizon">
<summary>有效时间跨度</summary>

忽略终止边界并假设无限长序列，折扣权重和为 $\sum_{l\ge0}\gamma^l=1/(1-\gamma)$；GAE 中 TD 残差权重和为 $\sum_{l\ge0}(\gamma\lambda)^l=1/(1-\gamma\lambda)$。这两式提供权重衰减的尺度，不是硬性的观察窗口，也不表示这些步的贡献相等。

实际计算还受到 rollout 长度、episode 边界、奖励分布和价值估计误差影响。因此，先根据任务与控制周期选择合理的折扣，再检查 target/优势；比较 rollout 或 lambda 时固定其余关键配置，并使用同一评估协议。

</details>

<span id="tips"></span>

### 5.2 损失函数与训练诊断

**策略 loss 接近零不能单独判断训练失败或收敛，也不是 skrl 独有的现象。** 在旧策略点，ratio 等于 1；若优势在所观察的整批样本上已中心化，则

$$
\mathcal L_\pi(\theta_{\mathrm{old}})
=-\frac1N\sum_t\widetilde A_t\approx0,
\qquad
\nabla_\theta\mathcal L_\pi
=-\frac1N\sum_t\widetilde A_t
\nabla_\theta\log\pi_\theta(a_t\mid o_t)
\quad\text{仍可能非零。}
$$

标量的正负项抵消，不代表梯度相互抵消。随机 minibatch 的均值、更新后的 loss 和跨多个 step 的日志均值也不必恰好等于零。skrl 2.1.1 与 Stable-Baselines3 v2.7.0 都采用样本均值构造策略 loss；“采用 mean”本身无法解释任务失败。[skrl 固定版本源码](https://github.com/Toni-SM/skrl/blob/2.1.1/skrl/agents/torch/ppo/ppo.py)、[SB3 固定版本源码](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/ppo/ppo.py)。

<details class="kt-derivation" id="ppo-near-zero-loss-diagnosis">
<summary>低损失训练失败的诊断</summary>

1. 确认看的是 policy、value、entropy 还是 total loss，以及是否乘系数、归一化、平滑。
2. 检查实际 optimizer step、梯度范数和参数变化。若没有执行更新，再查 KL 早停、learning rate、梯度被 detach 等原因。
3. 检查 reward、优势和 target 的分布；稀疏奖励或错误的终止处理可能让监督信号退化。
4. 检查 KL、clip fraction 和熵，再与固定评估集的行为表现对照。
5. 核对日志聚合分母。skrl **2.1.1** 的该 PPO 实现累加已处理 minibatch 的损失，却按配置的 `learning_epochs × mini_batches` 归一；发生 KL 提前跳出时，相对按实际完成次数求均值，日志值的绝对值可能被更大的分母压小。这是一处具体版本的统计口径，不能未经核对就套到其他版本或本地项目。

</details>

### 5.3 核心结论

- 先把观测、动作、边界和缓冲区定义清楚，再解释训练曲线。
- 用 $ET$ 核对采样量、用实际 optimizer step 核对更新量；两者分开记录。
- 固定本轮旧量与 target，逐批重算当前网络输出；复用数据的次数和策略偏移一起监控。
- 结合任务表现、策略变化、梯度和日志口径判断训练；保存足够的配置与状态，使结论可复核。

## 6. 关联与参考

### 6.1 知识关联

- **算法主线**：[PPO](../02_rl-algorithms/02_model-free-rl/03_ppo.md)——目标函数、GAE、梯度路径与教学代码。
- **实现边界**：[强化学习工程避坑指南](./02_uav-rl-task-design.md)——接口、采样边界、训练与恢复的一致性。
- **监控规范**：[TensorBoard 指标模板](./03_training-metrics-and-diagnosis.md)——指标含义和统计层级。
- **部署实践**：[多服务器多 GPU 训练](../../04_engineering-tools/02_training-infrastructure/01_multi-gpu-training-deployment-guide.md)——并行训练与运维入口。

### 6.2 参考资料

1. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017：交替采样与多轮 minibatch 优化。
2. Gymnasium, [Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)：终止、截断和 bootstrap。
3. skrl, [PPO 文档](https://skrl.readthedocs.io/en/latest/api/agents/ppo.html)、[2.1.1 版 PyTorch PPO 实现](https://github.com/Toni-SM/skrl/blob/2.1.1/skrl/agents/torch/ppo/ppo.py)：配置语义、更新循环与 loss 聚合。本文版本核对日：2026-11-06。
4. Stable-Baselines3, [v2.7.0 PPO 实现](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/ppo/ppo.py)：样本均值策略损失、KL 早停与日志实例。
5. PyTorch, [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)；skrl, [Saving, loading and logging](https://skrl.readthedocs.io/en/latest/intro/data.html)：模型、optimizer 和训练状态的保存入口。
