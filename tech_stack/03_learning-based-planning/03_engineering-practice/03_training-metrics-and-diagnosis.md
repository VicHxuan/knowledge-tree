# 训练指标与诊断

## 1. 概述

训练诊断沿着任务结果、数据采样、策略更新和系统性能逐层定位问题。每个指标同时规定计算对象、分子分母、统计窗口和横轴，使曲线能对应到一次实际训练变化。

## 2. 基础与定义

### 2.1 统计单位

区分环境转移、完成回合、优化更新和墙钟时间。训练横轴默认采用全局有效环境转移数；GPU 扩展同时报告墙钟耗时。回合指标在结束时写入，更新指标在参数更新后写入，资源指标按固定时间采集。

多个 Rank 的样本量可能不同。设局部和为 $S_r$、有效样本数为 $N_r$，总体均值为

$$
\bar x=\frac{\sum_rS_r}{\sum_rN_r}.
$$

分位数使用合并样本或可合并直方图；各 Rank 分位数的平均值通常不对应总体分位数。

### 2.2 任务结果

对同一统计窗口中的 $N$ 个已完成回合，按预先规定的优先级归入成功、碰撞、超时和其他失败四类：

$$
p_k=\frac{N_k}{N},\qquad \sum_kN_k=N.
$$

若同一步同时满足到达和碰撞，按固定规则确定主终止原因，另保留原始事件标记。窗口内没有完成回合时记录样本数为零，跳过该比率或记录缺失值。

## 3. 核心原理

### 3.1 导航表现与奖励

| 分组 | 核心指标 | 含义 |
|---|---|---|
| `episode/*` | 回报、长度、到达时间 | 完成回合的结果分布 |
| `termination/*` | 成功、碰撞、超时、其他失败 | 同一回合集合上的互斥统计 |
| `reward/*` | 每步均值、回合累计、各加权分量 | 奖励实际由哪些项主导 |
| `task/*` | 净间距、路径长度、跟踪误差 | 无人机任务的物理表现 |
| `curriculum/*` | 等级、难度、晋级样本数 | 训练分布的变化 |
| `eval/*` | 固定场景成功率与安全指标 | 独立于训练分布的导航能力 |

回报同时受回合长度、奖励权重与终止方式影响。将总体曲线与分终止原因的曲线并列，检查策略是否通过提前结束或延长回合获得不同累计奖励。

### 3.2 PPO 策略更新

设旧策略对动作的对数概率为 $\ell_i^{\mathrm{old}}$，当前值为 $\ell_i$，定义 $\Delta\ell_i=\ell_i-\ell_i^{\mathrm{old}}$、$\rho_i=e^{\Delta\ell_i}$。常用诊断量为

$$
\widehat D_{\mathrm{KL}}=\frac1N\sum_i[(\rho_i-1)-\Delta\ell_i],\qquad
f_{\mathrm{clip}}=\frac1N\sum_i\mathbf1\{|\rho_i-1|>\epsilon\}.
$$

这里的 KL 估计以旧策略样本为基础；它与精确分布 KL、以及仅使用 $-\Delta\ell$ 的样本估计分别命名。连续动作的联合对数概率先沿动作维求和。

<details class="kt-derivation" id="kl-estimator">
<summary>KL 估计与概率比</summary>

在支持集一致且比率可积时，$\mathbb E_{\pi_{\mathrm{old}}}[\rho]=1$，因此

$$
\mathbb E_{\pi_{\mathrm{old}}}[(\rho-1)-\log\rho]
=\mathbb E_{\pi_{\mathrm{old}}}\!\left[\log\frac{\pi_{\mathrm{old}}}{\pi}\right]
=D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi).
$$

每个样本满足 $x-1-\log x\ge0$。对小 $\Delta\ell$，可用 `expm1(delta_logp) - delta_logp` 减轻消减误差。极大概率比先记录触发位置与有限性，避免只靠日志裁剪隐藏异常。

</details>

同时记录策略熵、动作均值与标准差、概率比范围、实际更新轮数和提前停止次数。`clip_fraction` 表示概率比越界频率；样本是否对截断目标产生梯度还与优势符号有关。

### 3.3 价值估计与梯度

设训练目标为 $y_i$、预测为 $V_i$，解释方差为

$$
\mathrm{EV}=1-\frac{\operatorname{Var}(y-V)}{\operatorname{Var}(y)}.
$$

目标方差接近零时报告该指标未定义，并记录目标方差。EV 对常数偏差不敏感，应与价值 MSE、预测均值、目标均值和回报尺度共同分析。

| 分组 | 核心指标 |
|---|---|
| `critic/*` | EV、价值均值、目标均值、目标标准差 |
| `loss/*` | Actor、Critic、熵项及加权总损失 |
| `optimizer/*` | 学习率、裁剪前梯度范数、裁剪后梯度范数 |
| `diagnostics/*` | 非有限观测、动作、损失与梯度计数 |
| `policy/*` | 熵、连续动作尺度或离散动作使用频率 |

优势归一化、奖励归一化和多 Critic 分量会改变数值尺度；按实际计算位置标记指标。冻结模块应核对梯度与运行统计，Dropout 和 BatchNorm 的训练模式也应与采样概率计算保持一致。

### 3.4 吞吐与分布式统计

| 分组 | 指标与用途 |
|---|---|
| `rollout/*` | 有效步数、回合边界、优势与回报分布 |
| `performance/*` | 环境步进、推理、反向、通信、写盘耗时 |
| `distributed/*` | 各 Rank 样本量、等待时间、更新次数 |
| `resources/*` | GPU 显存、利用率、CPU 与存储占用 |
| `model/*` | 选择性记录激活、特征尺度和冻结状态 |

吞吐采用全局有效转移数除以真实墙钟时间。参数同步、全局归一化和不等样本梯度权重见[分布式训练架构](/knowledge-tree/engineering-tools/ppo-in-multi-gpu/distributed-rl-training-architecture/)。

## 4. 实现与应用

### 4.1 记录流程

采样时累计奖励分量和步数；回合结束时提交结果记录；优化后记录本轮统计；所有聚合量带上计数，再由主写入进程输出。训练恢复时沿用已保存的全局步数，并明确事件文件衔接方式。

<details class="kt-derivation" id="logging-schema">
<summary>指标记录示例</summary>

```text
run_id, code_commit, config_hash, model_checkpoint
step/env_transitions, step/updates, time/wall_seconds
episode/count, episode/return_mean, episode/length_mean
termination/success_count, termination/collision_count
termination/timeout_count, termination/other_count
ppo/approx_kl, ppo/clip_fraction, policy/entropy
critic/explained_variance, critic/target_variance
optimizer/grad_norm_before_clip, optimizer/grad_norm_after_clip
performance/rollout_seconds, performance/update_seconds
```

事件命名可沿用项目已有约定，但分母、单位和统计时刻必须一致。离散候选任务额外记录有效候选数、选择分布、无候选次数和参考评分遗憾；每轮候选集合变化时，按几何或语义分组分析动作频率。

</details>

### 4.2 诊断顺序

| 现象 | 首先验证 | 后续对照 |
|---|---|---|
| 训练无法启动或出现 NaN | 首个非有限张量、观测范围、除零、梯度 | 固定输入的单轮前向与更新 |
| 回报增长而成功率不变 | 奖励分量、回合长度、任务分布 | 固定场景独立评估 |
| KL 快速增大 | 旧 log-prob、动作变换、优势尺度 | 减少更新轮数或步长的单因素实验 |
| Critic 误差增大 | 自举边界、奖励尺度、目标计算 | 固定轨迹上的价值拟合 |
| 策略很快确定化 | 熵、动作饱和、有效候选数 | 检查可行动作与探索范围 |
| 多卡掉速或停滞 | 最早失败 Rank、采样负载、通信时序 | 单卡与双卡同配置短测 |

先运行简化场景和短训练，确认采样、真实反向更新、检查点读取均可完成。每个实验只改变明确因素，保留代码、配置、随机种子与相同评估条件。

### 4.3 指标字段参考

以下保留既有指标字段，接入时按项目需要选择。带 `latest_` 的终止指标使用每环境最近一次结果：$N_s/N_{\mathrm{env}}$、$N_f/N_{\mathrm{env}}$、$N_t/N_{\mathrm{env}}$，三者之和为已产生终止记录的环境覆盖率。它描述环境最新状态分布；主线中的回合成功率使用已完成回合数作分母。课程切换后分别重置或按等级分开统计。

<details class="kt-derivation" id="metric-fields-2">
<summary>回合表现 episode/*</summary>

| 指标 | 含义 |
|---|---|
| `episode/return_mean` | 最近 K 个完成 Episode 的平均累计奖励 |
| `episode/return_std` | 最近 K 个完成 Episode 的累计奖励标准差 |
| `episode/length_mean` | 最近 K 个完成 Episode 的平均 RL 决策步数 |
| `episode/window_count` | 当前统计窗口中的有效 Episode 数量 |

</details>

<details class="kt-derivation" id="metric-fields-3">
<summary>终止结果 termination/*</summary>

| 数值 | 含义 |
|---|---|
| `1` | Success |
| `-1` | Failure |
| `0` | Timeout |
| `-10` | 尚无有效终止记录 |

| 指标 | 含义 |
|---|---|
| `termination/latest_success_rate` | 最近一次终止为 Success 的环境数 / 全局环境数 |
| `termination/latest_failure_rate` | 最近一次终止为 Failure 的环境数 / 全局环境数 |
| `termination/latest_timeout_rate` | 最近一次终止为 Timeout 的环境数 / 全局环境数 |
| `termination/latest_coverage` | 已产生过有效终止记录的环境数 / 全局环境数 |

| 指标 | 含义 |
|---|---|
| `termination/latest_reasons/<name>_rate` | 最近一次终止为指定原因的环境数 / 全局环境数 |

</details>

<details class="kt-derivation" id="metric-fields-4">
<summary>奖励统计 reward/*</summary>

| 指标 | 含义 |
|---|---|
| `reward/step_mean` | 当前 Rollout 所有 Transition 的平均原始奖励 |
| `reward/step_std` | 当前 Rollout 所有 Transition 的原始奖励标准差 |

| 指标 | 含义 |
|---|---|
| `reward/components/<name>_step_mean` | 当前 Rollout 中指定奖励分量的每步均值 |

</details>

<details class="kt-derivation" id="metric-fields-5">
<summary>采样统计 rollout/*</summary>

| 指标 | 含义 |
|---|---|
| `rollout/transitions` | 本次 Rollout 采集的全局 Transition 数量 |

</details>

<details class="kt-derivation" id="metric-fields-6">
<summary>策略分布 policy/*</summary>

| 指标 | 含义 |
|---|---|
| `policy/discrete/entropy` | 当前 Categorical 分布的信息熵，单位 nats |
| `policy/discrete/normalized_entropy` | 当前离散动作分布的归一化熵 |
| `policy/discrete/max_probability` | 每个样本最大动作概率的均值 |
| `policy/discrete/logits_std` | 每个样本有效动作 Logits 的标准差，再对样本求均值 |

| 指标 | 含义 |
|---|---|
| `policy/continuous/entropy` | 当前连续动作分布的微分熵 |
| `policy/continuous/action_std_mean` | 策略输出动作分布标准差的均值 |
| `policy/continuous/action_std_min` | 策略输出动作分布标准差的最小值 |
| `policy/continuous/action_std_max` | 策略输出动作分布标准差的最大值 |
| `policy/continuous/action_saturation_fraction` | 实际执行动作接近动作上下限的比例 |

| 指标 | 含义 |
|---|---|
| `policy/log_prob_mean` | 采样动作在当前策略下的平均 Log Probability |

</details>

<details class="kt-derivation" id="metric-fields-7">
<summary>PPO 更新诊断 ppo/*</summary>

| 指标 | 含义 |
|---|---|
| `ppo/approx_kl` | PPO 更新前后策略分布的近似 KL 散度 |
| `ppo/clip_fraction` | Probability Ratio 超出 PPO 裁剪区间的样本比例 |
| `ppo/ratio_mean` | 新旧策略动作概率比的均值 |
| `ppo/ratio_std` | 新旧策略动作概率比的标准差 |
| `ppo/advantage_mean_raw` | Advantage 归一化之前的均值 |
| `ppo/advantage_std_raw` | Advantage 归一化之前的标准差 |
| `ppo/advantage_positive_fraction` | 原始 Advantage 大于零的样本比例 |
| `ppo/early_stop_fraction` | 因 KL 阈值触发提前停止的 PPO 更新比例 |

</details>

<details class="kt-derivation" id="metric-fields-8">
<summary>价值网络 critic/*</summary>

| 指标 | 含义 |
|---|---|
| `critic/value_mean` | Critic 预测 State Value 的均值 |
| `critic/value_std` | State Value 预测结果的标准差 |
| `critic/target_mean` | Critic 训练目标 Return 的均值 |
| `critic/target_std` | Critic 训练目标 Return 的标准差 |
| `critic/explained_variance` | Critic 对目标 Return 方差的解释能力 |

</details>

<details class="kt-derivation" id="metric-fields-9">
<summary>训练损失 loss/*</summary>

| 指标 | 含义 |
|---|---|
| `loss/actor` | PPO Actor 的策略优化损失 |
| `loss/critic` | Critic 的价值回归损失 |
| `loss/entropy` | Entropy Regularization 对应的损失项 |
| `loss/total` | 实际参与反向传播的总损失 |

</details>

<details class="kt-derivation" id="metric-fields-10">
<summary>优化过程 optimizer/*</summary>

| 指标 | 含义 |
|---|---|
| `optimizer/learning_rate_actor` | Actor 当前学习率 |
| `optimizer/learning_rate_critic` | Critic 当前学习率 |
| `optimizer/grad_norm_actor_preclip` | Actor 参数梯度裁剪前的 L2 范数 |
| `optimizer/grad_norm_critic_preclip` | Critic 参数梯度裁剪前的 L2 范数 |
| `optimizer/grad_clip_fraction` | 当前 Update 中触发梯度裁剪的优化步骤比例 |

</details>

<details class="kt-derivation" id="metric-fields-11">
<summary>数值诊断 diagnostics/*</summary>

| 指标 | 含义 |
|---|---|
| `diagnostics/obs_nonfinite_fraction` | Observation 中 NaN/Inf 元素的比例 |
| `diagnostics/reward_nonfinite_fraction` | Reward 中 NaN/Inf 元素的比例 |
| `diagnostics/action_nonfinite_fraction` | Action 中 NaN/Inf 元素的比例 |
| `diagnostics/optimizer_nonfinite_count` | 当前统计周期检测到非有限 Loss 或 Gradient 的次数 |
| `diagnostics/action_invalid_fraction` | 不满足环境动作有效性约束的动作比例 |
| `diagnostics/action_clip_fraction` | 连续动作经过边界裁剪的比例 |

</details>

<details class="kt-derivation" id="metric-fields-12">
<summary>独立评估 eval/*</summary>

| 指标 | 含义 |
|---|---|
| `eval/return_mean` | 当前评估 Episode 集合的平均累计奖励 |
| `eval/return_std` | 当前评估 Episode 累计奖励的标准差 |
| `eval/length_mean` | 当前评估 Episode 的平均决策步数 |
| `eval/success_rate` | 当前评估 Episode 中成功回合的比例 |
| `eval/episode_count` | 本次评估完成的 Episode 数量 |

</details>

<details class="kt-derivation" id="metric-fields-13">
<summary>训练性能 performance/*</summary>

| 指标 | 含义 |
|---|---|
| `performance/transitions_per_second` | 完整训练迭代对应的全局 Transition 吞吐率 |
| `performance/iteration_time_sec` | 一轮完整训练迭代的墙钟耗时 |
| `performance/rollout_time_sec` | 当前 Rollout 采样阶段耗时 |
| `performance/update_time_sec` | 当前 PPO Update 阶段耗时 |
| `performance/total_training_hours` | 累计训练墙钟时间，单位小时 |

</details>

<details class="kt-derivation" id="metric-fields-14">
<summary>分布式训练 distributed/*</summary>

| 指标 | 含义 |
|---|---|
| `distributed/world_size` | 训练进程 / Rank 数量 |
| `distributed/global_env_count` | 全局环境数量 |
| `distributed/local_env_count` | 当前 Rank 的环境数量 |
| `distributed/rank_iteration_time_min` | 各 Rank 最短迭代耗时 |
| `distributed/rank_iteration_time_max` | 各 Rank 最长迭代耗时 |
| `distributed/rank_rollout_time_min` | 各 Rank 最短 Rollout 耗时 |
| `distributed/rank_rollout_time_max` | 各 Rank 最长 Rollout 耗时 |

</details>

<details class="kt-derivation" id="metric-fields-15">
<summary>硬件资源 resources/*</summary>

| 指标 | 含义 |
|---|---|
| `resources/gpu_utilization` | GPU 利用率 |
| `resources/gpu_memory_used` | GPU 显存占用 |
| `resources/cpu_utilization` | CPU 利用率 |
| `resources/process_rss_memory` | 训练进程 RSS 内存占用 |

</details>

<details class="kt-derivation" id="metric-fields-16">
<summary>课程学习 curriculum/*</summary>

| 指标 | 含义 |
|---|---|
| `curriculum/level` | 当前 Curriculum Level |
| `curriculum/level_transition_count` | 从训练开始累计发生的 Level 切换次数 |
| `curriculum/episodes_since_level_change` | 当前 Level 下已经完成的 Episode 数量 |

</details>

<details class="kt-derivation" id="metric-fields-17">
<summary>网络结构诊断 model/*</summary>

| 指标 | 含义 |
|---|---|
| `model/nonfinite_activation_fraction` | 网络中非有限 Activation 元素的比例 |
| `model/cnn/feature_norm` | CNN 特征范数 |
| `model/transformer/latent_norm` | Transformer 输出特征范数 |
| `model/transformer/attention_entropy` | Attention 分布熵 |
| `model/transformer/valid_token_count` | Transformer 有效 Token 数量 |
| `model/rnn/hidden_norm` | RNN Hidden State 范数 |

</details>

<details class="kt-derivation" id="metric-fields-18">
<summary>任务指标 task/*</summary>

| 指标 | 含义 |
|---|---|
| `task/tracking/position_error` | 位置跟踪误差 |
| `task/tracking/velocity_error` | 速度跟踪误差 |
| `task/obstacle/nearest_distance` | 最近障碍物距离 |
| `task/execution/action_override_rate` | 动作被安全模块覆盖或修改的比例 |

</details>

<details class="kt-derivation" id="metric-fields-20">
<summary>其他算法扩展指标</summary>

| 指标 | 含义 |
|---|---|
| `sac/alpha` | 当前 Entropy Temperature |
| `sac/alpha_loss` | 自动温度调整损失 |
| `sac/target_entropy` | 目标策略熵 |
| `sac/q1_mean` | Q1 网络预测均值 |
| `sac/q2_mean` | Q2 网络预测均值 |
| `sac/td_error_abs_mean` | TD Error 绝对值均值 |

| 指标 | 含义 |
|---|---|
| `dqn/epsilon` | Epsilon-Greedy 探索概率 |
| `dqn/q_selected_mean` | 当前选择动作的 Q 值均值 |
| `dqn/q_target_mean` | TD Target 均值 |
| `dqn/td_error_abs_mean` | TD Error 绝对值均值 |

| 指标 | 含义 |
|---|---|
| `replay/size` | 当前 Replay Buffer 样本数量 |
| `replay/capacity_fraction` | Replay Buffer 当前容量占比 |
| `replay/update_to_data_ratio` | 梯度更新次数相对新增环境 Transition 数的比例 |

</details>

## 5. 分析与总结

训练诊断需要同时观察导航结果与优化过程。单个损失变小、单条奖励上升或 GPU 利用率升高，都只描述闭环的一部分。性能改进同时报告样本效率和墙钟效率，课程变化处单独标记。

其他算法沿同一统计结构扩展：SAC 增加温度、双 Q 与目标值；DQN 增加 TD 误差、探索率和目标网络更新；经验回放记录容量、采样年龄与更新数据比。PPO 主线保持独立。

## 6. 关联与参考

- [PPO 工程走读](./01_ppo-walkthrough.md)。
- [无人机强化学习任务设计](./02_uav-rl-task-design.md)。
- [导航评估方法](/knowledge-tree/simulation-evaluation/evaluation-methodology/navigation-metrics/)。
- [PPO 原论文](https://arxiv.org/abs/1707.06347)。
- [PyTorch TensorBoard 接口](https://docs.pytorch.org/docs/stable/tensorboard.html)。
