---
title: "通用强化学习 TensorBoard 指标模板"
sidebar:
  order: 3
---

**版本：v1.2**

**默认算法：PPO（Proximal Policy Optimization）**

**适用范围：离散动作 / 连续动作、单环境 / 向量化环境、单 GPU / 多 GPU、带或不带 Curriculum Learning 的强化学习任务。**

## 1. 设计原则

1. **通用性**：基础指标不依赖具体任务，任务特异性指标单独扩展。
2. **功能分类**：仅按照统计功能组织，不区分 Core、Recommended 等优先级。
3. **算法基准**：以 PPO 为默认算法，SAC、DQN 等作为扩展。
4. **动作类型**：离散动作与连续动作指标分组，按照实际策略类型选择。
5. **统计口径**：明确每个指标的统计范围、计算方法及重置条件。
6. **分布式一致性**：多 GPU 训练时，任务与回合统计使用全局聚合结果。
7. **统一横轴**：TensorBoard Scalar 默认使用全局累计 Environment Transitions 作为横轴。

---

## 2. Episode：回合表现 `episode/*`

### 2.1 指标清单

| 指标 | 含义 |
|---|---|
| `episode/return_mean` | 最近 K 个完成 Episode 的平均累计奖励 |
| `episode/return_std` | 最近 K 个完成 Episode 的累计奖励标准差 |
| `episode/length_mean` | 最近 K 个完成 Episode 的平均 RL 决策步数 |
| `episode/window_count` | 当前统计窗口中的有效 Episode 数量 |

### 2.2 统计规则

维护一个长度为 K 的 Episode 滑动窗口。

每次 Episode 完成时，将该 Episode 的累计奖励和实际长度加入窗口。

- 窗口未满时，使用现有有效 Episode 统计。
- 窗口满后，新增 Episode 覆盖最旧 Episode。
- Curriculum Level 改变时，清空窗口。
- 窗口为空时，不写入 Return Mean、Return Std 和 Length Mean。
- 分布式训练使用全局完成的 Episode 共同维护窗口。

K 为可配置参数，建议初始使用：

`K = max(100, global_num_envs)`

**注意**：Episode Return 是完整回合奖励之和，不是每步奖励均值。

---

## 3. Termination：终止结果 `termination/*`

采用 **Per-Env Latest Termination** 机制。

### 3.1 状态向量

维护长度为全局环境数量 N 的向量：

`latest_termination[N]`

每个位置对应一个环境，取值定义如下：

| 数值 | 含义 |
|---|---|
| `1` | Success |
| `-1` | Failure |
| `0` | Timeout |
| `-10` | 尚无有效终止记录 |

初始化时全部填充 `-10`。

每个环境完成 Episode 时，仅更新该环境对应位置，其他位置不变。

Curriculum Level 全局切换时，将全部元素重新填充为 `-10`。

### 3.2 指标清单

| 指标 | 含义 |
|---|---|
| `termination/latest_success_rate` | 最近一次终止为 Success 的环境数 / 全局环境数 |
| `termination/latest_failure_rate` | 最近一次终止为 Failure 的环境数 / 全局环境数 |
| `termination/latest_timeout_rate` | 最近一次终止为 Timeout 的环境数 / 全局环境数 |
| `termination/latest_coverage` | 已产生过有效终止记录的环境数 / 全局环境数 |

### 3.3 统计公式

设：

- `Ns`：最近一次终止为 Success 的环境数量
- `Nf`：最近一次终止为 Failure 的环境数量
- `Nt`：最近一次终止为 Timeout 的环境数量
- `N`：全局环境数量

则：

`latest_success_rate = Ns / N`

`latest_failure_rate = Nf / N`

`latest_timeout_rate = Nt / N`

`latest_coverage = (Ns + Nf + Nt) / N`

始终满足：

`latest_success_rate + latest_failure_rate + latest_timeout_rate = latest_coverage`

### 3.4 终止原因扩展

对于需要细分 Failure 类型的任务，可增加：

| 指标 | 含义 |
|---|---|
| `termination/latest_reasons/<name>_rate` | 最近一次终止为指定原因的环境数 / 全局环境数 |

例如：

- `termination/latest_reasons/collision_rate`
- `termination/latest_reasons/out_of_bounds_rate`
- `termination/latest_reasons/numerical_failure_rate`

终止原因使用任务定义的互斥最终原因，避免重复统计同一次终止。

**注意**：

- Latest Termination 记录的是各环境最近一次终止结果的分布，不是最近 K 个 Episode 的经验成功率。
- Coverage 表示终止记录覆盖率，不是环境空间探索覆盖率。
- 采用全局统一 Curriculum Level 时，Level 切换后全部清空。如果各环境拥有独立 Level，则需要按 Level 分开统计。
- PPO 内部仍需保留真正终止 `terminated` 与时间截断 `truncated` 的区别，用于 GAE 和 Value Bootstrap；这里只是取消对应的 TensorBoard Scalar。

---

## 4. Reward：奖励统计 `reward/*`

### 4.1 基础奖励

| 指标 | 含义 |
|---|---|
| `reward/step_mean` | 当前 Rollout 所有 Transition 的平均原始奖励 |
| `reward/step_std` | 当前 Rollout 所有 Transition 的原始奖励标准差 |

### 4.2 奖励分量

通过任务动态注册：

| 指标 | 含义 |
|---|---|
| `reward/components/<name>_step_mean` | 当前 Rollout 中指定奖励分量的每步均值 |

例如：

- `reward/components/progress_step_mean`
- `reward/components/collision_step_mean`
- `reward/components/energy_step_mean`

**统计规则**：

- 所有 Reward 指标使用当前 Rollout 的 Transition 作为统计样本。
- 每次 Rollout 后重新统计，不进行从训练开始的累计平均。
- 多 GPU 时使用全局 Reward 数据聚合。
- 奖励分量名称由具体任务定义。
- 如果使用 Reward Normalization 或 Reward Clipping，必须明确记录的是原始环境奖励还是变换后的奖励。

---

## 5. Rollout：采样统计 `rollout/*`

| 指标 | 含义 |
|---|---|
| `rollout/transitions` | 本次 Rollout 采集的全局 Transition 数量 |

该模块仅保留采样量，不重复记录 Episode 完成数及终止比例。

---

## 6. Policy：策略分布 `policy/*`

根据 PPO Actor 的动作空间类型，选择以下两种分组之一。

### 6.1 离散动作策略（Discrete Action）

适用于 Categorical Policy，例如在多个候选动作中选择一个。

| 指标 | 含义 |
|---|---|
| `policy/discrete/entropy` | 当前 Categorical 分布的信息熵，单位 nats |
| `policy/discrete/normalized_entropy` | 当前离散动作分布的归一化熵 |
| `policy/discrete/max_probability` | 每个样本最大动作概率的均值 |
| `policy/discrete/logits_std` | 每个样本有效动作 Logits 的标准差，再对样本求均值 |

归一化熵定义：

`normalized_entropy = entropy / ln(valid_action_count)`

当有效动作数量为 1 时，约定归一化熵为 0。

如果存在 Action Mask，应针对各样本的有效动作集合计算。

### 6.2 连续动作策略（Continuous Action）

适用于 Gaussian Policy、Squashed Gaussian Policy 等。

| 指标 | 含义 |
|---|---|
| `policy/continuous/entropy` | 当前连续动作分布的微分熵 |
| `policy/continuous/action_std_mean` | 策略输出动作分布标准差的均值 |
| `policy/continuous/action_std_min` | 策略输出动作分布标准差的最小值 |
| `policy/continuous/action_std_max` | 策略输出动作分布标准差的最大值 |
| `policy/continuous/action_saturation_fraction` | 实际执行动作接近动作上下限的比例 |

对于高斯策略，动作标准差的统计应基于策略分布本身，而不是仅对执行动作求经验标准差。

如果使用 Tanh Squashing，需明确 Entropy 和 Std 是针对变换前还是变换后的动作分布。

### 6.3 两种策略通用

| 指标 | 含义 |
|---|---|
| `policy/log_prob_mean` | 采样动作在当前策略下的平均 Log Probability |

**说明**：

- 离散动作与连续动作只启用对应策略分组。
- Entropy 与 Normalized Entropy 并不是互斥指标；离散策略可以同时记录二者。
- 连续动作不采用离散动作的 `ln(valid_action_count)` 归一化方式。
- 若使用混合动作空间，需要分别统计各动作分支。

---

## 7. PPO：策略更新诊断 `ppo/*`

### 7.1 指标清单

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

### 7.2 统计规则

PPO Probability Ratio：

`ratio = exp(new_log_prob - old_log_prob)`

对应裁剪区间：

`[1 - clip_epsilon, 1 + clip_epsilon]`

- KL、Clip Fraction、Ratio 等指标在 PPO Update 阶段统计。
- 根据当前 Update 的有效 Minibatch 样本聚合。
- Advantage 统计使用归一化前的数据。
- `early_stop_fraction` 仅在实现 KL Early Stopping 时启用。
- 近似 KL 的具体估计公式应固定，并在代码中保持一致。

---

## 8. Critic：价值网络 `critic/*`

### 8.1 指标清单

| 指标 | 含义 |
|---|---|
| `critic/value_mean` | Critic 预测 State Value 的均值 |
| `critic/value_std` | State Value 预测结果的标准差 |
| `critic/target_mean` | Critic 训练目标 Return 的均值 |
| `critic/target_std` | Critic 训练目标 Return 的标准差 |
| `critic/explained_variance` | Critic 对目标 Return 方差的解释能力 |

### 8.2 Explained Variance

定义：

`EV = 1 - Var(Return - Value) / Var(Return)`

通常：

- 接近 1：能够解释较多目标变化。
- 接近 0：预测解释能力较弱。
- 小于 0：误差方差大于目标方差。

当目标 Return 方差接近零时，Explained Variance 可能不稳定或无定义，应跳过该次记录。

Value Mean 本身不是 Critic 准确率，需结合 Value Target、Value Loss 和 Explained Variance 分析。

---

## 9. Loss：训练损失 `loss/*`

| 指标 | 含义 |
|---|---|
| `loss/actor` | PPO Actor 的策略优化损失 |
| `loss/critic` | Critic 的价值回归损失 |
| `loss/entropy` | Entropy Regularization 对应的损失项 |
| `loss/total` | 实际参与反向传播的总损失 |

统计使用当前 PPO Update 中的有效 Minibatch。

应统一损失项的正负号约定，并明确 `loss/entropy` 是否已经乘以熵系数。

**注意**：Actor Loss、Critic Loss、Entropy Loss 的绝对值大小不能直接作为相互比较的依据。

---

## 10. Optimizer：优化过程 `optimizer/*`

| 指标 | 含义 |
|---|---|
| `optimizer/learning_rate_actor` | Actor 当前学习率 |
| `optimizer/learning_rate_critic` | Critic 当前学习率 |
| `optimizer/grad_norm_actor_preclip` | Actor 参数梯度裁剪前的 L2 范数 |
| `optimizer/grad_norm_critic_preclip` | Critic 参数梯度裁剪前的 L2 范数 |
| `optimizer/grad_clip_fraction` | 当前 Update 中触发梯度裁剪的优化步骤比例 |

### 10.1 梯度统计规则

推荐在执行 Gradient Clipping 之前记录梯度范数。

如果使用：

`clip_grad_norm_(parameters, max_norm)`

应使用其返回的裁剪前 Total Norm 或等效实现。

对于共享 Actor-Critic Backbone 的网络，可以额外记录：

`optimizer/grad_norm_shared_preclip`

如果 Actor 与 Critic 共享优化器，则可以将学习率和梯度监控改为实际使用的参数分组，避免重复统计。

---

## 11. Diagnostics：通用数值诊断 `diagnostics/*`

| 指标 | 含义 |
|---|---|
| `diagnostics/obs_nonfinite_fraction` | Observation 中 NaN/Inf 元素的比例 |
| `diagnostics/reward_nonfinite_fraction` | Reward 中 NaN/Inf 元素的比例 |
| `diagnostics/action_nonfinite_fraction` | Action 中 NaN/Inf 元素的比例 |
| `diagnostics/optimizer_nonfinite_count` | 当前统计周期检测到非有限 Loss 或 Gradient 的次数 |
| `diagnostics/action_invalid_fraction` | 不满足环境动作有效性约束的动作比例 |
| `diagnostics/action_clip_fraction` | 连续动作经过边界裁剪的比例 |

其中：

- `action_invalid_fraction` 仅适用于具有显式动作有效性约束的任务。
- `action_clip_fraction` 适用于存在动作裁剪的连续控制任务。
- 对 NaN/Inf 不应只记录比例，还应配合训练异常处理。

---

## 12. Evaluation：独立评估 `eval/*`

| 指标 | 含义 |
|---|---|
| `eval/return_mean` | 当前评估 Episode 集合的平均累计奖励 |
| `eval/return_std` | 当前评估 Episode 累计奖励的标准差 |
| `eval/length_mean` | 当前评估 Episode 的平均决策步数 |
| `eval/success_rate` | 当前评估 Episode 中成功回合的比例 |
| `eval/episode_count` | 本次评估完成的 Episode 数量 |

### 12.1 评估规则

- 使用独立于训练采样的评估流程。
- 固定评估 Episode 数量。
- 固定评估随机种子集合、任务分布及环境配置。
- 明确使用 Deterministic 还是 Stochastic Policy。
- Curriculum 任务应明确评估的是当前 Level，还是跨 Level 固定测试集。
- 无任务成功定义时，不记录 `eval/success_rate`。

这里的评估成功率与 `termination/latest_success_rate` 是不同统计口径。

---

## 13. Performance：训练性能 `performance/*`

| 指标 | 含义 |
|---|---|
| `performance/transitions_per_second` | 完整训练迭代对应的全局 Transition 吞吐率 |
| `performance/iteration_time_sec` | 一轮完整训练迭代的墙钟耗时 |
| `performance/rollout_time_sec` | 当前 Rollout 采样阶段耗时 |
| `performance/update_time_sec` | 当前 PPO Update 阶段耗时 |
| `performance/total_training_hours` | 累计训练墙钟时间，单位小时 |

统一定义：

`transitions_per_second = global_transitions_this_iteration / iteration_wall_time_sec`

其中完整迭代时间应包含采样、PPO 更新与必要同步开销。

---

## 14. Distributed：分布式训练 `distributed/*`

| 指标 | 含义 |
|---|---|
| `distributed/world_size` | 训练进程 / Rank 数量 |
| `distributed/global_env_count` | 全局环境数量 |
| `distributed/local_env_count` | 当前 Rank 的环境数量 |
| `distributed/rank_iteration_time_min` | 各 Rank 最短迭代耗时 |
| `distributed/rank_iteration_time_max` | 各 Rank 最长迭代耗时 |
| `distributed/rank_rollout_time_min` | 各 Rank 最短 Rollout 耗时 |
| `distributed/rank_rollout_time_max` | 各 Rank 最长 Rollout 耗时 |

单进程训练不需要该模块。

多进程训练应正确处理跨 Rank 的统计聚合，而不是简单平均各 Rank 的统计结果。

---

## 15. Resources：硬件资源 `resources/*`

| 指标 | 含义 |
|---|---|
| `resources/gpu_utilization` | GPU 利用率 |
| `resources/gpu_memory_used` | GPU 显存占用 |
| `resources/cpu_utilization` | CPU 利用率 |
| `resources/process_rss_memory` | 训练进程 RSS 内存占用 |

多 GPU 时建议按 Rank 分别记录，也可以增加全局最大值与均值。

---

## 16. Curriculum：课程学习 `curriculum/*`

| 指标 | 含义 |
|---|---|
| `curriculum/level` | 当前 Curriculum Level |
| `curriculum/level_transition_count` | 从训练开始累计发生的 Level 切换次数 |
| `curriculum/episodes_since_level_change` | 当前 Level 下已经完成的 Episode 数量 |

### 16.1 Level 切换规则

当全局 Curriculum Level 变化时：

1. 重置 `latest_termination` 向量为 `-10`。
2. 清空 Episode Return / Length 滑动窗口。
3. 清空与旧 Level 绑定的其他临时 Episode 统计。
4. 更新 `curriculum/level`。
5. 增加 `curriculum/level_transition_count`。
6. 将 `curriculum/episodes_since_level_change` 重新置零。

如果环境可以在不同 Level 下独立运行，应使用按 Level 分组的统计方式，不直接混合不同难度的数据。

---

## 17. Model：网络结构诊断 `model/*`

根据实际网络结构添加。

| 指标 | 含义 |
|---|---|
| `model/nonfinite_activation_fraction` | 网络中非有限 Activation 元素的比例 |
| `model/cnn/feature_norm` | CNN 特征范数 |
| `model/transformer/latent_norm` | Transformer 输出特征范数 |
| `model/transformer/attention_entropy` | Attention 分布熵 |
| `model/transformer/valid_token_count` | Transformer 有效 Token 数量 |
| `model/rnn/hidden_norm` | RNN Hidden State 范数 |

这一模块不属于 PPO 特有统计，按网络结构扩展。

---

## 18. Task：任务特异性指标 `task/*`

新项目仅需在该模块添加任务相关指标。

例如无人机轨迹规划任务：

| 指标 | 含义 |
|---|---|
| `task/tracking/position_error` | 位置跟踪误差 |
| `task/tracking/velocity_error` | 速度跟踪误差 |
| `task/obstacle/nearest_distance` | 最近障碍物距离 |
| `task/execution/action_override_rate` | 动作被安全模块覆盖或修改的比例 |

如果任务具有候选动作、评分融合或轨迹历史机制，可继续增加：

- `task/candidate/*`
- `task/score/*`
- `task/history/*`
- `task/execution/*`

通用指标系统不对 Task 模块的内部结构施加限制。

---

## 19. TensorBoard 记录规范

### 19.1 统一横轴

所有 Scalar 使用：

`global_transitions`

定义：

`global_transitions = cumulative_environment_decision_steps`

即累计产生的全局环境 Transition 总数，而非 PPO Epoch 或 Optimizer Step。

并行环境下，必须累计所有环境实际产生的 Transition。

### 19.2 各指标统计周期

| 指标分组 | 统计范围或更新周期 |
|---|---|
| `episode/*` | 最近 K 个已完成 Episode |
| `termination/*` | 每个 Env 最近一次终止结果 |
| `reward/*` | 当前 Rollout |
| `rollout/*` | 当前 Rollout |
| `policy/*` | 当前采样观测或指定诊断批次 |
| `ppo/*` | 当前 PPO Update |
| `critic/*` | 当前指定诊断批次或 Update 数据 |
| `loss/*` | 当前 PPO Update |
| `optimizer/*` | 当前 PPO Update |
| `diagnostics/*` | 当前相应统计周期 |
| `eval/*` | 当前完整 Evaluation |
| `performance/*` | 当前训练迭代 |
| `distributed/*` | 当前训练迭代 |
| `resources/*` | 固定墙钟采样周期 |
| `curriculum/*` | Level 变化时及定期记录 |
| `model/*` | 指定网络诊断周期 |
| `task/*` | 任务自定义 |

### 19.3 分布式聚合

涉及 Episode、Termination、Reward 等全局统计时：

- Count：跨 Rank 求和。
- Rate：全局分子求和后除以全局分母。
- Mean：使用全局样本数量进行加权。
- Std：根据全局样本数、总和、平方和或等效充分统计量计算。
- Min / Max：跨 Rank 取对应极值。
- Episode 滑动窗口：统一收集全局完成回合。
- Latest Termination：各 Rank 保存本地 Env 状态，汇总全局计数。

Policy / Critic 的即时诊断若只使用 Rank 0，需要明确注明，不能解释成全局策略统计。

### 19.4 数值有效性

- 没有有效样本时，不强行写入零。
- 避免将无定义统计直接写成 NaN。
- 对于可选指标，未启用的模块不产生对应 Scalar。
- 相同 TensorBoard Tag 在不同任务中应保持相同定义。
- 显式说明数值单位、Reduction 方式及统计范围。

---

## 20. 其他算法扩展指标

以下指标仅作为未来从 PPO 切换到其他 RL 算法时的参考，不属于当前 PPO 基础模板。

### 20.1 SAC

| 指标 | 含义 |
|---|---|
| `sac/alpha` | 当前 Entropy Temperature |
| `sac/alpha_loss` | 自动温度调整损失 |
| `sac/target_entropy` | 目标策略熵 |
| `sac/q1_mean` | Q1 网络预测均值 |
| `sac/q2_mean` | Q2 网络预测均值 |
| `sac/td_error_abs_mean` | TD Error 绝对值均值 |

### 20.2 DQN

| 指标 | 含义 |
|---|---|
| `dqn/epsilon` | Epsilon-Greedy 探索概率 |
| `dqn/q_selected_mean` | 当前选择动作的 Q 值均值 |
| `dqn/q_target_mean` | TD Target 均值 |
| `dqn/td_error_abs_mean` | TD Error 绝对值均值 |

### 20.3 Replay Buffer

SAC、DQN、TD3 等离策略算法可增加：

| 指标 | 含义 |
|---|---|
| `replay/size` | 当前 Replay Buffer 样本数量 |
| `replay/capacity_fraction` | Replay Buffer 当前容量占比 |
| `replay/update_to_data_ratio` | 梯度更新次数相对新增环境 Transition 数的比例 |

---

## 21. 新项目使用规则

在新的 RL 训练项目中，按照以下顺序配置：

1. 启用 Episode、Termination、Reward、Rollout、Performance 和 Diagnostics。
2. 根据动作空间类型选择 Discrete Policy 或 Continuous Policy。
3. 使用 PPO 时启用 PPO、Critic、Loss 和 Optimizer。
4. 存在独立评估流程时启用 Evaluation。
5. 多 GPU 训练时启用 Distributed 和 Resources。
6. 使用 Curriculum Learning 时启用 Curriculum，并同步处理统计重置。
7. 根据网络结构添加 Model 指标。
8. 根据实际任务添加 Task 指标和 Reward Components。
9. 若更换算法，则在最后的算法扩展指标基础上替换 PPO 专属模块。

**最终原则：通用统计定义保持稳定，算法模块按需切换，任务指标独立扩展。**