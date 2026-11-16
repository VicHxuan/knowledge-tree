# POMDP 与循环策略

## 1. 概述

部分可观测决策中，单次观测只包含环境状态的一部分信息。循环策略用历史更新内部记忆，再根据记忆选择动作。策略学习因此同时优化历史表示与动作分布，价值估计也需要与可用信息对齐。

## 2. 基础与定义

### 2.1 状态与观测

POMDP 在状态转移 $p(s'\mid s,a)$ 与奖励机制之外，引入观测分布 $\Omega(o\mid s)$。智能体获得 $o_t$，其决策历史记为：

$$
\mathcal H_t=(o_0,a_0,o_1,a_1,\ldots,a_{t-1},o_t).
$$

动作取决于历史的策略写为 $\pi(a_t\mid\mathcal H_t)$。有限时域任务若以剩余时间决定最优决策，应将时间信息纳入状态定义。

### 2.2 信念状态

信念状态是给定历史的状态后验：

$$
b_t(s)=p(s_t=s\mid\mathcal H_t).
$$

已知模型时可通过贝叶斯滤波更新信念；循环网络则学习历史的有限维表示 $h_t$，用于近似决策所需的信息。

<details class="kt-derivation" id="pomdp-belief-update">
<summary>信念状态递推</summary>

在观测仅依赖新状态的模型下：

$$
b_{t+1}(s')=
\frac{\Omega(o_{t+1}\mid s')\sum_s p(s'\mid s,a_t)b_t(s)}
{\sum_{\bar s}\Omega(o_{t+1}\mid\bar s)\sum_s p(\bar s\mid s,a_t)b_t(s)}.
$$

先用转移模型预测状态分布，再根据新观测更新概率。若奖励本身也携带状态信息，则历史和观测更新应同时包含该奖励。

</details>

## 3. 核心原理

### 3.1 历史表示与网络结构

循环 Actor 可表示为：

$$
h_t=f_\psi(h_{t-1},o_t,a_{t-1}),\qquad
\pi_\theta(a_t\mid h_t).
$$

$\psi$ 为记忆网络参数，$\theta$ 为策略输出头参数。Critic 可以使用同一历史表示，或用独立循环网络构造 $V_\phi(\mathcal H_t)$。

帧堆叠保留固定长度的原始历史，循环网络通过递推压缩历史。二者的效果取决于历史是否包含所需信息，以及表示能否保留该信息。

### 3.2 循环 PPO 的策略损失

采样时保存旧策略对实际动作的对数概率。更新时，从序列起点沿原观测与动作历史重新展开当前网络，得到 $h_t^{\rm new}$。概率比率为：

$$
\rho_t=
\frac{\pi_\theta(a_t\mid h_t^{\rm new})}
{\pi_{\theta_{\rm old}}(a_t\mid h_t^{\rm old})}.
$$

给定固定优势 $\hat A_t$，采用 PPO 截断目标：

$$
L_\pi=-\hat{\mathbb E}_{t\in\mathcal V}
\left[\min(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\right].
$$

$\mathcal V$ 为有效时间步集合，排除 padding。当前记忆状态依赖序列内的网络参数，因此梯度沿时间展开传播。

<details class="kt-derivation" id="pomdp-recurrent-gradient">
<summary>时间展开与梯度截断</summary>

对序列内部的记忆状态：

$$
\frac{\partial h_t}{\partial\psi}
=\frac{\partial f_\psi}{\partial\psi}
+\frac{\partial f_\psi}{\partial h_{t-1}}
\frac{\partial h_{t-1}}{\partial\psi}.
$$

策略损失经 $\partial L/\partial h_t$ 更新记忆参数。截断时间反向传播在序列起点固定输入记忆，仅沿当前片段求导。使用旧策略保存的起点记忆是一种近似；重新展开历史或使用预热片段可缓解参数变化造成的起点记忆偏差，同时增加计算量。

</details>

### 3.3 价值估计与边界

历史价值定义为 $V^\pi(\mathcal H_t)=\mathbb E[G_t\mid\mathcal H_t]$。用旧价值构造 TD 残差、GAE 和固定回报目标，再以均方误差训练 Critic。

环境回合终止或重置时重置对应记忆；仅划分训练序列时，保留该位置的历史条件。外部截断从重置前的观测与记忆自举，GAE 累加在回合边界结束。

<details class="kt-derivation" id="pomdp-privileged-critic">
<summary>历史价值与特权信息</summary>

训练时可以为 Critic 提供额外状态信息，以改善价值估计。循环策略的未来动作同时依赖环境状态与内部记忆，因此仅给环境状态的 Critic 仍可能缺少预测未来策略所需的信息。设计估计目标时应明确条件变量，例如 $V^\pi(s_t,h_t)$。

Actor 的推理路径使用部署时可用的观测和记忆。独立参数下，固定优势作为策略梯度权重；共享特征时，价值损失还会直接更新共享表示。

</details>

## 4. 实现与应用

### 4.1 序列训练流程

1. 采集观测、动作、旧动作概率、价值和每个环境的记忆状态。
2. 按环境时间线计算优势与回报目标。
3. 按连续片段构造批次，保存序列起点记忆与回合边界。
4. 展开当前网络，使用有效样本掩码计算损失与统计量。
5. 反向传播并更新参数，采样时持续维护各环境的记忆。

### 4.2 时序验证

<details class="kt-derivation" id="pomdp-sequence-checks">
<summary>序列与掩码验证</summary>

- 相同参数和相同起点记忆下，批量序列展开应复现逐步前向的动作概率。
- 只重置结束回合的环境槽位，其他环境的记忆连续。
- padding 对策略损失、价值损失、熵及优势统计量的贡献为零。
- 区分序列截断位置与环境终止标记，并检查末状态的观测和记忆配对。
- 打乱完整序列的顺序，序列内部保持时间顺序。

</details>

## 5. 分析与总结

部分可观测性决定策略需要利用哪些历史信息，循环网络决定如何表示历史，序列训练决定梯度能传播多远。比较帧堆叠与循环策略时，应共同检查可用历史、记忆状态偏差和训练序列长度。

## 6. 关联与参考

### 6.1 相关知识

- [强化学习基础](./01_reinforcement-learning-foundations.md)：状态、价值与策略目标。
- [PPO](../02_model-free-rl/03_ppo.md)：概率比率、GAE 和固定训练目标。
- [循环神经网络](../../01_neural-network-architectures/03_rnn-lstm.md)：记忆结构。
- [多价值网络](./05_multi-critic.md)：估计对象与梯度路径。

### 6.2 参考文献

1. Kaelbling, Littman & Cassandra, [Planning and Acting in Partially Observable Stochastic Domains](https://doi.org/10.1016/S0004-3702(98)00023-X), 1998。
2. Hausknecht & Stone, [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527), 2015。
3. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017。
4. Stable-Baselines3 Contrib, [Recurrent PPO 实现](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py)：循环状态与序列训练。
