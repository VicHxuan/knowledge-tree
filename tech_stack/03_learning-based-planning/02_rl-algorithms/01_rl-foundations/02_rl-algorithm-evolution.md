# 强化学习算法演进与梳理

## 1. 概述

现代强化学习沿价值估计、策略优化与模型利用三条路线发展。各方法通过改变学习对象、训练信号或数据使用方式，处理估计误差、策略更新稳定性与交互成本。

## 2. 基础与定义

### 2.1 学习对象

| 对象 | 参数化形式 | 主要训练信号 |
| --- | --- | --- |
| 状态或动作价值 | $V_\phi(s),Q_\phi(s,a)$ | 采样回报或自举目标 |
| 策略 | $\pi_\theta(a\mid s),\mu_\theta(s)$ | 优势或动作价值梯度 |
| 环境模型 | $\hat p_\omega(s',r\mid s,a)$ | 状态转移与奖励数据 |

### 2.2 数据与目标

同策略方法围绕当前采样策略构造更新；离策略方法允许复用其他行为策略的数据。模型方法还需说明数据如何训练预测模型，以及预测结果如何进入策略优化。

## 3. 核心原理

### 3.1 蒙特卡洛策略梯度

REINFORCE 用采样回报加权动作对数概率的梯度。对有限回合的初始状态折扣目标：

$$
\hat g=\sum_{t=0}^{T-1}\gamma^tG_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

Williams 在 1992 年给出了这一类随机梯度算法。状态基线和价值估计随后为降低梯度方差提供了方法。

### 3.2 策略梯度与 Actor-Critic

策略梯度定理将优化方向写成策略对数梯度与动作价值的期望。Sutton 等在 NIPS 1999 的工作分析了函数近似下的兼容性条件。Actor-Critic 将策略更新与价值评估结合，用 TD 或多步估计提供训练信号。

<details class="kt-derivation" id="evolution-compatible-critic">
<summary>兼容函数近似</summary>

兼容近似器采用与策略得分函数一致的特征，例如：

$$
f_w(s,a)=w^\top\nabla_\theta\log\pi_\theta(a\mid s).
$$

在相应访问分布下满足加权平方误差最优条件时，正交残差使该近似器可代入策略梯度。该结论同时依赖特征形式和拟合条件；一般神经网络 Critic 的价值误差仍会影响梯度估计。

</details>

### 3.3 深度价值学习

DQN 的 2013 年预印本与 2015 年 Nature 论文将深度动作价值网络用于像素控制。经验回放复用数据，目标网络降低自举目标的快速漂移。

Rainbow 在 2017 年预印本中组合 Double DQN、优先回放、Dueling、多步目标、回报分布建模和参数化噪声，随后发表于 AAAI 2018。回报分布建模的对象是回报随机变量的分布。

<details class="kt-derivation" id="evolution-rainbow">
<summary>价值学习的改进机制</summary>

| 机制 | 改变的位置 |
| --- | --- |
| Double DQN | 下一动作选择与价值评估的网络 |
| 优先回放 | 样本抽取概率及重要性权重 |
| Dueling | 状态价值与动作优势的表示 |
| 多步目标 | 实际奖励与自举的时间范围 |
| 回报分布建模 | 从期望标量扩展到回报分布 |
| Noisy Networks | 参数化探索噪声 |

Rainbow 用消融实验分析组合效果；各组件的收益取决于任务、数据与其他组件的配合。

</details>

### 3.4 信赖域与策略截断

TRPO 在 2015 年将局部策略改进与 KL 约束结合，通过自然梯度方向和线搜索选择更新。理论改进界在其假设下成立，实用算法使用平均 KL 与有限样本近似。

PPO 在 2017 年给出便于一阶优化的替代目标，其中 PPO-Clip 为：

$$
L^{\rm CLIP}=\hat{\mathbb E}_t\left[\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

$\rho_t$ 为新旧策略对同一动作的概率比率。截断使部分样本的有利更新进入平坦区，实际策略变化仍需由 KL 等量诊断。

### 3.5 并行采样与优势估计

A3C 在 2016 年使用多个环境工作线程异步更新 Actor 与 Critic，结合多步回报和熵奖励。各工作线程的梯度基于其本地策略版本，异步更新可能产生参数滞后。

GAE 通过加权 TD 残差调节优势估计的采样范围。并行采样决定数据吞吐，优势估计决定策略梯度信号，两种机制可分别选择。

### 3.6 连续动作与最大熵学习

确定性策略梯度在 2014 年给出沿动作价值对动作求导的策略更新。DDPG 将其与深度网络、回放及目标网络结合；TD3 在 2018 年通过双 Q、目标动作平滑和延迟更新控制近似误差。

SAC 在 2018 年发展最大熵离策略 Actor-Critic。常用现代实现采用随机 Actor、双 Q 和自动温度调节，Actor 同时优化动作价值与熵。

### 3.7 策略扰动

RPO 的 2022 年预印本研究在 PPO 中引入策略分布扰动。其参考实现于更新阶段评估已有动作时扰动高斯均值，再计算策略概率和损失。扰动位置直接影响概率比率的含义。[参考实现说明](https://cleanrl.vercel.app/rl-algorithms/rpo/)

<details class="kt-derivation" id="evolution-rpo">
<summary>更新阶段的均值扰动</summary>

对当前均值 $\mu_\theta(s)$ 加入均匀扰动：

$$
z\sim\mathcal U(-\alpha,\alpha),\qquad
\widetilde\mu_\theta(s)=\mu_\theta(s)+z.
$$

更新时用扰动后的分布评估采样动作，旧动作概率仍来自采样策略。这改变了策略替代目标的估计过程，扰动范围是新增超参数。其经验效果需结合任务和预算评价。

</details>

### 3.8 模型学习与规划

| 方法 | 发表节点 | 模型用途 |
| --- | --- | --- |
| Dyna | 1990—1991 | 真实经验与模型生成经验共同更新价值 |
| PILCO | ICML 2011 | 高斯过程动力学与不确定性传播 |
| PETS | NeurIPS 2018 | 概率模型集成与轨迹采样规划 |
| PlaNet | ICML 2019 | 潜在动力学与在线规划 |
| MBPO | NeurIPS 2019 | 短模型轨迹训练离策略学习器 |
| MuZero | 预印本 2019，Nature 2020 | 价值相关的潜在模型与树搜索 |
| Dreamer | ICLR 2020 起 | 潜在想象轨迹上的行为学习 |

模型预测、价值自举和策略优化可共同组成一个系统，具体训练路径见[有模型与无模型强化学习](./03_model-free-vs-model-based.md)。

## 4. 实现与应用

### 4.1 阅读路径

先用[基础手册](./01_reinforcement-learning-foundations.md)统一回报和梯度，再按学习对象进入：

- 离散动作价值学习：[DQN](../02_model-free-rl/01_dqn.md)。
- 采样策略优化：[TRPO](../02_model-free-rl/02_trpo.md) → [PPO](../02_model-free-rl/03_ppo.md)。
- 连续动作价值梯度：[DDPG](../02_model-free-rl/04_ddpg.md) → [TD3](../02_model-free-rl/05_td3.md)，再对照 [SAC](../02_model-free-rl/06_sac.md)。
- 估计与结构扩展：[多价值网络](./05_multi-critic.md)、[循环策略](./04_pomdp-recurrent-policies.md)。

### 4.2 方法对照

比较两个算法时，分别记录网络输出、采样策略、训练数据、目标构造、梯度路径和参数更新时间。这样可以定位差异来自目标、估计器还是优化流程。

## 5. 分析与总结

价值学习围绕目标估计与数据复用展开，策略优化围绕梯度信号与更新幅度展开，模型方法围绕预测计算与真实交互的配合展开。算法的历史关系提供理解入口，具体适用性由任务与实现条件决定。

## 6. 关联与参考

### 6.1 相关知识

[基础手册](./01_reinforcement-learning-foundations.md)、[模型方法分类](./03_model-free-vs-model-based.md)与各算法专项文章共同构成理论阅读路径。

### 6.2 参考文献

1. Williams, [REINFORCE](https://doi.org/10.1007/BF00992696), 1992。
2. Sutton et al., [Policy Gradient Methods with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html), 1999。
3. Mnih et al., [DQN](https://www.nature.com/articles/nature14236), 2015；[A3C](https://proceedings.mlr.press/v48/mniha16.html), 2016。
4. Schulman et al., [TRPO](https://proceedings.mlr.press/v37/schulman15.html), 2015；[PPO](https://arxiv.org/abs/1707.06347), 2017。
5. Hessel et al., [Rainbow](https://arxiv.org/abs/1710.02298), 2017。
6. Haarnoja et al., [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html), 2018。
7. Rahman & Xue, [Robust Policy Optimization](https://arxiv.org/abs/2212.07536), 2022。
8. Schrittwieser et al., [MuZero](https://www.nature.com/articles/s41586-020-03051-4), 2020。
9. Hafner et al., [Dreamer](https://arxiv.org/abs/1912.01603), 2019。
