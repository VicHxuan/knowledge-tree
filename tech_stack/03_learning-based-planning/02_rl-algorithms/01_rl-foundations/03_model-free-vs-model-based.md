# 有模型与无模型强化学习

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

本篇是强化学习算法分支的**范式导航**：澄清 Model-Free 与 Model-Based 两条主线的本质区别，梳理各自的内部结构与典型用法，并把常与它混淆的「On/Off-Policy」维度剥离出去。读完本篇应能回答三个问题——「这里的『模型』到底指什么」「两条范式各自拿模型做什么」「给定任务该从哪条主线入手」。本篇只做导航与辨析，不重复具体算法（PPO、SAC、Dreamer、MBPO 等）的推导，推导见对应专项文章。

## 2. 基础与定义

### 2.1 环境转移模型

强化学习语境下的**模型（model）**特指智能体**从交互数据中学习得到**的环境转移规律预测器：

$$\hat p(s', r \mid s, a)$$

三条边界澄清：

- **模型是「学出来的」，不是环境本身。** 环境可以是物理仿真器、真实机器人或游戏引擎；只要智能体没有从数据中学习其转移规律并用于计算，算法就不是 Model-Based。
- **Model-Free 不是「没有环境模型可用」，而是「不在算法内学习、使用转移模型」。** 只用物理仿真器采样训练的 PPO，仍然是标准的 Model-Free 算法——仿真器是环境，不是模型。
- **经典 MPC 不一定属于 RL。** 若控制器直接使用已知的解析动力学方程（而非从数据学习），它通常被归为最优控制；界线在于模型是否**从数据中学习**。

### 2.2 有模型与无模型方法

- **Model-Free**：交互样本直接用于值函数或策略的估计与更新。样本可以被回放池复用（Off-Policy），但不会被任何学到的转移模型改写或外推。
- **Model-Based**：先（或同时）从样本学习 $\hat p$，再把 $\hat p$ 作为计算的中间层——用于在线规划、生成合成经验，或作为策略梯度的传导介质。

## 3. 核心原理

### 3.1 样本效率与模型误差

| 维度 | Model-Free | Model-Based |
| :--- | :--- | :--- |
| 样本效率 | 低——信息只来自真实交互 | 高——模型可无限采样外推 |
| 额外误差来源 | 函数近似误差 | 函数近似误差 + **模型误差** |
| 计算结构 | 直接、成熟稳定 | 多出「学模型 + 用模型」两级 |
| 典型代价 | 环境交互量大 | 模型偏差可能被策略利用（model exploitation） |

两条主线的一切算法设计，本质都是在各自一侧的格子里做工程取舍。

### 3.2 无模型方法分类

- **Value-Based**：学 $Q(s,a)$ 或 $V(s)$，策略隐式从值导出（$\epsilon$-greedy / argmax）——🚧 值函数方法与 DQN（撰写中）；
- **Policy-Based**：直接参数化并优化策略——REINFORCE 到 TRPO；
- **Actor-Critic**：策略与值函数联合学习——[PPO](../02_model-free-rl/04_ppo.md)、[SAC](../02_model-free-rl/07_sac.md)。

**红线澄清**：Model-Free ≠ 不用 Critic。Critic 是**值函数**，不是转移模型——它评估「当前策略下状态/动作的好坏」，不预测「环境下一步给什么」。Actor-Critic 是无模型的：它 critic 的是策略，不是环境。

### 3.3 模型使用方式

同一个学到的 $\hat p$，有三种被消费的方式，对应三族算法：

1. **在线规划（模型当「模拟器」用）**：每次决策在 $\hat p$ 上做滚动时域优化（CEM/MPC）——🚧 PETS（撰写中）；
2. **生成合成经验（模型当「数据增强」用）**：模型 rollout 出的转移混入回放池，交给 model-free 学习器消化——🚧 Dyna（撰写中）、🚧 MBPO（撰写中）；
3. **想象中直接优化策略（模型当「梯度通路」用）**：策略梯度穿过（冻结的）模型反传——🚧 Dreamer（撰写中）；🚧 MuZero（撰写中）介于规划与蒸馏之间。

**红线澄清**：Model-Based ≠ 必须通过动力学反传——MBPO 全程不反传；Model-Based ≠ 必须是神经网络世界模型——Dyna 的表格模型、PILCO 的高斯过程同样成立。

### 3.4 策略数据类型与模型分类

- **On/Off-Policy** 回答：「更新用的数据由谁产生」——当前策略产生（On），或更早/其他策略产生（Off，含回放池）；
- **MB/MF** 回答：「更新是否依赖学到的转移模型」。

四个象限里都有算法：Model-Free + Off-Policy（DQN、TD3、SAC），Model-Free + On-Policy（REINFORCE、TRPO、PPO）；Model-Based 各族的学习器继承其数据消费方式——MBPO 用回放池（Off），Dreamer 在当前策略的想象轨迹上更新（≈On）。因此「Model-Based 一定是 Off-Policy」「Actor-Critic 只属于无模型路线」都是错觉：**Actor-Critic 在两条路线都出现**（SAC 与 Dreamer 的 actor-critic 结构同构），两个维度可以自由组合。

### 3.5 方法选择

| 场景特征 | 建议起点 | 对应文章 |
| :--- | :--- | :--- |
| 环境交互廉价、可大规模并行 | Model-Free 直取 | [PPO](../02_model-free-rl/04_ppo.md) |
| 环境交互昂贵、样本预算紧 | 模型造数据 | 🚧 MBPO（撰写中） |
| 高维观测（像素）、需要通用性 | 潜在世界模型 | 🚧 Dreamer（撰写中） |
| 离散决策、长时程信用分配 | 模型内搜索 | 🚧 MuZero（撰写中） |
| 环境可微、任务简单 | 直接反传 | 🚧 可微动力学与直接策略优化（撰写中） |

## 4. 实现与应用

### 4.1 实现条件

1. 先问环境交互的真实成本：仿真器快、可并行，Model-Free 的工程收益通常更大；
2. 再问状态可观测性：像素级观测先考虑表征（世界模型），全状态观测可直接学转移模型；
3. 最后问部署形态：需要显式可导出策略（不能在线搜索）就排除纯规划路线（PETS、MuZero）。

### 4.2 混合方法

- 两条主线不是二选一：MBPO = 学到的模型（MB）+ SAC 学习器（MF）；Dyna = 模型 + Q-learning；
- 「用物理仿真器采样」不改变范式归属——判断只看算法内部是否学习并使用转移模型。

## 5. 分析与总结

- 范式的分水岭只有一个：**算法内部是否学习并使用转移模型**；与是否使用 Critic、是否 On-Policy、环境是否仿真器均无关；
- Model-Based 的三条用法（规划 / 造数据 / 传梯度）是比「Model-Based」这个标签更有信息量的分类轴；
- 样本效率的收益要用模型误差的风险来支付：短 rollout、系综不确定性、终端价值截断都是这份保险的保费；
- 深入任一条线之前，先回到本篇确认自己的问题落在哪个象限。

## 6. 关联与参考

### 6.1 相关知识

- **本分支地基**：🚧 基于模型的强化学习基础（撰写中）——转移模型学习与不确定性度量
- **Model-Free 主线**：🚧 值函数方法与 DQN（撰写中）、[PPO 算法完整解析与代码实现](../02_model-free-rl/04_ppo.md)、[SAC 算法完整解析与代码实现](../02_model-free-rl/07_sac.md)
- **Model-Based 主线**：🚧 PETS（撰写中）、🚧 MBPO（撰写中）、🚧 Dreamer（撰写中）

### 6.2 参考资料

1. Sutton, "Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming"（ICML 1990）——Dyna 与两类范式关系的原始框架。正式链接待补充。
2. Chua et al., "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"（NeurIPS 2018，PETS）。正式链接待补充。
3. Janner et al., "When to Trust Your Model: Model-Based Policy Optimization"（NeurIPS 2019，MBPO）。正式链接待补充。
4. Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination"（ICLR 2020，Dreamer）。正式链接待补充。
