# POMDP 与循环策略

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

部分可观测与循环网络策略。本篇大纲先行，正文完成后进入正式发布：

## 2. 基础与定义

### POMDP 定义

部分可观测马尔可夫决策过程（POMDP）在 MDP 之上增加一层**观测机制**：状态 $s_t$ 对智能体不可直接观测，每个时刻只能获得观测 $o_t\sim\Omega(\cdot\mid s_t)$（$\Omega$ 为观测函数），且观测可能带噪、低维或有遮挡。马尔可夫性建立在 $s_t$ 上，而 $s_t$ 不可见，因此直接把 $o_t$ 当作状态的算法不再满足马尔可夫假设。

**信念状态（belief state）**把 POMDP 归约回 MDP：

$$b_t(s)=p\left(s_t=s\mid o_{1:t},\,a_{1:t-1}\right)$$

即观测与动作历史下的状态后验分布。以 $b_t$ 为扩充后的状态，转移由贝叶斯滤波给出，过程重新满足马尔可夫性。显式维护 $b_t$（贝叶斯滤波、粒子滤波）是经典 POMDP 求解路线；深度 RL 则通常把「滤波」交给网络——帧堆叠或循环网络从历史中隐式重建充分统计量。

## 3. 核心原理

### 帧堆叠与循环网络

历史信息编码

### Recurrent PPO

时序批处理与隐藏状态管理

## 4. 实现与应用

### 无人机导航应用

视觉/深度输入的时序依赖

## 5. 分析与总结

POMDP 是**环境性质**的刻画，而非某一算法范式的专属问题——它与 Model-Free / Model-Based 维度正交，两条路线各有自己的处理方式：

- **Model-Free 路线**：帧堆叠、循环网络（DRQN、Recurrent PPO）把历史压缩进隐状态，网络隐式承担滤波职能；
- **Model-Based 路线**：世界模型本身就在学 belief state——Dreamer 的 RSSM、MuZero 的表征网络 $h$ 都是参数化的历史压缩器；
- **经典规划路线**：显式维护 belief 分布，并在 belief 空间上做规划与求解。

工程启示：状态可观测性不足时，先问「历史里有多少信息被丢掉了」，再决定用记忆网络补、还是用世界模型补。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](./01_reinforcement-learning-foundations.md)——MDP 定义是 POMDP 的起点
- **网络载体**：[循环神经网络（RNN/LSTM）](../../01_neural-network-architectures/03_rnn-lstm.md)——历史信息编码的默认实现
- **范式对照**：🚧 Dreamer（撰写中）——RSSM 即学习到的 belief state；🚧 MuZero（撰写中）——表征网络消费观测历史

### 6.2 参考资料

1. Kaelbling, Littman & Cassandra, "Planning and Acting in Partially Observable Stochastic Domains"（Artificial Intelligence，1998）。正式链接待补充。
2. Hausknecht & Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs"（AAAI 2015 Fall Symposium，DRQN）。正式链接待补充。
