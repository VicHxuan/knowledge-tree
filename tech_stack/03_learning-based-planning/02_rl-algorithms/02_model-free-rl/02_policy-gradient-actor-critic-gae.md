# 策略梯度、Actor-Critic 与 GAE

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。
>
> **策略数据类型**：On-Policy（随机策略梯度，回报/优势来自当前策略的同批采样）。

从 REINFORCE 到优势估计。本篇大纲先行，正文完成后进入正式发布：

## 2. 基础与定义

> 待补充。

## 3. 核心原理

### REINFORCE

蒙特卡洛策略梯度与高方差

### 基线与优势函数

方差缩减

### Actor-Critic

值函数辅助的策略梯度

### GAE

优势估计的偏差-方差权衡

## 4. 实现与应用

> 待补充。

## 5. 分析与总结

> 待补充。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——策略梯度定理的数学基础
- **进阶阅读**：[TRPO 算法完整解析与代码实现](./03_trpo.md)、[PPO 算法完整解析与代码实现](./04_ppo.md)
- **对照阅读**：🚧 值函数方法与 DQN（撰写中）——值方法与策略方法两条路线的对照

### TRPO 与 PPO 的关系

信赖域动机——策略梯度的方差问题与步长敏感，直接引出 TRPO 的信赖域约束与 PPO 的截断目标；完整推导见对应专项文章。

### 6.2 参考资料

1. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"（Machine Learning，1992，REINFORCE）。正式链接待补充。
2. Sutton et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation"（NeurIPS 1999 会议；论文集 2000 年出版）。正式链接待补充。
3. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"（ICLR 2016，GAE）。正式链接待补充。
