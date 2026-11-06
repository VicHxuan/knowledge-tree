# 值函数方法与 DQN

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。
>
> **策略数据类型**：Off-Policy（值方法，经验回放 + 目标网络）。

从 Q-learning 到深度 Q 网络。本篇大纲先行，正文完成后进入正式发布：

## 2. 基础与定义

> 待补充。

## 3. 核心原理

### 值函数与贝尔曼方程

动态规划基础

### Q-learning

off-policy 时序差分

### DQN

经验回放与目标网络

### 改进变体

Double/Dueling/Prioritized

## 4. 实现与应用

> 待补充。

## 5. 分析与总结

### 连续动作空间的局限

引出确定性策略方法

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——Bellman 方程与 TD 学习
- **进阶阅读**：🚧 策略梯度、Actor-Critic 与 GAE（撰写中）——值方法之外的另一条主线；🚧 Dyna（撰写中）——Dyna-Q 同样以 Q-learning 为学习器，只是经验由模型生成
- **范式定位**：🚧 有模型与无模型强化学习（撰写中）——只用环境交互样本学习的 DQN 是标准的 Model-Free 方法

### 6.2 参考资料

1. Watkins & Dayan, "Q-learning"（Machine Learning，1992）。正式链接待补充。
2. Mnih et al., "Playing Atari with Deep Reinforcement Learning"（arXiv 2013，DQN 技术报告）。正式链接待补充。
3. Mnih et al., "Human-level Control through Deep Reinforcement Learning"（Nature 2015）。正式链接待补充。
4. 改进变体原始论文：van Hasselt et al.（Double，AAAI 2016）、Wang et al.（Dueling，ICML 2016）、Schaul et al.（Prioritized，ICLR 2016）。正式链接待补充。
