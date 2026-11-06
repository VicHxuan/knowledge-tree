# DDPG 与确定性策略梯度

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。
>
> **策略数据类型**：Off-Policy（确定性策略梯度，经验回放）。

确定性策略梯度与连续控制。本篇大纲先行，正文完成后进入正式发布：

## 2. 基础与定义

> 待补充。

## 3. 核心原理

### 确定性策略梯度定理

DPG 的推导

### DDPG 架构

critic 与 target 网络

### 探索策略

OU 噪声与参数噪声

### TD3 改进机制

过估计问题的来源

## 4. 实现与应用

> 待补充。

## 5. 分析与总结

> 待补充。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——确定性策略梯度定理的推导基础
- **同族对照**：🚧 策略梯度、Actor-Critic 与 GAE（撰写中）——随机策略梯度与确定性策略梯度的对照
- **进阶阅读**：[TD3 算法完整解析与代码实现](./06_td3.md)——对 DDPG 过估计问题的三项修复；[SAC 算法完整解析与代码实现](./07_sac.md)——随机化与最大熵的替代路线

### 6.2 参考资料

1. Silver et al., "Deterministic Policy Gradient Algorithms"（ICML 2014）。正式链接待补充。
2. Lillicrap et al., "Continuous Control with Deep Reinforcement Learning"（ICLR 2016，DDPG）。正式链接待补充。
