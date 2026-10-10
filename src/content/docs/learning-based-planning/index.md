---
title: "学习式规划"
description: "sensor → 特征提取 → RL 推理 → control 的学习式管线"
sidebar:
  order: 1
---

学习式规划管线用网络替代或增强经典管线中的迭代环节：

<div class="pipeline">
	<span class="node">sensor</span><span class="arrow">→</span>
	<span class="node">特征提取</span><span class="arrow">→</span>
	<span class="node">RL 推理</span><span class="arrow">→</span>
	<span class="node">control</span>
</div>

四个子领域对应管线的四个环节，再落到轨迹规划本身：

1. [神经网络架构](/knowledge-tree/learning-based-planning/neural-network-architectures/)：MLP、CNN、RNN/LSTM、Transformer——把传感器数据变成特征
2. [强化学习算法](/knowledge-tree/learning-based-planning/rl-algorithms/)：值方法、策略梯度、TRPO/PPO、DDPG/TD3/SAC——从特征到决策
3. [训练工程实践](/knowledge-tree/learning-based-planning/engineering-practice/)：PPO 走读、避坑指南、TensorBoard——让训练真正跑起来
4. [学习式轨迹规划](/knowledge-tree/learning-based-planning/learning-based-trajectory-planning/) 🚧：YOPO、候选选择、残差优化——回到轨迹本身

## 推荐

- 入门顺序见[强化学习学习路径](/knowledge-tree/learning-paths/reinforcement-learning/)
- 与经典管线的对照见[传统规划控制](/knowledge-tree/classical-planning-control/)
- 概率与优化前置见[数学基础](/knowledge-tree/mathematics/)
