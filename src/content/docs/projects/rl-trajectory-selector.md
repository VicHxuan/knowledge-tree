---
title: "RL 轨迹选择器"
description: "用 PPO 在仿真中训练轨迹候选选择策略——训练基础设施与工程实践"
sidebar:
  order: 3
---

## 项目定位

用强化学习训练一个**轨迹候选选择器**：上游给出若干条候选轨迹（经典规划器或生成网络输出），策略网络从中选出一条执行。这是「学习式」与「经典」路线的折中——不追求端到端，而是把学习模块插在管线中最有价值的决策点上。

## 系统与知识关系

### 技术栈

| 环节 | 选型 |
| --- | --- |
| 训练算法 | PPO（on-policy，并行采样） |
| 物理仿真 | Isaac Sim |
| RL 框架 | skrl 2.1.0（原版） |
| 训练基础设施 | 多服务器多 GPU，TensorBoard 监控 |

选型理由与训练链路的完整走读：

- [PPO 算法](/knowledge-tree/learning-based-planning/rl-algorithms/model-free-rl/ppo/)：一次迭代的数据流；
- [PPO 全流程走读与调参](/knowledge-tree/learning-based-planning/engineering-practice/ppo-walkthrough/)：从训练脚本到 checkpoint；
- [多服务器多 GPU 训练](/knowledge-tree/engineering-tools/ppo-in-multi-gpu/multi-gpu-training-deployment-guide/)：跨机环境的搭建与排障；
- [训练指标与诊断](/knowledge-tree/learning-based-planning/engineering-practice/tensorboard-metrics/)：判断训练是否在正确轨道上的速查表。

### 工程要点

这一类项目的重心常常不在算法而在训练工程，实践中反复被验证的几条：

1. **先跑通最小闭环再上规模**：单进程单环境验证 reward 收敛趋势，再扩展到并行采样；
2. **训练与评估分别记录**：联合观察奖励分量、优化诊断和固定场景中的成功率、安全性与效率；
3. **配置即代码**：训练配置（网络结构、超参、环境参数）与训练脚本一起进版本管理，保证可复现。

详细展开见[无人机强化学习任务设计](/knowledge-tree/learning-based-planning/engineering-practice/rl-engineering-pitfalls/)。

## 实践入口

[轨迹候选生成与选择](/knowledge-tree/learning-based-planning/learning-based-trajectory-planning/trajectory-candidate-selection/)说明候选观测、概率比、梯度和生成质量与选择质量的区别。

从 [PPO 算法主线](/knowledge-tree/learning-based-planning/rl-algorithms/model-free-rl/ppo/)进入，按[训练工程走读](/knowledge-tree/learning-based-planning/engineering-practice/ppo-walkthrough/)检查环境接口、采样、更新和记录。复现实验时固定源码版本、任务配置、checkpoint 与评估协议。

## 进展与证据

项目持续迭代中。任务设计（state / action / reward 抽象）的方法论整理在[学习式无人机规划](/knowledge-tree/learning-based-planning/learning-based-trajectory-planning/learning-based-planning-overview/)。

本页整理知识与复现入口；具体实验结论需附配置、评估协议及结果记录后补充。
