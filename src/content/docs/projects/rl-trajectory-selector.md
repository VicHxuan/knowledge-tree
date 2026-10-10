---
title: "RL 轨迹选择器"
description: "用 PPO 在仿真中训练轨迹候选选择策略——训练基础设施与工程实践"
sidebar:
  order: 3
---

## 项目定位

用强化学习训练一个**轨迹候选选择器**：上游给出若干条候选轨迹（经典规划器或生成网络输出），策略网络从中选出一条执行。这是「学习式」与「经典」路线的折中——不追求端到端，而是把学习模块插在管线中最有价值的决策点上。

## 技术栈

| 环节 | 选型 |
| --- | --- |
| 训练算法 | PPO（on-policy，并行采样） |
| 物理仿真 | Isaac Sim |
| RL 框架 | skrl 2.1.0（原版） |
| 训练基础设施 | 多服务器多 GPU，TensorBoard 监控 |

选型理由与训练链路的完整走读：

- [PPO 算法](/knowledge-tree/learning-based-planning/rl-algorithms/ppo/)：一次迭代的数据流；
- [PPO 全流程走读与调参](/knowledge-tree/learning-based-planning/engineering-practice/ppo-walkthrough/)：从训练脚本到 checkpoint；
- [多服务器多 GPU 训练](/knowledge-tree/engineering-tools/multi-server-gpu-training/)：跨机环境的搭建与排障；
- [TensorBoard 指标模板](/knowledge-tree/learning-based-planning/engineering-practice/tensorboard-metrics/)：判断训练是否在正确轨道上的速查表。

## 工程要点

这一类项目的重心常常不在算法而在训练工程，实践中反复被验证的几条：

1. **先跑通最小闭环再上规模**：单进程单环境验证 reward 收敛趋势，再扩展到并行采样；
2. **reward 曲线比单点指标可靠**：用 TensorBoard 的滑动平均看趋势，警惕只看最终成功率；
3. **配置即代码**：训练配置（网络结构、超参、环境参数）与训练脚本一起进版本管理，保证可复现。

详细展开见[强化学习工程避坑指南](/knowledge-tree/learning-based-planning/engineering-practice/rl-engineering-pitfalls/)。

## 状态

项目持续迭代中。任务设计（state / action / reward 抽象）的方法论整理在[小脑 Work Pipeline](/knowledge-tree/learning-based-planning/engineering-practice/work-pipeline/)。
