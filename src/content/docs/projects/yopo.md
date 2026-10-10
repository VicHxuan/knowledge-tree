---
title: "YOPO：一次性规划网络"
description: "用网络替代迭代优化的端到端轨迹规划——论文走读与技术跟踪"
sidebar:
  order: 2
---

## 项目定位

YOPO（You Only Plan Once）是学习式轨迹规划方向的代表性工作：**用一次前向传播直接输出可执行轨迹**，替代经典管线中「搜索 → 优化」的迭代过程。项目页跟踪该论文族的阅读与实践，对应的站点章节（学习式轨迹规划总览、YOPO、NavRL）正在撰写中 🚧。

## 技术要点

从当前整理的资料看，YOPO 的关键设计可以概括为：

1. **感知骨干 + 轨迹候选生成**：网络从传感器数据直接生成轨迹候选，而非生成中间表示（如 ESDF）再优化；
2. **监督信号的设计空间**：与 behavior cloning 只模仿专家不同，YOPO 的 loss 可以引入**不属于网络输入的环境信息**（guidance learning）——这让监督训练获得类似 reward 的环境反馈，而不必等待完整 RL 闭环；
3. **数据来源**：训练数据从轻量级仿真器快速提取，绕开 RL 在线交互的数据效率瓶颈。

这套「用环境信息求 loss」的思路，与 RL 的 reward 设计一脉相承，具体的对比分析在[小脑 Work Pipeline](/knowledge-tree/learning-based-planning/engineering-practice/work-pipeline/) 中有展开。

## 相关阅读

- 论文原文与衍生工作（YOPO-swarm 等多机变体）走读计划收录在[学习式轨迹规划](/knowledge-tree/learning-based-planning/learning-based-trajectory-planning/)板块；
- 训练基础设施相关：[多服务器多 GPU 训练](/knowledge-tree/engineering-tools/multi-server-gpu-training/)、[PPO 全流程走读](/knowledge-tree/learning-based-planning/engineering-practice/ppo-walkthrough/)。

## 状态

论文走读与架构分析撰写中（对应知识树的 `learning-based-planning/learning-based-trajectory-planning` 目录），本页随进展更新。
