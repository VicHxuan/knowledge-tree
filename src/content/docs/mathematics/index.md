---
title: "数学基础"
description: "无人机规划控制的数学底座：线性代数与几何、微积分与优化、概率统计、动力学、最优控制"
---

数学是整棵知识树的根。规划器里的每一个 QP、滤波器里的每一次更新、策略梯度里的每一次反向传播，底下都是这几块数学。

本板块按 **线性代数与几何 → 微积分与优化 → 概率统计 → 动力学 → 最优控制** 组织，为[传统规划控制](/knowledge-tree/classical-planning-control/)与[学习式规划](/knowledge-tree/learning-based-planning/)提供共同底座。

## 子领域

- [线性代数与几何](/knowledge-tree/mathematics/linear-algebra-geometry/)：矩阵分解、坐标变换、SO(3)/SE(3) 与四元数——状态估计与旋转求导的语言
- [微积分与优化](/knowledge-tree/mathematics/calculus-optimization/)：梯度/雅可比/海塞、优化理论、约束优化与 KKT、数值优化——轨迹优化 QP 的直接前置
- [概率统计](/knowledge-tree/mathematics/probability-statistics/) 🚧：概率与贝叶斯推断、高斯与状态估计——卡尔曼滤波族的理论底座
- [动力学](/knowledge-tree/mathematics/dynamics/) 🚧：刚体与四旋翼建模——微分平坦性的出发点
- [最优控制基础](/knowledge-tree/mathematics/optimal-control-foundations/) 🚧：最优控制建模、Hamilton 与 Pontryagin——LQR / MPC 的理论根源

## 推荐顺序

1. 先修[线性代数与几何](/knowledge-tree/mathematics/linear-algebra-geometry/)与[微积分与优化](/knowledge-tree/mathematics/calculus-optimization/)（[数学优化理论](/knowledge-tree/mathematics/calculus-optimization/optimization-theory/)已完成）
2. 按目标选深入方向：
   - 做**状态估计 / SLAM** → [概率统计](/knowledge-tree/mathematics/probability-statistics/) → [卡尔曼滤波系列](/knowledge-tree/classical-planning-control/perception-slam/kalman-filters/)
   - 做**轨迹优化** → 直接进[轨迹优化](/knowledge-tree/classical-planning-control/trajectory-optimization/)，缺什么回这里补
   - 做**控制** → [动力学](/knowledge-tree/mathematics/dynamics/) → [最优控制基础](/knowledge-tree/mathematics/optimal-control-foundations/) → [LQR](/knowledge-tree/classical-planning-control/tracking-control/lqr/)

## 关联

- [强化学习核心公式手册](/knowledge-tree/learning-based-planning/rl-algorithms/rl-foundations/)假设你熟悉本板块的概率与优化语言
- 对应学习路径：[强化学习学习路径](/knowledge-tree/learning-paths/reinforcement-learning/)、[经典轨迹规划学习路径](/knowledge-tree/learning-paths/classical-trajectory-planning/)
