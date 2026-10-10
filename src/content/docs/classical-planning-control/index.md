---
title: "传统规划控制"
description: "sensor → SLAM → 搜索 → 优化 → 控制的经典管线"
sidebar:
  order: 1
---

经典规划控制管线：**sensor → SLAM → 路径搜索 → 轨迹优化 → 跟踪控制**，按数据流顺序组织。

1. [感知与SLAM](/knowledge-tree/classical-planning-control/perception-slam/)：状态估计、占据网格与 ESDF、动态障碍物跟踪
2. [路径搜索](/knowledge-tree/classical-planning-control/path-search/)：启发式 / 采样 / 增量式搜索
3. [轨迹优化](/knowledge-tree/classical-planning-control/trajectory-optimization/)：轨迹表示、Minimum Snap、主流规划器
4. [跟踪控制](/knowledge-tree/classical-planning-control/tracking-control/)：PID / LQR / MPC 与 PX4 实机控制器

> 数学优化理论已归入[数学基础板块](/knowledge-tree/mathematics/calculus-optimization/optimization-theory/)。
