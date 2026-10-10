---
title: "轨迹优化"
description: "后端优化：轨迹表示、Minimum Snap、B 样条、SFC 与主流规划器"
sidebar:
  order: 3
---

后端优化把路径变成时间参数化、动力学可行的可飞轨迹：从轨迹表示与微分平坦出发，经 Minimum Snap、贝塞尔 / B 样条、安全飞行走廊，到 Fast-Planner、EGO-Planner、GCOPTER 等主流框架。

- [轨迹表示与参数化](/knowledge-tree/classical-planning-control/trajectory-optimization/trajectory-representation-parameterization/)：轨迹的多项式表示与参数化选择
- [微分平坦与 Minimum Snap](/knowledge-tree/classical-planning-control/trajectory-optimization/differential-flatness-minimum-snap/)：微分平坦性与 Minimum Snap QP 求解
- [贝塞尔与 B 样条](/knowledge-tree/classical-planning-control/trajectory-optimization/bezier-bspline/)：贝塞尔曲线与 B 样条基础
- [安全飞行走廊](/knowledge-tree/classical-planning-control/trajectory-optimization/safe-flight-corridor/)：飞行安全走廊的生成与约束
- 优化问题建模 🚧：决策变量、目标函数与约束体系（撰写中）
- MINCO 参数化 🚧：MINCO 表示与空间-时间联合优化（撰写中）
- [Fast-Planner](/knowledge-tree/classical-planning-control/trajectory-optimization/fast-planner/)：Fast-Planner 前后端规划框架
- [EGO-Planner](/knowledge-tree/classical-planning-control/trajectory-optimization/ego-planner/)：EGO-Planner 梯度轨迹优化
- [GCOPTER-MINCO](/knowledge-tree/classical-planning-control/trajectory-optimization/gcopter-minco/)：GCOPTER 与 MINCO 联合优化
- 重规划与连续性 🚧：在线重规划与轨迹衔接（撰写中）
