---
title: "经典轨迹规划学习路径"
description: "从轨迹表示到控制器实现的推荐学习顺序"
sidebar:
  order: 2
---

按依赖关系推荐的学习顺序（标注 🚧 的内容尚未撰写）：

1. 数学底座：[数学优化理论](/knowledge-tree/mathematics/calculus-optimization/optimization-theory/)
2. [轨迹表示与参数化](/knowledge-tree/classical-planning-control/trajectory-optimization/trajectory-representation-parameterization/)：先理解轨迹如何被数学表示与参数化
3. [微分平坦与 Minimum Snap](/knowledge-tree/classical-planning-control/trajectory-optimization/differential-flatness-minimum-snap/)：轨迹生成的基本范式
4. [轨迹优化](/knowledge-tree/classical-planning-control/trajectory-optimization/)：安全飞行走廊、贝塞尔 / B 样条 / MINCO、Fast-Planner、EGO-Planner、GCOPTER
5. [跟踪控制](/knowledge-tree/classical-planning-control/tracking-control/)：PID、LQR、MPC，以及 [PX4 控制器](/knowledge-tree/classical-planning-control/tracking-control/px4-controller-attitude-rate/)如何跟踪并执行期望轨迹
6. 应用视角：[小脑 Work Pipeline](/knowledge-tree/learning-based-planning/engineering-practice/work-pipeline/)
