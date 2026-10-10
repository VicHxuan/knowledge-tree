---
title: 无人机规划与控制学习路径
description: 从轨迹表示到控制器实现的推荐学习顺序
sidebar:
  order: 2
---

按依赖关系推荐的学习顺序（标注 🚧 的页面尚未撰写）：

1. [Trajectory Representation and Parameterization](/knowledge-tree/classical-planning-control/trajectory-optimization/trajectory-representation-parameterization/)：先理解轨迹如何被数学表示与参数化
2. [轨迹优化](/knowledge-tree/classical-planning-control/trajectory-optimization/)：微分平坦与 Minimum Snap、SFC 安全走廊、贝塞尔 / B 样条 / MINCO、Fast-Planner、EGO-Planner 等
3. [跟踪控制](/knowledge-tree/classical-planning-control/tracking-control/)：PID、LQR、MPC，以及 [PX4 控制器](/knowledge-tree/classical-planning-control/tracking-control/px4-controller-attitude-rate/)如何跟踪并执行期望轨迹
4. 应用视角：[小脑 Work Pipeline](/knowledge-tree/rl-end-to-end/engineering-practice/work-pipeline/)
