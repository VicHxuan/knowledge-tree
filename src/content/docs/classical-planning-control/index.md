---
title: "传统规划控制"
description: "sensor → SLAM → 搜索 → 优化 → 控制的经典管线"
sidebar:
  order: 1
---

经典规划控制管线按**数据流**组织：

<div class="pipeline">
	<span class="node">sensor</span><span class="arrow">→</span>
	<span class="node">SLAM / 建图</span><span class="arrow">→</span>
	<span class="node">路径搜索</span><span class="arrow">→</span>
	<span class="node">轨迹优化</span><span class="arrow">→</span>
	<span class="node">跟踪控制</span>
</div>

每个阶段是一个子领域，按管线顺序学习最顺畅：

1. [感知与SLAM](/knowledge-tree/classical-planning-control/perception-slam/)：状态估计、占据网格与 ESDF、动态障碍物跟踪——规划的眼睛
2. [路径搜索](/knowledge-tree/classical-planning-control/path-search/)：启发式 / 采样 / 增量式 / Kinodynamic 搜索——在地图里找可行路径
3. [轨迹优化](/knowledge-tree/classical-planning-control/trajectory-optimization/)：轨迹表示、Minimum Snap、B 样条 / MINCO、Fast-Planner / EGO-Planner / GCOPTER——把路径变成可飞轨迹
4. [跟踪控制](/knowledge-tree/classical-planning-control/tracking-control/)：PID / LQR / MPC 与 PX4 实机控制器——让无人机跟上轨迹

## 推荐

- 完整学习顺序见[经典轨迹规划学习路径](/knowledge-tree/learning-paths/classical-trajectory-planning/)
- 数学前置见[数学基础](/knowledge-tree/mathematics/)（数学优化理论已归入该板块）
