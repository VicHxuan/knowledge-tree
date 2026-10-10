---
title: "MINCO 规划管线"
description: "以 MINCO 轨迹参数化为核心的经典轨迹优化方法族梳理与实现"
sidebar:
  order: 4
---

## 项目定位

以 **MINCO 轨迹参数化**为核心的经典轨迹优化管线：把轨迹表示成可微分的时间分配 + 空间多项式组合，再交给优化器求解。这是 GCOPTER 系列工作的数学基础，也是理解现代轨迹规划器（Fast-Planner、EGO-Planner、GCOPTER）之间差异的钥匙。

## 方法族谱

经典规划管线自上而下：

<div class="pipeline">
	<span class="node">表示</span><span class="arrow">→</span>
	<span class="node">时间分配</span><span class="arrow">→</span>
	<span class="node">空间优化</span><span class="arrow">→</span>
	<span class="node">跟踪控制</span>
</div>

各家方法在这一管线上的取舍不同：

| 方法 | 轨迹表示 | 特点 |
| --- | --- | --- |
| Minimum Snap | 单段多项式 | 微分平坦理论的起点 |
| Fast-Planner | B 样条 | 安全飞行走廊内优化 |
| EGO-Planner | B 样条（梯度版） | 免 ESDF 的局部优化 |
| GCOPTER | MINCO | 联合优化时间与空间，凸包性质保证安全 |

对应站点章节：[轨迹表示与参数化](/knowledge-tree/classical-planning-control/trajectory-optimization/trajectory-representation-parameterization/)、[微分平坦与 Minimum Snap](/knowledge-tree/classical-planning-control/trajectory-optimization/differential-flatness-minimum-snap/)、[GCOPTER-MINCO 框架](/knowledge-tree/classical-planning-control/trajectory-optimization/gcopter-minco/)；MINCO 轨迹参数化的专题章节撰写中（🚧）。

## 与学习式路线的关系

MINCO 管线在这个知识体系里承担两个角色：

1. **基线**：学习式方法的对照——规划质量、计算时间、鲁棒性的参照系；
2. **组件**：轨迹候选生成器本身可以由 MINCO 类方法提供，交由学习模块选择（见 [RL 轨迹选择器](/knowledge-tree/projects/rl-trajectory-selector/)）。

两条路线的完整对比在[学习式轨迹规划总览](/knowledge-tree/learning-based-planning/learning-based-trajectory-planning/)（🚧 撰写中）中展开。

## 状态

GCOPTER-MINCO 框架走读已发布；MINCO 参数化专题与实现笔记撰写中。
