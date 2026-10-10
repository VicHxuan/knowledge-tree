---
title: "感知与SLAM"
description: "状态估计与建图：滤波、里程计、占据网格与 ESDF、动态障碍物跟踪"
sidebar:
  order: 1
---

感知与建图是规划的眼睛：从贝叶斯滤波基础到激光/视觉里程计，再到供规划使用的占据网格、ESDF 地图与动态障碍物跟踪。

- [卡尔曼滤波系列](/knowledge-tree/classical-planning-control/perception-slam/kalman-filters/)：贝叶斯滤波到 KF/EKF/PF 的完整推导
- [FAST-LIO](/knowledge-tree/classical-planning-control/perception-slam/fast-lio/)：FAST-LIO2 框架与迭代误差状态卡尔曼滤波
- VIO 🚧：视觉与 IMU 紧耦合状态估计（撰写中）
- [占据网格与 ESDF](/knowledge-tree/classical-planning-control/perception-slam/occupancy-grid-esdf/)：体素地图与增量式 ESDF 构建（Voxblox）
- 动态障碍物跟踪 🚧：动态目标检测、跟踪与预测（撰写中）
