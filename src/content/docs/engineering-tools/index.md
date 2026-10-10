---
title: "工程工具链"
description: "PX4 / MAVROS、ROS、Docker、Git 与多服务器 GPU 训练"
sidebar:
  order: 1
---

从飞控到训练集群的工程工具箱。按"上机 → 开发 → 训练"的使用顺序分组：

## 飞控与机体

- [PX4 / MAVROS](/knowledge-tree/engineering-tools/px4-mavros/)：PX4 与 MAVROS 面试要点
- [遥控器配置](/knowledge-tree/engineering-tools/remote-control-setup/)：QGC 遥控器校准
- [PX4 控制器代码](/knowledge-tree/classical-planning-control/tracking-control/px4-controller-attitude-rate/)（在传统规划控制板块）：姿态/角速度级联实现

## 机器人中间件

- [ROS1](/knowledge-tree/engineering-tools/ros1/)：ROS1 面试要点
- ROS2 🚧：DDS 中间件与 ROS1 迁移（撰写中）
- PX4 Offboard 🚧：uXRCE-DDS 桥接与 offboard 指令流（撰写中）

## 训练与实验

- [多服务器 GPU 训练](/knowledge-tree/engineering-tools/multi-server-gpu-training/)：多服务器部署与 torchrun 多卡训练
- Docker 环境 🚧：容器化训练环境与 GPU 支持（撰写中）
- 性能与调试 🚧：系统观测与训练瓶颈定位（撰写中）
- 可复现实验 🚧：seed、配置与实验追踪（撰写中）

## 版本控制

- [Git 手册](/knowledge-tree/engineering-tools/git-manual/)：Git 常用指令速查
