---
title: "工程工具链"
description: "Git、Docker、PX4 与 ROS、多 GPU 训练"
sidebar:
  order: 1
---

从飞控到训练集群的工程工具箱：版本控制、容器环境、PX4 与 ROS 中间件，以及多 GPU 训练基础设施。

## 版本控制与容器

- [Git 手册](/knowledge-tree/engineering-tools/git-manual/)：Git 常用指令速查
- Docker 环境 🚧：容器化训练环境与 GPU 支持（撰写中）

## PX4 与 ROS

- [ROS1](/knowledge-tree/engineering-tools/px4-ros/ros1/)：ROS1 面试要点
- ROS2 🚧：DDS 中间件与 ROS1 迁移（撰写中）
- [PX4 / MAVROS](/knowledge-tree/engineering-tools/px4-ros/px4-sitl/)：PX4 与 MAVROS 面试要点
- [PX4 控制器代码](/knowledge-tree/classical-planning-control/tracking-control/px4-controller-attitude-rate/)（在传统规划控制板块）：姿态/角速度级联实现

## 训练与实验

- [多 GPU 训练部署](/knowledge-tree/engineering-tools/ppo-in-multi-gpu/multi-gpu-training-deployment-guide/)：服务器审计、存储、容器与多卡启动（在 [PPO 多卡训练](/knowledge-tree/engineering-tools/ppo-in-multi-gpu/) 子板块）
- [分布式 RL 训练架构](/knowledge-tree/engineering-tools/ppo-in-multi-gpu/distributed-rl-training-architecture/)：同步数据并行 PPO、梯度同步与 Rollout 调度

遥控器配置等实飞内容已移至[真机飞行](/knowledge-tree/real-flight/)板块。
