# 无人机轨迹规划控制 — 知识库（V2）

**个人学习知识库 + 简历 + 面试复习主线**

> **Author**: Jiarui Hua
> **方向**: 无人机轨迹规划控制 (Trajectory Planning & Control)
> **核心管线**: 数学物理基础 / 经典规划控制 / 学习式规划 / 工程工具 / 仿真与评估

本目录是知识树网站的**原件库**：网站发布副本由 `scripts/sync-docs.mjs` 依据
`src/data/content-map.json` 从这里确定性生成，正文修改一律改原件。

---

## 知识库结构（五个一级领域）

### [01_math-physic-base](01_math-physic-base/) — 数学物理基础

| 子领域 | 内容 | 状态 |
|------|------|------|
| [01_linear-algebra-geometry](01_math-physic-base/01_linear-algebra-geometry/) | 矩阵分解、坐标变换、SO(3)/SE(3) 与四元数 | 撰写中 |
| [02_calculus-optimization](01_math-physic-base/02_calculus-optimization/) | 梯度/雅可比、优化理论、无约束/约束优化、数值优化 | 优化理论已完成，其余撰写中 |
| [03_probability-statistics](01_math-physic-base/03_probability-statistics/) | 概率与贝叶斯推断、高斯与状态估计 | 撰写中 |
| [04_dynamics](01_math-physic-base/04_dynamics/) | 刚体动力学、四旋翼动力学 | 撰写中 |
| [05_optimal-control-foundations](01_math-physic-base/05_optimal-control-foundations/) | 最优控制建模、Hamilton 与 Pontryagin | 撰写中 |

### [02_classical-planning-control](02_classical-planning-control/) — Pipeline: sensor → SLAM → 搜索 → 优化 → 控制

| 阶段 | 目录 | 内容 | 状态 |
|------|------|------|------|
| 感知与建图 | [01_perception-slam](02_classical-planning-control/01_perception-slam/) | 卡尔曼滤波系列、FAST-LIO、VIO、占据网格与 ESDF、动态障碍物跟踪 | 大部分完成 |
| 前端搜索 | [02_path-search](02_classical-planning-control/02_path-search/) | 启发式、采样法、增量式、Kinodynamic 搜索 | 大部分完成 |
| 后端优化 | [03_trajectory-optimization](02_classical-planning-control/03_trajectory-optimization/) | 轨迹表示、微分平坦与 Minimum Snap、贝塞尔与 B 样条、SFC、问题建模、MINCO、Fast-Planner、EGO-Planner、GCOPTER、重规划 | 核心已完成 |
| 跟踪控制 | [04_tracking-control](02_classical-planning-control/04_tracking-control/) | PID、LQR、MPC、PX4 控制器、几何控制、延迟与执行器 | 核心已完成 |

### [03_learning-based-planning](03_learning-based-planning/) — Pipeline: sensor → 特征提取 → RL 推理 → control

| 阶段 | 目录 | 内容 | 状态 |
|------|------|------|------|
| 特征提取 | [01_neural-network-architectures](03_learning-based-planning/01_neural-network-architectures/) | MLP、CNN、RNN/LSTM、Attention/Transformer、多尺度特征、时序融合 | 已完成 |
| 决策推理 | [02_rl-algorithms](03_learning-based-planning/02_rl-algorithms/) | MDP 基础、算法演进、值方法、策略梯度/GAE、TRPO、PPO、DDPG、TD3、SAC、POMDP、有模型 RL | 核心已完成 |
| 工程实践 | [03_engineering-practice](03_learning-based-planning/03_engineering-practice/) | PPO 走读、避坑指南、TensorBoard、小脑工作流、观测奖励设计、训练调试 | 已完成 |
| 学习式规划 | [04_learning-based-trajectory-planning](03_learning-based-planning/04_learning-based-trajectory-planning/) | 总览、YOPO、NavRL、候选选择、残差优化 | 撰写中 |

### [04_engineering-tools](04_engineering-tools/)

PX4 / MAVROS、ROS1、遥控器配置、Git 手册、多服务器多 GPU 训练（已完成）；
ROS2、PX4 Offboard、Docker、性能调试、可复现实验（撰写中）。

### [05_simulation-evaluation](05_simulation-evaluation/)

仿真平台、时序与接口、场景设计、域随机化、评估指标、基线与 Oracle、泛化与消融、Sim-to-Real（撰写中）。

### [references](references/) — 参考资料（仅本地，不入库）

论文与课程笔记，通过正式来源链接引用。

### [personal](personal/) — 简历与面试材料（仅本地，不入库）

---

## 撰写约定

- 文件名数字前缀 = 领域内章节顺序；网站侧栏顺序由 `content-map.json` 决定。
- 新增章节先写大纲（正文中标注"状态：草稿"），完成后进入正式发布。
- 正文修改在本目录进行，网站副本由同步脚本生成，不要手改 `src/content/docs` 下的生成页面。

*Keep Building, Keep Flying.*
