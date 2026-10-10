# 无人机轨迹规划控制 — 知识库

**个人学习知识库 + 简历 + 面试复习主线**

> **Author**: Jiarui Hua
> **方向**: 无人机轨迹规划控制 (Trajectory Planning & Control)
> **核心管线**: 传统规划控制 / 强化学习端到端 / 模仿学习与VLA

---

## 知识库结构

### [01_classical-planning-control](01_classical-planning-control/) — Pipeline: sensor → SLAM → 搜索 → 优化 → 控制

| 阶段 | 目录 | 内容 | 状态 |
|------|------|------|------|
| 感知与建图 | [01_perception-slam](01_classical-planning-control/01_perception-slam/) | 卡尔曼滤波系列 (BF/KF/EKF/PF)、FAST-LIO系列 | 已完成 |
| 前端搜索 | [02_path-search](01_classical-planning-control/02_path-search/) | 启发式 (Dijkstra, A*, IDA*, JPS)、采样法 (RRT系列)、增量法 | 已完成 |
| 后端优化 | [03_trajectory-optimization](01_classical-planning-control/03_trajectory-optimization/) | 轨迹表示与参数化、微分平坦 & Minimum Snap QP、SFC安全走廊、贝塞尔曲线 & B样条、基础优化理论、ESDF地图、Fast-Planner、EGO-Planner、MINCO | 大部分完成 |
| 跟踪控制 | [04_tracking-control](01_classical-planning-control/04_tracking-control/) | PID、LQR、MPC、PX4 姿态/角速度控制器代码讲解 | 已完成 |

### [02_rl-end-to-end](02_rl-end-to-end/) — Pipeline: sensor → 特征提取 → RL推理 → control

| 阶段 | 目录 | 内容 | 状态 |
|------|------|------|------|
| 特征提取 | [01_feature-extraction-networks](02_rl-end-to-end/01_feature-extraction-networks/) | MLP、CNN、RNN-LSTM、Attention-Transformer | 仅 MLP 完成 |
| 决策推理 | [02_rl-algorithms](02_rl-end-to-end/02_rl-algorithms/) | MDP & Bellman 基础、算法演进、TRPO、PPO、SAC、TD3 | 已完成 |
| 工程实践 | [03_engineering-practice](02_rl-end-to-end/03_engineering-practice/) | PPO 全流程走读与调参、RL 工程避坑、TensorBoard 指标模板、小脑工作流 | 已完成 |

### [03_engineering-tools](03_engineering-tools/)

PX4 / MAVROS、ROS1、遥控器与接收机配置 (QGC 校准)、Git 常用指令手册、多服务器部署与单机多 GPU 训练

### [references](references/) — 参考资料

| 类型 | 目录 | 内容 |
|------|------|------|
| 论文 | [papers](references/papers/) | EGO-Planner、Fast-Planner、GCOPTER、NavRL、PPO、SAC v1、YOPO |
| 课程笔记 | [rl-course-notes](references/rl-course-notes/) | 强化学习数学基础系列 11 讲（原「西湖笔记」） |

### [personal](personal/) — 简历与面试材料（仅本地保留，不入库）

个人简历 (cv.md)、面试准备、简历深挖问答、技术脉络、研究生实习简历 PDF、ICGNC / ICRA 论文与演示视频。

---

## 规划中（待填充）

以下主题为此前的空占位文件，内容待撰写：

- **轨迹优化**：时间重分配 (time re-allocation)
- **模仿学习与 VLA**：模仿学习 / 监督学习、VLA 大模型端到端控制
- **数学基础**：李群与李代数、旋转矩阵与四元数、凸优化、坐标变换、微分平坦、微分流形
- **工程工具链**：Docker、ROS2、CUDA / TensorRT / ONNX、PyTorch
- **仿真器**：Isaac Sim、Gazebo、AirSim
- **C++ 工程**：现代 C++ 特性、多线程与并发、嵌入式部署 (Jetson)
- **RL 工程实践**：参数手册 (param handbook)
- **RL 常规工具**：RL 训练辅助工具
- **项目与论文**：无人机室内巡检项目、Airsim-RL-ICGNC 论文、Isaac-Sim-RL 论文
- **面试复习**：C++ 八股

---

## 关键文件

- [personal/cv.md](personal/cv.md) — 个人简历

---

*Keep Building, Keep Flying.*
