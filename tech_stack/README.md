# 无人机规划控制知识库

这里维护知识树网站的技术文章原件。每篇文章采用同一主体结构：**概述 → 基础与定义 → 核心原理 → 实现与应用 → 分析与总结 → 关联与参考**。主线默认展开，长推导、算例和实现细节可展开阅读。

## 六大领域与两条技术主线

| 领域 | 内容与作用 |
|---|---|
| [数学基础](01_math-physic-base/) | 线性代数、优化、概率、动力学和最优控制，共同的概念底座 |
| [传统规划控制](02_classical-planning-control/) | 状态估计与建图 → 路径搜索 → 轨迹优化 → 跟踪控制 |
| [学习式规划](03_learning-based-planning/) | 观测 → 表示学习 → 策略或规划网络 → 动作/轨迹 → 反馈 |
| [工程工具链](04_engineering-tools/) | Git、容器、ROS/PX4、训练部署与分布式架构 |
| [仿真与评估](05_simulation-evaluation/) | 平台、场景、固定评估协议、泛化、真机迁移 |
| [真机飞行](06_real-flight/) | 设备配置、校准与实飞工程记录 |

数学为两条技术主线提供语言；工程、评估和实飞将方法连接成可复现系统。具体文章的可阅读/撰写中状态，以网站分类目录为准。

## 强化学习的阅读入口

`03_learning-based-planning/02_rl-algorithms/` 按三个分支组织：

1. `01_rl-foundations/`：公式手册、算法演进、范式导航、部分可观测问题。
2. `02_model-free-rl/`：DQN；PG/Actor-Critic/GAE → TRPO/PPO；DDPG → TD3/SAC。
3. `03_model-based-rl/`：模型学习、Dyna、PETS、MBPO、MuZero、Dreamer 和可微模型优化。

[PPO 文章](03_learning-based-planning/02_rl-algorithms/02_model-free-rl/04_ppo.md)集中说明一轮算法的数据与参数更新；[工程走读](03_learning-based-planning/03_engineering-practice/01_ppo-walkthrough.md)负责训练循环、配置与诊断。基础专题尚未发布时，PPO 保留理解主线所需的推导。

## 写作与维护

- 按[技术文章模板](TECHNICAL_ARTICLE_TEMPLATE.md)编写正文；草稿也保留六模块，未知部分如实标注。
- 原件文件名可用数字前缀排序；网站的排序、标题、状态、URL 在 `src/data/content-map.json` 维护。
- 注册、移动与验证按[贡献指南](../CONTRIBUTING.md)执行；移动原件默认保留公开 URL。
- 网站生成副本不可手改；完整可运行示例在仓库 `examples/`，同步脚本生成网页下载。
- `references/` 和 `personal/` 为本地资料，不进入网站发布。文章用正式论文/文档链接引用参考资料。
