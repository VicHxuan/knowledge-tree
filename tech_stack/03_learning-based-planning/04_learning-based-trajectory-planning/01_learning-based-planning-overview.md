# 学习式无人机规划

## 1. 概述

学习式规划将传感器观测、运动状态与导航目标映射为动作或轨迹。系统沿“观测构造 → 网络推理 → 轨迹或指令解码 → 控制跟踪 → 新观测”循环运行。设计首先确定网络负责的决策层级，再确定训练信号与执行接口。

## 2. 基础与定义

### 2.1 系统接口

设观测为 $o_t$、目标为 $g_t$、网络输出为 $z_t=f_\theta(o_t,g_t)$，执行指令为

$$
u_t=\mathcal D(z_t,\hat s_t),
$$

其中 $\hat s_t$ 是状态估计，$\mathcal D$ 完成坐标转换、轨迹求解、限幅和控制接口适配。轨迹输出可包含位置、速度与加速度参考，速度动作则交给速度跟踪控制器。

### 2.2 方法分类

| 网络输出 | 代表组织方式 | 主要接口问题 |
|---|---|---|
| 连续控制参考 | NavRL 导航策略 | 动作坐标、时延、低层响应 |
| 多条轨迹及评分 | YOPO 规划网络 | 边界条件、代价、候选选择 |
| 候选选择概率 | RL 轨迹选择器 | 候选可观测性、有效掩码 |
| 名义解的修正 | 残差规划 | 修正范围、约束与重新验证 |

## 3. 核心原理

### 3.1 训练信号

模仿学习用专家动作或轨迹 $z^*$ 构造 $L_{\mathrm{IL}}=\mathbb E[\ell(f_\theta(o),z^*)]$。可微代价学习使用环境信息 $\mathcal M$ 评价输出，得到 $L_{\mathrm{cost}}=\mathbb E[J(\mathcal D(f_\theta(o)),\mathcal M)]$。强化学习以交互回报定义 $J_{\mathrm{RL}}=\mathbb E[\sum_t\gamma^tr_{t+1}]$。

三种训练方式分别提供标签、输出空间梯度与轨迹回报。它们可以按预训练、交互优化和独立验证组合。训练中的地图真值可用于代价或 Critic，部署网络输入保持可获得性。

### 3.2 学习与优化结合

给定名义轨迹参数 $z_0$，残差模型产生

$$
z=z_0+S\Delta z_\theta(o),
$$

其中 $S$ 规定允许修改的变量与尺度。起点状态、时间分配和约束根据所选参数化显式处理。投影到可行集合时可定义

$$
z^*=\arg\min_{z\in\mathcal C}\frac12\|z-z_0-S\Delta z_\theta(o)\|^2.
$$

集合 $\mathcal C$ 的构造决定约束含义；在线求解失败与不可行情况需要明确的执行行为。

<details class="kt-derivation" id="residual-gradient">
<summary>残差参数的梯度</summary>

在未加入投影且代价可微时，链式法则给出

$$
\nabla_\theta J
=\left(\frac{\partial\Delta z_\theta}{\partial\theta}\right)^{\mathsf T}S^{\mathsf T}\nabla_zJ.
$$

加入优化层后，还需乘上最优解 $z^*$ 对输入的导数。活跃约束切换处可能不可微，求解器输出的梯度需与具体最优性条件对应。该机制连接[数值优化](/knowledge-tree/mathematics/calculus-optimization/numerical-optimization/)和[MINCO](/knowledge-tree/classical-planning-control/trajectory-optimization/minco-parameterization/)。

</details>

### 3.3 执行闭环

规划器按新观测反复产生有限时域计划，控制器以自身频率跟踪当前有效指令。重规划时明确使用实测状态还是参考状态作为新轨迹边界，并记录交接误差。候选评分、独立几何检查和控制可执行性分别提供证据。

## 4. 实现与应用

### 4.1 开发流程

1. 定义到达、碰撞、超时与控制输出，固定观测和动作接口。
2. 建立可重放的仿真场景，先验证简单控制器与数据链路。
3. 选择标签、可微代价或回报作为训练信号，保存版本与配置。
4. 在固定测试场景中比较成功率、间距、时间和计算延迟。
5. 对机载推理、控制接口与传感器误差逐项验证。

### 4.2 方法阅读

[YOPO](./02_yopo.md)解释轨迹生成与代价梯度；[NavRL](./03_navrl.md)解释障碍表示、速度动作与策略训练；[候选选择](./04_trajectory-candidate-selection.md)解释如何用 PPO 学习离散决策。具体实验和权重记录放在[项目页](/knowledge-tree/projects/)。

## 5. 分析与总结

方法比较同时固定观测条件、运动限制、控制器和评估预算。网络推理时间只是端到端延迟的一部分，还需计入传感器更新、预处理、轨迹解码、通信与跟踪。局部轨迹质量和长期导航结果分别评价。

## 6. 关联与参考

- [任务设计](/knowledge-tree/learning-based-planning/engineering-practice/rl-engineering-pitfalls/)。
- [导航评估](/knowledge-tree/simulation-evaluation/evaluation-methodology/navigation-metrics/)。
- [YOPO 原论文](https://tju-air-lab.github.io/projects/YOPO/YOPO.pdf)。
- [NavRL 原论文](https://arxiv.org/abs/2409.15634)。
