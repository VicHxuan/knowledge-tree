# SuperMimic：可微世界模型与运动控制

## 1. 概述

SuperMimic 用一个学习的可微世界模型支撑多种运动的策略训练。仿真器提供实际交互数据，世界模型预测动作后的状态，Actor 通过模型内的短时域回报求梯度，Critic 补充片段之外的长期价值。

其核心联系是可微策略梯度与短时域 Actor-Critic。运动模仿提供任务目标，共享动力学模型复用不同动作序列中的物理经验。

## 2. 基础与定义

### 2.1 系统结构

令 $x_t$ 表示预测所需的物理状态，$c_t$ 表示影响转移的环境条件，$g_t$ 表示任务参考，$o_t$ 表示 Actor 可用观测：

$$
a_t=\pi_\theta(o_t,g_t),\qquad
\hat x_{t+1}=f_\phi(\hat x_t,a_t,c_t).
$$

Critic $V_\psi$ 使用其训练输入估计任务价值。不同参考动作改变奖励与策略输入；质量、摩擦或外力等物理条件影响状态转移。

| 模块 | 输入与输出 | 学习信号 |
|---|---|---|
| 世界模型 | 状态、动作、环境条件 → 下一状态 | 仿真轨迹中的状态误差 |
| Actor | 观测、任务参考 → 动作 | 模型内回报的路径梯度 |
| Critic | 价值输入 → 未来回报 | 多步回报目标 |

### 2.2 数据与目标

经验库保存仿真器产生的连续状态与动作。世界模型沿这些动作预测状态序列；Actor 训练从经验状态开始，后续动作由当前策略产生。任务奖励 $r(\hat x_t,a_t,g_t)$ 衡量跟踪质量与控制代价。

## 3. 核心原理

### 3.1 世界模型学习

模型训练使用多步自回归预测：

$$
\hat x_0=x_0,\qquad \hat x_{t+1}=f_\phi(\hat x_t,a_t,c_t),
$$

$$
L_M(\phi)=\frac1K\sum_{t=1}^{K}\sum_j w_j
\left\|e_j(\hat x_t,x_t)\right\|^2.
$$

$j$ 表示状态分组，$e_j$ 按该组的表示与尺度计算误差，$w_j$ 控制训练权重。模型后续输入来自自身预测，多步损失因而约束误差沿轨迹传播的结果。

姿态、位置、速度和执行器状态具有不同几何结构与尺度。误差定义和状态更新应与这些结构一致，再通过权重平衡各组学习信号。

<details class="kt-derivation" id="supermimic-model-gradient">
<summary>多步模型损失的梯度</summary>

固定经验动作序列，令 $E_t=\partial\hat x_t/\partial\phi$。模型梯度满足：

$$
E_0=0,\qquad E_{t+1}=f_xE_t+f_\phi.
$$

于是：

$$
\nabla_\phi L_M=\frac2K\sum_{t,j}w_j e_j^\top
\frac{\partial e_j}{\partial\hat x_t}E_t.
$$

每个预测步的参数既影响当前预测，也影响后续模型输入。只对单步教师强制预测计算损失，训练信号主要约束实际状态附近的局部预测。

官方世界模型实现按状态组计算均方误差，并结合组维数与权重聚合；窗口中的首个状态作为起点，后续状态用于监督预测。

</details>

### 3.2 Actor 优化

对于长度为 $H$ 的连续片段，固定世界模型和 Critic 参数，最大化：

$$
J_\pi=\mathbb E\left[\sum_{t=0}^{H-1}\gamma^t r(\hat x_t,a_t,g_t)
+\gamma^H V_\psi(\hat x_H,g_H)\right].
$$

此处以状态与任务参考简写 Critic 输入。Actor 损失为 $L_\pi=-J_\pi$，梯度沿以下路径传播：

$$
\theta\longrightarrow a_t\longrightarrow\hat x_{t+1}
\longrightarrow\{r_{t+1},a_{t+1},V_\psi(\hat x_H,g_H)\}.
$$

世界模型提供 $\partial f/\partial x$ 和 $\partial f/\partial a$，Critic 提供末端价值对状态的导数。模型在 Actor 更新阶段固定，在后续数据学习阶段继续更新。

<details class="kt-derivation" id="supermimic-survival-return">
<summary>终止概率与回报</summary>

若模型内任务使用终止概率 $d_t$，可以定义存活权重：

$$
S_0=1,\qquad S_{t+1}=S_t(1-d_t).
$$

对应目标为：

$$
J_\pi=\mathbb E\left[\sum_{t=0}^{H-1}\gamma^t S_t r_t+
\gamma^H S_HV_\psi(\hat x_H,g_H)\right].
$$

官方实现通过配置控制存活权重的累积，并在末端处理终止；复现时需同时核对奖励、终止与价值的掩码。如果 $d_t$ 可微且保留计算图，目标还包含存活权重对策略的导数；硬终止通常按分段边界处理。

</details>

### 3.3 Critic 学习

Critic 使用分离后的模型轨迹构造目标。记 $v_t$ 为目标构造时固定的价值预测：

$$
\delta_t=r_t+\gamma(1-d_t)v_{t+1}-v_t,
$$

$$
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},\qquad
G_t^\lambda=v_t+\hat A_t.
$$

$$
L_V(\psi)=\frac12\mathbb E[(V_\psi(x_t,g_t)-\operatorname{sg}(G_t^\lambda))^2].
$$

这里 GAE 用于构造 Critic 的回报目标。Actor 的主更新信号来自模型内回报的路径梯度；PPO 则将优势乘到概率比值目标中。

<details class="kt-derivation" id="supermimic-gradient-boundaries">
<summary>官方实现的梯度边界</summary>

官方实现提交 `7e94b073940b7a19b9a82ff01e191cfb35c4ba91` 中：

- `policy_learner.py` 在 Actor 展开期间保留奖励和末端价值的计算图，并保存分离后的轨迹供 Critic 训练。
- 末端价值计算冻结 Critic 参数，保留价值对输入的导数。
- `learner_utils.py` 从末端价值反向计算 GAE 回报。
- Critic 拟合时固定这些回报目标，再启用 Critic 参数梯度。
- `wm_learner.py` 单独使用仿真经验训练世界模型。

该分工对应三组参数的不同任务：$\phi$ 拟合动力学，$\theta$ 改善回报，$\psi$ 估计价值。

</details>

## 4. 实现与应用

### 4.1 训练闭环

1. 在仿真器中运行当前策略，保存连续物理状态、动作和环境条件。
2. 从经验库抽取时间窗口，更新世界模型的多步预测。
3. 从经验状态出发，在模型中执行当前策略并计算短时域回报。
4. 固定模型与 Critic，反向传播更新 Actor。
5. 用分离后的回报目标更新 Critic。
6. 回到仿真器评估新策略，补充新访问区域的数据。

模型训练、策略优化与仿真评估共同决定效率。模型内回报上升时，应同步检查仿真回报及策略新访问区域的预测误差。

### 4.2 无人机迁移分析

该结构用于无人机时，可按以下对应关系设计：

| 运动控制要素 | 无人机中的对应对象 |
|---|---|
| 物理状态 | 位置、姿态、速度、角速度与执行器内部状态 |
| 任务参考 | 期望轨迹、航点及速度指令 |
| 动力学条件 | 质量、惯量、风场、气动参数及电机特性 |
| 模型监督 | 仿真或实飞连续状态与动作 |
| 奖励 | 跟踪误差、碰撞风险与控制代价 |

先对短时域姿态和轨迹跟踪建立闭环，再增加气动与执行器残差模型。Actor 使用部署时可获得的观测；训练模型可使用仿真真值及环境条件。这里给出结构迁移设计，效果需由无人机任务实验验证。

## 5. 分析与总结

共享世界模型复用不同任务中的动力学数据，短时域梯度提供局部行为改进，Critic 连接长期目标。模型误差会同时影响预测轨迹与策略梯度，需比较实际动力学下的回报、轨迹偏差和局部动作响应。

| 相关方向 | 在本主题中的位置 |
|---|---|
| PILCO、SVG | 可微模型与策略梯度基础 |
| SHAC | 短时域回报与末端价值的优化结构 |
| Dreamer | 学习模型内的行为优化，使用潜在状态表示 |
| SuperTrack、ControlVAE、SuperDyno | 可微代理动力学与物理运动控制的相关工作 |
| DeepMimic、PHC、BeyondMimic、SONIC | 运动模仿与通用跟踪策略的任务背景 |


## 6. 关联与参考

- 前置：[可微模型与策略梯度](07_differentiable-policy-optimization.md)、[SHAC](08_shac.md)。
- 对照：[MBPO](03_mbpo.md)、[Dreamer](09_dreamer.md)。
- [SuperMimic 官方项目](https://supermimic.github.io/)：论文入口、系统说明与实验展示。
- [SuperMimic 官方实现](https://github.com/HybridRobotics/supermimic/tree/7e94b073940b7a19b9a82ff01e191cfb35c4ba91)：本文梯度边界与训练分工的实现依据。
