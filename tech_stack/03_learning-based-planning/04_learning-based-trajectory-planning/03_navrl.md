# NavRL 导航策略

## 1. 概述

NavRL 将静态障碍与动态障碍转换为结构化观测，用 PPO 学习速度决策，并在部署阶段通过安全修正处理策略动作。它把导航系统组织为“障碍感知 → 状态编码 → 速度策略 → 安全修正 → 跟踪控制”。[原论文](https://arxiv.org/abs/2409.15634)

## 2. 基础与定义

### 2.1 障碍表示

静态部分由占据表示提取距离信息；动态部分使用目标位置、速度与尺寸。网络将两类障碍特征与无人机运动状态、目标信息结合。感知输出需要有效性与时效约定，便于策略区分缺测和真实空旷区域。

### 2.2 策略动作

Actor 输出有界动作分布，经解码得到期望速度。速度参考交给低层控制器实现。本文统一记作 $v_t^{\mathrm{raw}}=\mathcal D(a_t)$，具体坐标变换、缩放及分布参数以所复现版本为准。[官方实现](https://github.com/Zhefan-Xu/NavRL)

## 3. 核心原理

### 3.1 观测与策略网络

可将信息流表示为

$$
z_t^{\mathrm{static}}=f_s(o_t^{\mathrm{static}}),\quad
z_t^{\mathrm{dynamic}}=f_d(o_t^{\mathrm{dynamic}}),\quad
\pi_\theta(a_t\mid z_t^{\mathrm{static}},z_t^{\mathrm{dynamic}},x_t).
$$

$x_t$ 为无人机自身状态及目标信息。结构化障碍表示提供从仿真到真实传感器的共同接口；迁移质量取决于检测、跟踪和数据预处理的一致性。

### 3.2 PPO 训练

训练中根据动作推进无人机环境，累计任务回报，计算价值目标与优势，再更新 Actor 和 Critic。奖励围绕目标进度、障碍风险和运动平滑表达任务偏好。其优化机制见 [PPO](/knowledge-tree/learning-based-planning/rl-algorithms/model-free-rl/ppo/)，环境侧重点是速度动作对真实转移的影响。

### 3.3 速度障碍与动作修正

设障碍物相对位置为 $r=q-p$，双方速度为 $v_o,v$，组合半径为 $R$。有限时域速度障碍集合定义为

$$
\mathrm{VO}_{\tau}=\{v:\exists t\in(0,\tau],\ \|r+(v_o-v)t\|<R\}.
$$

NavRL 的安全层利用速度障碍构造限制，再通过线性规划修正策略速度。安全层的输入、预测模型与控制约束共同决定结果。[安全层说明](https://arxiv.org/html/2409.15634v2#S3.SS4)

<details class="kt-derivation" id="closest-approach">
<summary>匀速模型的最近接近距离</summary>

令 $w=v_o-v$，则距离平方为

$$
f(t)=\|r+wt\|^2,\qquad f'(t)=2w^{\mathsf T}(r+wt).
$$

当 $\|w\|>0$ 时，有限时域的最小距离时刻为

$$
t^*=\operatorname{clip}\!\left(-\frac{r^{\mathsf T}w}{\|w\|^2},0,\tau\right),
\qquad d_{\min}=\|r+wt^*\|.
$$

相对速度为零时，距离保持 $\|r\|$。这一推导用于理解和测试匀速风险判断；完整安全层还要处理几何近似、多个障碍和速度限制。

</details>

## 4. 实现与应用

### 4.1 复现流程

固定论文对应的代码版本，核对静态距离表示、动态目标数量与掩码、坐标变换、动作范围和奖励配置。先用理想状态验证策略执行，再接入检测与跟踪误差。同步保存原始策略速度、修正速度和实测速度。

### 4.2 系统评估

分别评价策略本身和带安全修正的完整系统：成功率、碰撞率、修正频率、速度改变量、到达时间与感知延迟。固定相同障碍运动轨迹，对照静态场景、迎面相遇、交叉运动和遮挡恢复。

## 5. 分析与总结

NavRL 的关键是把动态障碍信息连接到策略与执行层。速度安全判断依赖有限时域预测，实际安全还受跟踪误差、目标漏检和执行延迟影响。安全层改善后的系统表现应与策略原始表现分别报告，便于定位改进来源。

## 6. 关联与参考

- [无人机任务设计](/knowledge-tree/learning-based-planning/engineering-practice/rl-engineering-pitfalls/)。
- [动态障碍跟踪](/knowledge-tree/classical-planning-control/perception-slam/dynamic-obstacle-tracking/)。
- [Xu 等：NavRL](https://arxiv.org/abs/2409.15634)。
- [Zhefan-Xu/NavRL](https://github.com/Zhefan-Xu/NavRL)。
