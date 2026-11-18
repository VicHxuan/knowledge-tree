# TD-MPC：潜在动力学与价值规划

## 1. 概述

TD-MPC 在任务相关的潜在空间中进行短时域规划，并用动作价值函数评价规划终点。模型学习同时受到奖励预测、价值学习和潜在状态一致性的约束，使表示保留决策所需的信息。

## 2. 基础与定义

### 2.1 网络结构

系统包含观测编码器 $h_\phi$、潜在动力学 $d_\phi$、奖励模型 $R_\phi$、动作价值函数 $Q_\psi$ 和辅助策略 $\pi_\theta$：

$$
z_t=h_\phi(o_t),\qquad
\hat z_{t+1}=d_\phi(\hat z_t,a_t),\qquad
\hat r_t=R_\phi(\hat z_t,a_t).
$$

辅助策略生成部分候选轨迹，并提供末端价值查询的动作。实际执行的动作由滚动规划器选择。

### 2.2 规划目标

从当前编码 $z_0$ 出发，候选动作序列的评分为：

$$
\hat J(A)=\sum_{t=0}^{H-1}\gamma^tR_\phi(\hat z_t,a_t)
+\gamma^H Q_\psi(\hat z_H,\pi_\theta(\hat z_H)).
$$

短展开描述近期状态与奖励，末端价值补充后续控制收益。双 Q 实现使用较小值进行保守估计。

## 3. 核心原理

### 3.1 任务相关模型学习

从经验库采样连续片段，令 $\hat z_0=h_\phi(o_0)$，后续沿经验动作展开。单步损失包含：

$$
\ell_t=c_r(R_\phi(\hat z_t,a_t)-r_t)^2
+c_q(Q_\psi(\hat z_t,a_t)-y_t)^2
+c_z\|\hat z_{t+1}-\operatorname{sg}(h_{\bar\phi}(o_{t+1}))\|^2.
$$

目标编码器参数为 $\bar\phi$。TD 目标使用实际下一观测的编码与目标价值网络：

$$
y_t=\operatorname{sg}\left[r_t+\gamma(1-d_t)
Q_{\bar\psi}(z_{t+1},\pi_\theta(z_{t+1}))\right],
\qquad z_{t+1}=h_\phi(o_{t+1}).
$$

$d_t$ 标记真正终止，目标构造整体停止梯度。整个训练片段的损失按预测深度加权：

$$
L_{M,Q}=\sum_{t=0}^{H-1}\eta^t\ell_t,\qquad 0<\eta\leq1.
$$

这里 $\eta$ 为训练损失权重，与回报折扣 $\gamma$ 分别控制不同对象。

<details class="kt-derivation" id="tdmpc-representation-loss">
<summary>潜在一致性与任务监督</summary>

一致性损失使模型预测 $d_\phi(\hat z_t,a_t)$ 接近实际下一观测的目标编码。目标编码停止梯度，预测分支接收误差信号。

奖励与价值损失同时作用于当前表示。若表示把奖励或未来价值不同的状态映射到同一点，相关预测损失会推动编码器区分这些状态。因此，任务监督约束表示保留决策信息。

编码器、动力学和价值函数通过联合损失协同学习。原始 TD-MPC 的目标侧编码、价值目标与在线编码的参数使用方式可在官方实现的更新函数中核对。

</details>

### 3.2 辅助策略学习

辅助策略通过动作价值梯度训练：

$$
L_\pi=-\mathbb E_z[Q_\psi(z,\pi_\theta(z))],
$$

$$
\nabla_\theta L_\pi=-\mathbb E_z\left[
\nabla_aQ_\psi(z,a)|_{a=\pi_\theta(z)}\nabla_\theta\pi_\theta(z)\right].
$$

策略更新时固定潜在训练状态与 Q 参数。该梯度改善候选动作先验，在线规划再比较完整候选序列。

### 3.3 滚动规划

规划器混合辅助策略产生的轨迹和从动作序列分布采样的轨迹，按模型回报与末端价值排序，再更新采样分布。原始实现对精英序列使用回报相关权重拟合均值与方差。

执行第一个动作后，利用新观测重建起点，并平移前一次动作序列作为搜索初始化。

## 4. 实现与应用

### 4.1 训练与控制

1. 编码当前观测，在潜在空间规划并执行动作。
2. 将实际转移加入经验库。
3. 采样连续片段，展开潜在动力学。
4. 更新奖励、价值和潜在一致性损失，再训练辅助策略。
5. 更新目标网络，重复规划与交互。

无人机应用需同时验证规划延迟、实际闭环回报与模型外推误差。碰撞与硬约束需要单独的预测或约束接口，任务相关潜在状态可能缺少直接检查几何约束的物理量。

### 4.2 TD-MPC2

TD-MPC2 延续潜在动力学、价值估计和短时域规划结构，扩展模型规模与多任务学习能力，并改进归一化、价值学习及策略先验等训练设计。

阅读时先掌握 TD-MPC 的模型用途与损失，再对照 TD-MPC2 的具体实现。本文中的标量平方损失用于讲解原始方法；TD-MPC2 的分布预测与正则化按其版本单独核对。

## 5. 分析与总结

TD-MPC 的模型既服务于近期预测，也由长期价值任务塑造。与 PETS 相比，它增加潜在表示和末端价值；与 SHAC 相比，实际动作主要通过在线采样规划确定，辅助策略使用 Q 的动作梯度学习。

## 6. 关联与参考

- 前置：[PETS](04_pets.md)。
- 对照：[MuZero](06_muzero.md)、[SHAC](08_shac.md)、[Dreamer](09_dreamer.md)。
- Hansen 等：[Temporal Difference Learning for Model Predictive Control](https://proceedings.mlr.press/v162/hansen22a.html)，ICML 2022。
- [TD-MPC 官方实现](https://github.com/nicklashansen/tdmpc)。
- [TD-MPC2 官方项目](https://www.tdmpc2.com/)，ICLR 2024；[官方实现](https://github.com/nicklashansen/tdmpc2)。
