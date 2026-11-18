# 可微模型与策略梯度

## 1. 概述

可微策略优化把策略、动力学和奖励连接成计算图，通过未来回报对动作的导数更新策略。PILCO 从概率模型中的期望代价求导；SVG 用重参数化连接随机策略、随机动力学与价值函数；SHAC 将这种梯度限制在短时域，并用 Critic 估计后续回报。

## 2. 基础与定义

### 2.1 模型与策略

设状态为 $s_t$，动作为 $a_t$，策略参数为 $\theta$，动力学参数为 $\phi$，价值参数为 $\psi$。先考虑确定性闭环：

$$
a_t=\pi_\theta(s_t),\qquad s_{t+1}=f_\phi(s_t,a_t).
$$

模型可以来自解析动力学、可微仿真器或数据拟合。Actor 更新时固定 $\phi$ 与 $\psi$，保留模型和价值函数对输入的导数。初始状态来自采样分布，在一次更新中视为固定。

### 2.2 优化目标

对长度为 $H$ 的轨迹，奖励 $r_t=r(s_t,a_t)$，折扣因子为 $\gamma$：

$$
J_H(\theta)=\mathbb E\left[\sum_{t=0}^{H-1}\gamma^t r(s_t,a_t)+\gamma^H V_\psi(s_H)\right],\qquad L_\pi=-J_H.
$$

前一项由模型展开计算，末端价值估计 $H$ 步之后的回报。有限任务真正终止时，后续价值取零。

## 3. 核心原理

### 3.1 策略梯度

定义 $D_t=\partial s_t/\partial\theta$，$U_t=\partial a_t/\partial\theta$。以下 $f_s,f_a$ 为动力学对状态和动作的雅可比，$\pi_s,\pi_\theta$ 为策略对状态和参数的雅可比。省略求值点，有：

$$
U_t=\pi_sD_t+\pi_\theta,\qquad
D_{t+1}=f_sD_t+f_aU_t=(f_s+f_a\pi_s)D_t+f_a\pi_\theta.
$$

因此，目标梯度为：

$$
\nabla_\theta J_H=
\mathbb E\left[\sum_{t=0}^{H-1}\gamma^t(r_sD_t+r_aU_t)+\gamma^H V_s(s_H)D_H\right].
$$

策略既改变当前动作，也通过状态转移影响后续动作和末端价值。反向传播按这条依赖链累积各时刻贡献。

<details class="kt-derivation" id="pathwise-two-step">
<summary>二步回报的梯度</summary>

固定 $s_0$ 时，$D_0=0$，$U_0=\pi_\theta(s_0)$。第一步与第二步状态导数分别为：

$$
D_1=f_{a,0}\pi_{\theta,0},\qquad
D_2=(f_{s,1}+f_{a,1}\pi_{s,1})D_1+f_{a,1}\pi_{\theta,1}.
$$

对 $J_2=r_0+\gamma r_1+\gamma^2V_\psi(s_2)$ 求导：

$$
\nabla_\theta J_2=r_{a,0}\pi_{\theta,0}
+\gamma\left[r_{s,1}D_1+r_{a,1}(\pi_{s,1}D_1+\pi_{\theta,1})\right]
+\gamma^2V_s(s_2)D_2.
$$

末项把 Critic 对末状态的评价传回前两步动作。对 $V_\psi(s_2)$ 整体停止梯度会删除这一项。

</details>

### 3.2 随机价值梯度

SVG 将噪声从可学习映射中分离：

$$
a_t=g_\theta(s_t,\epsilon_t),\qquad
s_{t+1}=f_\phi(s_t,a_t,\xi_t),
$$

其中 $\epsilon_t,\xi_t$ 来自与参数无关的基础分布。在允许交换微分与期望的条件下，固定一次噪声样本即可对整条路径求导，再用多条路径估计期望梯度。

一步价值展开给出动作梯度的来源：

$$
Q(s,a)=\mathbb E_\xi[r(s,a)+\gamma V_\psi(f_\phi(s,a,\xi))],
$$

$$
\nabla_a Q=\mathbb E_\xi\left[r_a+\gamma V_s f_a\right].
$$

| 方法 | 价值信号 | 动力学导数的用途 |
|---|---|---|
| SVG(0) | 学习的 $Q(s,a)$ | 通过 $Q_a$ 更新策略 |
| SVG(1) | 单步奖励与下一状态价值 | 通过 $f_a$ 传播一次价值梯度 |
| 多步 SVG | 多步奖励与后续价值 | 沿轨迹传播状态与动作导数 |

原始 SVG 还利用实际环境轨迹及推断的噪声，在访问过的状态上计算模型局部导数，减少完全依赖长预测轨迹造成的状态偏移。

### 3.3 PILCO

PILCO 学习高斯过程动力学，并在策略作用下传播状态分布。若以高斯分布近似 $p(s_t)$，可用矩匹配计算下一状态的均值和协方差，再优化期望代价：

$$
J_c(\theta)=\sum_{t=1}^{T}\mathbb E_{p(s_t;\theta)}[c(s_t)].
$$

梯度经过“策略参数 → 状态分布矩 → 期望代价”。模型不确定性参与分布传播，影响策略对未来风险的评价。

<details class="kt-derivation" id="pilco-moment-gradient">
<summary>分布矩的梯度</summary>

令 $m_t=(\mu_t,\Sigma_t)$ 表示近似状态分布，矩传播记为 $m_{t+1}=F(m_t,\theta)$。则：

$$
\frac{d m_{t+1}}{d\theta}=F_m\frac{d m_t}{d\theta}+F_\theta,
$$

$$
\frac{dJ_c}{d\theta}=\sum_t\left[
\frac{\partial\mathbb E[c]}{\partial\mu_t}\frac{d\mu_t}{d\theta}
+\left\langle\frac{\partial\mathbb E[c]}{\partial\Sigma_t},\frac{d\Sigma_t}{d\theta}\right\rangle_F\right].
$$

矩匹配对所选高斯过程核及输入分布计算预测矩，再以高斯分布近似传播结果。近似来自分布表示；该机制与只在均值处进行一阶线性化有不同的误差来源。

</details>

## 4. 实现与应用

### 4.1 训练流程

1. 收集环境状态、动作与下一状态，训练模型；使用已知可微动力学时直接调用模型。
2. 从环境数据采样起点，固定本轮模型及 Critic 参数。
3. 用当前策略展开 $H$ 步，计算奖励与末端价值。
4. 对 $L_\pi$ 反向传播，更新 Actor。
5. 用停止梯度的回报目标更新 Critic，重新采集数据并刷新模型。

冻结参数时，自动微分仍需记录模型与价值函数的输入依赖。模型输出、下一状态或末端价值的整体 `detach` 会切断对应路径。

### 4.2 梯度验证

选择短轨迹和固定噪声，用中心差分检查自动微分：

$$
\frac{\partial J}{\partial\theta_i}\approx
\frac{J(\theta+\varepsilon e_i)-J(\theta-\varepsilon e_i)}{2\varepsilon}.
$$

同时检查模型预测误差与局部导数误差。对无人机动力学，优先核对姿态更新、推力响应、动作饱和及气动载荷在当前飞行包线内的导数。

## 5. 分析与总结

路径梯度利用动力学的局部结构。其质量同时依赖模型预测、模型雅可比、奖励导数与末端价值导数。

长时域反向传播包含多个闭环雅可比 $f_s+f_a\pi_s$ 的乘积；其放大或收缩可能导致梯度爆炸或消失。缩短展开时域减少这条链的长度，同时增加对 Critic 的依赖。接触切换、碰撞终止与硬裁剪需要按具体动力学检查可微性。

PILCO 强调不确定性的解析传播，SVG 建立随机路径梯度框架，SHAC 将短时域路径梯度与长期价值估计结合。

## 6. 关联与参考

- 前置：[Model-Based RL 基础](01_model-based-rl-foundations.md)。
- 后续：[SHAC](08_shac.md)、[Dreamer](09_dreamer.md)、[SuperMimic](10_supermimic.md)。
- Deisenroth & Rasmussen：[PILCO](https://icml.cc/2011/papers/323_icmlpaper.pdf)，ICML 2011。
- Heess 等：[Learning Continuous Control Policies by Stochastic Value Gradients](https://proceedings.neurips.cc/paper_files/paper/2015/hash/148510031349642de5ca0c544f31b2ef-Abstract.html)，NeurIPS 2015。
