# SHAC：短时域 Actor-Critic

## 1. 概述

SHAC 使用可微仿真器计算短时域策略梯度，并用 Critic 估计截断位置之后的回报。Actor 同时接收即时奖励梯度和末端价值梯度，从而把长任务分解为连续的短时域优化。

## 2. 基础与定义

### 2.1 网络结构

系统由 Actor $\pi_\theta$、Critic $V_\psi$、目标 Critic $V_{\bar\psi}$ 与可微动力学 $f$ 构成：

$$
a_t=\pi_\theta(s_t),\qquad s_{t+1}=f(s_t,a_t).
$$

Actor 输出动作，Critic 估计当前策略的未来折扣回报，目标 Critic 为自举提供缓慢变化的估计。原始方法通过可微仿真器获得动力学导数。

### 2.2 轨迹边界

设本次展开长度为 $H$，奖励记为 $r_t=r(s_t,a_t)$，折扣为 $\gamma$。轨迹片段从采样状态开始，片段之间截断计算图。真正终止时末端价值为零；外部时间限制和计算截断使用重置前末状态的价值。

## 3. 核心原理

### 3.1 Actor 目标

对未提前终止的片段，Actor 最小化：

$$
L_\pi(\theta)=-\mathbb E\left[\sum_{t=0}^{H-1}\gamma^t r_t+\gamma^H V_{\bar\psi}(s_H)\right].
$$

优化时固定目标 Critic 参数，保留 $s_H$ 对策略参数的依赖：

$$
\nabla_\theta L_\pi=-\mathbb E\left[
\sum_{t=0}^{H-1}\gamma^t\frac{d r_t}{d\theta}
+\gamma^H\frac{\partial V_{\bar\psi}(s_H)}{\partial s_H}\frac{d s_H}{d\theta}\right].
$$

第一项评价已展开的动作效果，第二项评价这些动作留下的末端状态。长时间任务的信息由末端价值传入短片段。

<details class="kt-derivation" id="shac-terminal-gradient">
<summary>末端价值梯度</summary>

取标量系统 $s_1=s_0+\theta$、$s_0=1$，奖励为零，$V_w(s)=-ws^2$，$w=2$，$\gamma=0.9$。一步目标为：

$$
J_1=-0.9\times2(1+\theta)^2,\qquad
\left.\frac{dJ_1}{d\theta}\right|_{\theta=0.5}=-5.4.
$$

```python
import torch

theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
w = torch.tensor(2.0, dtype=torch.float64)  # 固定价值函数参数
s1 = 1.0 + theta
value = -w * s1.square()
objective = 0.9 * value
(gradient,) = torch.autograd.grad(objective, theta)
assert torch.allclose(gradient, torch.tensor(-5.4, dtype=torch.float64))
assert not w.requires_grad
```

若把 `value` 替换为 `value.detach()`，当前示例的目标失去全部策略依赖。冻结 Critic 参数保留了状态梯度，分离价值输出则删除了该梯度。

</details>

### 3.2 Critic 目标

Critic 通过回归有限时域 $\lambda$-return 学习长期价值。记自举掩码 $b_t$ 为真正终止时取零、其余取一，片段内反向递推：

$$
G_H^\lambda=V_{\bar\psi}(s_H),
$$

$$
G_t^\lambda=r_t+\gamma b_t\left[(1-\lambda)V_{\bar\psi}(s_{t+1})+\lambda G_{t+1}^\lambda\right].
$$

递推在当前片段边界结束，重置后的轨迹作为新片段处理。价值损失和目标参数更新为：

$$
L_V(\psi)=\frac12\mathbb E\left[(V_\psi(s_t)-\operatorname{sg}(G_t^\lambda))^2\right],
$$

$$
\bar\psi\leftarrow(1-\tau)\bar\psi+\tau\psi.
$$

$\operatorname{sg}$ 表示停止梯度。Actor 用价值对状态的导数改进行为，Critic 用固定回报目标校准价值。

<details class="kt-derivation" id="shac-lambda-return">
<summary>有限时域回报展开</summary>

在连续片段内，令剩余长度为 $N=H-t$，定义 $n$ 步回报：

$$
G_t^{(n)}=\sum_{j=0}^{n-1}\gamma^j r_{t+j}+\gamma^nV_{\bar\psi}(s_{t+n}).
$$

有限时域混合为：

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{N-1}\lambda^{n-1}G_t^{(n)}+\lambda^{N-1}G_t^{(N)}.
$$

各项权重之和为一。提取首个奖励并合并剩余回报，可得到正文递推式。$\lambda=0$ 使用一步自举，$\lambda=1$ 使用整个片段的回报与末端价值。

</details>

### 3.3 梯度分工

| 更新对象 | 目标来源 | 保留的梯度路径 |
|---|---|---|
| Actor | 片段奖励与末端价值 | 策略 → 动作 → 状态 → 奖励和价值 |
| Critic | 固定的 $\lambda$-return | Critic 参数 → 当前价值预测 |
| 目标 Critic | Critic 参数 | 参数滑动平均 |

PPO 通过概率比值和优势更新策略；SHAC 通过动力学导数更新策略。二者均可使用 Critic 和多步回报，但 Actor 的梯度来源不同。

## 4. 实现与应用

### 4.1 更新流程

1. 从并行环境的当前状态开始，截断跨片段的旧计算图。
2. 固定目标 Critic，执行 $H$ 步策略与可微仿真。
3. 在终止或截断处完成当前片段的回报，汇总 Actor 损失并反向传播。
4. 保存分离后的状态、奖励和边界信息，构造 Critic 目标。
5. 更新 Critic 和目标 Critic，继续推进环境。

同一批并行环境可以在不同时间终止。每个环境分别维护片段边界，避免把重置状态的价值接到终止轨迹上。

### 4.2 无人机控制

对姿态与轨迹跟踪，短展开覆盖电机响应和局部动力学，Critic 补充后续跟踪误差与任务进度。动作经过执行器模型后进入刚体方程，奖励对位置、速度、姿态及控制代价求导。

验证应同时记录任务回报、Actor 梯度范数、Critic 预测误差与展开时域。比较不同时域时保持交互量和计算预算可比，并在部署使用的仿真器中重新评估策略。

## 5. 分析与总结

缩短时域减少长链反向传播的数值困难，也把更多长期判断交给 Critic。价值预测接近回报，并不自动保证其状态导数准确；Actor 的更新还需要检查末端价值梯度是否稳定。

SHAC 适合作为“短时域路径梯度 + 长期价值”的结构基础。将可微仿真器替换为学习模型后，还需处理模型误差、数据覆盖与策略利用模型偏差的问题。

## 6. 关联与参考

- 前置：[可微模型与策略梯度](07_differentiable-policy-optimization.md)。
- 后续：[SuperMimic](10_supermimic.md)。
- 对照：[PPO](../02_model-free-rl/03_ppo.md)、[TD-MPC](05_td-mpc.md)。
- Xu 等：[Accelerated Policy Learning with Parallel Differentiable Simulation](https://openreview.net/forum?id=ZSKRQMvttc)，ICLR 2022。
- 官方项目：[SHAC](https://short-horizon-actor-critic.github.io/)，官方实现：[NVlabs/DiffRL](https://github.com/NVlabs/DiffRL)。
