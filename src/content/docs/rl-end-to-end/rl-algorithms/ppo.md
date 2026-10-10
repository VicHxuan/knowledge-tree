---
title: "PPO 算法完整解析与代码实现"
sidebar:
  order: 4
---

PPO（Proximal Policy Optimization，近端策略优化）是一类同策略（On-Policy）Actor-Critic 算法。它从基础策略梯度出发，先用重要性采样比率描述新旧策略之间的变化，再用截断替代目标抑制过大的策略更新。这样既避免了 TRPO 的二阶约束优化，又允许对同一批同策略轨迹进行若干轮小批量更新。

本文的符号和基础结论承接 [强化学习核心公式手册](/knowledge-tree/rl-end-to-end/rl-algorithms/rl-foundations/)，算法改造依据 [PPO 论文](https://arxiv.org/abs/1707.06347)：策略网络记为 $\pi_\theta$，价值网络记为 $V_\phi$，折扣因子为 $\gamma$，终止标记为 $d_t\in\{0,1\}$。

---

## 1. 从策略梯度到 PPO

### 1.1 起点：优势形式的策略梯度

基础策略梯度定理可写为

$$
\nabla_\theta J(\theta)
\propto
\mathbb{E}_{s_t,a_t\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,
A^{\pi_\theta}(s_t,a_t)
\right].
$$

用采样得到的优势估计 $\hat A_t$ 代替真实优势函数，可构造需要最大化的经验目标

$$
L^{\mathrm{PG}}(\theta)
=
\hat{\mathbb{E}}_t
\left[
\log\pi_\theta(a_t\mid s_t)\hat A_t
\right].
$$

当 $\hat A_t>0$ 时，应提高动作 $a_t$ 的概率；当 $\hat A_t<0$ 时，应降低该动作的概率。问题在于：若在同一批轨迹上连续进行多轮更新，参数改变后，数据仍来自采样时的旧策略 $\pi_{\theta_{\mathrm{old}}}$，直接反复优化 $L^{\mathrm{PG}}$ 可能使新策略一次偏离过远，破坏训练稳定性。

### 1.2 用重要性采样比率复用旧策略数据

采样阶段固定策略参数为 $\theta_{\mathrm{old}}$，并保存旧策略对实际动作给出的对数概率。更新时定义新旧策略的概率比率

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
=
\exp\!\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
\right).
$$

因为

$$
\mathbb{E}_{a_t\sim\pi_{\theta_{\mathrm{old}}}}
\left[r_t(\theta)f(a_t)\right]
=
\mathbb{E}_{a_t\sim\pi_\theta}
\left[f(a_t)\right],
$$

所以可将策略梯度目标改写为重要性采样替代目标

$$
L^{\mathrm{CPI}}(\theta)
=
\hat{\mathbb{E}}_t
\left[r_t(\theta)\hat A_t\right].
$$

在 $\theta=\theta_{\mathrm{old}}$ 处有 $r_t=1$，该目标的一阶梯度与原策略梯度一致。但是，只最大化该目标仍可能让概率比率发生过大的变化。

### 1.3 从 TRPO 的硬约束到 PPO 的截断目标

TRPO 通过平均 KL 散度显式限制新旧策略的距离：

$$
\begin{aligned}
\max_\theta\quad
&\hat{\mathbb{E}}_t
\left[r_t(\theta)\hat A_t\right],\\
\text{s.t.}\quad
&\hat{\mathbb{E}}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\,\|\,
\pi_\theta(\cdot\mid s_t)
\right)
\right]
\leq\delta.
\end{aligned}
$$

该方法需要近似求解带二阶信息的约束优化问题。PPO-Clip 将其改造成可直接使用一阶优化器训练的无约束目标：

$$
L^{\mathrm{CLIP}}(\theta)
=
\hat{\mathbb{E}}_t
\left[
\min\!\left(
r_t(\theta)\hat A_t,
\operatorname{clip}\!\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat A_t
\right)
\right].
$$

其作用分两种情况理解：

- 当 $\hat A_t>0$ 时，提高该动作概率是有利的，但当 $r_t>1+\epsilon$ 后，目标不再提供继续提高概率的收益。
- 当 $\hat A_t<0$ 时，降低该动作概率是有利的，但当 $r_t<1-\epsilon$ 后，目标不再提供继续降低概率的收益。

因此，$\min$ 选取的是较保守的替代目标。需要注意：截断限制的是目标函数提供的优化激励，并非把策略概率比率硬性投影在 $[1-\epsilon,1+\epsilon]$ 内。

由于代码使用梯度下降，实际策略损失取最大化目标的相反数：

$$
L_{\pi}(\theta)
=
-L^{\mathrm{CLIP}}(\theta).
$$

## 2. 优势估计、价值拟合与联合目标

### 2.1 从优势定义推导 GAE 与 Critic target

这一部分要解决两个不同但相连的问题：

1. Actor 需要估计优势 $A^{\pi_{\mathrm{old}}}(s_t,a_t)$，判断动作 $a_t$ 比旧策略在 $s_t$ 下的平均表现好还是差。
2. Critic 需要一个回报目标 $\hat R_t$，使 $V_\phi(s_t)$ 向“从 $s_t$ 出发能够获得的折扣回报”拟合。

#### 2.1.1 TD 残差为什么可以估计优势

优势函数的定义为

$$
A^{\pi_{\mathrm{old}}}(s_t,a_t)
=
Q^{\pi_{\mathrm{old}}}(s_t,a_t)
-V^{\pi_{\mathrm{old}}}(s_t).
$$

对动作价值函数进行一步贝尔曼展开：

$$
Q^{\pi_{\mathrm{old}}}(s_t,a_t)
=
\mathbb{E}
\left[
r_{t+1}
+\gamma(1-d_t)V^{\pi_{\mathrm{old}}}(s_{t+1})
\mid s_t,a_t
\right].
$$

将其代回优势定义：

$$
A^{\pi_{\mathrm{old}}}(s_t,a_t)
=
\mathbb{E}
\left[
r_{t+1}
+\gamma(1-d_t)V^{\pi_{\mathrm{old}}}(s_{t+1})
-V^{\pi_{\mathrm{old}}}(s_t)
\mid s_t,a_t
\right].
$$

因此，用旧 Critic 近似真实价值函数后，单条轨迹上的 TD 残差

$$
\boxed{
\delta_t
=
r_{t+1}
+\gamma(1-d_t)V_{\phi_{\mathrm{old}}}(s_{t+1})
-V_{\phi_{\mathrm{old}}}(s_t)
}
$$

就是单步优势的样本估计。若旧 Critic 完全准确，则

$$
\mathbb{E}[\delta_t\mid s_t,a_t]
=
A^{\pi_{\mathrm{old}}}(s_t,a_t).
$$

#### 2.1.2 从单步 TD 残差到多步 GAE

只使用 $\delta_t$ 自举程度高、方差小，但会继承 Critic 的估计偏差。为了加入后续真实奖励，可将连续 TD 残差累加。为简化符号，以下将 $V_{\phi_{\mathrm{old}}}$ 简记为 $V_{\mathrm{old}}$。忽略中途终止时，前几项为

$$
\begin{aligned}
\hat A_t^{(1)}
&=\delta_t,\\
\hat A_t^{(2)}
&=\delta_t+\gamma\delta_{t+1},\\
\hat A_t^{(3)}
&=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}.
\end{aligned}
$$

例如二步优势展开后，中间的价值项会抵消：

$$
\begin{aligned}
\hat A_t^{(2)}
&=r_{t+1}+\gamma V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t)\\
&\quad+\gamma\left[r_{t+2}+\gamma V_{\mathrm{old}}(s_{t+2})-V_{\mathrm{old}}(s_{t+1})\right]\\
&=r_{t+1}+\gamma r_{t+2}+\gamma^2V_{\mathrm{old}}(s_{t+2})-V_{\mathrm{old}}(s_t).
\end{aligned}
$$

这说明“多步 TD 残差之和”等于“多步自举回报减去当前旧价值”。GAE 使用 $\lambda$ 对这些不同步数的估计做指数加权；计入终止标记后为

$$
\boxed{
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l
\left(
\prod_{j=0}^{l-1}(1-d_{t+j})
\right)
\delta_{t+l}
}.
$$

程序中采用等价的倒序递推形式：

$$
\boxed{
\hat A_t
=
\delta_t
+\gamma\lambda(1-d_t)\hat A_{t+1}
}.
$$

其中 $\lambda=0$ 时退化为单步 TD 优势；$\lambda\to1$ 时趋近当前 rollout 内的多步回报减去旧价值。如果轨迹在 rollout 末端尚未终止，最后仍使用 $V_{\phi_{\mathrm{old}}}(s_T)$ 自举；只有到达真正终止状态时，末端自举值才为 $0$。

#### 2.1.3 Critic target 为什么是 $\hat A_t+V_{\mathrm{old}}(s_t)$

由前面的抵消关系，GAE 可以统一写成

$$
\hat A_t
=
\hat R_t^{(\lambda)}
-V_{\phi_{\mathrm{old}}}(s_t),
$$

其中 $\hat R_t^{(\lambda)}$ 是 GAE 隐含的 $\lambda$-return。移项立即得到

$$
\boxed{
\hat R_t^{(\lambda)}
=
\hat A_t
+V_{\phi_{\mathrm{old}}}(s_t)
}.
$$

这不是额外规定的经验公式，而是从“优势 = 回报 − 基线”直接恢复出的回报目标。由于轨迹由 $\pi_{\mathrm{old}}$ 采样，对同一状态的动作和后继转移取平均后，$\hat R_t^{(\lambda)}$ 近似 $V^{\pi_{\mathrm{old}}}(s_t)$，因此可以作为 Critic 的监督目标。

下文将 $\hat R_t^{(\lambda)}$ 简记为 $\hat R_t$。

计算 target 时需要注意：

- $\hat R_t^{(\lambda)}$ 由旧价值估计构造，在优化阶段作为常数停止梯度。
- Actor 可以使用标准化后的优势，但 Critic target 必须由**未标准化**的 $\hat A_t$ 构造，否则会破坏回报的真实数值尺度。
- 下方代码中的 `returns = advantages + values_buffer` 正是上述等式的实现。

### 2.2 价值函数损失

最基本的价值拟合目标是

$$
L_V(\phi)
=
\frac{1}{2}
\hat{\mathbb{E}}_t
\left[
\left(V_\phi(s_t)-\hat R_t\right)^2
\right].
$$

当前代码采用常见的价值截断扩展。先定义

$$
V_t^{\mathrm{clip}}
=
V_{\phi_{\mathrm{old}}}(s_t)
+\operatorname{clip}\!\left(
V_\phi(s_t)-V_{\phi_{\mathrm{old}}}(s_t),
-\epsilon,
\epsilon
\right),
$$

再取普通误差与截断误差中较大的一个：

$$
L_V^{\mathrm{clip}}(\phi)
=
\frac{1}{2}
\hat{\mathbb{E}}_t
\left[
\max\!\left(
\left(V_\phi(s_t)-\hat R_t\right)^2,
\left(V_t^{\mathrm{clip}}-\hat R_t\right)^2
\right)
\right].
$$

价值截断不是 PPO 论文中截断策略目标所必需的组成部分，而是许多工程实现用于限制 Critic 单次变化的附加措施。

### 2.3 熵奖励与总损失

策略熵为

$$
\mathcal H\!\left(\pi_\theta(\cdot\mid s_t)\right)
=
-\mathbb{E}_{a\sim\pi_\theta}
\left[\log\pi_\theta(a\mid s_t)\right],
$$

用于避免策略过早变得确定。PPO 论文以“最大化目标”的形式写作

$$
\max_{\theta,\phi}\quad
L^{\mathrm{CLIP}}(\theta)
-c_vL_V(\phi)
+c_e\hat{\mathbb{E}}_t[\mathcal H_t].
$$

当前代码使用梯度下降，因此等价地最小化

$$
L_{\mathrm{total}}
=
L_\pi(\theta)
+c_vL_V^{\mathrm{clip}}(\phi)
-c_e\hat{\mathbb{E}}_t[\mathcal H_t].
$$

## 3. PPO-Clip 的完整训练流程

### 步骤 1：初始化网络与超参数

初始化 Actor 参数 $\theta$、Critic 参数 $\phi$ 和优化器，并设置 rollout 长度 $T$、折扣因子 $\gamma$、GAE 参数 $\lambda$、PPO 截断系数 $\epsilon$、更新轮数 $K$、小批量大小 $M$、价值损失系数 $c_v$ 与熵系数 $c_e$。

### 步骤 2：固定本轮旧策略并收集 rollout

将采样开始时的策略和价值函数记为 $\pi_{\theta_{\mathrm{old}}}$、$V_{\phi_{\mathrm{old}}}$，使用它们与环境交互 $T$ 步：

$$
a_t\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),
\qquad t=0,1,\ldots,T-1.
$$

下方实现没有额外复制一套旧网络，而是在整个 rollout 结束前不更新参数；因此采样期间的当前网络就是旧网络。

### 步骤 3：保存后续更新所需数据

对每个时间步保存

$$
\left(
s_t,
a_t,
r_{t+1},
d_t,
\ell_t^{\mathrm{old}},
v_t^{\mathrm{old}}
\right),
$$

其中

$$
\ell_t^{\mathrm{old}}
=
\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t),
\qquad
v_t^{\mathrm{old}}
=
V_{\phi_{\mathrm{old}}}(s_t).
$$

保存旧对数概率是为了在更新时固定重要性采样分母；保存旧价值是为了计算 GAE、回报 target 和价值截断。

### 步骤 4：倒序计算 GAE 与 Critic target

若 rollout 的最后状态 $s_T$ 并非终止状态，则用旧 Critic 自举：

$$
v_T^{\mathrm{boot}}
=
V_{\phi_{\mathrm{old}}}(s_T);
$$

若最后一步到达终止状态，则令 $v_T^{\mathrm{boot}}=0$。初始化 $\hat A_T=0$，然后对 $t=T-1,T-2,\ldots,0$ 依次计算以下中间变量：

$$
m_t=1-d_t,
$$

$$
v_{t+1}^{\mathrm{next}}
=
\begin{cases}
v_{t+1}^{\mathrm{old}}, & t<T-1,\\
v_T^{\mathrm{boot}}, & t=T-1,
\end{cases}
$$

$$
\delta_t
=
r_{t+1}
+\gamma m_t v_{t+1}^{\mathrm{next}}
-v_t^{\mathrm{old}},
$$

$$
\hat A_t
=
\delta_t
+\gamma\lambda m_t\hat A_{t+1},
$$

$$
\boxed{
\hat R_t
=
\hat A_t+v_t^{\mathrm{old}}
}.
$$

其中 $m_t$ 会在终止边界切断价值自举和 GAE 的跨回合传播。此处的 $\hat A_t$ 尚未标准化，因而 $\hat R_t$ 保留原始回报尺度。

严格实现时应区分 `terminated` 与 `truncated`：真正的 MDP 终止通常令 $m_t=0$，仅由时间上限造成的截断通常仍应从截断状态自举。下方精简代码用逻辑或合并了二者，适合展示核心流程，但实际训练可按环境语义分别处理。

### 步骤 5：标准化 Actor 优势并构造小批量

先打乱 $T$ 个索引并划分为大小为 $M$ 的小批量。下方代码在每个小批量 $\mathcal M$ 内，只对传入 Actor 损失的优势进行标准化：

$$
\bar A_{\mathcal M}
=
\frac{1}{M}\sum_{t\in\mathcal M}\hat A_t,
\qquad
\sigma_{\mathcal M}
=
\sqrt{
\frac{1}{M}
\sum_{t\in\mathcal M}
\left(\hat A_t-\bar A_{\mathcal M}\right)^2
},
$$

$$
\tilde A_t
=
\frac{
\hat A_t-\bar A_{\mathcal M}
}{
\sigma_{\mathcal M}+\eta
},
\qquad t\in\mathcal M,
$$

其中 $\eta$ 是防止除零的小常数。Critic 仍使用步骤 4 中未改变尺度的 $\hat R_t$。有些实现会在整个 rollout 上统一标准化，作用相近；这里给出的是与下方代码一致的小批量标准化。

### 步骤 6：在同一 rollout 上进行 $K$ 轮小批量更新

对每个小批量，先用更新中的当前网络重新计算

$$
\ell_t^{\mathrm{new}}
=
\log\pi_\theta(a_t\mid s_t),
\qquad
v_t^{\mathrm{new}}
=
V_\phi(s_t),
$$

再计算对数概率差和概率比率：

$$
\Delta\ell_t
=
\ell_t^{\mathrm{new}}-\ell_t^{\mathrm{old}},
\qquad
r_t(\theta)=\exp(\Delta\ell_t).
$$

策略目标的两个候选项为

$$
s_t^{(1)}=r_t(\theta)\tilde A_t,
$$

$$
s_t^{(2)}
=
\operatorname{clip}\!\left(r_t(\theta),1-\epsilon,1+\epsilon\right)
\tilde A_t,
$$

因此小批量策略损失为

$$
L_\pi
=
-\frac{1}{M}
\sum_{t\in\mathcal M}
\min\!\left(s_t^{(1)},s_t^{(2)}\right).
$$

价值截断中间量与损失为

$$
v_t^{\mathrm{clip}}
=
v_t^{\mathrm{old}}
+\operatorname{clip}\!\left(
v_t^{\mathrm{new}}-v_t^{\mathrm{old}},
-\epsilon,
\epsilon
\right),
$$

$$
L_V^{\mathrm{clip}}
=
\frac{1}{2M}
\sum_{t\in\mathcal M}
\max\!\left[
\left(v_t^{\mathrm{new}}-\hat R_t\right)^2,
\left(v_t^{\mathrm{clip}}-\hat R_t\right)^2
\right].
$$

计算当前策略熵均值

$$
\bar{\mathcal H}
=
\frac{1}{M}
\sum_{t\in\mathcal M}
\mathcal H\!\left(\pi_\theta(\cdot\mid s_t)\right),
$$

最终最小化

$$
\boxed{
L_{\mathrm{total}}
=
L_\pi+c_vL_V^{\mathrm{clip}}-c_e\bar{\mathcal H}
}.
$$

### 步骤 7：裁剪梯度并更新参数

反向传播得到联合参数梯度 $g=\nabla_{\theta,\phi}L_{\mathrm{total}}$。若最大梯度范数为 $g_{\max}$，全局范数裁剪可表示为

$$
g
\leftarrow
g\cdot
\min\!\left(1,\frac{g_{\max}}{\lVert g\rVert_2}\right).
$$

然后使用 Adam 更新 Actor 与 Critic。随着每个 Epoch 的更新，$\ell_t^{\mathrm{new}}$ 和 $v_t^{\mathrm{new}}$ 会改变，但缓冲区中的 $\ell_t^{\mathrm{old}}$、$v_t^{\mathrm{old}}$、$\hat A_t$ 与 $\hat R_t$ 在本轮始终固定。

### 步骤 8：丢弃旧 rollout 并重新采样

完成 $K$ 轮更新后丢弃本轮轨迹，用更新后的策略重新采样。旧数据只在本轮被有限复用，不能像经验回放数据那样跨多个策略版本长期使用，因此 PPO 仍是同策略算法。

## 4. 公式与下方代码的对应关系

| 数学量或步骤 | 代码中的对应实现 | 作用 |
| --- | --- | --- |
| $\pi_\theta(a\mid s)$ | `PPOAgent.actor`、`get_action_and_value` | 输出高斯策略并采样动作 |
| $V_\phi(s)$ | `PPOAgent.critic`、`get_value` | 估计状态价值 |
| $\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$ | 采样时保存的 `logprobs` | 固定重要性采样分母 |
| $\delta_t$ 与 $\hat A_t$ | 反向循环中的 `delta`、`lastgaelam` | 递推计算 GAE |
| $\hat R_t$ | `returns = advantages + values` | 构造 Critic 的监督目标 |
| $r_t(\theta)$ | `ratio = (newlogprob - old_logprob).exp()` | 衡量新旧策略对同一动作的概率变化 |
| $L_\pi$ | `pg_loss1`、`pg_loss2`、`torch.max(...)` | 最小化 $-\min(r_t\hat A_t,\operatorname{clip}(r_t)\hat A_t)$ |
| $L_V^{\mathrm{clip}}$ | `v_loss_unclipped`、`v_loss_clipped` | 限制价值预测单次变化 |
| $\mathcal H$ | `entropy.mean()` | 鼓励策略探索 |
| 多轮小批量更新 | `update_epochs` 与 `minibatch_size` | 有限复用本轮同策略轨迹 |

优势标准化、价值截断、Adam 和梯度范数裁剪属于稳定训练的工程措施；PPO 的核心算法改造是对新旧策略概率比率构造截断替代目标。

---

## 5. 核心代码架构实现

以下代码剔除了所有日志记录、命令行参数解析及环境包裹器的冗余代码，高度抽象了算法核心逻辑，完全符合 Python 规范与 PyTorch 工作流，用于展示连续动作空间下 PPO 算法的本质架构。

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# =======================================
# 1. 策略与价值网络定义
# =======================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    """PPO 智能体：包含独立的 Actor 和 Critic 网络（连续动作空间）"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Critic：评估状态价值
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor：输出动作均值
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # 动作对数标准差（作为可学习的独立参数）
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# =======================================
# 2. 核心算法流程演示
# =======================================

def ppo_training_loop(env, num_iterations, num_steps, batch_size, minibatch_size, update_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
    # 算法超参数
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5

    # 数据存储容器
    obs_buffer = torch.zeros((num_steps, obs_dim)).to(device)
    actions_buffer = torch.zeros((num_steps, action_dim)).to(device)
    logprobs_buffer = torch.zeros((num_steps,)).to(device)
    rewards_buffer = torch.zeros((num_steps,)).to(device)
    dones_buffer = torch.zeros((num_steps,)).to(device)
    values_buffer = torch.zeros((num_steps,)).to(device)

    next_obs = torch.Tensor(env.reset()[0]).to(device)
    next_done = torch.zeros(1).to(device)

    for iteration in range(num_iterations):
        # -----------------------------------
        # 步骤 2：环境交互与数据收集 (Rollout)
        # -----------------------------------
        for step in range(num_steps):
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buffer[step] = value.flatten()
            
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            # 执行动作，获取环境反馈
            next_obs_np, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminated, truncated)
            
            rewards_buffer[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor([next_done_np]).to(device)
            
            if next_done_np:
                next_obs = torch.Tensor(env.reset()[0]).to(device)

        # -----------------------------------
        # 步骤 3：广义优势估计 (GAE)
        # -----------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                
                # TD 误差
                delta = rewards_buffer[t] + gamma * nextvalues * nextnonterminal - values_buffer[t]
                # GAE 递归计算
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            # 计算目标回报 Return
            returns = advantages + values_buffer

        # -----------------------------------
        # 步骤 4 & 5：多轮小批量网络参数更新
        # -----------------------------------
        b_inds = np.arange(batch_size)
        
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # 获取当前网络的最新评估结果
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_buffer[mb_inds], actions_buffer[mb_inds]
                )
                
                # 计算概率比率 r_t
                logratio = newlogprob - logprobs_buffer[mb_inds]
                ratio = logratio.exp()

                # 优势值标准化
                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 1. 计算策略损失 (Clipped Surrogate Objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 2. 计算价值损失 (Clipped Value Loss)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values_buffer[mb_inds] + torch.clamp(
                    newvalue - values_buffer[mb_inds], -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # 3. 计算熵奖励
                entropy_loss = entropy.mean()

                # 汇总总损失
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                # 梯度反向传播与更新
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
```

---

## 6. 补充证明

### 6.1 重要性采样下两个期望相等的证明

第 1.2 节使用了

$$
\mathbb{E}_{a_t\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)}
\left[r_t(\theta)f(a_t)\right]
=
\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}
\left[f(a_t)\right],
$$

其中

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

该等式是在**给定同一个状态 $s_t$** 时，对动作分布进行换测度。成立条件是：凡是 $\pi_\theta(a\mid s_t)f(a)\neq0$ 的动作，都必须满足 $\pi_{\theta_{\mathrm{old}}}(a\mid s_t)>0$，否则概率比率没有定义。也就是说，新策略需要在旧策略的动作分布支撑集内。

#### 离散动作空间

根据离散期望的定义，逐步代入概率比率：

$$
\begin{aligned}
&\mathbb{E}_{a_t\sim\pi_{\theta_{\mathrm{old}}}}
\left[r_t(\theta)f(a_t)\right]\\
&=
\sum_{a_t}
\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
r_t(\theta)f(a_t)\\
&=
\sum_{a_t}
\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
f(a_t)\\
&=
\sum_{a_t}
\pi_\theta(a_t\mid s_t)f(a_t)\\
&=
\mathbb{E}_{a_t\sim\pi_\theta}
\left[f(a_t)\right].
\end{aligned}
$$

核心只是旧策略概率在乘积中被约掉，从而把旧策略下的期望转换成新策略下的期望。

#### 连续动作空间

连续动作空间将求和换为积分，证明完全相同：

$$
\begin{aligned}
&\mathbb{E}_{a_t\sim\pi_{\theta_{\mathrm{old}}}}
\left[r_t(\theta)f(a_t)\right]\\
&=
\int
\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
f(a_t)\,\mathrm da_t\\
&=
\int
\pi_\theta(a_t\mid s_t)f(a_t)\,\mathrm da_t\\
&=
\mathbb{E}_{a_t\sim\pi_\theta}
\left[f(a_t)\right].
\end{aligned}
$$

令 $f(a_t)=\hat A_t$，就得到 PPO 使用的替代目标

$$
\hat{\mathbb E}_t
\left[r_t(\theta)\hat A_t\right].
$$

需要注意：上述严格等式只校正了固定状态下的**动作分布**。rollout 中的状态仍来自旧策略的访问分布 $\rho^{\pi_{\mathrm{old}}}(s)$，而不是新策略的 $\rho^{\pi_\theta}(s)$。TRPO/PPO 依赖“小步更新”使两种状态分布保持接近，因此使用旧策略采样状态构造局部替代目标；概率比率并没有精确校正整条轨迹的状态分布变化。

### 6.2 TRPO 中两个策略分布的期望 KL 散度

#### 6.2.1 单个状态上的 KL 散度

对固定状态 $s$，TRPO 使用从旧策略到新策略的 KL 散度

$$
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)
\,\|\,
\pi_\theta(\cdot\mid s)
\right)
=
\mathbb{E}_{a\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)}
\left[
\log
\frac{\pi_{\theta_{\mathrm{old}}}(a\mid s)}
{\pi_\theta(a\mid s)}
\right].
$$

离散动作时，它等于

$$
\sum_a
\pi_{\theta_{\mathrm{old}}}(a\mid s)
\left[
\log\pi_{\theta_{\mathrm{old}}}(a\mid s)
-\log\pi_\theta(a\mid s)
\right];
$$

连续动作时，只需将求和替换为对动作的积分。KL 散度满足 $D_{\mathrm{KL}}\geq0$，且仅当两个策略分布几乎处处相同时取 $0$。注意 KL 不对称，交换新旧策略后通常会得到不同结果；TRPO 约束采用这里写出的“旧策略 $\|$ 新策略”方向。

#### 6.2.2 再对旧策略的状态分布取期望

TRPO 不是只约束某一个状态，而是约束旧策略访问到的状态上的平均 KL：

$$
\begin{aligned}
\bar D_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)
&=
\mathbb{E}_{s\sim\rho^{\pi_{\mathrm{old}}}}
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)
\,\|\,
\pi_\theta(\cdot\mid s)
\right)
\right]\\
&=
\mathbb{E}_{\substack{s\sim\rho^{\pi_{\mathrm{old}}}\\
a\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)}}
\left[
\log\pi_{\theta_{\mathrm{old}}}(a\mid s)
-\log\pi_\theta(a\mid s)
\right].
\end{aligned}
$$

这就是“先在每个状态上对动作求 KL，再对状态访问分布求平均”的两层期望。TRPO 的约束为

$$
\bar D_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)
\leq\delta.
$$

给定旧策略收集的 $N$ 个状态，若能直接计算两个完整策略分布的 KL，则经验估计为

$$
\widehat{\bar D}_{\mathrm{KL}}
=
\frac{1}{N}
\sum_{t=1}^{N}
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\,\|\,
\pi_\theta(\cdot\mid s_t)
\right).
$$

若每个状态只使用 rollout 中实际采到的一个动作，则可作蒙特卡洛估计：

$$
\widehat{\bar D}_{\mathrm{KL}}^{\mathrm{MC}}
=
\frac{1}{N}
\sum_{t=1}^{N}
\left[
\ell_t^{\mathrm{old}}-\ell_t^{\mathrm{new}}
\right]
=
-\frac{1}{N}
\sum_{t=1}^{N}\log r_t(\theta).
$$

单个样本的对数概率差可能为负，但其总体期望是非负的 KL 散度。常见的非负样本近似还可以写成

$$
\widehat{\bar D}_{\mathrm{KL}}^{\mathrm{approx}}
=
\frac{1}{N}
\sum_{t=1}^{N}
\left[
r_t(\theta)-1-\log r_t(\theta)
\right],
$$

因为对任意 $x>0$ 都有 $x-1-\log x\geq0$；同时在精确期望下 $\mathbb E_{a\sim\pi_{\mathrm{old}}}[r_t]=1$，所以它与 $-\mathbb E[\log r_t]$ 对应同一 KL 量。

#### 6.2.3 对角高斯策略的 KL 闭式计算

下方 PPO 代码的连续动作策略是对角高斯分布。先考虑一维情况：

$$
\pi_{\mathrm{old}}
=
\mathcal N(\mu_0,\sigma_0^2),
\qquad
\pi_{\mathrm{new}}
=
\mathcal N(\mu_1,\sigma_1^2).
$$

把两个高斯密度的对数代入 KL 定义：

$$
\begin{aligned}
D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_{\mathrm{new}})
&=
\mathbb E_{a\sim\pi_{\mathrm{old}}}
\left[
\log\frac{\sigma_1}{\sigma_0}
+\frac{(a-\mu_1)^2}{2\sigma_1^2}
-\frac{(a-\mu_0)^2}{2\sigma_0^2}
\right].
\end{aligned}
$$

在旧分布下有

$$
\mathbb E[(a-\mu_0)^2]=\sigma_0^2,
$$

以及

$$
\mathbb E[(a-\mu_1)^2]
=
\sigma_0^2+(\mu_0-\mu_1)^2.
$$

因此

$$
\boxed{
D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_{\mathrm{new}})
=
\log\frac{\sigma_1}{\sigma_0}
+\frac{\sigma_0^2+(\mu_0-\mu_1)^2}{2\sigma_1^2}
-\frac{1}{2}
}.
$$

若动作有 $D$ 个相互独立的维度，对角高斯的联合 KL 等于各维 KL 之和：

$$
\boxed{
D_{\mathrm{KL}}
\left(
\pi_{\mathrm{old}}(\cdot\mid s)
\,\|\,
\pi_{\mathrm{new}}(\cdot\mid s)
\right)
=
\sum_{j=1}^{D}
\left[
\log\frac{\sigma_{1,j}(s)}{\sigma_{0,j}(s)}
+\frac{
\sigma_{0,j}^2(s)
+\left(\mu_{0,j}(s)-\mu_{1,j}(s)\right)^2
}{2\sigma_{1,j}^2(s)}
-\frac{1}{2}
\right]
}.
$$

将每个状态上的闭式 KL 对 rollout 状态求算术平均，就得到 TRPO 约束中的经验期望：

$$
\boxed{
\widehat{\bar D}_{\mathrm{KL}}
=
\frac{1}{N}
\sum_{t=1}^{N}
\sum_{j=1}^{D}
\left[
\log\frac{\sigma_{1,j}(s_t)}{\sigma_{0,j}(s_t)}
+\frac{
\sigma_{0,j}^2(s_t)
+\left(\mu_{0,j}(s_t)-\mu_{1,j}(s_t)\right)^2
}{2\sigma_{1,j}^2(s_t)}
-\frac{1}{2}
\right]
}.
$$

若新旧策略的标准差完全相同，即 $\sigma_{0,j}=\sigma_{1,j}=\sigma_j$，单个状态上的公式进一步化为

$$
D_{\mathrm{KL}}
=
\frac{1}{2}
\sum_{j=1}^{D}
\frac{
\left(\mu_{0,j}(s)-\mu_{1,j}(s)\right)^2
}{\sigma_j^2}.
$$

PPO-Clip 不需要像 TRPO 一样把平均 KL 写成硬约束，所以下方核心代码没有显式计算该量；实际工程中仍可监控经验 KL，并在它超过预设阈值时提前结束本轮 Epoch。
