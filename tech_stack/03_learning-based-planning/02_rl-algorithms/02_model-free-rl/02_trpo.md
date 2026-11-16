# TRPO 与信赖域策略优化

## 1. 概述

TRPO 采用 Actor-Critic 架构，在旧策略采集的轨迹上提高策略替代目标，并用平均 KL 散度约束策略变化。Actor 通过自然梯度方向和回溯线搜索更新，Critic 通过回报回归提供优势估计。

## 2. 基础与定义

### 2.1 网络结构

采用两个独立网络：Actor $\pi_\theta(a\mid s)$ 表示动作分布，Critic $V_\phi(s)$ 估计状态价值。采样策略参数记为 $\theta_{\mathrm{old}}$，本轮更新使用固定的优势估计 $\hat A_t$ 与价值目标 $\hat R_t$。

### 2.2 符号约定

| 符号 | 含义 |
| --- | --- |
| $\rho_t(\theta)$ | 动作概率比率 $\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$ |
| $\hat L(\theta)$ | 经验策略替代目标 |
| $\bar D_{\mathrm{KL}}(\theta)$ | 采样状态上的平均 $D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_\theta)$ |
| $\delta$ | 平均 KL 上限 |
| $g,H$ | 旧参数处的目标梯度与 KL Hessian |

$\hat{\mathbb E}_t$ 表示对本轮轨迹样本取均值。KL 使用完整动作分布；多维独立动作的 KL 先按动作维求和，再按样本求均值。

## 3. 核心原理

<span id="1-完整的数学公式与算法流程"></span>
<span id="步骤-1构建替代目标函数-surrogate-objective"></span>

### 3.1 策略替代目标

Actor 最大化旧样本上的加权优势：

$$
\hat L(\theta)=\hat{\mathbb E}_t[\rho_t(\theta)\hat A_t],
\qquad
\rho_t(\theta)=\exp\!\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right).
$$

固定旧策略与优势，目标梯度为

$$
\nabla_\theta\hat L
=\hat{\mathbb E}_t[\rho_t\hat A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)].
$$

在旧参数处 $\rho_t=1$，得到样本策略梯度。概率比率校正给定状态下的动作分布，状态仍由旧策略采样；TRPO 用信赖域控制这个替代目标的适用范围。

<span id="步骤-2引入信任区域约束-trust-region-constraint"></span>
<span id="步骤-3泰勒展开与二次近似"></span>

### 3.2 KL 信赖域约束

Actor 的约束优化问题为

$$
\max_\theta\ \hat L(\theta)
\quad\text{s.t.}\quad
\bar D_{\mathrm{KL}}(\theta)
=\hat{\mathbb E}_t\!\left[D_{\mathrm{KL}}\!\left(\pi_{\mathrm{old}}(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t)\right)\right]\le\delta.
$$

令 $\Delta\theta=\theta-\theta_{\mathrm{old}}$。在旧参数处对目标作一阶展开，对 KL 作二阶展开：

$$
\hat L(\theta)\approx\hat L(\theta_{\mathrm{old}})+g^\top\Delta\theta,
\qquad
\bar D_{\mathrm{KL}}(\theta)\approx\frac12\Delta\theta^\top H\Delta\theta.
$$

KL 在相同策略处的值与一阶导数均为零。满足通常的可微与支撑集条件时，$H$ 是旧策略状态分布下的 Fisher 信息矩阵，描述策略分布对参数变化的敏感程度。

<span id="步骤-4共轭梯度法求搜索方向-conjugate-gradient"></span>
<span id="步骤-5计算最大更新步长"></span>

### 3.3 自然梯度与共轭梯度

局部问题为

$$
\max_{\Delta\theta}\ g^\top\Delta\theta
\quad\text{s.t.}\quad
\frac12\Delta\theta^\top H\Delta\theta\le\delta.
$$

当 $H$ 正定且 $g\ne0$ 时，先解 $Hx=g$，再缩放方向：

$$
\Delta\theta_{\mathrm{full}}=\sqrt{\frac{2\delta}{x^\top Hx}}\,x.
$$

<details class="kt-derivation" id="trpo-natural-step">
<summary>自然梯度与步长推导</summary>

拉格朗日函数为

$$
\mathcal F(\Delta\theta,\mu)
=g^\top\Delta\theta-\mu\left(\frac12\Delta\theta^\top H\Delta\theta-\delta\right).
$$

由驻点条件 $g-\mu H\Delta\theta=0$，得到 $\Delta\theta=\mu^{-1}H^{-1}g$。代入约束边界：

$$
\frac{1}{2\mu^2}g^\top H^{-1}g=\delta,
\qquad
\Delta\theta=\sqrt{\frac{2\delta}{g^\top H^{-1}g}}H^{-1}g.
$$

令 $x=H^{-1}g$，有 $g^\top H^{-1}g=x^\top Hx$，得到正文结果。

共轭梯度通过矩阵向量积近似求解线性方程。该乘积可由两次自动微分计算：

$$
Hv=\nabla_\theta\left[(\nabla_\theta\bar D_{\mathrm{KL}})^\top v\right]_{\theta=\theta_{\mathrm{old}}}.
$$

数值实现用 $H_\eta=H+\eta I$ 缓解曲率病态，$\eta\ge0$ 为阻尼系数。以 $H_\eta$ 求方向和缩放步长后，候选策略仍用原始经验 KL 验收。

</details>

<span id="步骤-6回溯线搜索-backtracking-line-search"></span>

### 3.4 回溯线搜索

从完整步长开始，按 $0<\alpha<1$ 依次缩小：

$$
\theta_j=\theta_{\mathrm{old}}+\alpha^j\Delta\theta_{\mathrm{full}},\qquad j=0,1,\ldots.
$$

接受目标与 KL 均有限且满足下列条件的候选：

$$
\hat L(\theta_j)>\hat L(\theta_{\mathrm{old}}),
\qquad
\bar D_{\mathrm{KL}}(\theta_j)\le\delta.
$$

全部候选失败时恢复旧 Actor 参数。这样将局部二次近似得到的方向与候选策略的实际分布变化连接起来。

### 3.5 价值回归与优势估计

Critic 用固定回报目标拟合状态价值：

$$
L_V(\phi)=\hat{\mathbb E}_t[(V_\phi(s_t)-\hat R_t)^2],
\qquad
\nabla_\phi L_V=2\hat{\mathbb E}_t[(V_\phi(s_t)-\hat R_t)\nabla_\phi V_\phi(s_t)].
$$

可用旧 Critic 计算 GAE，再构造 $\hat R_t=\hat A_t+V_{\mathrm{old}}(s_t)$。两者在本轮拟合与线搜索期间固定；估计量的定义与边界处理见 [PPO 的优势估计](./03_ppo.md#34-优势估计与价值目标)。

## 4. 实现与应用

<span id="2-核心代码架构实现"></span>

### 4.1 训练流程

1. 用旧策略采样，保存动作、旧对数概率以及重建旧动作分布所需的参数。
2. 用旧 Critic 计算固定的优势与回报目标。
3. 计算 Actor 的目标梯度，以共轭梯度求自然梯度方向。
4. 缩放步长并回溯，接受满足目标改进和 KL 约束的候选。
5. 拟合独立 Critic，以更新后的策略采集下一轮轨迹。

### 4.2 实现示例

<details class="kt-derivation" id="trpo-pytorch-update">
<summary>TRPO 的 PyTorch 实现</summary>

Actor 使用未压缩的对角高斯分布，输入动作来自该分布。输入张量位于同一设备：`obs` 为 `[N, obs_dim]`，`actions` 为 `[N, action_dim]`，旧对数概率、回报与优势均为 `[N]`。采样与 GAE 由调用方提供。

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


def flat_grad(grads):
    return torch.cat([grad.reshape(-1) for grad in grads])


def get_flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


@torch.no_grad()
def set_flat_params(model, flat_params):
    offset = 0
    for param in model.parameters():
        count = param.numel()
        param.copy_(flat_params[offset:offset + count].view_as(param))
        offset += count


def hessian_vector_product(actor, obs, old_dist, vector, damping=0.1):
    """At theta_old: return (H + damping * I) @ vector."""
    mean, logstd = actor(obs)
    dist = Normal(mean, logstd.exp())
    # Joint-action KL: sum action dimensions, then average samples.
    kl = kl_divergence(old_dist, dist).sum(-1).mean()
    first = torch.autograd.grad(kl, tuple(actor.parameters()), create_graph=True)
    product = torch.dot(flat_grad(first), vector)
    second = torch.autograd.grad(product, tuple(actor.parameters()))
    return flat_grad(second).detach() + damping * vector


def conjugate_gradient(matvec, b, nsteps=10, residual_tol=1e-10):
    """Approximately solve A x = b for a positive-definite matvec."""
    x = torch.zeros_like(b)
    residual = b.clone()
    direction = residual.clone()
    squared_residual = torch.dot(residual, residual)
    for _ in range(nsteps):
        if squared_residual <= residual_tol:
            break
        image = matvec(direction)
        curvature = torch.dot(direction, image)
        if not torch.isfinite(curvature) or curvature <= 0:
            break
        step = squared_residual / curvature
        x = x + step * direction
        residual = residual - step * image
        next_squared = torch.dot(residual, residual)
        if not torch.isfinite(next_squared):
            return torch.zeros_like(b)
        direction = residual + (next_squared / squared_residual) * direction
        squared_residual = next_squared
    return x


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs):
        return self.net(obs), self.logstd.expand(obs.shape[0], -1)


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class TRPOAgent:
    def __init__(self, obs_dim, action_dim, device="cpu"):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        # Illustrative values, not a task-specific tuned configuration.
        self.max_kl = 0.01
        self.damping = 0.1
        self.cg_iters = 10
        self.critic_iters = 10
        self.line_search_coef = 0.5
        self.line_search_iters = 10

    def surrogate(self, obs, actions, old_logprobs, advantages):
        mean, logstd = self.actor(obs)
        logprobs = Normal(mean, logstd.exp()).log_prob(actions).sum(-1)
        return (torch.exp(logprobs - old_logprobs) * advantages).mean()

    def trpo_update(self, obs, actions, old_logprobs, returns, advantages):
        obs, actions = obs.detach(), actions.detach()
        old_logprobs = old_logprobs.detach().reshape(-1)
        returns, advantages = returns.detach().reshape(-1), advantages.detach().reshape(-1)
        if not all(torch.isfinite(x).all() for x in (obs, actions, old_logprobs, returns, advantages)):
            raise ValueError("Rollout contains non-finite values")
        if any(x.shape != (obs.shape[0],) for x in (old_logprobs, returns, advantages)):
            raise ValueError("Scalar targets must have one value per observation")
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        old_params = get_flat_params(self.actor)
        with torch.no_grad():
            old_mean, old_logstd = self.actor(obs)
            old_dist = Normal(old_mean.detach().clone(), old_logstd.exp().detach().clone())
            reference_logp = old_dist.log_prob(actions).sum(-1)
            if not torch.allclose(reference_logp, old_logprobs, atol=1e-5, rtol=1e-4):
                raise ValueError("Actor and old log-prob do not match the rollout policy")

        # Independent Critic: fitting it cannot move the Actor snapshot.
        for _ in range(self.critic_iters):
            critic_loss = F.mse_loss(self.critic(obs), returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        objective = self.surrogate(obs, actions, old_logprobs, advantages)
        old_objective = objective.detach().item()
        g = flat_grad(torch.autograd.grad(objective, tuple(self.actor.parameters()))).detach()
        result = {"accepted": False, "kl": 0.0, "improvement": 0.0}
        if not torch.isfinite(g).all() or torch.dot(g, g) <= 1e-12:
            return result

        def matvec(vector):
            return hessian_vector_product(self.actor, obs, old_dist, vector, self.damping)

        direction = conjugate_gradient(matvec, g, self.cg_iters)
        curvature = torch.dot(direction, matvec(direction))
        if not torch.isfinite(curvature) or curvature <= 0:
            return result
        full_step = torch.sqrt(2 * self.max_kl / curvature) * direction
        if not torch.isfinite(full_step).all():
            return result

        accepted = False
        try:
            for i in range(self.line_search_iters):
                fraction = self.line_search_coef ** i
                set_flat_params(self.actor, old_params + fraction * full_step)
                with torch.no_grad():
                    mean, logstd = self.actor(obs)
                    scale = logstd.exp()
                    if not (torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale > 0).all()):
                        continue
                    candidate = Normal(mean, scale)
                    logp = candidate.log_prob(actions).sum(-1)
                    score = (torch.exp(logp - old_logprobs) * advantages).mean().item()
                    kl = kl_divergence(old_dist, candidate).sum(-1).mean().item()
                if math.isfinite(score) and math.isfinite(kl) and score > old_objective and kl <= self.max_kl:
                    accepted = True
                    result = {"accepted": True, "kl": kl, "improvement": score - old_objective}
                    break
        finally:
            if not accepted:
                set_flat_params(self.actor, old_params)
        return result


def main_training_loop(agent, env_rollout_data):
    # Targets and old log-probs were already computed by the rollout stage.
    obs, actions, logprobs, returns, advantages = env_rollout_data
    return agent.trpo_update(obs, actions, logprobs, returns, advantages)
```

</details>

<details class="kt-derivation" id="trpo-update-checks">
<summary>更新一致性检验</summary>

- 旧参数处 KL 接近零；Hessian 向量积与方向有限差分一致。
- 旧对数概率与旧动作分布来自同一采样策略；线搜索期间固定其快照。
- 接受候选时，经验 KL 满足上限，替代目标相对旧目标增加。
- 全零优势保持 Actor 参数不变；全部候选失败时完整恢复参数。
- 多维动作使用联合 KL；Hessian、步长与候选检查的聚合尺度一致。
- 含 Dropout 或 BatchNorm 的网络需固定推断模式及相关状态，使候选评估可重复。

</details>

## 5. 分析与总结

### 5.1 优化方法比较

| 方法 | 策略更新方式 | 计算结构 |
| --- | --- | --- |
| 普通策略梯度 | 沿目标梯度调整参数 | 一阶更新 |
| 自然策略梯度 | 用 Fisher 曲率调整方向 | 线性方程近似求解 |
| TRPO | 自然梯度方向与 KL 回溯 | 曲率计算、候选评估 |
| PPO-Clip | 截断优势目标 | 多轮小批量一阶更新 |

### 5.2 近似误差

理论改进界涉及状态分布变化与最坏状态的策略差异。实用 TRPO 使用平均 KL、有限样本、估计优势和局部二次模型；线搜索验收的是本批样本上的替代目标与经验 KL。真实回报由后续环境评估衡量。

### 5.3 核心结论

TRPO 用优势决定改进方向，用 KL 曲率调整参数位移，再通过回溯确定实际步长。Critic 为策略更新提供固定的价值与优势信号。

## 6. 关联与参考

### 6.1 知识关联

- [强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)：策略梯度与 Actor-Critic。
- [数学优化理论](../../../01_math-physic-base/02_calculus-optimization/02_optimization-theory.md)：拉格朗日乘子与约束优化。
- [PPO](./03_ppo.md)：截断目标、GAE 与回报目标。
- [多价值网络与价值估计](../01_rl-foundations/05_multi-critic.md)：多个优势信号的合成。

### 6.2 参考资料

1. Schulman et al., [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html), ICML 2015。
2. OpenAI Spinning Up, [TRPO 算法说明](https://spinningup.openai.com/en/latest/algorithms/trpo.html)、[TRPO 源码](https://spinningup.openai.com/en/latest/_modules/spinup/algos/tf1/trpo/trpo.html)。
3. Kakade, [A Natural Policy Gradient](https://proceedings.neurips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html), NeurIPS 2001。
4. Kakade & Langford, [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf), ICML 2002。
5. Peters & Schaal, [Reinforcement Learning of Motor Skills with Policy Gradients](https://doi.org/10.1016/j.neunet.2008.02.003), Neural Networks, 2008。
