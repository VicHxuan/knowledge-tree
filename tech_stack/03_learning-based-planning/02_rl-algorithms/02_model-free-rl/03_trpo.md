# TRPO (Trust Region Policy Optimization) 算法完整解析与代码实现

## 1. 概述

TRPO（Trust Region Policy Optimization，信赖域策略优化）用策略分布的变化限制一次更新：**先在旧策略样本上估计改进方向，再沿该方向寻找满足 KL 约束且提高替代目标的步长**。它把自然梯度、共轭梯度与回溯线搜索连接成一个策略更新过程。

> **策略数据类型**：On-Policy。本轮由旧策略采样，Actor 根据整批数据尝试一次受约束的更新；候选步长评估和 Critic 拟合可以重复使用这批数据。

TRPO 论文先给出带假设的策略改进下界，再通过平均 KL、有限样本和局部近似得到实用算法。实际训练不能据此保证每轮真实环境回报都单调增加。[原始论文](https://proceedings.mlr.press/v37/schulman15.html)。

本文默认展示“替代目标 → KL 局部模型 → 自然梯度方向 → 线搜索”的主线；拉格朗日推导和完整 PyTorch 更新示例可按需展开。

## 2. 基础与定义

### 2.1 输入与输出

输入是一批由 $\pi_{\theta_{\mathrm{old}}}$ 采集的状态、动作、旧 log-prob、优势估计 $\hat A_t$ 和价值回归目标 $\hat R_t$。Actor 更新输出新策略参数，Critic 则用固定的 $\hat R_t$ 拟合状态价值。

计算 KL 需要旧策略在每个采样状态上的**完整动作分布**；仅保存被选中动作的一个 log-prob 通常不够。例如，对角高斯策略还需旧均值与标准差，或者保留可重建旧分布的冻结 Actor。优势估计可以采用 GAE，通用定义见[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)。

### 2.2 符号与约定

| 符号 | 含义 |
|---|---|
| $\theta_{\mathrm{old}},\theta$ | 采样策略参数与候选策略参数 |
| $d_{\mathrm{old}}(s)$ | 旧策略的归一化折扣状态访问分布 |
| $\hat L(\theta)$ | 旧样本上的经验替代目标，本文最大化它 |
| $\bar D_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)$ | 旧状态上的平均 $D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_\theta)$ |
| $g,H$ | 在旧参数处的目标梯度与 KL Hessian |
| $\delta$ | 平均 KL 上限；数值随任务和实现选择 |
| $\eta$ | Hessian-vector product 的非负阻尼系数 |
| $\alpha\in(0,1)$ | 回溯步长衰减系数 |

下文固定 KL 方向为 old → new，梯度、曲率和候选检查使用同一约定。部分资料写 new → old；两者在相同策略点的二阶曲率一致，但候选离开该点后 KL 数值会不同，不应中途混用。独立动作维度的联合 log-prob 与 KL 都先沿动作维求和，再沿样本维求均值；改变动作维度时，不应无意间改变 KL 阈值的含义。

## 3. 核心原理

<span id="1-完整的数学公式与算法流程"></span>
<span id="步骤-1构建替代目标函数-surrogate-objective"></span>

### 3.1 策略替代目标

用旧策略的状态分布固定采样测度，通过动作概率比率评价候选策略：

$$
L(\theta)=\mathbb E_{s\sim d_{\mathrm{old}},\ a\sim\pi_{\mathrm{old}}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}
A^{\pi_{\mathrm{old}}}(s,a)\right],
\qquad
\hat L(\theta)=\frac1N\sum_t
\exp\!\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right)\hat A_t.
$$

旧参数点的 ratio 为 1，替代目标的梯度与策略梯度具有相同方向；具体尺度取决于状态访问分布的归一化约定。比率只校正动作分布，采样状态仍来自旧策略，因此大幅移动策略后，替代目标可能不再可靠。

<span id="步骤-2引入信任区域约束-trust-region-constraint"></span>
<span id="步骤-3泰勒展开与二次近似"></span>

### 3.2 KL 信赖域约束

TRPO 将改进限制在旧策略附近：

$$
\max_\theta\ \hat L(\theta)
\quad\text{s.t.}\quad
\bar D_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)
=\frac1N\sum_t D_{\mathrm{KL}}\!\left(
\pi_{\mathrm{old}}(\cdot\mid s_t)\,\|\,\pi_\theta(\cdot\mid s_t)
\right)\le\delta.
$$

平均 KL 约束的是采样状态上整体的分布变化，不是每个参数的欧氏位移，也不是每个状态各自的最大 KL。局部求解时，用 $\Delta\theta=\theta-\theta_{\mathrm{old}}$ 表示位移：

$$
\hat L(\theta)\approx\hat L(\theta_{\mathrm{old}})+g^\top\Delta\theta,
\qquad
\bar D_{\mathrm{KL}}\approx\frac12\Delta\theta^\top H\Delta\theta.
$$

KL 在相同策略处为零且一阶导为零，所以其首个非零局部项为二阶项。在通常的可微与支撑集条件下，$H$ 对应旧策略状态分布下的 Fisher 信息矩阵；它描述策略分布对参数变化的敏感程度。

<span id="步骤-4共轭梯度法求搜索方向-conjugate-gradient"></span>
<span id="步骤-5计算最大更新步长"></span>

### 3.3 自然梯度与共轭梯度

忽略常数项后的局部问题为

$$
\max_{\Delta\theta}\ g^\top\Delta\theta
\quad\text{s.t.}\quad
\frac12\Delta\theta^\top H\Delta\theta\le\delta.
$$

当 $H$ 正定且 $g\ne0$ 时，解沿 $H^{-1}g$ 方向。先求 $Hx=g$，再按信赖域半径缩放：

$$
\beta=\sqrt{\frac{2\delta}{x^\top Hx}},
\qquad
\Delta\theta_{\mathrm{full}}=\beta x.
$$

<details class="kt-derivation" id="trpo-natural-step">
<summary>自然梯度与步长推导</summary>

对局部约束问题构造

$$
\mathcal F(\Delta\theta,\mu)
=g^\top\Delta\theta-\mu\left(\frac12\Delta\theta^\top H\Delta\theta-\delta\right),\qquad\mu>0.
$$

对位移求导得到 $g-\mu H\Delta\theta=0$，即 $\Delta\theta=\mu^{-1}H^{-1}g$。将它代入边界条件：

$$
\frac{1}{2\mu^2}g^\top H^{-1}g=\delta
\quad\Rightarrow\quad
\Delta\theta=\sqrt{\frac{2\delta}{g^\top H^{-1}g}}\,H^{-1}g.
$$

令 $x=H^{-1}g$，则 $g^\top H^{-1}g=x^\top Hx$，得到正文的缩放形式。实际网络的曲率可能病态或奇异，通常改用 $H_\eta=H+\eta I$ 求解；这已是带数值正则的方向，最终仍要检查未加阻尼的真实经验 KL。

</details>

深度网络无法经济地显式构造 $H$。共轭梯度（CG）只需要 $Hv$，可以通过两次自动微分得到：

$$
Hv=\nabla_\theta\left[
\left(\nabla_\theta\bar D_{\mathrm{KL}}\right)^\top v
\right]_{\theta=\theta_{\mathrm{old}}}.
$$

实际计算返回 $H_\eta v=Hv+\eta v$，CG 迭代近似求解 $H_\eta x=g$；全文代码在该方向上用同一个 $H_\eta$ 缩放步长。阻尼和 CG 迭代数的具体实现可对照 [Spinning Up TRPO 源码](https://spinningup.openai.com/en/latest/_modules/spinup/algos/tf1/trpo/trpo.html)。

<span id="步骤-6回溯线搜索-backtracking-line-search"></span>

### 3.4 回溯线搜索

从全步长开始，依次尝试

$$
\theta_j=\theta_{\mathrm{old}}+\alpha^j\Delta\theta_{\mathrm{full}},\qquad j=0,1,\ldots.
$$

接受候选必须同时满足：

$$
\hat L(\theta_j)>\hat L(\theta_{\mathrm{old}}),
\qquad
\bar D_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta_j)\le\delta,
$$

且目标与 KL 都是有限数。**比较的是目标的实际增量，而不是只检查 $\hat L(\theta_j)>0$。** 只有旧目标恰好为零时，这两个写法才等价。若所有候选都不合格，恢复旧 Actor 参数。

整轮流程是：旧策略采样 → 固定优势与价值目标 → 求 $g$ → CG 求方向 → 缩放 → 回溯接受或回滚 → 用后续新采样继续训练。Critic 通过常规回归拟合 $\hat R_t$，本文示例使用独立网络与 Adam；Critic 更新不会移动 Actor 的参数。

## 4. 实现与应用

<span id="2-核心代码架构实现"></span>

### 4.1 实现一致性

1. **旧策略快照**：old log-prob 和用于 KL 的旧分布来自同一个采样 Actor，在 CG 与线搜索期间固定。
2. **KL 标度**：HVP、步长缩放、候选检查使用一致的样本与动作维聚合方式；阻尼只用于数值求解。
3. **参数恢复**：所有候选从同一份 `old_params` 出发，失败时完整回滚，不能在连续失败候选上累加步长。

以下示例保留 Actor、Critic、HVP、CG 与 TRPOAgent 的完整连接。动作分布为未压缩的对角高斯；输入 actions 是该分布下的动作。带边界的动作变换需要对应的概率处理，不能把环境裁剪后的动作直接视为原分布样本。

<details class="kt-derivation" id="trpo-pytorch-update">
<summary>TRPO 的 PyTorch 实现</summary>

代码输入均为同设备的 batch 张量：`obs: [N, obs_dim]`、`actions: [N, action_dim]`，其余三个目标为 `[N]`。采样、GAE 和环境交互由调用方提供；这里专注于 TRPO 的受约束更新，不复制 PPO 的采样/GAE 实现。

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

### 4.2 更新验证

先在小型网络、二维以上动作空间中检查数学接口，再接长时间训练：

- 在旧参数点，KL 接近零；HVP 的方向有限差分与自动微分结果一致。
- 接受更新时，经验 KL 不超过 `max_kl`，替代目标严格提高；记录实际回溯次数。
- 优势全零时跳过 Actor 更新；强制所有候选失败时，Actor 参数与旧快照完全相同。
- Critic 更新前后，固定的 returns、advantages、old log-prob 和旧分布没有变化。

这些检查只验明更新是否按公式执行。策略是否学会任务仍需独立评估，不能用 surrogate 的单步提高替代真实回报评估。

## 5. 分析与总结

### 5.1 理论保证与实现近似

理论改进界控制的是策略变化对状态访问分布与优势评价的影响；实用 TRPO 将最坏状态的约束近似为平均 KL，再用样本、近似优势和局部二次模型求解。因此，线搜索通过表示“本批样本上的目标与约束通过检查”，不等于所有状态都满足同样的 KL 上限，也不保证下一批实际回报提高。论文明确区分理论算法和这些实用近似。[TRPO 原文第 3–5 节](https://proceedings.mlr.press/v37/schulman15.pdf)。

| 方法 | 控制策略变化的方式 | 主要实现代价 |
|---|---|---|
| 普通策略梯度 | learning rate 控制参数步长 | 一阶梯度更新 |
| 自然策略梯度 | Fisher 曲率调整方向和局部步长 | HVP / 线性方程近似求解 |
| TRPO | 曲率方向 + 实际 KL 与目标的回溯检查 | CG、多个候选前向、参数回滚 |
| PPO-Clip | 截断替代目标，多轮 minibatch 一阶更新 | 普通 optimizer；实践中仍监控 KL |

PPO-Clip 没有把每次更新投影回 TRPO 的 KL 可行域；理解 TRPO 的价值在于看清“为什么只按参数 learning rate 约束步长不够”。

### 5.2 常见实现错误

- HVP 把动作维也平均掉，线搜索却使用联合动作 KL：两处阈值尺度不一致。
- 更新 Actor 后才建立 old distribution，或共享编码器被 Critic 改动：比较基准已变化。
- 将 `surrogate > 0` 当作通用接受条件：旧目标可能不是零，应检查实际增量。
- CG 分母非正、梯度为零仍强行计算步长，或线搜索失败未回滚：产生无意义的更新。
- 在存在 Dropout/BatchNorm 状态变化的网络中直接多次前向：同一参数未必对应同一策略分布，应先明确训练/评估模式与状态冻结方式。

### 5.3 核心结论

TRPO 的核心是**用 KL 曲率决定局部方向，再用原目标与原 KL 检查实际步长**。CG 降低二阶求解的存储成本，回溯修正局部近似误差，旧策略快照和失败回滚保证检查具有一致参照。三者共同构成一次完整的策略更新。

## 6. 关联与参考

### 6.1 知识关联

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——策略梯度、价值函数与优势。
- **局部约束优化**：[数学优化理论](../../../01_math-physic-base/02_calculus-optimization/02_optimization-theory.md)——拉格朗日乘子、曲率和约束问题。
- **算法对比**：[PPO](./04_ppo.md)——用截断替代目标构造一阶优化闭环。
- **工程后续**：[PPO 全流程走读与调参](../../03_engineering-practice/01_ppo-walkthrough.md)——采样、更新计数、日志和恢复。

### 6.2 参考资料

1. Schulman et al., [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html), ICML 2015；[论文 PDF](https://proceedings.mlr.press/v37/schulman15.pdf)：理论改进界与实用平均 KL 约束。
2. OpenAI Spinning Up, [TRPO 算法说明](https://spinningup.openai.com/en/latest/algorithms/trpo.html)、[TensorFlow 1 实现源码](https://spinningup.openai.com/en/latest/_modules/spinup/algos/tf1/trpo/trpo.html)：CG、阻尼、线搜索与参数恢复的参考；本文 PyTorch 示例独立表达同一流程。
3. Kakade, [A Natural Policy Gradient](https://proceedings.neurips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html), NeurIPS 2001：自然策略梯度的来源。
4. Kakade & Langford, [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf), ICML 2002：保守策略迭代的理论背景。
5. Peters & Schaal, [Reinforcement Learning of Motor Skills with Policy Gradients](https://doi.org/10.1016/j.neunet.2008.02.003), Neural Networks 21(4), 2008：策略梯度与自然梯度在控制任务中的讨论。
