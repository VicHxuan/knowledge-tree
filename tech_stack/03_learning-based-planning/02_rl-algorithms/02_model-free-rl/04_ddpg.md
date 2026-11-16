# DDPG 与确定性策略梯度

## 1. 概述

DDPG 用确定性 Actor 输出连续动作，用 Critic 估计动作价值。Critic 通过贝尔曼回归学习，Actor 沿 Critic 提供的动作梯度提高价值；经验回放与目标网络连接两种更新。

## 2. 基础与定义

### 2.1 网络结构

| 网络 | 输出 | 参数 |
| --- | --- | --- |
| Actor | 连续动作 $\mu_\theta(s)$ | $\theta$ |
| Critic | 动作价值 $Q_\phi(s,a)$ | $\phi$ |
| 目标 Actor | 自举动作 $\mu_{\bar\theta}(s')$ | $\bar\theta$ |
| 目标 Critic | 自举动作价值 $Q_{\bar\phi}(s',a')$ | $\bar\phi$ |

在线网络接收梯度更新，目标网络由在线参数缓慢更新。

### 2.2 经验数据

回放池 $\mathcal D$ 保存状态转移。$r$ 为执行动作后得到的奖励，$d$ 标记真正终止，$s'$ 为实际后继状态。行为动作在 Actor 输出上增加探索噪声，并映射到合法动作范围。

## 3. 核心原理

### 3.1 价值损失

Critic 拟合固定的目标 $y$：

$$
L_Q(\phi)=\frac12\mathbb E_{\mathcal D}[(Q_\phi(s,a)-y)^2],
\qquad
y=r+\gamma(1-d)Q_{\bar\phi}(s',\mu_{\bar\theta}(s')).
$$

目标由实际奖励、目标动作及目标价值构成，在当前 Critic 更新中停止梯度。

### 3.2 策略损失与动作梯度

Actor 在回放状态上最大化当前 Critic 对自身动作的评价：

$$
L_\pi(\theta)=-\mathbb E_{s\sim\mathcal D}[Q_\phi(s,\mu_\theta(s))].
$$

应用链式法则：

$$
\nabla_\theta L_\pi
=-\mathbb E_{s\sim\mathcal D}\left[
\left(\nabla_\theta\mu_\theta(s)\right)^\top
\nabla_aQ_\phi(s,a)\big|_{a=\mu_\theta(s)}\right].
$$

更新 Actor 时固定 Critic 参数，保留 Q 值对动作的导数。该路径将动作空间中的价值变化转换为 Actor 参数变化。

<details class="kt-derivation" id="ddpg-policy-gradient">
<summary>确定性策略梯度与经验目标</summary>

确定性策略梯度定理在其访问分布及可微条件下，将真实性能梯度写成动作价值梯度与策略雅可比的期望。DDPG 用回放状态分布和近似 Critic 构造可优化的经验目标。

令 $\mu_\theta(s)=\theta s$，$Q(s,a)=-(a-a_*)^2$，则：

$$
L_\pi=(\theta s-a_*)^2,\qquad
\frac{\partial L_\pi}{\partial\theta}=2(\theta s-a_*)s.
$$

梯度下降使输出动作接近 Critic 评价中的最优动作 $a_*$。实际 Critic 的近似误差也会沿这条路径影响 Actor。

</details>

### 3.3 目标网络更新

在线网络更新后，目标网络通过 Polyak 平均跟随：

$$
\bar\theta\leftarrow\tau\theta+(1-\tau)\bar\theta,\qquad
\bar\phi\leftarrow\tau\phi+(1-\tau)\bar\phi.
$$

$\tau\in(0,1]$ 控制跟随速度。较慢的目标变化减少自举目标对单次参数更新的敏感程度。

### 3.4 探索与数据复用

采样动作可写为：

$$
a=\operatorname{clip}(\mu_\theta(s)+\epsilon,a_{\min},a_{\max}).
$$

探索噪声负责扩大动作覆盖，策略更新使用当前 Actor 的确定性动作。回放数据来自历史探索策略，因此 DDPG 属于离策略学习。

## 4. 实现与应用

### 4.1 训练流程

1. 初始化在线与目标网络，建立回放池。
2. 用带噪动作交互，记录奖励、后继状态和终止标记。
3. 采样批次，用目标网络计算价值回归目标。
4. 更新 Critic，再用当前 Critic 的动作梯度更新 Actor。
5. 软更新目标网络，继续采样。

### 4.2 更新示例

<details class="kt-derivation" id="ddpg-update">
<summary>价值与策略更新</summary>

```python
with torch.no_grad():
    next_action = target_actor(next_obs)
    target = rewards + gamma * (~terminated).float() * target_critic(next_obs, next_action)
value_loss = 0.5 * (critic(obs, actions) - target).square().mean()
critic_optimizer.zero_grad()
value_loss.backward()
critic_optimizer.step()

for parameter in critic.parameters():
    parameter.requires_grad_(False)
policy_loss = -critic(obs, actor(obs)).mean()
actor_optimizer.zero_grad()
policy_loss.backward()
actor_optimizer.step()
for parameter in critic.parameters():
    parameter.requires_grad_(True)
```

示例采用参数独立且全部可训练的 Actor 与 Critic；奖励、终止掩码和价值输出统一为 `[B,1]`。冻结 Critic 参数保留了对动作输入的求导。目标网络在之后执行软更新。

</details>

## 5. 分析与总结

### 5.1 价值误差与策略更新

Actor 会倾向于 Critic 评价较高的动作，局部高估可能被策略反复利用。TD3 从双 Q 目标、目标动作平滑和更新频率三个位置缓解这一问题。

### 5.2 方法关系

DDPG 将离策略价值回归与确定性动作梯度结合。PPO 用固定优势加权概率梯度，DDPG 则在 Actor 更新时沿当前 Q 函数对动作求导；两者的 Critic 作用路径不同。

## 6. 关联与参考

### 6.1 相关知识

- [强化学习基础](../01_rl-foundations/01_reinforcement-learning-foundations.md)：价值回归与策略梯度。
- [TD3](./05_td3.md)：确定性 Actor-Critic 的改进。
- [SAC](./06_sac.md)：随机策略与最大熵优化。
- [PPO](./03_ppo.md)：固定优势的策略更新。

### 6.2 参考文献

1. Silver et al., [Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.html), 2014。
2. Lillicrap et al., [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), 2015。
3. OpenAI, [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)：目标与实现。
