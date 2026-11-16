# 值函数方法与 DQN

## 1. 概述

DQN 用神经网络估计离散动作的价值，通过贝尔曼目标回归学习，再从动作价值中提取策略。在线网络负责当前预测，目标网络提供自举估计，经验回放组织训练样本。

## 2. 基础与定义

### 2.1 网络结构

在线网络 $Q_\phi(s,\cdot)$ 输入状态，输出各离散动作的价值。目标网络 $Q_{\bar\phi}$ 具有相同结构，其参数定期从在线网络复制。贪婪动作满足：

$$
a^*(s)\in\arg\max_a Q_\phi(s,a).
$$

行为策略以概率 $\epsilon$ 随机探索，以概率 $1-\epsilon$ 选择贪婪动作。

### 2.2 数据与符号

回放池 $\mathcal D$ 保存 $(s_t,a_t,r_{t+1},s_{t+1},d_t)$，其中 $d_t=1$ 表示任务真正终止。外部截断转移保留重置前后继状态，并允许价值自举。$\gamma$ 为折扣因子。

## 3. 核心原理

### 3.1 回归损失与梯度

对回放样本构造固定目标 $y_t$，在线网络最小化：

$$
L_Q(\phi)=\frac12\hat{\mathbb E}_{\mathcal D}[(Q_\phi(s_t,a_t)-y_t)^2].
$$

梯度只经过当前动作的价值预测：

$$
\nabla_\phi L_Q
=\hat{\mathbb E}_{\mathcal D}[(Q_\phi(s_t,a_t)-y_t)\nabla_\phi Q_\phi(s_t,a_t)].
$$

<details class="kt-derivation" id="dqn-semigradient">
<summary>表格更新与网络梯度</summary>

记 $\delta_t=y_t-Q_\phi(s_t,a_t)$。梯度下降给出：

$$
\phi\leftarrow\phi+\alpha\hat{\mathbb E}_{\mathcal D}[\delta_t\nabla_\phi Q_\phi(s_t,a_t)].
$$

若每个状态动作对由一个独立参数表示，$\nabla Q$ 选中对应表项，退化为表格 Q-learning 的误差校正。神经网络共享参数，使一次更新同时影响其他状态动作的预测。

</details>

### 3.2 贝尔曼目标

最优动作价值满足：

$$
Q^*(s,a)=\mathbb E\left[r_{t+1}+\gamma\max_{a'}Q^*(s_{t+1},a')\mid s_t=s,a_t=a\right].
$$

用单次转移与目标网络近似右端，得到：

$$
y_t=r_{t+1}+\gamma(1-d_t)\max_{a'}Q_{\bar\phi}(s_{t+1},a').
$$

目标网络在若干次训练中固定，使回归目标变化较慢；之后执行 $\bar\phi\leftarrow\phi$。

### 3.3 经验回放与策略改进

回放池随机抽取历史转移，提供数据复用并降低相邻样本的时序相关性。目标中的下一动作按贪婪价值选择，采样动作则来自探索策略，因此 DQN 属于离策略学习。

更新后的 Q 值决定新的贪婪动作，形成“经验 → 价值回归 → 行为策略 → 新经验”的循环。

### 3.4 价值估计变体

Double DQN 用在线网络选择下一动作，再用目标网络评价：

$$
a^*=\arg\max_{a'}Q_\phi(s_{t+1},a'),\qquad
y_t^{\rm DDQN}=r_{t+1}+\gamma(1-d_t)Q_{\bar\phi}(s_{t+1},a^*).
$$

这将动作选择与目标评价分开，以缓解最大值选择引起的过估计。

<details class="kt-derivation" id="dqn-extensions">
<summary>价值表示与采样改进</summary>

Dueling 将状态价值与动作优势组合为：

$$
Q(s,a)=V(s)+A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a').
$$

优先回放改变样本抽取分布，可用重要性权重调整损失；多步目标增加实际奖励的覆盖范围；回报分布建模学习回报的分布。Rainbow 将这些机制与 Double DQN、参数化探索噪声结合。

</details>

## 4. 实现与应用

### 4.1 训练流程

1. 初始化在线网络、目标网络与回放池。
2. 用探索策略选择动作，存储真实转移与终止标记。
3. 从回放池抽取小批量，计算停止梯度的贝尔曼目标。
4. 选择批次中已执行动作的 Q 值，计算损失并更新在线参数。
5. 定期同步目标网络，继续采样。

### 4.2 更新示例

<details class="kt-derivation" id="dqn-update">
<summary>小批量损失计算</summary>

```python
with torch.no_grad():
    next_q = target_q(next_obs).max(dim=-1).values
    target = rewards + gamma * (~terminated).float() * next_q
predicted = q_network(obs).gather(1, actions[:, None]).squeeze(1)
loss = 0.5 * (predicted - target).square().mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`actions` 为整数索引，奖励、目标和选中的 Q 值均为 `[B]`。`next_obs` 使用实际后继状态，回合重置由采集器处理。Huber 损失可替代平方误差，以改变大残差对梯度的影响。

</details>

## 5. 分析与总结

### 5.1 适用条件

离散动作可直接枚举网络输出，动作数量很大时计算成本上升。连续动作中的最大化可通过额外优化或参数化 Actor 完成，由此连接到 DDPG 等方法。

### 5.2 估计误差

自举误差、函数近似和回放分布共同影响训练。目标网络与经验回放改善训练条件；价值预测误差和任务回报应同时评估。

## 6. 关联与参考

### 6.1 相关知识

- [强化学习基础](../01_rl-foundations/01_reinforcement-learning-foundations.md)：Q-learning 与半梯度。
- [DDPG](./04_ddpg.md)：连续动作的参数化最大化。
- [多价值网络](../01_rl-foundations/05_multi-critic.md)：估计器数量与组合规则。

### 6.2 参考文献

1. Mnih et al., [Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236), 2015。
2. van Hasselt et al., [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), 2015。
3. Wang et al., [Dueling Network Architectures](https://proceedings.mlr.press/v48/wangf16.html), 2016。
4. Hessel et al., [Rainbow](https://arxiv.org/abs/1710.02298), 2017。
