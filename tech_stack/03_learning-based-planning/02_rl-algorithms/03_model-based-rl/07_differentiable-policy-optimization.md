# 可微动力学与直接策略优化

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

除了「模型造数据」（Dyna/MBPO）与「模型内搜索」（PETS/MuZero），还有第三种用法：**把状态转移当作可微计算图，用反向传播（BPTT）直接求策略梯度**。这类方法统称可微动力学/直接策略优化——策略回报对动作的敏感度经由 $\partial s'/\partial a$、$\partial s'/\partial s$ 链式传播到策略参数，即 pathwise（路径导数）策略梯度。SHAC（Short-Horizon Actor-Critic，2022）是这条路线中兼顾稳定性与样本效率的代表：短时域 rollout + critic 终端自举，抑制长链梯度的爆炸与偏差。

三个必须先说清的边界（也是分支内反复强调的区分）：

1. **可微动力学不是所有 Model-Based RL 的必要条件**——Dyna/MBPO/PETS/MuZero 都不需要动力学可微；
2. **冻结世界模型参数 ≠ 切断经过世界模型的梯度**——Dreamer 冻结模型却反传策略梯度；「冻结」指不更新模型权重，梯度仍然流经模型函数；
3. **使用模型生成样本与通过模型反传梯度是两种不同的技术机制**——前者改数据的量，后者改梯度的路。

## 2. 基础与定义

### 2.1 可微状态转移

设转移 $s_{t+1}=f(s_t,a_t)$ 可微，则 rollout 是一个复合函数计算图。两类来源：

- **可微物理仿真器**：以可微算子实现刚体/软体动力学（分析梯度或自动微分），常见于 GPU 加速仿真与图形学；
- **学习得到的可微代理动力学**：神经网络拟合的 $\hat f_\theta$ 天然可微——但梯度质量取决于模型在训练分布之外的行为。

### 2.2 状态与动作雅可比矩阵

反传的核心变量是两个雅可比：

$$A_t=\frac{\partial f}{\partial a}\Big|_{(s_t,a_t)},\qquad S_t=\frac{\partial f}{\partial s}\Big|_{(s_t,a_t)}$$

$k$ 步 BPTT 中，梯度按 $\prod_{j}S_j$ 复合——**状态雅可比的谱半径决定梯度爆炸/消失**：这是该路线一切稳定性设计的根源。

### 2.3 路径导数策略梯度

对随机策略 $a\sim\pi_\phi(s)$，用重参数化把随机性从优化变量中剥离后，可直接对期望回报求导：

$$\nabla_\phi J=\mathbb E\Big[\nabla_\phi\big(\textstyle\sum_{t=0}^{k-1}r(s_t,a_t)+V(s_k)\big)\Big]$$

与 score-function（REINFORCE 型）梯度相比，pathwise 梯度方差低、但要求模型可微且光滑假设更强。

## 3. 核心原理

### 3.1 时域截断与 BPTT

对短时域想象 rollout（长度 $k$），从真实起点 $(s_0)$ 出发：

$$s_{t+1}=f_\theta(s_t,a_t),\quad a_t=\pi_\phi(s_t),\quad J_k=\sum_{t=0}^{k-1}r(s_t,a_t)+V_\psi(s_k)$$

对 $J_k$ 做 BPTT：梯度依次穿过 $V_\psi$、$\prod S$、奖励函数，回到 $\pi_\phi$。截断在 $k$ 步而非整回合，是稳定性设计的第一层。

### 3.2 终端价值自举与 SHAC

SHAC 的关键组合：**短时域想象 + 终端价值自举**。

- rollout 只展开 $k$ 步（如 3~5），梯度链长度受控；
- 第 $k$ 步不假设「自然结束」，而是用学习到的 critic $V_\psi$ 估计余下回报，作为 BPTT 目标的终端；
- critic 用标准 TD/actor-critic 目标训练，对 $V$ 的梯度**不**穿过动力学（否则会被 $\prod S$ 的病态谱污染）。

直觉：短链保住梯度的方向感（真实敏感度），bootstrap 补回被截断的长时视野。

### 3.3 梯度截断与数值稳定性

工程上的四件套：

1. **时域截断**（如上）；
2. **梯度范数裁剪**：对 BPTT 梯度按范数截断，防 $S$ 谱半径 > 1 时的爆炸；
3. **对 $V$ 通路 stop-gradient**：终端 bootstrap 只提供标量价值，不回传过模型；
4. **模型侧正则**：训练可微代理模型时约束 Lipschitz/雅可比幅度，压低 $\|S\|$。

### 3.4 动力学误差对策略梯度的影响

与「造数据」路线不同，梯度式路线中模型误差直接**改写梯度方向**：$\partial\hat f/\partial a\neq\partial f/\partial a$ 时，策略沿着模型以为的最优方向优化，而真实环境可能并不买账。光滑性失配（接触、摩擦切换等不连续动力学）会让 $\hat f$ 的雅可比在真实环境的行为附近剧烈错误。这解释了该路线的成功域：**光滑动力学、短视野、连续控制**；而在接触丰富任务上，「造数据」类方法（MBPO）往往更稳。

## 4. 实现与应用

### 4.1 可微策略梯度实现

```python
import torch

def short_horizon_policy_grad(f_model, policy, critic, s0, k=4):
    """SHAC 式短时域直接策略优化（示意：批评家终端自举，价值通路不反传）."""
    s, J = s0, 0.0
    for t in range(k):
        a = policy(s)                       # 确定性策略（或重参数化采样）
        s = f_model(s, a)                   # 可微动力学：梯度将穿过这里
        J += reward(s, a)
    J = J + critic(s).detach()              # 终端 bootstrap：critic 提供标量，不传梯度
    (-J).backward()                         # 最大化回报 = 最小化 -J
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)   # 梯度裁剪
    # f_model.parameters() 的 requires_grad=False：冻结模型，梯度仍流经其计算图
```

### 4.2 应用场景

- 可微仿真器内的策略预训练/轨迹优化（图形学、机器人滚动接触等光滑阶段）；
- 与 model-free 方法级联：短时域 BPTT 预训练 + model-free 精调；
- 与学习型世界模型结合时，注意代理模型的雅可比质量决定可用时域上限。

## 5. 分析与总结

- 该路线的赌注：**模型的雅可比比模型的输出更有用**。成立时光滑任务上梯度质量远超 score-function；不成立时（不连续、模型外推区）反噬同样剧烈。
- SHAC 的启示与 MBPO/Dreamer 一致：短 rollout + 终端价值，是所有现代 MBRL 控制误差的共同手段——只是控制的对象从「数据分布」变成了「梯度链」。
- 方法选型速览：模型造数据（最稳，样本效率中等）→ 模型传梯度（样本效率最高，稳定性最依赖模型光滑性）→ 模型内搜索（推理开销大，无策略滞后）。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——策略梯度定理与重参数化
- **本分支地基**：🚧 基于模型的强化学习基础（撰写中）——模型使用的三种流派
- **进阶阅读**：🚧 Dreamer（撰写中）——同样反传经过模型、但作用于潜在世界模型；🚧 策略梯度、Actor-Critic 与 GAE（撰写中）——pathwise 与 score-function 梯度的对照

### 6.2 参考资料

1. SHAC（Short-Horizon Actor-Critic，2022）——短时域想象 + critic 终端自举的直接策略优化。正式链接待补充。
2. Schulman et al., "Gradient Estimation Using Stochastic Computation Graphs"（NeurIPS 2015）——pathwise 与 score-function 梯度的统一视角。正式链接待补充。
3. 可微物理仿真器的代表工作（GPU 分析梯度仿真、DiffTaichi 一线）。具体文献待补充。
