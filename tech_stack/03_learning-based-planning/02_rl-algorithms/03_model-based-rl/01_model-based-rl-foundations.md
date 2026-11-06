# 基于模型的强化学习基础

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

Model-Based RL（基于模型的强化学习）的核心思想：显式学习一个环境模型，让策略额外从「模型产生的经验」中学习，用模型误差换取样本效率。本篇是 Model-Based RL 分支的技术地基：定义环境模型与奖励模型的形态、世界模型的组成、真实与想象 rollout 的区别，以及模型误差如何影响策略优化。范式层面的对比与选型（Model-Free 与 Model-Based 的本质区别、两条路线的适用场景）在《有模型与无模型强化学习》中展开，本篇不重复。

## 2. 基础与定义

### 2.1 动力学模型与奖励模型

MDP 的转移核与奖励函数是环境的完整描述，Model-Based RL 的「模型」即对它们的学习或利用：

- **动力学模型（dynamics model）**：$\hat p(s'\mid s,a)$ 或确定性映射 $\hat s' = f_\theta(s,a)$，预测状态转移；
- **奖励模型（reward model）**：$\hat r(s,a)$ 或 $\hat r(s'\mid s,a)$，预测即时奖励；
- **终止/继续信号（continue prediction）**：$\hat d(s,a)$，预测回合是否结束，决定想象轨迹的折扣与截断。

### 2.2 已知动力学与学习动力学

- **已知动力学**：规则完全已知的棋类、可解析的刚体/轨道动力学——模型给定，问题退化为规划（经典 MPC、MCTS 属于这一形态，它们本身不必然是强化学习）。
- **学习动力学**：真实环境未知，从交互数据中拟合 $\hat p,\hat r$——本分支各篇主要讨论这一类。
- **中间形态**：存在近似物理模型但与真实环境失配（model mismatch），只学习残差或做少量参数校准。

### 2.3 确定性模型与概率模型

- **确定性模型**：输出单点预测 $\hat s' = f_\theta(s,a)$。实现简单，但无法表达环境固有的随机性（偶然不确定性），多步展开时容易把「平均」错当「真实」。
- **概率模型**：输出分布 $\hat p_\theta(s'\mid s,a)$（常用高斯：预测均值与方差）。PILCO / PETS 一线的工作表明：概率模型是长时域规划稳定性的关键，方差输出同时为不确定性度量提供基础。

### 2.4 显式状态与潜在状态

- **显式模型**：在原始观测空间直接预测下一观测（PETS、MBPO），状态语义清楚、监督信号直接。
- **潜在模型（world model）**：先把观测编码为隐状态 $z_t$，在隐空间中建模转移（Dreamer 的 RSSM、MuZero 的隐式状态）。优点是压缩表示、天然适配图像观测；代价是需要为「隐状态是否学到位」设计间接监督（重建、奖励预测、价值预测等）。

## 3. 核心原理

### 3.1 真实轨迹与模型轨迹

- **真实 rollout**：策略与环境真实交互产生的 $(s_t,a_t,r_t,s_{t+1})$ 序列——数据「贵但无模型偏差」。
- **想象 rollout（imagined rollout）**：从某个（通常是真实的）状态出发，在模型中逐步展开：

$$\hat s_{t+1}\sim\hat p_\theta(\cdot\mid \hat s_t,a_t),\qquad \hat r_t=\hat r_\theta(\hat s_t,a_t)$$

配齐奖励与终止信号后即得到合成轨迹——数据「便宜但携带模型偏差」。

Model-Based RL 的各变体，本质都是在回答三个问题：想象 rollout 从哪开始、走多远、产生的数据（或梯度）交给谁使用。

### 3.2 预测误差与累积误差

模型每步的小误差 $\varepsilon_t$ 会沿 rollout 复合放大——即使单步误差有界，$k$ 步后状态分布也会显著偏离真实。两个工程后果：

1. **短 rollout**：把想象轨迹限制在模型可信的时域内（MBPO 用线性递增的 rollout 长度控制偏差增长）；
2. **单步监督优先**：训练世界模型时以单步（或少数几步）转移预测为主，而非长序列端到端拟合。

### 3.3 模型不确定性与误差利用

- **认知不确定性（epistemic）**：数据不足所致，随数据增加而下降。工程上用系综（ensemble 成员间的方差）或贝叶斯网络度量。
- **偶然不确定性（aleatoric）**：环境本身随机，只能用概率模型表达，不会随数据消失。
- **Model exploitation（模型利用 / 模型钻空子）**：策略优化会把模型预测「过于乐观」的区域当作真实可达的高回报区域，主动 exploiting 模型误差——模型偏差从数值问题升级为被优化目标。经典对策：系综不确定性加权、限制 rollout 长度、对想象数据降低置信度。

### 3.4 世界模型学习与策略优化

模型学习的目标（极大似然、变分推断、重建损失）与策略优化目标是分离的：模型只需「好到能产生有用的数据或梯度」。这解释了两个经验现象：

- 不追求完美模型——策略关心的区域与任务相关，短时域内够用即可；
- 模型质量与最终性能之间不是单调关系——更强的模型若误差分布被策略利用，性能反而可能下降。

### 3.5 训练流程

1. 策略与环境交互，收集真实数据进 replay buffer；
2. 用真实数据训练世界模型（单步/多步转移监督）；
3. 用模型生成想象 rollout（起点通常取自真实数据）；
4. 想象数据交给策略学习器——规划搜索、model-free 更新、或梯度反传，三种用法见 4.2；
5. 回到第 1 步，交替进行。

Dyna、PETS、MBPO、MuZero、Dreamer 的差异全部体现在第 3–4 步的「怎么用模型」上。

## 4. 实现与应用

### 4.1 概率系综模型实现

```python
import torch
import torch.nn as nn

class ProbabilisticDynamics(nn.Module):
    """单个系综成员：输出高斯 (mean, log_var)，表达偶然不确定性."""
    def __init__(self, s_dim, a_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
        )
        self.mean = nn.Linear(h, s_dim)
        self.log_var = nn.Linear(h, s_dim)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.mean(self.net(x)), self.log_var(self.net(x)).clamp(-10, 2)

class DynamicsEnsemble(nn.Module):
    """N 个独立初始化、独立训练的成员，成员间方差表达认知不确定性."""
    def __init__(self, s_dim, a_dim, n=5):
        super().__init__()
        self.members = nn.ModuleList(
            ProbabilisticDynamics(s_dim, a_dim) for _ in range(n))

    def step(self, s, a):
        preds = [m(s, a) for m in self.members]
        mean = torch.stack([p[0] for p in preds])   # [N, B, s_dim]
        log_var = torch.stack([p[1] for p in preds])
        return mean, log_var
```

### 4.2 模型使用方式

| 流派 | 模型用途 | 代表方法 |
| :--- | :--- | :--- |
| 规划式 | 在模型内搜索动作序列/搜索树 | PETS（CEM）、MuZero（MCTS） |
| 数据生成式 | 模型 rollout 产出合成经验，交给 model-free 学习器 | Dyna、MBPO |
| 梯度式 | 策略目标函数经过可微模型反传梯度 | Dreamer、SHAC |

三种用法不互斥：MuZero 用搜索树产生比初始策略更强的训练信号（规划式），Dreamer 在冻结的世界模型中反传策略梯度（梯度式），MBPO 则把模型当「经验放大器」（数据生成式）。分支内的各篇文档按此脉络展开。

## 5. 分析与总结

- **主线是交换**：以模型偏差与额外计算，换取真实交互样本量。样本贵（真实机器人、风洞、真实场景）时收益最大；样本便宜（仿真器大规模并行）时 model-free 往往更简单可靠。
- **设计要点**：概率模型表达偶然不确定性；系综表达认知不确定性；短 rollout 控制误差复合；起点取自真实数据分布。
- **准确理解边界**：Model-Based 不要求神经网络世界模型（表格 Dyna-Q 也是 model-based）；Model-Based 不要求通过动力学反传梯度（MBPO 恰恰不反传）；On-Policy / Off-Policy 与 Model-Based / Model-Free 是两个独立的分类维度。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——MDP、贝尔曼方程与策略梯度
- **范式导航**：🚧 有模型与无模型强化学习（撰写中）——两条范式的对比与选型
- **进阶阅读**：🚧 Dyna（撰写中）、🚧 PETS（撰写中）、🚧 MBPO（撰写中）、🚧 Dreamer（撰写中）——按 4.2 的三种模型用法逐篇展开

### 6.2 参考资料

1. Sutton & Barto，《Reinforcement Learning: An Introduction》（第 2 版，2018）第 8 章：planning and learning with models。正式链接待补充。
2. Sutton, "Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming"（ICML 1990）——Dyna 架构原始文献。正式链接待补充。
3. Hafner et al., "Learning Latent Dynamics for Planning from Pixels"（ICML 2019，PlaNet）——潜在世界模型与想象 rollout。正式链接待补充。
