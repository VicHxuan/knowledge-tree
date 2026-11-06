# PETS：概率动力学模型与轨迹采样规划

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

PETS（Probabilistic Ensembles with Trajectory Sampling，Chua et al.，NeurIPS 2018）是规划式 Model-Based RL 的代表：训练一个**概率系综动力学模型**，然后在模型内用**交叉熵方法（CEM）**对动作序列做在线优化，按 MPC 的滚动时域方式执行。它没有显式的策略网络——「策略」就是每一步重新做一次的轨迹优化。PETS 的谱系可追溯到 PILCO（Deisenroth & Rasmussen，2011）的高斯过程动力学模型；PETS 用「概率神经网络 + 系综」替代 GP，把数据效率带到高维观测任务。

注意与经典 MPC 的区分：MPC 本身是控制框架，动力学已知（由物理方程给出）时它不是强化学习；PETS 的特殊性在于**动力学从数据中学习**，且模型的不确定性被显式纳入轨迹评估。

## 2. 基础与定义

### 2.1 概率动力学模型

模型输出下一状态的分布而非单点：

$$\hat s_{t+1}\sim\mathcal N\big(\mu_\theta(s_t,a_t),\ \Sigma_\theta(s_t,a_t)\big)$$

- 均值网络给出预测中心，方差网络表达**偶然不确定性**（环境固有噪声）；
- 训练用负对数似然：$\mathcal L(\theta)=-\log p_\theta(s_{t+1}\mid s_t,a_t)$，而不是均方误差——避免模型对多模态转移输出「错误的平均」。

### 2.2 概率系综

单一神经网络无法表达**认知不确定性**（数据没覆盖的区域它也敢给出自信预测）。PETS 用 $N$ 个独立初始化、独立训练（各自 bootstrap 子集）的成员组成系综：

- 成员间预测的分歧 = 认知不确定性的度量；
- 轨迹评估时让不同成员参与不同粒子/时刻的预测，把「不知道」传播到代价估计中。

### 2.3 MPC 与滚动时域

模型预测控制（Model Predictive Control）：每个时刻求解未来 $H$ 步的动作序列最优解，**只执行第一步**，下个时刻携带新观测重新求解——用滚动重规划吸收模型误差。

## 3. 核心原理

### 3.1 轨迹采样

对模型做 $H$ 步展开有两条路：

- **解析传播**：假设分布为高斯并在成员间线性化传递（PILCO 的路线）——高维下不可行；
- **轨迹采样（TS）**：维护 $P$ 个粒子，每步从某系综成员的预测分布中**采样**下一状态。PETS 给出两种粒度：TS$^\infty$（粒子间、时间步间独立换成员）与 TS$^1$（同一粒子始终用同一成员，捕获模型偏差的时序相关性）。

粒子集合近似了 belief 状态 $p(\hat s_{t+k})$，其离散程度就是「模型对未来知道多少」。

### 3.2 CEM 动作序列优化

CEM 是分布迭代优化：维护长度 $H$ 的动作序列高斯分布 $\mathcal N(\mu,\sigma^2)$，循环若干轮：

1. 采样 $K$ 条候选序列 $A^{(1)},\dots,A^{(K)}$；
2. 每条序列用 $P$ 个粒子按 TS 展开至 $H$ 步，累计代价取粒子均值；
3. 取表现最好的前 $\lceil \rho K\rceil$（elite）序列；
4. 用 elite 集合重新拟合 $(\mu,\sigma^2)$，收缩采样分布。

执行 elite 均值序列的第一个动作，随后整个规划在下一时刻从头再来。

### 3.3 代价函数与模型预测范围

$$J(A)=\mathbb E\!\left[\sum_{t=0}^{H-1} c\big(\hat s_t^{(p)},a_t\big)\right]$$

其中 $c$ 通常取负奖励（如距离目标的平方代价），期望在粒子与系综成员上平均。评估长度 $H$ 就是想象 rollout 的时域上限——PETS 通过每步重规划，把「模型多步误差累积」限制在 $H$ 步内。

### 3.4 无显式 Actor 的规划架构

规划式 MBRL 的谱系内有一类方法完全省去策略网络：PETS 用 CEM、PlaNet 用 CEM 变体在潜在空间搜索。优点是「策略」始终与最新模型同步、无策略滞后；代价是每个决策都要做一次优化，无法做快速反应式推理，也难以做带随机起始状态的大规模想象训练。MuZero 把规划保留在**推理与训练信号**中、同时用网络提供先验来加速搜索，是这条线的另一种平衡。

## 4. 实现与应用

### 4.1 CEM 规划器实现

```python
import torch

@torch.no_grad()
def cem_plan(model, s0, horizon, dim_a, pop=400, elites=40, iters=5):
    """model: DynamicsEnsemble.step(s, a) -> 高斯参数（见 MB 基础篇 4.1）"""
    mu = torch.zeros(horizon, dim_a)
    std = torch.ones(horizon, dim_a)

    def rollout_cost(seqs):                      # seqs: [K, H, dim_a]
        K = seqs.shape[0]
        s = s0.repeat(K, 1)                      # 粒子=候选序列并行
        cost = torch.zeros(K)
        for t in range(horizon):
            a = seqs[:, t]
            mean, log_var = model.step(s, a)     # [N, K, s_dim]
            idx = torch.randint(mean.shape[0], (K,))   # TS1: 每序列抽一个成员
            m, lv = mean[idx, torch.arange(K)], log_var[idx, torch.arange(K)]
            s = m + torch.exp(0.5 * lv) * torch.randn_like(m)   # 采样下一状态
            cost += ((s - goal) ** 2).sum(-1)    # 示意：二次代价
        return cost

    for _ in range(iters):
        seqs = mu + std * torch.randn(pop, horizon, dim_a)
        costs = rollout_cost(seqs)
        top = costs.topk(elites, largest=False).indices
        elite = seqs[top]
        mu, std = elite.mean(0), elite.std(0) + 1e-3
    return mu[0]                                  # 只执行第一个动作
```

### 4.2 适用场景

- 低维状态（机器人关节角、CartPole 类）连续控制任务；
- 样本预算紧、允许较慢决策的场景（每步一次 CEM）；
- 高维图像观测需要先经表征压缩（PlaNet 的路线：在潜在空间执行同样的规划）。

## 5. 分析与总结

- PETS 的两个支柱：**概率模型**处理偶然不确定性，**系综**处理认知不确定性，二者都通过轨迹采样进入代价估计——这使它在极少真实样本（数十条轨迹量级）下可工作。
- 优点：无策略训练的超参与不稳定性；模型改进立刻反映在行为上。
- 局限：每步重规划的计算成本高（实时性差）；长时域下 CEM 采样分布退化；没有可离线导出的反应式策略。
- 与分支内其他方法的关系：MBPO 保留了它的系综概率模型，但把「在线搜索动作」换成「模型生成数据训练 SAC」；MuZero 用网络先验把搜索做便宜。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——奖励、折扣与最优控制的目标形式
- **本分支地基**：🚧 基于模型的强化学习基础（撰写中）——概率/确定性模型、系综与不确定性
- **进阶阅读**：🚧 MuZero（撰写中）——同样以规划为中心、但用网络先验加速搜索
- **相关方法**：🚧 MBPO（撰写中）——同一系综模型思想的「数据生成」用法

### 6.2 参考资料

1. Chua et al., "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"（NeurIPS 2018，PETS）。正式链接待补充。
2. Deisenroth & Rasmussen, "PILCO: A Model-Based and Data-Efficient Approach to Policy Search"（ICML 2011）——解析不确定性传播的前身。正式链接待补充。
3. Hafner et al., "Learning Latent Dynamics for Planning from Pixels"（ICML 2019，PlaNet）——潜在空间 CEM 规划。正式链接待补充。
