# Dreamer：基于潜在世界模型的想象策略学习

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

Dreamer（Hafner et al.，ICLR 2020）是「潜在世界模型 + 想象策略优化」路线的代表作：在图像等高维观测上学一个循环状态空间世界模型（RSSM），在其隐空间中**想象**批量 rollout，并在这些想象轨迹上直接训练 actor-critic——策略梯度**穿过（冻结参数的）世界模型**反传。谱系上：PlaNet（ICML 2019）建立 RSSM 与潜在空间 CEM 规划，Dreamer 把「在线搜索动作」换成「学习显式策略」，DreamerV2（ICLR 2021）改用离散分类隐变量，DreamerV3（arXiv 2023，Nature 2025）以固定超参数跨任务首次在无人类数据 Minecraft 中采到钻石，证明该范式的规模可扩展性。

## 2. 基础与定义

### 2.1 观测编码器与 RSSM

- **观测编码器 $e_t=\mathrm{enc}(o_t)$**：把图像等观测嵌入到特征空间；
- **RSSM（Recurrent State-Space Model）**：隐状态由「确定性 + 随机性」两部分组成：

$$h_t=\mathrm{GRU}(h_{t-1},z_{t-1},a_{t-1}),\qquad z_t\sim p_\theta(z_t\mid h_t,e_t)$$

- $h_t$（deterministic）：GRU 传递的长期记忆，保证梯度在时间上可传播；
- $z_t$（stochastic）：吸收单步观测信息的随机瓶颈——V1 用高斯，V2/V3 用多个分类变量拼合，训练更稳。

### 2.2 世界模型与 Actor-Critic

Dreamer 由两半组成，**训练目标彼此独立**：

- **世界模型**（encoder/RSSM/解码器/奖励头/继续头）：学习「环境长什么样」；
- **Actor-Critic**：在世界模型冻结后产生的想象轨迹上学「怎么行为」。

### 2.3 POMDP 与隐状态

RSSM 本质是学到的 belief state：$h_t$ 聚合历史 $(o_{1:t},a_{1:t-1})$，对部分可观测环境给出充分统计量的近似——这也是 POMDP 一篇在 Model-Based 分支中反复出现的原因。

## 3. 核心原理

### 3.1 世界模型训练

想象 rollout 之前，模型先在真实序列上训练，损失为（V3 形状的要点）：

$$\mathcal L_{\text{world}}=\underbrace{-\log \hat p(o_t\mid h_t,z_t)}_{\text{观测重建}}+\underbrace{\big\|\hat r_t-r_t\big\|^2}_{\text{奖励预测}}+\underbrace{\mathrm{BCE}(\hat c_t,c_t)}_{\text{continue 预测}}+\underbrace{\beta\,\mathrm{KL}\big[\,p_\theta(z_t\mid h_t,e_t)\,\Vert\,q_\phi(z_t\mid h_{t-1},a_{t-1})\,\big]}_{\text{后验→先验一致（含 KL balancing）}}$$

- **观测重建（observation reconstruction）**：解码器把 $(h_t,z_t)$ 还原观测，为隐状态提供「信息不丢失」的监督；
- **奖励预测（reward prediction）**、**继续预测（continue prediction）**：想象 rollout 需要的全部任务信号；$\hat c_t$ 预测回合是否继续，决定折扣因子 $\gamma$；
- **KL balancing**（V2/V3）：先验偏向信任后验，减轻模型过早收缩。

注意：重建损失属于**模型学习**；actor 的优化目标（3.3）与它无关——两者经常被混淆，但必须分开。

### 3.2 潜在空间轨迹采样

从真实 buffer 采起点 $(h_0,z_0)$（warm-up 若干步编码真实观测），然后纯在隐空间批量展开 $H$ 步：

$$h_{t+1}=\mathrm{GRU}(h_t,z_t,a_t),\quad z_{t+1}\sim q_\phi(z_{t+1}\mid h_{t+1}),\quad a_t\sim\pi_\phi(z_t),\quad \hat r_t=\hat r(z_t,a_t)$$

无解码、无观测——整条想象轨迹的计算只涉及小隐状态，可大批量并行。

### 3.3 模型内 Actor-Critic

对每条想象轨迹估计 λ-returns $\lambda G_t$：前向迭代 TD 目标，终点（$H$ 步处）不再继续展开，而由 **critic 给出的 value bootstrap** 截断——短时域 + 终端自举同时控制了计算与模型误差的复合。

- **Actor 目标**：最大化想象回报的 λ-return：

$$\max_\phi\ \mathbb E\Big[\textstyle\sum_t \lambda G_t\Big],\qquad \nabla_\phi \lambda G_t=\frac{\partial \lambda G_t}{\partial a_{t:H-1}}\frac{\partial a_{t:H-1}}{\partial \phi}$$

策略梯度**经过世界模型**（$z_{t+1}$ 依赖 $a_t$）链式反传，但**模型参数不在策略更新中被修改**——冻结参数 ≠ 切断梯度，梯度流经的是模型函数而不更新模型权重（这是 Dreamer 与 MBPO 的机制分水岭）。

- **Critic 目标**：回归 λ-returns（V3 用离散化 value bin 交叉熵，改善回归稳定性）；actor 与 critic 更新频率错开（actor 慢、critic 快）。

### 3.4 模型误差与策略优化

模型误差进入策略的通道是 λ-return：误差使梯度方向偏移，策略随之漂向模型「想象出来的好处」。Dreamer 的缓解：终端 bootstrap 把反传时域压短（与 MBPO 的短 rollout 同一动机）；RSSM 的先验-后验约束抑制隐空间的过度外推；V3 的 value bin 与 free bits 等正则进一步提升鲁棒性。

## 4. 实现与应用

### 4.1 RSSM 与世界模型损失实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, s_dim, z_dim, a_dim, h_dim):
        super().__init__()
        self.gru = nn.GRUCell(z_dim + a_dim, h_dim)
        self.post = nn.Linear(h_dim + s_dim, 2 * z_dim)   # 后验 p(z_t|h_t,e_t)
        self.prior = nn.Linear(h_dim, 2 * z_dim)          # 先验 q(z_t|h_t)

    def step(self, h, z, a, e=None, sample=True):
        h = self.gru(torch.cat([z, a], -1), h)
        mu, lv = (self.post(torch.cat([h, e], -1)) if e is not None
                  else self.prior(h)).chunk(2, -1)
        lv = lv.clamp(-7, 7)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu) if sample else mu
        return h, z, mu, lv

def world_model_loss(rssm_out, recon, r_hat, c_hat, o, r, c, beta=1.0):
    """重建 + 奖励 + continue + KL（KL balancing 细节从简）."""
    L_recon = F.mse_loss(recon, o)
    L_reward = F.mse_loss(r_hat, r.squeeze(-1))
    L_cont = F.binary_cross_entropy_with_logits(c_hat, c.squeeze(-1))
    L_kl = beta * kl_divergence_posterior_prior(rssm_out)   # posterior vs prior
    return L_recon + L_reward + L_cont + L_kl
```

### 4.2 模型内策略训练

1. 从真实 buffer 采批序列，warm-up 编码得 $(h_0,z_0)$；
2. `for t in range(H): a=actor(z_t); h,z,..=rssm.step(h,z,a); r=r_head(z,a); c=c_head(z,a)`；
3. 计算 λ-returns（critic 终端 bootstrap）；
4. actor 反传（retain 模型参数 `requires_grad=False`，但让梯度经过计算图）、critic 回归；
5. 每步把世界模型更新与想象更新交错（模型先训，策略后学）。

### 4.3 适用场景

- 高维观测（像素）、跨任务通用性优先的基准研究（V3 的「一套超参」）；
- 单 GPU/少量算力预算下的视觉控制任务（想象批量并行的高性价比）；
- 需要显式可导出策略（不能每步在线搜索）的部署形态。

## 5. 分析与总结

- Dreamer 把 Model-Based RL 的模型用途推到「梯度式」：模型不仅造数据，还是策略梯度的传导介质。
- 三步技术链条：RSSM 压缩观测为 belief 状态 → 隐空间想象几乎免费 → actor-critic 在想象上以任意规模训练。
- 与 MBPO 的对照：两者都冻结模型，但 Dreamer 的策略梯度**经过**模型反传，MBPO 完全不反传、只把模型当经验源。
- 局限：模型误差的梯度通道更隐蔽（policy exploitation 表现为训练不稳而非显式误差）；对任务信号稀疏到只剩成败奖励时仍需较大算力；离散分类隐变量对连续精细控制的表达能力存在争议（待补充）。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——Actor-Critic 与 λ-return
- **本分支地基**：🚧 基于模型的强化学习基础（撰写中）——潜在世界模型与想象 rollout
- **部分可观测视角**：🚧 POMDP 与循环策略（撰写中）——RSSM 即学到的 belief state 编码
- **进阶阅读**：🚧 可微动力学与直接策略优化（撰写中）——「梯度经过模型」路线的极端形态与稳定性设计

### 6.2 参考资料

1. Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination"（ICLR 2020，Dreamer）。正式链接待补充。
2. Hafner et al., "Learning Latent Dynamics for Planning from Pixels"（ICML 2019，PlaNet）。正式链接待补充。
3. Hafner et al., "Mastering Diverse Domains through World Models"（arXiv:2301.04104，2023；Nature 2025，DreamerV3）：[arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
4. DreamerV3 官方开源实现：[github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)
5. Hafner et al., "DreamerV2: Mastering Atari with Discrete World Models"（ICLR 2021）。正式链接待补充。
