# MBPO：基于模型生成经验的策略优化

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

MBPO（Model-Based Policy Optimization，Janner et al.，NeurIPS 2019，"When to Trust Your Model"）把 Dyna 的思想在现代深度 RL 中做实：训练**概率系综动力学模型**，从真实 replay buffer 的状态出发做**短时域模型 rollout**，把合成经验写入单独的 synthetic replay buffer，交给标准的 model-free 学习器（SAC）消化。它的标题即方法论——**何时信任模型**：rollout 长度随训练线性增长，让「信任模型走多远」成为显式受控的量。

必须明确的一点：**MBPO 使用世界模型生成合成经验，策略优化时仍然是标准 SAC 的 model-free 更新——不存在通过世界模型的 BPTT。**「用模型生成样本」与「通过模型反传梯度」是两种不同的技术机制，MBPO 属于前者。

## 2. 基础与定义

### 2.1 动力学系综

与 PETS 同型的概率系综模型：$N$ 个成员各自输出高斯转移 $\hat p_{\theta_i}(s'\mid s,a)$，独立训练（各自的数据子集/独立优化器状态）。成员间分歧用于：

- rollout 时随机选成员（把认知不确定性注入合成数据）；
- 分析模型在状态空间的可靠程度。

### 2.2 短时域模型采样

想象 rollout 的起点**从真实 replay buffer 中采样**（策略真实到过的状态），而非模型自己上一步的输出——这保证 rollout 始终「贴着」真实数据分布展开。长度控制在 $k$ 步（1~5 量级）内：

$$s_0\sim \mathcal D_{\text{real}},\qquad \hat s_{t+1}\sim \hat p_\theta(\cdot\mid \hat s_t,a_t),\ \ a_t\sim\pi_\phi(\hat s_t),\ \ t=0,\dots,k-1$$

### 2.3 合成经验回放池

模型 rollout 产生的 $(\hat s,\,a\sim\pi(\hat s),\,\hat r,\,\hat s')$ 存入独立缓冲区；SAC 每次更新按固定比例**混合**采样真实 buffer 与合成 buffer 的数据。

## 3. 核心原理

### 3.1 训练循环

1. **真实交互**：当前策略与环境交互，数据进 $\mathcal D_{\text{real}}$；
2. **模型训练**：用 $\mathcal D_{\text{real}}$ 单步转移监督训练系综成员；
3. **生成合成经验**：从 $\mathcal D_{\text{real}}$ 采起点，$k$ 步 rollout（每步随机选成员），写入 $\mathcal D_{\text{model}}$；
4. **策略更新**：SAC 学习器在混合 mini-batch 上做标准更新（critic 软贝尔曼、actor 重参数化+熵正则、温度自适应）；
5. 回到 1。

### 3.2 采样时域与模型偏差

多步误差累积（见分支基础篇 3.2）意味着固定长度的 rollout 早晚越界。MBPO 的答案是把 $k$ 随训练进度**线性递增**（从 1 步 ramp 到任务相关的上限）：模型在数据丰富的状态区越准，越允许走远。这与「对模型信任度」的贝叶斯分析一致——系综分歧大的区域应短走。

### 3.3 与 SAC 的结合

MBPO 对 SAC 的修改仅在数据来源上：损失函数完全不变。

- critic 更新：$\mathcal D_{\text{real}}\cup\mathcal D_{\text{model}}$ 混合采样的 $(s,a,r,s')$，目标 $r+\gamma(1-d)\,[\min_{j=1,2}Q_{\bar\theta_j}(s',\tilde a')-\alpha\log\pi(\tilde a'\mid s')]$；
- actor/温度更新照旧（在合成数据上学到的 Q 上重参数化）。

正因策略学习器是「黑盒 model-free 消费者」，MBPO 才能即插即用地跟进 SAC 的改进——这也再次印证它是 Dyna（模型喂经验）而非 Dreamer（模型传梯度）的血统。

### 3.4 模型偏差与策略学习稳定性

模型偏差进入策略学习有两条通道：合成数据的奖励/转移误差，以及策略对误差区域的主动利用（model exploitation）。MBPO 的防御是组合拳：概率模型（不过度自信）+ 系综（分歧可见）+ 短 rollout（误差复合受限）+ 起点贴真实分布（不外推到没见过的状态）。即便如此，合成/真实数据的采样比例仍是需要调的关键超参——比例失衡时策略性能先升后降。

## 4. 实现与应用

### 4.1 模型轨迹生成

```python
import torch

@torch.no_grad()
def generate_model_rollouts(ensemble, policy, real_buffer, k, n_branch):
    """从真实数据起点出发的 k 步想象 rollout（每步随机选系综成员）.

    Returns: List[(s, a, r, s', d)] 合成转移样本
    """
    s0, _ = real_buffer.sample(n_branch)          # 起点必须来自真实数据
    samples, s = [], s0
    for t in range(k):
        a = policy.sample(s)                      # 当前策略在想象状态上作用
        mean, log_var = ensemble.step(s, a)       # [N, B, s_dim]
        idx = torch.randint(mean.shape[0], (s.shape[0],))
        m = mean[idx, torch.arange(s.shape[0])]
        lv = log_var[idx, torch.arange(s.shape[0])]
        s_next = m + torch.exp(0.5 * lv) * torch.randn_like(m)
        r = ensemble.reward(s, a, s_next)         # 奖励模型（或真实奖励函数）
        samples.extend(zip(s, a, r, s_next))
        s = s_next
    return samples
```

### 4.2 长度调度与数据配比

```python
def rollout_length(step, ramp_start, ramp_end, k_max):
    """随训练步数线性递增的 rollout horizon（MBPO 原文设定）."""
    if step < ramp_start: return 1
    frac = (step - ramp_start) / max(1, ramp_end - ramp_start)
    return int(round(1 + frac * (k_max - 1)))

# SAC 更新时的混合采样：每个 mini-batch 按 95% 合成 / 5% 真实（原文典型配比）
```

### 4.3 适用场景

- 中低维连续控制、真实交互昂贵的任务（MBPO 的原始动机）；
- 已有稳定 model-free 学习器（SAC），想在不改损失的前提下提升样本效率；
- 状态分布会随策略变化而漂移、需要模型持续在线更新的在线学习设定。

## 5. 分析与总结

- MBPO 的定位：**经验放大器**。模型不参与梯度，只把每条真实轨迹「繁殖」成更多覆盖邻近状态转移的样本。
- 与 Dyna 的差异在工程而非思想：概率系综替代表格、短 rollout 替代单步、SAC 替代 Q-learning。
- 与 PETS 的差异是模型的使用方式：同样（几乎同样）的系综模型，PETS 用它在每步搜索动作，MBPO 用它造数据训练显式策略——后者推理快、可与任意 model-free 算法组合。
- 与 Dreamer 的差异是机制层面的：Dreamer 反传经过模型，MBPO 不反传。二者不应混为一谈。
- 开放问题：合成/真实配比、horizon 上限等超参的任务相关性较强；模型在非平稳环境下的持续有效性。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)；[SAC 算法完整解析与代码实现](../02_model-free-rl/07_sac.md)——作为学习器被嵌入
- **本分支地基**：🚧 基于模型的强化学习基础（撰写中）
- **思想渊源**：🚧 Dyna（撰写中）——「模型生成经验」的原始架构
- **相关方法**：🚧 PETS（撰写中）——同款系综模型的规划式用法

### 6.2 参考资料

1. Janner et al., "When to Trust Your Model: Model-Based Policy Optimization"（NeurIPS 2019，MBPO）。正式链接待补充。
2. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"（ICML 2018，SAC）——被嵌入的学习器。正式链接待补充。
3. Chua et al.（NeurIPS 2018，PETS）——系综概率动力学模型的来源。正式链接待补充。
