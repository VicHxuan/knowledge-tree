# MuZero：潜在动力学学习与模型规划

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

MuZero（Schrittwieser et al.，arXiv 2019 年 11 月首次公开，ICLR 2020 正式发表）把 AlphaGo Zero 一线的「模型 + 搜索」范式推进到**不知道环境规则**的设定：不学习重建观测，而是学习一个**价值等效（value-equivalent）的潜在动力学模型**——只预测与决策相关的量（奖励、策略先验、价值），再用 Monte Carlo Tree Search 在这个隐式模型上规划。规划与学习互相供给：搜索产生比策略网络自身更强的决策与训练信号，网络又让搜索更聚焦、更便宜。

## 2. 基础与定义

### 2.1 表征、动力学与预测网络

| 网络 | 输入 → 输出 | 作用 |
| :--- | :--- | :--- |
| 表征网络 $h$ | 观测 $o_{1:t}$ → 隐状态 $z_t$ | 把（可能是部分可观测的）观测编码为模型内部状态 |
| 动力学网络 $g$ | $(z_{t-1},a_t)$ → $(\hat r_t, z_t)$ | 隐空间的状态转移 + 奖励预测——MuZero 的「环境模型」 |
| 预测网络 $f$ | $z_t$ → $(\mathbf p_t, v_t)$ | 策略先验 $\mathbf p$（动作分布）与价值估计 $v$ |

隐状态 $z_t$ 完全是「为预测价值相关量服务」的内部构造：没有任何约束要求它重建观测。

### 2.2 隐式潜在状态

- $z_t$ 由 $h$ 与 $g$ 增量产生，维度固定、语义开放；
- 与显式/重建型世界模型（PETS 的观测空间、Dreamer 的带解码器潜在空间）相对照：MuZero 放弃「模型要像环境」，只要求「模型对预测 reward/value/policy 有用」；
- 代价：模型所学为何、何处失效不易解释；收益：容量全部花在决策相关的预测上。

### 2.3 与 POMDP 的关系

$h$ 网络消费观测历史 $o_{1:t}$，本质是对 belief state 的参数化压缩——MuZero 在结构上就是「POMDP + 学习的 belief 编码 + 模型规划」。

## 3. 核心原理

### 3.1 潜在空间中的 MCTS

每次决策从根 $z_0=h(o_{1:t})$ 出发，反复模拟：

1. **选择（Selection）**：从根下行，按 PUCT 选子节点——先验 $\mathbf p$ 加价值 $v$ 减访问次数惩罚；
2. **展开（Expansion）**：叶节点用 $g$ 展开：$z' ,\hat r=g(z,a)$；
3. **评估（Evaluation）**：$f(z')$ 给出 $(\mathbf p', v')$——$\mathbf p'$ 写入子节点先验，$v'$ 用于回传；
4. **回传（Backup）**：沿路径累积 $\hat r$ 与终止价值折扣，更新各节点价值统计与访问计数。

树深度即「规划时域」，但每条路径都由网络评价，搜索不需要环境真实响应。

### 3.2 联合训练目标

对批真实轨迹，沿已执行的动作用 $h,g$ 展开 $K$ 步，逐步产生预测三元组 $(\hat r_t,\mathbf{\hat p}_t,\hat v_t)$，与三个真值对齐：

$$\mathcal L=\sum_{t=1}^{K}\Big[\underbrace{\big\|\hat v_t-z_t^{\text{target}}\big\|^2}_{\text{价值}}+\underbrace{\big\|\hat r_t-u_t\big\|^2}_{\text{奖励}}+\underbrace{\mathbf p_t\log \mathbf{\hat p}_t}_{\text{策略先验}}\Big]+\lambda\|\theta\|^2$$

其中策略真值 $\mathbf p_t$ 取**搜索的访问计数分布**、价值真值取搜索的价值估计（自博弈 + n-step/蒙特卡洛混合）——即「蒸馏搜索」：模型被训练成搜索的压缩形式。

### 3.3 搜索与策略改进

- 推理时：搜索产生比 $f$ 的原始先验更强的决策（访问计数选动作）；
- 训练时：搜索结果作为目标回灌网络；
- 网络变强 → 先验更准 → 同样的模拟次数下搜索质量更高 → 训练目标更强。这是 AlphaZero 一脉确立的「policy improvement operator」循环，MuZero 把其中的真环境换成了学到的 $g,f$。

### 3.4 与 PETS、Dreamer 的比较

| 维度 | PETS | MuZero | Dreamer |
| :--- | :--- | :--- | :--- |
| 模型空间 | 观测空间 | 隐式（不重建） | 潜在空间（带重建监督） |
| 规划方式 | CEM 动作序列 | MCTS | 不在线规划 |
| 策略 | 无显式 actor | 网络先验 + 搜索 | 想象中训练的显式 actor |
| 模型用途 | 评估候选动作 | 生成搜索树 + 训练信号 | 想象 rollout 反传策略梯度 |

MuZero 保留了 PETS 的「规划中心」，但把每步优化的成本用网络先验摊薄；它与 Dreamer 同用潜在状态，但策略不在想象中被显式优化，而是搜索与蒸馏交替。

## 4. 实现与应用

### 4.1 训练伪代码

```python
import torch
import torch.nn.functional as F

def muzero_train_step(batch, h, g, f, K=5, gamma=0.997):
    """batch: (o_1:t, a_1..T, r_1..T) 一条真实轨迹"""
    loss = 0.0
    z = h(batch.obs_history)                       # 根隐状态
    for t in range(K):                             # 展开已执行的动作
        z, r_hat = g(z, batch.a[t])                 # 动力学：转移 + 奖励
        p_hat, v_hat = f(z)                        # 预测：策略先验 + 价值
        # 真值来自搜索：p_target=访问计数分布，v_target=搜索值，r_target=真实奖励
        loss += F.mse_loss(v_hat, batch.search_value[t])
        loss += F.mse_loss(r_hat, batch.r[t])
        loss += F.cross_entropy(p_hat, batch.search_policy[t])
    return loss
```

（完整实现还需自博弈/经验回放、搜索目标的 n-step 价值混合与重分析机制；此处只示意联合监督的形状。开源参考实现见 6.2。）

### 4.2 适用场景

- 离散动作、奖励稀疏、需要长时程信用分配的决策任务（棋类、Atari）；
- 环境规则不可知或不可微、但模拟/采样可行的设定；
- 推理预算允许每步有限次模拟的场合（连续动作扩展与推理蒸馏属于后续工作，待补充）。

## 5. 分析与总结

- MuZero 的核心洞见：**决策导向的模型**。模型的好坏以「能否支撑搜索与价值预测」衡量，而非「是否还原环境」——价值等效模型是 Model-Based RL 中与「重建型世界模型」并列的另一极。
- 优点：无需环境规则、无需可观测性假设的完整重建；规划与学习的正反馈在实践里极强。
- 局限：搜索带来可观的训练/推理计算；原始形态面向离散动作与确定回合制奖励结构；隐式模型的可解释性与安全性分析困难。
- 在分支内的位置：与 PETS 同属「规划式」，与 Dreamer 同属「潜在模型」——它同时站在两条线索的交点上。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——价值函数与策略改进
- **本分支地基**：🚧 基于模型的强化学习基础（撰写中）——潜在状态空间与想象 rollout
- **相关方法**：🚧 PETS（撰写中）——规划式的观测空间对照；🚧 Dreamer（撰写中）——潜在模型 + 显式 actor 的对照

### 6.2 参考资料

1. Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"（arXiv:1911.08265，2019；ICLR 2020）——MuZero 原始论文：[arxiv.org/abs/1911.08265](https://arxiv.org/abs/1911.08265)
2. MuZero 官方开源实现：[github.com/google-deepmind/muzero](https://github.com/google-deepmind/muzero)
3. Silver et al., "A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play"（Science 2018，AlphaZero）——搜索与网络共生的前身。正式链接待补充。
