# Model-Based RL 基础

## 1. 概述

Model-Based RL 使用环境模型辅助决策与学习。模型的主要用途包括生成训练经验、评价候选动作序列，以及传播策略梯度。不同方法的关键区别在于：模型预测什么、模型如何参与优化，以及哪些梯度经过模型。

## 2. 基础与定义

### 2.1 环境与模型

设状态为 $s_t$，动作为 $a_t$，转移分布为 $p(s_{t+1}\mid s_t,a_t)$，奖励为 $r_t$，折扣因子为 $\gamma$。策略优化目标为：

$$
J(\theta)=\mathbb E_{\pi_\theta,p}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

模型 $p_\phi$ 近似环境转移，并在需要时预测奖励与终止。模型可以来自已知物理方程、交互数据学习，或“物理模型 + 学习残差”。实际环境可以是物理系统，也可以是提供交互数据的高保真仿真器。

### 2.2 状态表示

| 表示 | 预测对象 | 典型方法 |
|---|---|---|
| 物理状态 | 位置、速度、姿态等可解释变量 | PETS、MBPO、SuperMimic |
| 潜在状态 | 从观测或历史编码的任务相关变量 | TD-MPC、MuZero、Dreamer |

世界模型描述用于预测环境演化的模型系统，可以使用物理状态或潜在状态。部分可观测任务需要把足够的历史信息纳入状态表示，以支撑未来预测。

## 3. 核心原理

### 3.1 模型用途与梯度

| 路线 | 模型的作用 | 决策或学习目标 | 策略梯度经过动力学 |
|---|---|---|---|
| 生成经验 | 产生额外转移样本 | TD 学习或离策略 Actor-Critic | Dyna、MBPO 的常规更新不经过 |
| 在线规划 | 评价未来动作序列 | 模型内累计回报与末端价值 | PETS、TD-MPC 主要采用采样优化 |
| 模型内策略学习 | 展开当前策略 | 想象轨迹的回报 | PILCO、SVG、SHAC、SuperMimic 使用 |

Dreamer 在潜在模型中学习行为，具体梯度随版本变化。MuZero 用搜索结果训练策略与价值预测，其网络训练梯度与搜索决策过程分别处理。

模型可微性、状态表示和模型用途是三个独立的分类维度。可微神经模型也可以只用于生成样本。

### 3.2 模型学习

从实际环境收集经验 $\mathcal D=\{s_t,a_t,r_t,s_{t+1},d_t\}$。概率转移模型通过负对数似然学习：

$$
L_M(\phi)=-\mathbb E_{\mathcal D}\left[\log p_\phi(s_{t+1}\mid s_t,a_t)\right].
$$

对于对角高斯模型，预测均值 $\mu_\phi$ 与协方差 $\Sigma_\phi$：

$$
L_M=\frac12\mathbb E\left[
(s'-\mu_\phi)^\top\Sigma_\phi^{-1}(s'-\mu_\phi)+\log\det\Sigma_\phi\right]+C.
$$

残差模型可预测 $\Delta s=s'-s$；姿态等流形状态需采用相应的更新与误差表示。奖励可由任务定义直接计算，也可增加奖励预测损失。

<details class="kt-derivation" id="model-ensemble-uncertainty">
<summary>概率系综的不确定性</summary>

设 $B$ 个模型输出 $\mu_b,\Sigma_b$，以均匀混合表示预测分布。总协方差满足：

$$
\operatorname{Cov}(s')=\frac1B\sum_b\Sigma_b+
\frac1B\sum_b(\mu_b-\bar\mu)(\mu_b-\bar\mu)^\top,
\qquad \bar\mu=\frac1B\sum_b\mu_b.
$$

第一项表示各成员预测的条件噪声，第二项表示成员之间的分歧。不同初始化和自助采样使成员产生差异。分歧是否反映真实预测风险，需要通过留出数据上的覆盖率与误差检验。

单个高斯输出表示一个条件高斯分布；多个成员的混合可以表达更复杂的预测形状。

</details>

### 3.3 模型使用时域

用模型展开 $H$ 步时，误差随状态分布改变而累积。短展开减少预测链长度，末端价值补充更长时间的回报：

$$
\hat J_H=\mathbb E_{p_\phi,\pi_\theta}\left[
\sum_{t=0}^{H-1}\gamma^t r_t+\gamma^H V_\psi(s_H)\right].
$$

增加 $H$ 会改变模型误差与价值误差的相对影响。适当时域取决于模型精度、任务时间尺度和规划或训练预算。

<details class="kt-derivation" id="model-rollout-error">
<summary>多步预测误差</summary>

考虑固定动作序列，真实映射为 $f$，模型为 $\hat f$。若相关状态区域内单步误差至多 $\varepsilon$，且 $f$ 对状态的 Lipschitz 常数为 $L$，则：
$$
e_{t+1}=\|\hat s_{t+1}-s_{t+1}\|\leq\varepsilon+Le_t.
$$

相同起点下：

$$
e_t\leq\varepsilon\sum_{j=0}^{t-1}L^j.
$$

闭环预测中，动作也随预测状态变化，此时应分析策略与动力学组合后的映射。轨迹误差与梯度误差是不同的量，可微策略优化还需要检查局部雅可比。

</details>

## 4. 实现与应用

### 4.1 学习闭环

1. 用当前策略或规划器访问实际环境，更新经验库。
2. 用训练数据拟合模型，用独立轨迹检验单步和多步预测。
3. 依据路线，用模型生成样本、在线规划或更新策略。
4. 在实际环境中评价改进，并补充新的状态与动作区域。

训练与验证按轨迹或实验批次划分，避免相邻帧泄漏导致过度乐观的预测指标。

### 4.2 无人机应用

| 需求 | 优先研究的机制 | 主要验证对象 |
|---|---|---|
| 数据昂贵，希望复用交互 | MBPO 的短模型展开 | 合成数据对实际回报的影响 |
| 在线修正动作序列 | PETS、TD-MPC | 规划时延、模型误差与约束 |
| 利用动力学导数训练控制器 | SHAC、SuperMimic | 雅可比、末端价值梯度与闭环稳定性 |
| 从图像历史建立预测状态 | Dreamer | 潜在预测、观测编码与任务泛化 |

无人机模型状态应覆盖影响后续转移的执行器与环境因素。观测延迟、推力响应和风场状态是否进入模型，直接影响预测闭环。

## 5. 分析与总结

模型质量由任务中的用途决定：生成经验关注转移分布与样本偏差，规划关注候选动作的排序，可微策略学习还关注导数。实际环境回报与同预算对照实验是检验模型收益的共同标准。

## 6. 关联与参考

- 生成经验：[Dyna](02_dyna.md) → [MBPO](03_mbpo.md)。
- 在线规划：[PETS](04_pets.md) → [TD-MPC](05_td-mpc.md)，补充 [MuZero](06_muzero.md)。
- 策略梯度：[可微模型与策略梯度](07_differentiable-policy-optimization.md) → [SHAC](08_shac.md) → [SuperMimic](10_supermimic.md)。
- 潜在模型：[Dreamer](09_dreamer.md)。
- Chua 等：[Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)，NeurIPS 2018。
- Janner 等：[When to Trust Your Model: Model-Based Policy Optimization](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html)，NeurIPS 2019。
