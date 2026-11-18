# MBPO：短模型展开与策略优化

## 1. 概述

MBPO 从实际经验中的状态开始短时域模型展开，把生成的转移交给离策略强化学习算法。原始方法使用概率系综动力学和 SAC，通过控制展开长度平衡额外数据的收益与模型误差。

## 2. 基础与定义

### 2.1 模型与网络

系统包含概率动力学模型 $p_\phi(s',r\mid s,a)$、随机 Actor $\pi_\theta$、两个动作价值网络 $Q_{\psi_1},Q_{\psi_2}$ 及其目标网络。实际数据保存于 $\mathcal D_{\mathrm{env}}$，模型生成数据保存于 $\mathcal D_{\mathrm{model}}$。

动力学模型由实际转移监督学习；Actor 与 Critic 从两种数据构成的训练分布中采样更新。

### 2.2 展开起点

每条模型轨迹的起点取自实际经验：

$$
\hat s_0\sim\mathcal D_{\mathrm{env}},\qquad
\hat a_t\sim\pi_\theta(\cdot\mid\hat s_t),\qquad
(\hat s_{t+1},\hat r_t)\sim p_\phi(\cdot\mid\hat s_t,\hat a_t).
$$

后续状态由模型预测，展开最多 $k$ 步。遇到真正终止时结束该条轨迹，并在价值目标中关闭自举。

## 3. 核心原理

### 3.1 策略与价值损失

设熵温度为 $\alpha$，转移样本为 $(s,a,r,s',d)$，$a'\sim\pi_\theta(\cdot\mid s')$。SAC 的目标值为：

$$
y=r+\gamma(1-d)\left[\min_jQ_{\bar\psi_j}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right].
$$

$$
L_{Q_i}=\frac12\mathbb E[(Q_{\psi_i}(s,a)-\operatorname{sg}(y))^2],
$$

$$
L_\pi=\mathbb E_{s,\epsilon}\left[\alpha\log\pi_\theta(a_\theta\mid s)
-\min_iQ_{\psi_i}(s,a_\theta)\right],\qquad a_\theta=g_\theta(s,\epsilon).
$$

模型转移作为固定训练样本。Actor 的动作梯度来自 $Q$ 和熵项，动力学模型通过训练数据影响 Critic 的估计。

<details class="kt-derivation" id="mbpo-gradient-path">
<summary>模型数据与 Actor 梯度</summary>

固定采样状态和 Critic 参数时：

$$
\nabla_\theta L_\pi=
\mathbb E\left[\alpha\frac{d\log\pi_\theta(a_\theta\mid s)}{d\theta}
-\left.\nabla_aQ_{\psi_{i^*}}(s,a)\right|_{a=a_\theta}\frac{d a_\theta}{d\theta}\right].
$$

$i^*$ 为当前较小的价值预测分支。该式使用固定缓冲区状态，更新过程中没有重新构造 $s'=f_\phi(s,a_\theta)$ 的动力学路径。模型误差仍通过数据和 TD 目标影响学到的 $Q$。

这与直接优化 $r(s,a)+\gamma V(f_\phi(s,a))$ 的可微路径梯度具有不同的计算图。

</details>

### 3.2 概率系综学习

各模型成员根据实际经验拟合转移与奖励的条件分布：

$$
L_M(\phi_b)=-\mathbb E_{\mathcal D_b}
[\log p_{\phi_b}(s',r\mid s,a)].
$$

$\mathcal D_b$ 是成员使用的训练数据。实际实现可预测状态增量，并依据留出误差选择参与采样的成员。生成转移时从成员预测分布采样，保留模型不确定性。

### 3.3 短展开

从实际状态分支生成短轨迹，使预测集中在经验分布附近。随着策略访问区域和模型精度变化，可按训练进度调整 $k$。原始方法使用预设时域调度，具体长度需要通过任务验证。

<details class="kt-derivation" id="mbpo-horizon-schedule">
<summary>展开长度调度</summary>

给定训练进度 $e$、调度区间 $e_0,e_1$ 与长度上下限 $k_{\min},k_{\max}$，一种线性调度为：

$$
u=\operatorname{clip}\left(\frac{e-e_0}{e_1-e_0},0,1\right),\qquad
k(e)=\left\lfloor k_{\min}+u(k_{\max}-k_{\min})\right\rfloor.
$$

这里 $e_1>e_0$。调度区间之外仍满足长度边界。采用该调度时，还应记录各长度对应的多步验证误差和实际环境回报，判断增加时域是否有效。

</details>

## 4. 实现与应用

### 4.1 训练流程

1. 在实际环境中采样，更新实际经验库。
2. 用实际经验训练和验证概率系综。
3. 采样一批实际状态，用当前策略展开最多 $k$ 步。
4. 将模型转移分离计算图后写入模型缓冲区，按终止标志结束轨迹。
5. 混合采样实际与模型数据，执行 SAC 更新。
6. 定期刷新模型数据，控制旧模型与旧策略生成经验的保留时间。

### 4.2 无人机实验

可从低维状态的姿态或速度控制开始，先检验单步预测和终止语义，再比较 SAC 与 MBPO。报告实际交互量、模型训练时间、策略更新时间和任务成功率，单独统计模型样本数量。

把 PPO 接到同一缓冲区需要重新处理策略分布和数据时效；SAC 的离策略更新机制与这里的数据复用方式更直接匹配。

## 5. 分析与总结

MBPO 将模型作用限制在数据生成阶段，通过短展开控制误差传播。模型内数据增加并不等于有效信息增加；展开长度、混合比例、缓冲区时效和实际环境数据覆盖共同决定训练结果。

## 6. 关联与参考

- 前置：[Dyna](02_dyna.md)、[SAC](../02_model-free-rl/06_sac.md)。
- 对照：[PETS](04_pets.md)、[可微模型与策略梯度](07_differentiable-policy-optimization.md)。
- Janner 等：[When to Trust Your Model: Model-Based Policy Optimization](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html)，NeurIPS 2019。
- [MBPO 官方实现](https://github.com/jannerm/mbpo)。
