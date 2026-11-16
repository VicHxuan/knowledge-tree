# 多价值网络与价值估计

## 1. 概述

Multi-Critic 用多个价值估计器为策略学习提供信号。设计首先取决于各估计器的对象：同一回报的多个估计、不同奖励分量的回报，或不同任务与约束的价值。本文以奖励分解下的 PPO 为主线，从网络与损失出发，分析优势合成、标准化和梯度更新。

## 2. 基础与定义

### 2.1 估计对象

| 结构 | 各 Critic 的估计对象 | 组合位置 |
| --- | --- | --- |
| 同目标双估计器或集成 | 同一策略或控制目标的价值 | 自举目标、价值或优势 |
| 奖励分量 Critic | 同一策略下不同奖励分量的回报 | 分量优势与 Actor 目标 |
| 任务或约束 Critic | 不同任务的回报，或奖励与成本 | 任务损失、约束与乘子 |

TD3 的双 Q 对同一目标提供两个估计，以较小的目标值抑制过估计。奖励分解则为每个奖励分量学习价值；HRA 是这一思路的代表性工作。两者的组合规则分别由其优化目标决定。[TD3](https://proceedings.mlr.press/v80/fujimoto18a.html)、[HRA](https://proceedings.neurips.cc/paper_files/paper/2017/hash/1264a061d82a2edae1574b07249800d6-Abstract.html)

### 2.2 网络与奖励分量

设一个 Actor 为 $\pi_\theta(a\mid s)$，$K$ 个 Critic 为 $V_{\phi_k}^{(k)}(s)$。先采用参数独立的网络，每个 Critic 拟合同一策略下一个奖励分量的折扣回报：

$$
V_k^\pi(s)=\mathbb E_\pi\left[\sum_{l=0}^{\infty}\gamma^l r_{t+l+1}^{(k)}\mid s_t=s\right].
$$

给定本轮固定的非负权重 $w_k$，总奖励与价值为：

$$
r_{t+1}=\sum_{k=1}^K w_k r_{t+1}^{(k)},\qquad
V^\pi(s)=\sum_{k=1}^K w_k V_k^\pi(s).
$$

该关系使用同一策略、折扣和回合定义。奖励分组应记录分量包含的基础奖励项及其系数。独立网络与共享特征多头是两种参数组织方式，分组与网络结构可分别选择。

## 3. 核心原理

### 3.1 优化目标与梯度

令 $S_t$ 为合成后的标量优势，$\hat R_{t,k}$ 为各 Critic 的固定回报目标。PPO 总损失为：

$$
L=L_\pi(\theta;S)+\sum_{k=1}^K\beta_k L_{V,k}(\phi_k)-c_e\bar{\mathcal H}(\theta),
\qquad
L_{V,k}=\frac12\hat{\mathbb E}_t\left[(V_{\phi_k}^{(k)}(s_t)-\hat R_{t,k})^2\right].
$$

$\beta_k$ 控制各价值回归损失的权重，$c_e$ 控制熵奖励。Actor 权重 $w_k$ 决定优势组合；$\beta_k$ 决定 Critic 拟合力度。

本轮的优势、回报目标和旧策略概率停止梯度，独立参数的梯度为：

$$
\nabla_\theta L=\nabla_\theta L_\pi-c_e\nabla_\theta\bar{\mathcal H},
\qquad
\nabla_{\phi_k}L
=\beta_k\hat{\mathbb E}_t\left[(V_{\phi_k}^{(k)}-\hat R_{t,k})\nabla_{\phi_k}V_{\phi_k}^{(k)}\right].
$$

Critic 通过后续轮次的价值预测影响优势估计；当前轮次的 Actor 梯度由固定的 $S_t$ 加权。

### 3.2 分量优势与回报目标

记采样时的旧价值为 $v_{t,k}$，实际后继状态的旧价值为 $v_{t+1,k}^{\rm next}$。$b_t$ 在真正终止时为零，其余情况为一；$c_t$ 在终止或外部截断时为零，其余情况为一。各分量分别计算：

$$
\delta_{t,k}=r_{t+1}^{(k)}+\gamma b_t v_{t+1,k}^{\rm next}-v_{t,k},
$$

$$
\hat A_{t,k}=\delta_{t,k}+\gamma\lambda c_t\hat A_{t+1,k},
\qquad
\hat A_{T,k}=0,\qquad
\hat R_{t,k}=\hat A_{t,k}+v_{t,k}.
$$

$T$ 为采样长度。外部截断使用重置前末状态的价值；残差累加截至当前回合边界。各 Critic 的回报目标先由原始优势构造，Actor 所需的标准化在此之后进行。

<details class="kt-derivation" id="multi-critic-gae-linearity">
<summary>分量 GAE 的线性关系</summary>

设总价值估计由分量估计定义为 $v_t^\Sigma=\sum_k w_kv_{t,k}$，且各分量使用相同的 $\gamma,\lambda,b_t,c_t$。代入总奖励的 TD 残差：

$$
\delta_t^\Sigma
=\sum_k w_kr_{t+1}^{(k)}+\gamma b_t\sum_k w_kv_{t+1,k}^{\rm next}-\sum_k w_kv_{t,k}
=\sum_k w_k\delta_{t,k}.
$$

设从 $t$ 到当前片段末端共有 $H$ 步，则：

$$
\hat A_t^\Sigma
=\sum_{l=0}^{H-1}(\gamma\lambda)^l\delta_{t+l}^\Sigma
=\sum_k w_k\sum_{l=0}^{H-1}(\gamma\lambda)^l\delta_{t+l,k}
=\sum_k w_k\hat A_{t,k}.
$$

因此 $\hat R_t^\Sigma=\sum_kw_k\hat R_{t,k}$。这是给定同一组分量价值预测时的代数恒等式。另行训练的单 Critic 有自身的函数近似误差，其输出与分量价值之和可以不同。

分量采用不同折扣、不同 $\lambda$ 或不同终止语义时，应分别定义估计目标，再解释组合后的优化含义。

</details>

### 3.3 优势合成与标准化

对固定采样集合 $\mathcal B$，定义标准化算子：

$$
\mathcal N(x_t)=\frac{x_t-\mu_x}{\sigma_x+\varepsilon},
\qquad \varepsilon>0.
$$

$\mu_x,\sigma_x$ 为该集合上的均值和标准差。本轮统计量保持固定。两种常见处理为：

$$
S_t^{\rm raw}=\mathcal N\left(\sum_k w_k\hat A_{t,k}\right),
$$

$$
S_t^{\rm group}=\mathcal N\left(\sum_k w_k\mathcal N(\hat A_{t,k})\right).
$$

| 处理 | Actor 使用的相对尺度 | 权重含义 |
| --- | --- | --- |
| 原始优势合成后标准化 | 保留分量的原始回报尺度，再统一缩放 | 原始回报单位下的相对权重 |
| 分组标准化后合成 | 每组先按自身标准差缩放 | 标准化信号之间的相对权重 |

分组标准化会使小尺度分量获得更大的相对系数。改变分组、组内方差或相关性，都可能改变合成优势的正负及大小，进而改变 PPO 的样本梯度。

<details class="kt-derivation" id="multi-critic-effective-weights">
<summary>有效权重与分组变化</summary>

令 $\mu_k,\sigma_k$ 为分量优势统计量。组内标准化后的加权和为：

$$
B_t=\sum_k\frac{w_k}{\sigma_k+\varepsilon}(\hat A_{t,k}-\mu_k).
$$

忽略数值稳定项并固定统计量，有效相对权重为 $w_k/\sigma_k$。最终标准化再施加共同尺度 $1/\sigma_B$。

若两个分量在某样本上的中心化优势为 $2$ 与 $-0.1$，标准差分别为 $10$ 与 $0.1$，且 $w_1=w_2=1$，则原始合成值为 $1.9$，分组标准化后为 $0.2-1=-0.8$。最终共同的正尺度保留两者各自的符号，Actor 得到相反的动作概率调整方向。

合组时的方差包含协方差：

$$
\operatorname{Var}(A_1+A_2)
=\sigma_1^2+\sigma_2^2+2\operatorname{Cov}(A_1,A_2).
$$

因此，将一个奖励组拆成两个独立标准化的组，会同时改变估计器和组合规则。各组归一化后再整体归一化，也会受到组间相关性的影响；若组合方差很小，数值稳定项或显式方差下限会决定实际缩放。

对于正比例缩放 $a>0$，在预测与目标同步缩放、忽略 $\varepsilon$ 时，$\mathcal N(aA)=\mathcal N(A)$。这解释了组内奖励整体缩放为何可能被优势标准化抵消。实际 Critic 拟合误差、价值损失尺度和优化轨迹仍会随缩放变化。

有限批次的均值与方差来自同一批样本，中心化还可能改变样本符号。应把这类标准化视为训练估计器的变换，而非任意情况下保持原始策略梯度完全不变的恒等式。

</details>

### 3.4 PPO 截断与策略梯度

定义单样本最大化目标：

$$
F(\rho,A)=\min\left(\rho A,\operatorname{clip}(\rho,1-\epsilon_\pi,1+\epsilon_\pi)A\right),
\qquad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm old}(a_t\mid s_t)}.
$$

先合成优势的 PPO 损失为：

$$
L_\pi=-\hat{\mathbb E}_t[F(\rho_t,S_t)].
$$

在截断拐点之外，其单样本梯度为：

$$
\nabla_\theta L_{\pi,t}
=-m_t S_t\rho_t\nabla_\theta\log\pi_\theta(a_t\mid s_t),
$$

其中 $m_t=0$ 对应 $S_t>0,\rho_t>1+\epsilon_\pi$ 或 $S_t<0,\rho_t<1-\epsilon_\pi$，其余为一。优势合成既决定梯度系数，也决定样本进入哪一侧的平坦区。

分别截断各分量再加权，则得到另一个目标 $-\hat{\mathbb E}_t[\sum_k w_kF(\rho_t,A_{t,k})]$；不同分量可以有不同的截断状态。

<details class="kt-derivation" id="multi-critic-clip-order">
<summary>截断顺序与梯度差异</summary>

先固定各分量优势并暂不标准化。记：

$$
F_{\rm joint}=F\left(\rho,\sum_k w_k A_k\right),\qquad
F_{\rm separate}=\sum_k w_k F(\rho,A_k).
$$

取 $A_1=2,A_2=-1,w_1=w_2=1,\epsilon_\pi=0.2,\rho=1.3$：

$$
F_{\rm joint}=F(1.3,1)=1.2,\qquad
F_{\rm separate}=2.4-1.3=1.1.
$$

此时：

$$
\frac{\partial F_{\rm joint}}{\partial\rho}=0,\qquad
\frac{\partial F_{\rm separate}}{\partial\rho}=-1.
$$

合成优势已进入正优势平坦区；分别截断时，负优势分量仍提供降低概率的梯度。两种目标在区间内部都等于 $\rho\sum_kw_kA_k$，在分量符号冲突且发生截断时通常产生差异。

</details>

### 3.5 价值标准化与梯度耦合

各 Critic 可使用独立的回报统计量，将原单位目标 $\hat R_{t,k}$ 转换为 $\widetilde R_{t,k}$：

$$
\widetilde R_{t,k}=\frac{\hat R_{t,k}-\mu_{R,k}}{\sigma_{R,k}+\varepsilon}.
$$

GAE 使用还原到奖励单位的价值预测；价值回归在同一套固定统计量下比较预测与目标。Actor 优势标准化和 Critic 回报标准化分别作用于两种损失。

<details class="kt-derivation" id="multi-critic-value-scaling">
<summary>价值回归尺度与截断</summary>

令 $s_k=\sigma_{R,k}+\varepsilon$。若用原单位预测 $V_k$ 表示标准化回归损失，则：

$$
L_{V,k}=\frac{1}{2s_k^2}\hat{\mathbb E}_t[(V_k-\hat R_k)^2],
\qquad
\nabla_{\phi_k}L_{V,k}
=\frac{1}{s_k^2}\hat{\mathbb E}_t[(V_k-\hat R_k)\nabla_{\phi_k}V_k].
$$

原单位表示下，各组回归误差得到不同尺度的权重。若网络直接输出标准化价值 $z_k$，实际参数梯度为 $(z_k-\widetilde R_k)\nabla z_k$；网络参数化与优化器状态也参与决定更新，不能只用一个常数概括两种训练轨迹。

在标准化空间使用价值截断阈值 $\epsilon_v$，对应原单位变化阈值 $s_k\epsilon_v$。旧预测、当前预测和目标必须用同一组统计量转换。本文的分析将统计量视为每轮固定；跨轮更新统计量时还需区分普通归一化与保持未归一化输出的 PopArt 变换。[PopArt](https://arxiv.org/abs/1602.07714)

</details>

<details class="kt-derivation" id="multi-critic-gradient-coupling">
<summary>共享特征与梯度裁剪</summary>

若各 Critic 共享特征 $z=f_\psi(s)$，共享参数接收：

$$
\nabla_\psi L=\sum_k\beta_k\nabla_\psi L_{V,k}.
$$

若 Actor 也使用该特征，再加上 $\nabla_\psi L_\pi-c_e\nabla_\psi\bar{\mathcal H}$。两个任务梯度的内积为负时，它们在共享参数空间中存在局部方向冲突。

独立参数仍可通过联合梯度范数裁剪发生数值耦合。令 $g_\theta$ 与 $g_k$ 为各网络梯度，联合裁剪使用：

$$
q=\min\left(1,\frac{g_{\max}}{\sqrt{\|g_\theta\|^2+\sum_k\|g_k\|^2}}\right).
$$

裁剪后 Actor 梯度为 $qg_\theta$，某个 Critic 的大梯度会影响这一尺度。分别裁剪时，各网络使用自身的缩放系数。

价值损失在 Critic 维度取平均时，隐含 $1/K$ 因子；取和时保留各分量尺度。改变组数需要同时核对这个归约方式。组的 Actor 权重为零时，该组仍可训练 Critic；共享特征或联合裁剪使其仍可能影响其他网络。

</details>

## 4. 实现与应用

### 4.1 训练流程

1. 固定采样策略与各价值估计器，记录奖励分量、旧价值、旧动作概率和边界。
2. 在每个环境的时间线上分别计算分量 GAE 与原单位回报目标。
3. 选择原始合成或分组标准化，得到一个固定的 Actor 优势。
4. 构造各 Critic 的回归目标，固定归一化统计量。
5. 小批量重新评估旧动作与当前价值，计算策略损失和各价值损失。
6. 按参数共享关系与裁剪范围执行更新，然后重新采样。

### 4.2 优势合成示例

本地参考实现使用 `raw_sum` 与 `per_group` 命名两种模式。下例仅表达通用的优势变换：输入最后一维为 Critic 维，其余维度组成同一统计集合。

<details class="kt-derivation" id="multi-critic-composition-code">
<summary>优势标准化与合成</summary>

```python
import torch

@torch.no_grad()
def compose_advantages(raw_advantages, weights, mode):
    axes = tuple(range(raw_advantages.ndim - 1))

    def normalize(x):
        count = x.numel() // x.shape[-1]
        std = x.std(dim=axes, keepdim=True, correction=1 if count > 1 else 0)
        return (x - x.mean(dim=axes, keepdim=True)) / (std + 1e-8)

    if mode == "raw_sum":
        components = raw_advantages
    elif mode == "per_group":
        components = normalize(raw_advantages)
    else:
        raise ValueError(mode)
    combined = (components * weights).sum(dim=-1, keepdim=True)
    return normalize(combined)
```

回报目标在调用此函数前由原始优势构造。带 padding 的序列只在有效样本上统计；分布式训练应明确采用全局统计还是各工作进程的局部统计。标准差的自由度、统计范围和数值下限都属于算法配置。

</details>

### 4.3 对照分析

| 对照 | 保持一致 | 重点观察 |
| --- | --- | --- |
| 单 Critic 与分量 Critic | 总奖励、原始合成规则、训练预算 | 价值近似误差与策略表现 |
| 原始合成与分组标准化 | 网络结构、奖励和采样数据 | 优势符号、相对权重与策略梯度 |
| 奖励组拆分与合并 | 基础奖励项及系数 | 分组方差、相关性、回归与合成变化 |
| 共享与独立特征 | 估计目标、Actor 目标 | 梯度内积与表示干扰 |
| 联合与独立梯度裁剪 | 损失及未裁剪梯度 | 各网络缩放系数 |

对照需分别记录每组的价值误差、原始优势均值与标准差、组合优势方差、PPO 截断比例与实际梯度范数。各组优势的方差或绝对值只是信号尺度，实际梯度还取决于概率比率、截断状态和策略的对数概率梯度。

## 5. 分析与总结

### 5.1 机制比较

| 改变 | 直接影响 | PPO 中的梯度后果 |
| --- | --- | --- |
| 增加同目标估计器 | 价值估计与误差相关性 | 经优势估计影响 Actor |
| 分解奖励 | 每个价值回归问题 | 改变近似误差与共享特征训练 |
| 分组标准化 | 优势的有效相对权重 | 改变样本系数及截断分支 |
| 分别截断分量 | Actor 目标的非线性组合 | 分量分别进入平坦区 |
| 回报标准化 | 价值回归与截断尺度 | 改变 Critic 梯度 |
| 参数共享 | 多损失的共同参数 | 梯度相加并可能冲突 |

<details class="kt-derivation" id="multi-critic-other-objectives">
<summary>同目标集成与约束价值</summary>

对同一回报的多个价值估计器，可先平均价值再计算 GAE。在相同奖励、折扣、边界和 $\lambda$ 下，GAE 对价值预测是仿射的，因此平均价值得到的 GAE 等于各估计器 GAE 的平均。取最小值则引入非线性选择，改变 TD 残差和估计偏差，使用时需重新定义其目的。

TD3 在目标 Q 上取最小值，而 PPO 使用状态价值构造优势。这两种位置的改变分别作用于自举回归目标和策略梯度信号，应根据各自的损失推导。

若任务要求最大化奖励并限制期望成本，可使用奖励 Critic 与成本 Critic。在拉格朗日形式下，给定乘子 $\eta\geq0$，Actor 的局部信号可由 $A^r-\eta A^c$ 构造；乘子更新对应成本约束，奖励和成本的量纲及阈值需共同定义。对成本优势单独标准化会改变其有效系数，因此需与乘子所对应的约束尺度一致。[CPO](https://proceedings.mlr.press/v70/achiam17a.html)

</details>

### 5.2 核心结论

Multi-Critic 的影响由估计对象、目标构造、优势合成和参数更新共同决定。解释某次变化时，先写出 Actor 最终使用的标量优势，再写出各 Critic 的回归损失，最后沿参数依赖分析梯度。

## 6. 关联与参考

### 6.1 相关知识

- [强化学习核心公式手册](./01_reinforcement-learning-foundations.md)：价值函数、基线和策略梯度。
- [PPO](../02_model-free-rl/03_ppo.md)：单 Critic、GAE 与截断目标。
- [TD3](../02_model-free-rl/05_td3.md) 与 [SAC](../02_model-free-rl/06_sac.md)：双 Q 的目标构造和动作梯度。
- [POMDP 与循环策略](./04_pomdp-recurrent-policies.md)：历史信息与序列估计。

### 6.2 参考文献

1. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017。
2. Schulman et al., [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015。
3. Fujimoto et al., [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html), 2018。
4. van Seijen et al., [Hybrid Reward Architecture](https://proceedings.neurips.cc/paper_files/paper/2017/hash/1264a061d82a2edae1574b07249800d6-Abstract.html), 2017。
5. van Hasselt et al., [Learning Values across Many Orders of Magnitude](https://arxiv.org/abs/1602.07714), 2016。
6. Achiam et al., [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html), 2017。

分组标准化、截断顺序与梯度耦合部分由本文给定的损失定义直接推导；`per_group` 与 `raw_sum` 是实现模式名称。
