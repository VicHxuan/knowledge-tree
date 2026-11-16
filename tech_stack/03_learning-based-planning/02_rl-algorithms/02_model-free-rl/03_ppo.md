# PPO 算法完整解析与代码实现

## 1. 概述

PPO 采用 Actor-Critic 架构。本文用两个独立网络分别表示策略与状态价值：Actor 通过 PPO-Clip 损失与熵奖励学习，Critic 通过均方误差损失学习。两种损失所需的优势估计与回报目标，由轨迹奖励和旧价值预测通过 GAE 构造。

## 2. 基础与定义

### 2.1 网络结构

| 网络 | 参数 | 输出 | 作用 |
| --- | --- | --- | --- |
| Actor | $\theta$ | 动作分布 $\pi_\theta(\cdot\mid s)$ | 根据状态选择动作 |
| Critic | $\phi$ | 状态价值 $V_\phi(s)$ | 估计策略的期望折扣回报 |

Actor 从动作分布中采样：

$$
a_t\sim\pi_\theta(\cdot\mid s_t).
$$

Critic 近似策略的状态价值：

$$
V_\phi(s_t)\approx V^\pi(s_t)
=\mathbb E_\pi\!\left[
\sum_{l=0}^{\infty}\gamma^l r_{t+l+1}\mid s_t
\right].
$$

$\gamma$ 为折扣因子；有限回合的回报求和截至终止时刻。

### 2.2 符号约定

一条状态转移记为 $(s_t,a_t,r_{t+1},s_{t+1})$：在状态 $s_t$ 执行动作 $a_t$，获得奖励 $r_{t+1}$，到达 $s_{t+1}$。

每轮训练先固定网络参数采样，再用这批数据更新网络。采样参数记为 $\theta_{\rm old},\phi_{\rm old}$，更新中的参数记为 $\theta,\phi$。$\hat{\mathbb E}_t$ 表示对批次样本取算术平均，帽子表示采样估计。

## 3. 核心原理

### 3.1 优化目标与梯度

两个网络最小化总损失：

$$
L_{\rm total}(\theta,\phi)
=L_\pi(\theta)+c_vL_V(\phi)-c_e\bar{\mathcal H}(\theta).
$$

$L_\pi$ 是策略损失，$L_V$ 是价值损失，$\bar{\mathcal H}$ 是平均策略熵；$c_v>0$ 与 $c_e\geq0$ 分别控制价值损失和熵奖励的权重。

本轮采样数据、旧策略概率、优势估计和回报目标作为固定训练数据。两套独立参数接收的梯度分别为：

$$
\nabla_\theta L_{\rm total}
=\nabla_\theta L_\pi-c_e\nabla_\theta\bar{\mathcal H},
\qquad
\nabla_\phi L_{\rm total}=c_v\nabla_\phi L_V.
$$

策略损失与熵奖励更新 Actor，价值损失更新 Critic。

### 3.2 策略损失

优势估计 $\hat A_t$ 衡量动作 $a_t$ 相对于状态 $s_t$ 下策略平均表现的好坏。PPO-Clip 使用以下策略损失：

$$
L_\pi(\theta)
=-\hat{\mathbb E}_t\left[
\min\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon_\pi,1+\epsilon_\pi)\hat A_t
\right)\right].
$$

其中 $\epsilon_\pi>0$ 为截断系数，$\rho_t$ 为当前策略与采样策略对同一动作的概率比率；连续动作使用概率密度：

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\rm old}}(a_t\mid s_t)}.
$$

- $\hat A_t>0$ 时，目标鼓励提高动作概率；比率达到 $1+\epsilon_\pi$ 后进入平坦区。
- $\hat A_t<0$ 时，目标鼓励降低动作概率；比率降至 $1-\epsilon_\pi$ 后进入平坦区。

截断使单样本目标在有利方向上饱和，从而抑制过大的策略更新激励。

<details class="kt-derivation" id="ppo-importance-sampling">
<summary>重要性采样与替代目标</summary>

令 $J(\theta)$ 为期望折扣回报。优势形式的策略梯度将动作概率的调整与真实优势联系起来：

$$
\nabla_\theta J(\theta)
\propto\mathbb E_{s_t,a_t\sim\pi_\theta}
\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)A^{\pi_\theta}(s_t,a_t)\right].
$$

这里的状态按策略的折扣访问分布取样，动作按该状态下的策略分布取样。PPO 更新使用旧策略收集的数据，因此需要构造基于旧数据的替代目标。

给定同一个状态 $s$，若新策略需要的动作均在旧策略的支撑集内，则对可积函数 $f$：

$$
\mathbb E_{a\sim\pi_{\rm old}(\cdot\mid s)}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)}f(a)\right]
=\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[f(a)].
$$

离散动作下，直接按期望定义展开：

$$
\begin{aligned}
\sum_a\pi_{\rm old}(a\mid s)
\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)}f(a)
&=\sum_a\pi_\theta(a\mid s)f(a).
\end{aligned}
$$

连续动作下把求和换成积分：

$$
\int\pi_{\rm old}(a\mid s)
\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)}f(a)\,\mathrm da
=\int\pi_\theta(a\mid s)f(a)\,\mathrm da.
$$

PPO 对旧策略采样的状态求平均，并使用旧策略优势的样本估计，得到局部替代目标。

未截断的概率比率目标与基础经验策略梯度目标分别为：

$$
L^{\rm CPI}(\theta)=\hat{\mathbb E}_t[\rho_t(\theta)\hat A_t],
$$

$$
L^{\rm PG}(\theta)=\hat{\mathbb E}_t[\log\pi_\theta(a_t\mid s_t)\hat A_t].
$$

由于 $\nabla_\theta\rho_t=\rho_t\nabla_\theta\log\pi_\theta$，且本轮固定 $\hat A_t$，有

$$
\left.\nabla_\theta L^{\rm CPI}\right|_{\theta_{\rm old}}
=\hat{\mathbb E}_t[\nabla_\theta\log\pi_{\theta_{\rm old}}(a_t\mid s_t)\hat A_t]
=\left.\nabla_\theta L^{\rm PG}\right|_{\theta_{\rm old}}.
$$

替代目标与经验策略梯度目标在旧策略处具有相同的一阶梯度。

</details>

<details class="kt-derivation" id="ppo-clip-cases">
<summary>截断目标的分段形式</summary>

单条样本的最大化目标可化简为

$$
L_t^{\rm CLIP}
=\begin{cases}
\hat A_t\min(\rho_t,1+\epsilon_\pi), & \hat A_t\geq0,\\
\hat A_t\max(\rho_t,1-\epsilon_\pi), & \hat A_t<0.
\end{cases}
$$

取 $\epsilon_\pi=0.2$：

| 优势 | ratio | 未截断项 | 截断项 | 取 min 后的目标 | 此样本是否进入平坦区 |
| --- | --- | --- | --- | --- | --- |
| $+2$ | 1.3 | 2.6 | 2.4 | 2.4 | 是：好动作继续增加的激励消失 |
| $-2$ | 0.7 | −1.4 | −1.6 | −1.6 | 是：差动作继续减少的激励消失 |
| $+2$ | 0.7 | 1.4 | 1.6 | 1.4 | 否：好动作概率降低，保留纠正梯度 |
| $-2$ | 1.3 | −2.6 | −2.4 | −2.6 | 否：差动作概率提高，保留纠正梯度 |

策略损失取上述最大化目标的负值。四条样本的比率都位于截断区间外，其中两条进入目标的平坦区。

</details>

熵奖励调节动作分布的随机性，其批次平均值为：

$$
\bar{\mathcal H}(\theta)
=\hat{\mathbb E}_t\left[
-\mathbb E_{a\sim\pi_\theta(\cdot\mid s_t)}
\log\pi_\theta(a\mid s_t)
\right].
$$

离散动作使用熵，连续动作使用微分熵。总损失中的负号使熵增大有利于损失下降。

### 3.3 价值损失

Critic 通过均方误差拟合回报目标 $\hat R_t$：

$$
L_V(\phi)
=\frac12\hat{\mathbb E}_t
\left[(V_\phi(s_t)-\hat R_t)^2\right].
$$

当前预测 $V_\phi(s_t)$ 随参数更新，目标 $\hat R_t$ 在本轮更新中保持固定。

<details class="kt-derivation" id="ppo-value-gradient">
<summary>价值损失的梯度</summary>

对当前预测应用链式法则：

$$
\nabla_\phi L_V
=\hat{\mathbb E}_t\left[
(V_\phi(s_t)-\hat R_t)\nabla_\phi V_\phi(s_t)
\right].
$$

预测误差决定梯度的方向与大小，梯度通过 $V_\phi$ 进入 Critic。

</details>

策略损失需要 $\hat A_t$，价值损失需要 $\hat R_t$。二者由同一批轨迹共同构造。

### 3.4 优势估计与价值目标

动作价值 $Q^\pi(s,a)$ 表示从状态 $s$ 执行动作 $a$，随后遵循策略 $\pi$ 的期望折扣回报。优势定义为：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

用采样奖励估计动作回报，再减去状态价值，即可构造优势估计。简记旧 Critic 为 $V=V_{\phi_{\rm old}}$，同一回合内的 $n$ 步回报与优势估计定义为：

$$
\hat R_t^{(n)}
\equiv\sum_{l=0}^{n-1}\gamma^lr_{t+l+1}+\gamma^nV(s_{t+n}),
\qquad
\hat A_t^{(n)}\equiv\hat R_t^{(n)}-V(s_t).
$$

前 $n$ 步使用实际奖励，后续回报由末状态价值估计。$n=1$ 时，优势估计等于 TD 残差：

$$
\hat A_t^{(1)}=\delta_t
=r_{t+1}+\gamma V(s_{t+1})-V(s_t).
$$

<details class="kt-derivation" id="ppo-td-advantage">
<summary>TD 残差与优势估计</summary>

在同一回合内，动作价值满足贝尔曼展开：

$$
Q^{\pi_{\rm old}}(s_t,a_t)
=\mathbb E\!\left[r_{t+1}+\gamma V^{\pi_{\rm old}}(s_{t+1})\mid s_t,a_t\right].
$$

减去状态价值，得到：

$$
A^{\pi_{\rm old}}(s_t,a_t)
=\mathbb E\!\left[
r_{t+1}+\gamma V^{\pi_{\rm old}}(s_{t+1})-V^{\pi_{\rm old}}(s_t)
\mid s_t,a_t\right].
$$

用旧 Critic 近似真实价值，括号内的单次采样值就是 $\delta_t$。当价值预测准确时，$\mathbb E[\delta_t\mid s_t,a_t]=A^{\pi_{\rm old}}(s_t,a_t)$；实际估计受采样噪声与价值预测误差影响。

</details>

广义优势估计 GAE 对不同步数的优势估计进行加权。设从时刻 $t$ 到本回合边界或采样末端共有 $H$ 步可用数据，则：

$$
\hat A_t=\sum_{l=0}^{H-1}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda\in[0,1]$ 控制多步估计的权重；$\lambda=0$ 使用一步 TD 残差，$\lambda=1$ 使用片段内全部残差的折扣和。末状态价值在真正终止时取零，在外部截断或采样末端使用旧 Critic 的预测。

将状态价值加回优势估计，得到 Critic 的 $\lambda$-return 目标：

$$
\hat R_t=\hat A_t+V_{\phi_{\rm old}}(s_t).
$$

<details class="kt-derivation" id="ppo-gae-derivation">
<summary>多步优势估计与有限时域 λ-return</summary>

考虑同一回合内由旧策略采集的连续片段，简记旧 Critic 为 $V$。

**二步回报**

从 $s_t$ 出发，实际观察两个时间步的奖励，再用 $V(s_{t+2})$ 估计剩余回报，定义二步自举回报：

$$
\hat R_t^{(2)}
\equiv r_{t+1}+\gamma r_{t+2}+\gamma^2V(s_{t+2})
$$

其中：

- $r_{t+1}$ 是执行当前动作 $a_t$ 后获得的奖励。
- $\gamma r_{t+2}$ 是下一时间步奖励折算到时刻 $t$ 的贡献。
- $\gamma^2V(s_{t+2})$ 是从 $s_{t+2}$ 开始的剩余回报估计，折算到时刻 $t$ 的贡献。

“二步”表示使用两个时间步的真实采样奖励。更远的未来仍通过 Critic 估计。

**二步优势估计**

用二步自举回报估计当前状态与动作对应的回报，再减去当前状态的价值基线，定义二步优势估计：

$$
\hat A_t^{(2)}\equiv\hat R_t^{(2)}-V(s_t)
$$

因此，它的直接计算式为：

$$
\hat A_t^{(2)}
=r_{t+1}+\gamma r_{t+2}+\gamma^2V(s_{t+2})-V(s_t)
$$

$\hat A_t^{(2)}$ 表示：**根据后续两步实际奖励和末状态价值，对时刻 $t$ 所选动作相对于状态基线的优势进行估计。**

上标 $(2)$ 表示估计使用的步数；它估计的对象仍是当前动作的优势 $A^{\pi_{\rm old}}(s_t,a_t)$。帽子表示这是采样估计值。

**TD 残差表示**

相邻两步的 TD 残差分别为：

$$
\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t),
$$

$$
\delta_{t+1}=r_{t+2}+\gamma V(s_{t+2})-V(s_{t+1}).
$$

将第二步残差折算到时刻 $t$，再与第一步相加：

$$
\begin{aligned}
\delta_t+\gamma\delta_{t+1}
&=r_{t+1}+\gamma V(s_{t+1})-V(s_t)\\
&\quad+\gamma r_{t+2}+\gamma^2V(s_{t+2})-\gamma V(s_{t+1})\\
&=r_{t+1}+\gamma r_{t+2}+\gamma^2V(s_{t+2})-V(s_t)\\
&=\hat R_t^{(2)}-V(s_t).
\end{aligned}
$$

中间状态的两项价值预测相互抵消，由此得到：

$$
\hat A_t^{(2)}=\delta_t+\gamma\delta_{t+1}
$$

这里，**“二步回报减去状态基线”说明估计量的含义，“两个 TD 残差的折扣和”给出等价的计算方式。**

同理，$n$ 步优势估计可以写为：

$$
\hat R_t^{(n)}
\equiv\sum_{l=0}^{n-1}\gamma^lr_{t+l+1}+\gamma^nV(s_{t+n}),
$$

$$
\hat A_t^{(n)}
\equiv\hat R_t^{(n)}-V(s_t)
=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}
$$

**有限时域 GAE**

设可用片段长为 $H$。正确的有限混合式将剩余权重赋给最长的 $H$ 步估计：

$$
\hat A_t
=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}\hat A_t^{(n)}
+\lambda^{H-1}\hat A_t^{(H)}.
$$

将 $\hat A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}$ 代入。对于第 $l$ 个残差，所有包含它的多步估计的混合权重之和为：

$$
(1-\lambda)\sum_{n=l+1}^{H-1}\lambda^{n-1}+\lambda^{H-1}=\lambda^l.
$$

因此：

$$
\hat A_t=\sum_{l=0}^{H-1}\gamma^l\lambda^l\delta_{t+l}
=\sum_{l=0}^{H-1}(\gamma\lambda)^l\delta_{t+l}.
$$

混合权重之和为 1，将同一个状态基线加回去得到：

$$
\begin{aligned}
\hat R_t^{(\lambda)}
&=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}\hat R_t^{(n)}
+\lambda^{H-1}\hat R_t^{(H)}\\
&=\hat A_t+V(s_t).
\end{aligned}
$$

$H=1$ 时使用一步回报。$\lambda=0$ 时仅保留一步估计；$\lambda=1$ 时得到：

$$
\hat R_t=\sum_{l=0}^{H-1}\gamma^lr_{t+l+1}+\gamma^HV(s_{t+H}).
$$

真正终止时，末状态价值为零，上式就是该段完整的采样折扣回报；外部截断或采样末端保留旧 Critic 的末值。

在连续片段内部，提出残差和的首项可得：

$$
\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}.
$$

加回状态价值并整理，得到等价的回报递推：

$$
\hat R_t=r_{t+1}+\gamma\left[(1-\lambda)V(s_{t+1})+\lambda\hat R_{t+1}\right].
$$

</details>

<details class="kt-derivation" id="ppo-gae-worked-example">
<summary>GAE 数值算例</summary>

设 $\gamma=0.9,\lambda=0.8$，片段内各步属于同一回合，末端为采样边界；奖励为 $[1,0,2,1]$，旧价值为 $[2,2.5,2,1.5]$，额外计算末状态价值 $V(s_4)=1$。

$$
\boldsymbol\delta=[1.25,-0.7,1.35,0.4],\qquad \gamma\lambda=0.72.
$$

从 $\hat A_4=0$ 开始反向计算：

$$
\begin{aligned}
\hat A_3&=0.4,\\
\hat A_2&=1.35+0.72\times0.4=1.638,\\
\hat A_1&=-0.7+0.72\times1.638=0.47936,\\
\hat A_0&=1.25+0.72\times0.47936=1.5951392.
\end{aligned}
$$

| $t$ | $r_{t+1}$ | $v_t^{\rm old}$ | $\delta_t$ | $\hat A_t$ | $\hat R_t=\hat A_t+v_t^{\rm old}$ |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2.0 | 1.25 | 1.5951392 | 3.5951392 |
| 1 | 0 | 2.5 | −0.70 | 0.47936 | 2.97936 |
| 2 | 2 | 2.0 | 1.35 | 1.638 | 3.638 |
| 3 | 1 | 1.5 | 0.40 | 0.4 | 1.9 |

第 1 步的 TD 残差为负，累加后续残差后，GAE 为正。

如果只把 $\lambda$ 改为 1，则

$$
\hat R_0=1+0.9\times0+0.9^2\times2+0.9^3\times1+0.9^4\times1=4.0051.
$$

最后一项来自末状态价值自举。$\hat A_4=0$ 结束本段残差累加，$V(s_4)=1$ 则保留末端的剩余回报估计。

再看截断：若当前奖励为 1，旧状态价值为 0，真实末状态价值为 10，$\gamma=0.9$，则外部截断处的优势为 $1+0.9\times10=10$。优势递推在此结束。若该转移为真正终止，则末端自举价值为零，优势为 1。

</details>

### 3.5 轨迹采样与边界处理

采样时固定 $\theta_{\rm old},\phi_{\rm old}$。Actor 产生动作并记录动作概率，环境提供奖励与后继状态，Critic 提供当前状态和后继状态的价值估计。整段轨迹采集完成后，计算 GAE 与回报目标。

轨迹边界决定末端价值和残差累加的范围：

| 边界 | 末端价值 | GAE 累加范围 |
| --- | --- | --- |
| 任务真正终止 | 零 | 截至终止转移 |
| 外部时间限制截断 | 重置前末状态的旧价值 | 截至截断转移 |
| 达到本轮采样长度 | 末状态的旧价值 | 截至本段最后一条转移 |

<details class="kt-derivation" id="ppo-boundary-masks">
<summary>边界掩码与反向递推</summary>

记 $v_t^{\rm old}=V_{\phi_{\rm old}}(s_t)$，$v_{t+1}^{\rm next}$ 为实际后继状态的旧价值。定义自举掩码 $b_t$ 与递推掩码 $c_t$：

$$
b_t=1-\mathbb 1[\mathrm{terminated}_t],\qquad
c_t=1-\mathbb 1[\mathrm{terminated}_t\lor\mathrm{truncated}_t].
$$

$\mathrm{terminated}_t$ 表示任务终止，$\mathrm{truncated}_t$ 表示外部限制引起的截断。TD 残差与 GAE 递推为：

$$
\delta_t=r_{t+1}+\gamma b_tv_{t+1}^{\rm next}-v_t^{\rm old},
$$

$$
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1},
\qquad\hat A_T=0.
$$

$T$ 为采样长度。$\hat A_T=0$ 结束本段残差累加，最后一步的自举价值通过 $\delta_{T-1}$ 保留。终止和截断处 $c_t=0$，各回合分别计算优势。

若有限时域属于任务定义，时间耗尽按终止处理，观测应包含剩余时间；外部时间限制按截断处理。参见 [Gymnasium 时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。

</details>

## 4. 实现与应用

### 4.1 训练流程

1. 固定采样参数，收集状态、动作、奖励、旧动作概率、旧价值及边界标记。
2. 沿时间倒序计算 GAE，并用原始优势构造固定回报目标。
3. 将数据划分为小批量，用当前网络重新评估已采样动作和状态价值。
4. 计算损失与梯度，更新 Actor 和 Critic；对本轮数据进行有限次遍历。
5. 使用更新后的策略采集下一批数据。

```mermaid
flowchart LR
    A["轨迹采样"] --> B["GAE 与回报目标"]
    B --> C["策略损失与价值损失"]
    C --> D["梯度与参数更新"]
    D -->|继续使用本轮数据| C
    D -->|完成本轮更新| A
```

### 4.2 实现示例

[下载 PPO 教学示例](/knowledge-tree/examples/ppo_core.py)。实现包含独立 Actor/Critic、GAE、损失计算与小批量更新。

<details class="kt-derivation" id="ppo-policy-distribution">
<summary>策略分布与动作概率</summary>

连续动作示例采用各维独立的高斯分布。Actor 输出均值，可学习的 `log_std` 决定标准差；示例中的标准差与状态无关。离散动作可使用分类分布。

采样时记录 $\ell_t^{\rm old}=\log\pi_{\theta_{\rm old}}(a_t\mid s_t)$，更新时在相同状态上评估原动作，得到 $\ell_t^{\rm new}$。概率比率由对数概率计算：

$$
\rho_t=\exp(\ell_t^{\rm new}-\ell_t^{\rm old}).
$$

多维动作的联合对数概率沿动作维度求和。示例使用 `Independent(Normal(mean, std), 1)`，联合对数概率与熵均为每条样本一个标量。

</details>

<details class="kt-derivation" id="ppo-data-layout">
<summary>数据结构与函数映射</summary>

使用 $N$ 个并行环境，每个环境采样 $T$ 步，共得到 $B=TN$ 条状态转移。先保留时间维与环境维计算 GAE，再按相同顺序展平字段。

| 字段 | 采样 / GAE 阶段 | 更新阶段 | 对应数学量 |
| --- | --- | --- | --- |
| `obs` | `[T,N,obs_dim]` | `[B,obs_dim]` | $s_t$ |
| `actions` | `[T,N,action_dim]` | `[B,action_dim]` | 已采样 $a_t$ |
| `old_logprobs` | `[T,N]` | `[B]` | $\ell_t^{\rm old}$ |
| `old_values` | `[T,N]` | `[B]` | $v_t^{\rm old}$ |
| `rewards`、`next_values` | `[T,N]` | 用于预先计算 target | $r_{t+1},v_{t+1}^{\rm next}$ |
| `terminated`、`truncated` | `[T,N]`，布尔类型 | 用于预先计算 target | 动作之后的边界 |
| `advantages`、`returns` | `[T,N]` | `[B]` | $\hat A_t,\hat R_t$ |

采样与训练的网络输入都保留 batch 维，即使只有一个环境也使用 `[1,obs_dim]`。联合对数概率和熵的形状均为 `[B]`；价值输出使用 `squeeze(-1)` 得到 `[B]`。

| 数学步骤 | 完整示例中的实现 |
| --- | --- |
| Actor / Critic 与可选共享特征 | `ActorCritic`，`distribution()`、`value()` |
| 固定旧策略采样 | `sample()` 与 `synthetic_rollout()` 中的 `no_grad()` |
| 两种边界 mask、TD、GAE、returns | `compute_gae()` |
| 全字段保持相同展平顺序并停止梯度 | `flatten_rollout()` 与 `PPOBatch` |
| 当前参数重新评估旧动作 | `evaluate(obs, actions)` |
| ratio、策略损失、价值损失、熵 | `ppo_loss()` |
| 多 epoch、minibatch、全局梯度裁剪 | `ppo_update()` |

</details>

<details class="kt-derivation" id="ppo-advantage-normalization">
<summary>优势标准化</summary>

先用原始优势构造回报目标 $\hat R_t=\hat A_t+v_t^{\rm old}$，再对策略损失使用的优势进行标准化。令 $\mathcal M$ 为含 $M$ 条数据的小批量，$\eta>0$ 为数值稳定项：

$$
\tilde A_t=\frac{\hat A_t-\bar A_{\mathcal M}}{\sigma_{\mathcal M}+\eta},
\quad
\bar A_{\mathcal M}=\frac1M\sum_{t\in\mathcal M}\hat A_t,
\quad
\sigma_{\mathcal M}^2=\frac1M\sum_{t\in\mathcal M}(\hat A_t-\bar A_{\mathcal M})^2.
$$

策略损失中的 $\hat A_t$ 替换为 $\tilde A_t$，截断方向也按 $\tilde A_t$ 的正负判断。回报目标保留原始回报尺度。示例采用小批量标准化与 `std(correction=0)`；整轮采样数据统一标准化是另一种实现选择。

</details>

<details class="kt-derivation" id="ppo-gae-code">
<summary>GAE 计算</summary>

在每个环境的时间序列上倒序计算优势。`rewards[t]` 对应 $r_{t+1}$，`next_values[t]` 对应重置前实际后继状态的旧价值。

```python
bootstrap_mask = (~terminated).to(rewards.dtype)
trace_mask = (~(terminated | truncated)).to(rewards.dtype)
advantages = torch.zeros_like(rewards)
last_gae = torch.zeros_like(rewards[0])  # [N]
for t in reversed(range(rewards.shape[0])):
    delta = rewards[t] + gamma * bootstrap_mask[t] * next_values[t] - values[t]
    last_gae = delta + gamma * gae_lambda * trace_mask[t] * last_gae
    advantages[t] = last_gae
returns = advantages + values
```

该计算在 `torch.no_grad()` 中执行。先保留时间与环境维度计算 GAE，再按一致顺序展平所有字段。

</details>

<details class="kt-derivation" id="ppo-loss-code">
<summary>损失计算与参数更新</summary>

```python
new_logprobs, entropy, new_values = model.evaluate(batch.obs, batch.actions)
logratio = new_logprobs - batch.old_logprobs
ratio = logratio.exp()
advantages = batch.advantages
advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)
unclipped = ratio * advantages
clipped = ratio.clamp(1 - policy_clip, 1 + policy_clip) * advantages
policy_loss = -torch.minimum(unclipped, clipped).mean()
value_loss = 0.5 * (new_values - batch.returns).square().mean()
total = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
```

`batch.advantages` 和 `batch.returns` 在采样结束后计算并停止梯度。当前网络重新计算动作对数概率、熵和价值，随后执行：

```python
optimizer.zero_grad(set_to_none=True)
total.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
```

`backward()` 计算梯度，`step()` 更新参数。全局梯度范数裁剪使用统一缩放系数：

$$
g=\nabla L_{\rm total},\qquad
g\leftarrow g\min\left(1,\frac{g_{\max}}{\lVert g\rVert_2}\right).
$$

实现处理零范数的数值情况。联合裁剪使两套网络共用梯度缩放系数；Adam 根据裁剪后的梯度及其动量更新参数。

</details>

<details class="kt-derivation" id="ppo-update-code">
<summary>小批量更新流程</summary>

一次 epoch 完整遍历本轮采样数据；每个 minibatch 用于一次参数更新。`batch.select()` 保持所有字段的索引一致。

```python
for _ in range(epochs):
    indices = torch.randperm(batch.obs.shape[0], device=batch.obs.device)
    for start in range(0, batch.obs.shape[0], minibatch_size):
        minibatch = batch.select(indices[start:start + minibatch_size])
        terms = ppo_loss(model, minibatch, policy_clip=policy_clip,
                         value_clip=value_clip, value_coef=value_coef,
                         entropy_coef=entropy_coef)
        optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True,
        )
        optimizer.step()
```

完整函数还校验 $B$ 可被 minibatch 大小整除、loss 有限，并累计每次更新前的诊断量。首次参数更新前 ratio 应接近 1，后续比率随策略更新而变化。

示例使用一个 Adam 管理两套网络参数：

```python
model = ActorCritic(obs_dim=4, action_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
```

示例使用两层 64 单元 Tanh MLP、正交初始化、独立可学习 `log_std`。

</details>

<details class="kt-derivation" id="ppo-environment-example">
<summary>示例运行与环境接口</summary>

在已安装 PyTorch 的环境中，从仓库根目录运行：

```bash
python examples/ppo_core.py
python -m unittest discover -s tests -p 'test_ppo_core.py' -v
```

下载单个文件后可运行 `python ppo_core.py`。示例使用合成轨迹检验数据与梯度流程，默认 $T=16,N=4$，共 64 条数据；执行 4 个 epoch，每批 16 条，共 16 次参数更新。

输出 JSON 的 `batch_size` 为 64，`optimizer_steps` 为 16，`initial_ratio_max_error` 应接近 0，损失与梯度应为有限值。

接入真实环境时，以采集器替换 `synthetic_rollout()`，提供 `[T,N,...]` 数据。自动重置环境需要从接口取得重置前的末状态，用于计算 `next_values`；新回合初始状态用于下一次动作采样。

示例策略为无界高斯。接入有界动作空间时，应保持动作变换与概率计算一致：使用压缩高斯时加入 Jacobian 修正；采用动作裁剪执行约定时，保存原始采样动作及其对数概率。

</details>

## 5. 分析与总结

### 5.1 核心结论

Actor 通过优势加权的截断目标调整动作概率，Critic 通过回归损失拟合回报。GAE 将轨迹奖励与旧价值预测组合成优势，加回状态价值后得到回报目标。每轮固定这些训练信号更新网络，再由新策略采集下一批数据。

### 5.2 训练诊断

任务回报反映策略表现，价值误差反映回报拟合程度，KL 与截断比例描述策略变化。

<details class="kt-derivation" id="ppo-training-diagnostics">
<summary>诊断指标</summary>

| 指标 | 应怎样解读 | 优先检查 |
| --- | --- | --- |
| 首次更新前 ratio | 同参数、同观测预处理、同动作时应接近 1 | 模型版本、动作变换、logprob 维度、随机层 |
| `policy_loss` | 当前 batch 的替代目标，不能单独代表任务表现 | 联看回报、KL、clipfrac 与优势分布 |
| `value_loss` | 当前价值对固定 return target 的误差 | 奖励尺度、边界处理、target 是否停止梯度 |
| `entropy` | 当前动作分布随机性 | 标准差、动作坐标与熵系数；连续熵允许为负 |
| `approx_kl` | 本轮策略漂移的采样诊断 | 先核对具体公式，再查学习率、epoch 数 |
| `clipfrac` | ratio 超出区间的样本比例 | 不等于实际进入 Clip 平坦区的比例 |
| `grad_norm_before_clip` | 全部参数裁剪前的梯度范数 | 与阈值比较，并留意两路损失的梯度规模 |
| `explained_variance` | $1-\operatorname{Var}(\hat R-V)/\operatorname{Var}(\hat R)$ | 区分新/旧价值；target 方差为零时无定义 |

本示例记录 loss、entropy、近似 KL、clipfrac 与裁剪前梯度范数等训练量，并在更新前单独检查 ratio。各 minibatch 指标是 step 前观测的均值，不应误读为整轮更新完成后的完整分布测量。

标准化优势在初始 ratio=1 时均值为 0，所以 $L_\pi=-\operatorname{mean}(\tilde A)\approx0$，而梯度 $-\operatorname{mean}(\tilde A\nabla\log\pi)$ 可以非零，需结合策略变化与任务指标判断学习进展。

Clip 作用于样本目标，实际比率仍可能超出截断区间。训练器可监控新旧策略的 KL，并在变化过大时提前停止本轮更新；本示例仅记录 KL 诊断量。

</details>

<details class="kt-derivation" id="ppo-kl-estimation">
<summary>KL 散度与采样估计</summary>

对固定状态，从旧策略到新策略的 KL 为

$$
D_{\rm KL}(\pi_{\rm old}(\cdot\mid s)\|\pi_\theta(\cdot\mid s))
=\mathbb E_{a\sim\pi_{\rm old}}
\left[\log\pi_{\rm old}(a\mid s)-\log\pi_\theta(a\mid s)\right].
$$

离散动作时显式求和，连续动作时积分。KL 非负且不对称。再对旧策略的状态访问分布 $d_{\rm old}$ 取平均，得到 TRPO 使用的约束量：

$$
\begin{aligned}
\bar D_{\rm KL}(\theta_{\rm old},\theta)
&=\mathbb E_{s\sim d_{\rm old}}[D_{\rm KL}(\pi_{\rm old}(\cdot\mid s)\|\pi_\theta(\cdot\mid s))]\\
&=\mathbb E_{\substack{s\sim d_{\rm old}\\a\sim\pi_{\rm old}(\cdot\mid s)}}
[\ell^{\rm old}-\ell^{\rm new}].
\end{aligned}
$$

对应优化问题为

$$
\max_\theta\ \hat{\mathbb E}_t[\rho_t\hat A_t]
\quad\text{s.t.}\quad\bar D_{\rm KL}(\theta_{\rm old},\theta)\leq\delta_{\rm KL}.
$$

如果保存了旧分布参数，可计算每个采样状态上的完整 KL，再取算术平均：

$$
\widehat{\bar D}_{\rm KL}
=\frac1B\sum_{t=1}^{B}D_{\rm KL}(\pi_{\rm old}(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t)).
$$

如果仅保存每个状态实际采到的动作及其 logprob，则常见诊断量有：

| 名称 | 样本表达式 | 解释 |
| --- | --- | --- |
| 对数比率估计 | $-\operatorname{mean}(\log\rho)$ | 期望为旧到新 KL，有限样本均值可能为负 |
| 非负比率估计 | $\operatorname{mean}(\rho-1-\log\rho)$ | 本文代码采用；同一精确期望，有限样本结果不同 |
| PPO2 二次近似 | $\frac12\operatorname{mean}[(\log\rho)^2]$ | 小变化下的近似诊断，不是精确 KL |

因为对 $x>0$ 有 $x-1-\log x\geq0$，第二种估计的单样本项非负；在相互兼容的支撑集条件下，$\mathbb E_{a\sim\pi_{\rm old}}[\rho]=1$，故其精确期望仍等于 $-\mathbb E[\log\rho]$。有限浮点运算可能有极小舍入误差。

本文代码使用非负比率估计，历史 [PPO2 model.py](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py) 使用二次近似；比较不同实现日志前要先核对公式和 KL 方向。

</details>

<details class="kt-derivation" id="ppo-gaussian-kl">
<summary>高斯策略的 KL 散度</summary>

先考虑一维旧策略 $\mathcal N(\mu_0,\sigma_0^2)$ 与新策略 $\mathcal N(\mu_1,\sigma_1^2)$。由高斯 log density 展开：

$$
\begin{aligned}
D_{\rm KL}(\pi_{\rm old}\|\pi_{\rm new})
&=\mathbb E_{a\sim\mathcal N(\mu_0,\sigma_0^2)}
\left[\log\frac{\sigma_1}{\sigma_0}
+\frac{(a-\mu_1)^2}{2\sigma_1^2}
-\frac{(a-\mu_0)^2}{2\sigma_0^2}\right].
\end{aligned}
$$

旧分布下有

$$
\mathbb E[(a-\mu_0)^2]=\sigma_0^2,
\qquad
\mathbb E[(a-\mu_1)^2]=\sigma_0^2+(\mu_0-\mu_1)^2.
$$

代入得到

$$
D_{\rm KL}(\pi_{\rm old}\|\pi_{\rm new})
=\log\frac{\sigma_1}{\sigma_0}
+\frac{\sigma_0^2+(\mu_0-\mu_1)^2}{2\sigma_1^2}-\frac12.
$$

对 $D$ 个独立动作维度，联合 KL 等于各维 KL 之和；再对 $B$ 个 rollout 状态取平均：

$$
\widehat{\bar D}_{\rm KL}
=\frac1B\sum_{t=1}^{B}\sum_{j=1}^{D}
\left[\log\frac{\sigma_{1,j}(s_t)}{\sigma_{0,j}(s_t)}
+\frac{\sigma_{0,j}^2(s_t)+(\mu_{0,j}(s_t)-\mu_{1,j}(s_t))^2}{2\sigma_{1,j}^2(s_t)}-\frac12\right].
$$

若新旧标准差相同，单个状态上的 KL 进一步化为

$$
D_{\rm KL}=\frac12\sum_{j=1}^{D}
\frac{(\mu_{0,j}(s)-\mu_{1,j}(s))^2}{\sigma_j^2}.
$$

该闭式计算需要旧、新两组均值和标准差。本文最小示例只保存旧动作 logprob，因此记录采样近似 KL，不把它当作完整高斯 KL。

</details>

<details class="kt-derivation" id="ppo-implementation-checks">
<summary>实现检查</summary>

- 更新阶段始终评估原采样动作，旧概率与当前概率对应相同状态和动作。
- 优势与回报目标提前计算并停止梯度，回报目标使用原始优势。
- 自举掩码与递推掩码分别处理终止和截断；截断使用重置前末状态。
- 每个并行环境独立计算 GAE，之后再展平；循环网络保留序列与隐藏状态。
- 分别记录策略比率截断、价值截断和梯度范数裁剪的配置。

</details>

### 5.3 实现变体

共享特征改变梯度流向，价值截断改变 Critic 的回归损失形式。

<details class="kt-derivation" id="ppo-gradient-flow">
<summary>独立网络与共享特征的梯度</summary>

独立网络各自拥有特征层与输出层，策略损失和价值损失分别更新对应参数。

共享架构使用 $z=f_\psi(s)$ 提取特征，再分别通过策略头 $\pi_\theta(a\mid z)$ 与价值头 $V_\phi(z)$ 输出。共享参数的梯度为：

$$
\nabla_\psi L_{\rm total}
=\nabla_\psi L_\pi+c_v\nabla_\psi L_V-c_e\nabla_\psi\bar{\mathcal H}.
$$

共享特征同时接收策略与价值梯度，优势和回报目标保持固定。示例默认使用独立网络，设置 `shared=True` 可启用共享特征。

</details>

<details class="kt-derivation" id="ppo-value-clipping">
<summary>价值截断</summary>

以旧价值为中心限制当前预测的变化：

$$
V_t^{\rm clip}=v_t^{\rm old}
+\operatorname{clip}(V_\phi(s_t)-v_t^{\rm old},-\epsilon_v,\epsilon_v).
$$

然后使用普通误差与截断误差的较大者：

$$
L_V^{\rm clip}
=\frac12\hat{\mathbb E}_t
\left[\max\left((V_\phi(s_t)-\hat R_t)^2,(V_t^{\rm clip}-\hat R_t)^2\right)\right].
$$

该损失抑制部分过大价值变化带来的误差改善，当前预测仍可能超出截断区间。

$\epsilon_\pi$ 限制无量纲的概率比率，$\epsilon_v$ 使用价值的回报单位。历史 PPO2 复用了同一个数值，其他实现允许独立配置或关闭。本文示例默认普通 MSE；设置 `value_clip=0.2` 可启用该变体。奖励缩放改变后应重新理解其尺度。

</details>

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——价值函数、贝尔曼方程、策略梯度与 Actor-Critic。
- **算法前身**：[TRPO 算法完整解析与代码实现](./02_trpo.md)——信赖域约束、二阶近似与线搜索。
- **工程实践**：[PPO 全流程走读与调参](../../03_engineering-practice/01_ppo-walkthrough.md)——训练配置、数据采集与工程经验。
- **路线对照**：[SAC 算法完整解析与代码实现](./06_sac.md)——连续控制中的 Off-Policy 方法。

### 6.2 参考文献

1. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017：替代目标、PPO-Clip 与多 epoch minibatch 优化。
2. Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), ICLR 2016：多步 residual、GAE 与偏差方差折中。
3. Schulman et al., [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html), ICML 2015：局部替代目标与信赖域优化。
4. OpenAI Baselines，代码对照固定为提交 `ea25b9e8b234e6ee1bca43083f8f3cf974143998`：[runner.py](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/runner.py)、[model.py](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py)、[ppo2.py](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py)。用于核对历史 PPO2 的 GAE、损失、共享参数视图和更新循环。
5. OpenAI Spinning Up，[Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：PPO-Clip 直觉、伪代码与 KL early stopping。
6. Farama Foundation，[Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)：终止、外部截断与 bootstrap 的区别。
