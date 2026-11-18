# Dreamer：潜在世界模型与行为学习

## 1. 概述

Dreamer 从观测序列学习潜在世界模型，在模型内生成想象轨迹并训练 Actor-Critic。实际交互持续修正模型，想象训练把模型预测转化为可直接执行的策略。

以下以 Dreamer V1 的连续控制机制建立主线，再比较 V2 和 V3 的表示与梯度设计。

## 2. 基础与定义

### 2.1 循环状态空间模型

RSSM 用确定性记忆 $h_t$ 和随机变量 $z_t$ 构成潜在状态 $x_t=(h_t,z_t)$：

$$
h_t=F_\phi(h_{t-1},z_{t-1},a_{t-1}),
$$

$$
z_t\sim q_\phi(z_t\mid h_t,o_t),\qquad
\hat z_t\sim p_\phi(z_t\mid h_t).
$$

后验 $q_\phi$ 利用实际观测修正状态，先验 $p_\phi$ 用于模型内预测。确定性记忆汇集历史，随机变量表达预测中的不确定性。

### 2.2 网络与奖励约定

世界模型还包含观测解码器和奖励预测头。Actor $\pi_\theta(a_t\mid x_t)$ 输出动作，Critic $V_\psi(x_t)$ 估计未来回报。

令 $r_t$ 表示从 $x_t$ 执行动作 $a_t$ 后获得的奖励。RSSM 推进到 $x_{t+1}$，奖励头从该后继状态预测 $\hat r_t$。模型学习用实际观测和奖励监督，行为学习用想象回报监督。

## 3. 核心原理

### 3.1 世界模型目标

以实际轨迹的后验状态训练世界模型，基础目标包含观测重建、奖励预测和先验匹配：

$$
L_M=\mathbb E_q\sum_t\left[
-\log p_\phi(o_t\mid x_t)
-\log p_\phi(r_{t-1}\mid x_t)
+\beta D_{\mathrm{KL}}\big(q_\phi(z_t\mid h_t,o_t)\|p_\phi(z_t\mid h_t)\big)
\right].
$$

奖励项只用于具有前一转移的状态。重建与奖励信号塑造状态表示，KL 项使无观测时的先验接近观测修正后的后验。包含终止预测时，增加继续概率的监督损失。

<details class="kt-derivation" id="dreamer-gaussian-kl">
<summary>高斯潜在变量的 KL 散度</summary>

V1 使用连续高斯潜在变量。对后验 $q=\mathcal N(\mu_q,\operatorname{diag}(\sigma_q^2))$ 与先验 $p=\mathcal N(\mu_p,\operatorname{diag}(\sigma_p^2))$：

$$
D_{\mathrm{KL}}(q\|p)=\frac12\sum_i\left[
\log\frac{\sigma_{p,i}^2}{\sigma_{q,i}^2}
+\frac{\sigma_{q,i}^2+(\mu_{q,i}-\mu_{p,i})^2}{\sigma_{p,i}^2}-1
\right].
$$

后验融合当前观测，先验只使用转移后的确定性记忆。二者的匹配帮助模型在想象阶段维持与实际数据一致的潜在分布。

</details>

### 3.2 想象回报

从实际序列推断的后验状态开始，固定模型参数，交替采样动作和先验状态。用预测奖励与 Critic 构造有限时域 $\lambda$-return：

$$
G_H^\lambda=V_\psi(x_H),
$$

$$
G_t^\lambda=\hat r_t+\gamma_t\left[(1-\lambda)V_\psi(x_{t+1})+\lambda G_{t+1}^\lambda\right].
$$

持续任务可取 $\gamma_t=\gamma$；带继续概率预测时，使用 $\gamma_t=\gamma\hat c_t$。想象过程中采用先验推进，奖励和价值直接在潜在状态上计算。

### 3.3 Actor 与 Critic

V1 的连续控制 Actor 最大化想象回报。令 $w_t=\prod_{j<t}\gamma_j$ 表示到达该步的折扣权重：

$$
L_\pi=-\mathbb E\left[\sum_{t=0}^{H-1}w_tG_t^\lambda\right],
\qquad
L_V=\frac12\mathbb E\left[\sum_{t=0}^{H-1}w_t
(V_\psi(x_t)-\operatorname{sg}(G_t^\lambda))^2\right].
$$

Actor 更新通过重参数化动作与状态传播回报梯度，模型和 Critic 参数固定；Critic 更新以分离后的状态和回报为训练数据。继续概率权重的梯度处理按实现中的停止梯度约定核对。

<details class="kt-derivation" id="dreamer-actor-critic-gradients">
<summary>想象学习的计算图</summary>

V1 Actor 的路径为：

$$
\theta\rightarrow a_t\rightarrow x_{t+1}\rightarrow
\{\hat r_t,V_\psi(x_{t+1}),a_{t+1}\}\rightarrow G_t^\lambda.
$$

冻结世界模型参数仍允许计算 $\partial x_{t+1}/\partial a_t$。冻结 Critic 参数仍允许计算 $\partial V_\psi/\partial x$。

Critic 拟合时，输入状态与目标回报作为固定数据；梯度路径为 $\psi\rightarrow V_\psi(x_t)\rightarrow L_V$。两种更新对同一回报张量采用不同的梯度边界。

</details>

### 3.4 版本演进

| 版本 | 潜在表示与训练设计 | Actor 梯度 |
|---|---|---|
| V1 | 连续随机状态，模型内连续控制 | 通过想象回报的路径梯度 |
| V2 | 离散随机状态、KL 平衡等改进 | 按动作空间与配置使用 REINFORCE、路径梯度或组合 |
| V3 | 离散随机状态、尺度稳健的奖励与价值预测、回报归一化 | Nature 版本采用 REINFORCE 与熵正则 |

V3 的策略目标可概括为：

$$
L_\pi^{\mathrm{V3}}=-\mathbb E\left[
\log\pi_\theta(a_t\mid x_t)\operatorname{sg}(\tilde A_t)
+\eta\mathcal H(\pi_\theta(\cdot\mid x_t))\right].
$$

$\tilde A_t$ 为按回报尺度处理的优势。世界模型提供状态、奖励与回报目标，Actor 使用固定优势的得分函数梯度。V3 的奖励和价值采用变换后离散支持上的分布预测，复现时应使用对应分类目标。

## 4. 实现与应用

### 4.1 学习流程

1. 在实际环境收集观测、动作、奖励与边界标志。
2. 从连续片段推断 RSSM 后验，更新世界模型。
3. 从后验状态开始，用先验和当前 Actor 生成想象轨迹。
4. 反向递推回报，按版本定义更新 Actor 与 Critic。
5. 回到实际环境，更新记忆状态并执行 Actor 输出。

观测编码和循环记忆参与部署，图像解码主要用于训练及诊断。回合重置时同步重置相关记忆状态。

### 4.2 无人机视觉控制

可将相机、惯性信息和历史动作编码为潜在状态，预测任务奖励并训练导航策略。模型应保留速度、障碍关系和执行器响应等影响控制的时序信息。

评估同时检查实际导航回报、先验多步预测、历史长度和延迟敏感性。图像重建清晰度只反映模型目标的一部分，不能替代闭环控制评价。

## 5. 分析与总结

Dreamer 将高维观测压缩为可递推状态，再用模型内数据训练行为。RSSM 提供任务相关的历史表示，其信息充分性取决于数据与训练目标。

V1 展示潜在模型中的路径梯度，V2 和 V3 进一步发展状态表示和行为学习。比较 Dreamer 与 SHAC、MBPO 时，应明确具体版本的 Actor 梯度与模型用途。

## 6. 关联与参考

- 前置：[Model-Based RL 基础](01_model-based-rl-foundations.md)、[可微模型与策略梯度](07_differentiable-policy-optimization.md)。
- 对照：[TD-MPC](05_td-mpc.md)、[SHAC](08_shac.md)、[SuperMimic](10_supermimic.md)。
- Hafner 等：[Dream to Control: Learning Behaviors by Latent Imagination](https://openreview.net/forum?id=S1lOTC4tDS)，ICLR 2020。
- Hafner 等：[Mastering Atari with Discrete World Models](https://openreview.net/forum?id=0oabwyZbOu)，ICLR 2021。
- Hafner 等：[Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2)，Nature 2025。
- [DreamerV3 官方实现](https://github.com/danijar/dreamerv3)。
