# 强化学习核心公式手册

## 1. 概述

强化学习通过交互数据学习长期决策。环境模型描述状态与奖励的生成，价值函数评价长期回报，策略决定动作。动态规划、价值学习与策略梯度分别给出利用模型、拟合价值和直接优化策略的计算方法。

## 2. 基础与定义

### 2.1 马尔可夫决策过程

状态、动作和奖励分别记为 $S_t,A_t,R_{t+1}$。环境由状态空间 $\mathcal S$、动作空间 $\mathcal A$、奖励空间 $\mathcal R$、联合转移分布 $p(s',r\mid s,a)$ 和折扣因子 $\gamma$ 描述；初始状态分布为 $p_0$。

马尔可夫性质表示：给定当前状态与动作，下一步状态及奖励的分布由当前信息决定。策略为 $\pi(a\mid s)$。一条轨迹写为：

$$
\tau=(S_0,A_0,R_1,S_1,A_1,R_2,\ldots).
$$

联合分布的边缘分布记为 $p(s'\mid s,a)$ 与 $p(r\mid s,a)$，期望即时奖励为 $r(s,a)=\mathbb E[R_{t+1}\mid s,a]$。奖励与后继状态的相关性由联合分布描述。

### 2.2 回报与优化目标

折扣回报与策略目标为：

$$
G_t=\sum_{l=0}^{\infty}\gamma^lR_{t+l+1},\qquad
G_t=R_{t+1}+\gamma G_{t+1},\qquad
J(\theta)=\mathbb E_{\pi_\theta}[G_0].
$$

持续任务采用 $\gamma<1$ 并假设奖励有界；有限回合的求和截至终止，回报可积时可取 $\gamma=1$。终止状态的后续价值为零。

<details class="kt-derivation" id="rl-symbols">
<summary>符号表</summary>

| 符号 | 含义 |
| :--- | :--- |
| $\mathcal S,\mathcal A,\mathcal R$ | 状态、动作、奖励空间 |
| $S_t,A_t,R_{t+1}$ | 时刻 $t$ 的状态、动作以及随后获得的奖励 |
| $p(s'\mid s,a)$ | 状态转移概率 |
| $p(r\mid s,a)$ | 奖励概率 |
| $r(s,a)$ | 状态—动作对的期望即时奖励 |
| $\pi(a\mid s)$ | 策略 |
| $\beta(a\mid s)$ | 行为策略 |
| $\gamma$ | 折扣因子 |
| $G_t$ | 时刻 $t$ 之后的折扣回报 |
| $v_\pi,q_\pi$ | 策略 $\pi$ 的真实状态价值与动作价值 |
| $v_*,q_*$ | 最优状态价值与最优动作价值 |
| $V,Q$ | 表格型价值估计 |
| $\hat v,\hat q$ | 参数化价值估计 |
| $\theta$ | 策略参数 |
| $\phi,w$ | 价值网络参数，$w$ 用于函数近似推导 |
| $\delta_t$ | 时序差分误差 |

</details>

## 3. 核心原理

### 3.1 价值函数与贝尔曼方程

价值函数把长期目标分解为状态与动作的评价：

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\qquad
q_\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

由回报递推得到：

$$
q_\pi(s,a)=r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s'),
\qquad
v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a).
$$

当前价值由即时奖励与后继价值组成。离散状态使用求和，连续状态使用对应积分。

<details class="kt-derivation" id="rl-bellman-derivation">
<summary>贝尔曼期望方程</summary>

将 $G_t=R_{t+1}+\gamma G_{t+1}$ 代入状态价值定义：

$$
v_\pi(s)
=\mathbb E_\pi\!\left[
R_{t+1}+\gamma v_\pi(S_{t+1})
\mid S_t=s
\right].
$$

分别展开策略、奖励概率和状态转移概率：

$$

v_\pi(s)
=\sum_a\pi(a\mid s)
\left[
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')
\right]
.
$$

动作价值的条件期望形式为：

$$
q_\pi(s,a)
=\mathbb E_\pi\!\left[
R_{t+1}
+\gamma q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
\right].
$$

展开后得到：

$$

q_\pi(s,a)
=\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)
\sum_{a'}\pi(a'\mid s')q_\pi(s',a')
.
$$

也可以先通过状态价值表示动作价值：

$$

q_\pi(s,a)
=r(s,a)
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')
.
$$

贝尔曼方程体现自举：使用后继价值推算当前价值。

</details>

<details class="kt-derivation" id="rl-policy-evaluation">
<summary>矩阵形式与策略评估</summary>

定义策略 $\pi$ 下的期望奖励向量和状态转移矩阵：

$$
r_\pi(s)=\sum_a\pi(a\mid s)r(s,a),
$$

$$
P_\pi(s,s')
=\sum_a\pi(a\mid s)p(s'\mid s,a).
$$

贝尔曼期望方程的矩阵形式为：

$$

v_\pi=r_\pi+\gamma P_\pi v_\pi
.
$$

当逆矩阵存在时，可闭式求解：

$$
v_\pi=(I-\gamma P_\pi)^{-1}r_\pi.
$$

更常用的是迭代求解：

$$
v_{k+1}=r_\pi+\gamma P_\pi v_k.
$$

</details>

### 3.2 动态规划

模型已知时，可交替进行策略评估与策略改进。策略评估求解当前策略的贝尔曼方程，策略改进选择动作价值最大的动作。值迭代直接更新最优价值估计：

$$
V_{j+1}(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_j(s')\right].
$$

有限状态、有限动作、有界奖励与 $\gamma<1$ 下，贝尔曼最优算子为压缩映射，反复更新收敛到唯一最优价值。

<details class="kt-derivation" id="rl-optimality">
<summary>贝尔曼最优方程</summary>

最优价值定义为：

$$
v_*(s)=\max_\pi v_\pi(s),
\qquad
q_*(s,a)=\max_\pi q_\pi(s,a).
$$

最优状态价值的条件期望形式为：

$$
v_*(s)
=\max_a
\mathbb E\!\left[
R_{t+1}+\gamma v_*(S_{t+1})
\mid S_t=s,A_t=a
\right].
$$

展开后得到：

$$

v_*(s)
=\max_a
\left[
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_*(s')
\right]
.
$$

最优动作价值方程为：

$$

q_*(s,a)
=\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)
\max_{a'}q_*(s',a')
.
$$

最优价值与最优策略满足：

$$
v_*(s)=\max_a q_*(s,a),
\qquad
\pi_*(s)\in\arg\max_a q_*(s,a).
$$

贝尔曼最优算子 $\mathcal T_*$ 在 $\gamma<1$ 时满足：

$$
\|\mathcal T_*u-\mathcal T_*v\|_\infty
\le\gamma\|u-v\|_\infty.
$$

因此它是压缩映射，存在唯一不动点 $v_*$，从任意初值反复应用该算子都可收敛至 $v_*$。

</details>

<details class="kt-derivation" id="rl-dynamic-programming">
<summary>策略迭代与值迭代</summary>

**策略迭代**

1. 任意初始化策略 $\pi$ 和价值函数 $V$。
2. 固定策略，反复执行策略评估：
   $$
   V_{j+1}(s)
   =\sum_a\pi(a\mid s)
   \left[
   r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_j(s')
   \right].
   $$
3. 评估收敛后执行策略改进：
   $$
   \pi_{\mathrm{new}}(s)
   \in\arg\max_a
   \left[
   r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V(s')
   \right].
   $$
4. 若策略不变则停止，否则继续评估新策略。

流程可写为：

$$
\pi_k
\xrightarrow{\text{策略评估}}
v_{\pi_k}
\xrightarrow{\text{策略改进}}
\pi_{k+1}.
$$

**值迭代**

值迭代直接反复应用贝尔曼最优算子：

$$

V_{k+1}(s)
=\max_a
\left[
r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_k(s')
\right]
.
$$

当

$$
\max_s|V_{k+1}(s)-V_k(s)|<\varepsilon
$$

时停止，并提取：

$$
\pi_{\mathrm{greedy}}(s)
\in\arg\max_a
\left[
r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V(s')
\right].
$$

**截断策略迭代**

截断策略迭代又称修正策略迭代。每次策略改进后只执行有限次策略评估：

$$
\pi_{k+1}=\operatorname{Greedy}(V_k),
\qquad
V_{k+1}=(\mathcal T_{\pi_{k+1}})^mV_k.
$$

* $m=1$ 时退化为值迭代；
* $m\to\infty$ 时趋近完整策略迭代；
* 有限 $m$ 在二者之间平衡评估成本与迭代次数。

</details>

### 3.3 蒙特卡洛与时序差分

交互样本可直接估计价值。两类目标为：

$$
Y_t^{\rm MC}=G_t,\qquad
Y_t^{\rm TD}=R_{t+1}+\gamma V(S_{t+1}).
$$

蒙特卡洛使用完整采样回报；时序差分使用后继价值自举。表格更新统一为：

$$
V(S_t)\leftarrow V(S_t)+\alpha[Y_t-V(S_t)].
$$

TD 残差为 $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$。MC 的采样回报在固定策略、完整回合下提供无偏价值样本；TD 的目标依赖当前价值近似，常以自举偏差换取较短的采样范围。

<details class="kt-derivation" id="rl-monte-carlo">
<summary>蒙特卡洛评估与控制</summary>

蒙特卡洛方法使用完整回合回报估计价值：

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],
\qquad
q_\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

若状态—动作对 $(s,a)$ 已观测 $N(s,a)$ 次：

$$
Q(s,a)
\approx
\frac{1}{N(s,a)}
\sum_{i=1}^{N(s,a)}G^{(i)}(s,a).
$$

增量更新形式为：

$$

Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+\alpha
\left[G_t-Q(S_t,A_t)\right]
.
$$

当 $\alpha=1/N(S_t,A_t)$ 时，该式等价于样本均值。

* First-visit MC：每个回合只更新状态或状态—动作对的第一次访问。
* Every-visit MC：更新该回合中的每一次访问。
* Exploring Starts：要求每个 $(s,a)$ 都有非零概率作为回合起点，理论清晰但实际较难实现。

实际控制通常使用 $\varepsilon$-贪婪策略。令 $a^*\in\arg\max_a Q(s,a)$：

$$
\pi(a\mid s)=
\begin{cases}
1-\varepsilon+\dfrac{\varepsilon}{|\mathcal A(s)|},
&a=a^*,\\[6pt]
\dfrac{\varepsilon}{|\mathcal A(s)|},
&a\ne a^*.
\end{cases}
$$

MC 控制反复执行：

$$
\text{采样回合}
\longrightarrow
\text{用 }G_t\text{ 更新 }Q
\longrightarrow
\varepsilon\text{-贪婪改进策略}.
$$

</details>

### 3.4 动作价值与控制

动作价值学习将更新对象改为 $Q(S_t,A_t)$：

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[Y_t-Q(S_t,A_t)].
$$

| 方法 | 目标 $Y_t$ | 下一动作来源 |
| --- | --- | --- |
| Sarsa | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ | 实际行为策略 |
| Expected Sarsa | $R_{t+1}+\gamma\sum_{a'}\pi(a'\mid S_{t+1})Q(S_{t+1},a')$ | 目标策略的期望 |
| Q-learning | $R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')$ | 贪婪目标 |

行为策略生成样本，目标策略定义所学习的价值；二者相同时为同策略学习，允许二者不同时为离策略学习。表中终止转移的后继项取零。

<details class="kt-derivation" id="rl-multistep-sarsa">
<summary>多步动作价值估计</summary>

动作价值算法统一写为：

$$

Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+\alpha
\left[Y_t-Q(S_t,A_t)\right]
.
$$

不同算法只是在目标 $Y_t$ 上不同：

| 方法 | 目标 $Y_t$ |
| :--- | :--- |
| Sarsa | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ |
| Expected Sarsa | $R_{t+1}+\gamma\sum_{a'}\pi(a'\mid S_{t+1})Q(S_{t+1},a')$ |
| Q-Learning | $R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')$ |

n-step Sarsa 的目标为：

$$
G_{t:t+n}
=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+\gamma^nQ(S_{t+n},A_{t+n}).
$$

其更新为：

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+\alpha
\left[G_{t:t+n}-Q(S_t,A_t)\right].
$$

当 $n=1$ 时得到单步 Sarsa；当 $n$ 延伸至回合终点时趋近蒙特卡洛目标。
若回合在 $t+n$ 之前终止，则后续奖励截断，并令末端自举项为 $0$。

* on-policy：行为策略与目标策略相同，例如 Sarsa。
* off-policy：行为策略负责探索，目标策略负责学习，例如 Q-Learning。

</details>

### 3.5 价值函数近似

连续或大规模状态空间使用参数化价值 $V_\phi(s)$。给定停止梯度的样本目标 $Y_t$，回归损失与梯度为：

$$
L_V(\phi)=\frac12\hat{\mathbb E}_t[(V_\phi(S_t)-Y_t)^2],
\qquad
\nabla_\phi L_V=\hat{\mathbb E}_t[(V_\phi(S_t)-Y_t)\nabla_\phi V_\phi(S_t)].
$$

TD 目标由当前或目标网络预测构造，求导时固定目标，形成半梯度更新。动作价值近似采用相同原则；DQN 结合经验回放与目标网络组织训练。

<details class="kt-derivation" id="rl-stochastic-approximation">
<summary>随机近似与半梯度</summary>

样本均值的递推形式为：

$$
\bar X_{n+1}
=\bar X_n+\frac{1}{n+1}(X_{n+1}-\bar X_n).
$$

强化学习中的增量更新通常具有 Robbins–Monro 形式：

$$
w_{k+1}
=w_k+\alpha_k
\left[\text{新目标}-\text{当前估计}\right].
$$

在满足采样覆盖、噪声与更新稳定性等条件的随机近似分析中，常用步长条件为：

$$
\sum_{k=0}^{\infty}\alpha_k=\infty,
\qquad
\sum_{k=0}^{\infty}\alpha_k^2<\infty.
$$

对于可微代价函数 $J(w)$，随机梯度下降为：

$$
w_{k+1}
=w_k-\alpha_k\widehat{\nabla_wJ(w_k)}.
$$



使用 $\hat v(s,w)$ 近似 $v_\pi(s)$。给定状态采样分布 $\mu$，均方误差目标为：

$$
J(w)
=\frac12
\mathbb E_{S\sim\mu}
\left[
\left(v_\pi(S)-\hat v(S,w)\right)^2
\right].
$$

其梯度为：

$$
\nabla_wJ(w)
=-\mathbb E_{S\sim\mu}
\left[
\left(v_\pi(S)-\hat v(S,w)\right)
\nabla_w\hat v(S,w)
\right].
$$

真实价值未知时，以样本目标 $U_t$ 替换，得到通用半梯度更新：

$$

w_{t+1}
=w_t+\alpha
\left[U_t-\hat v(S_t,w_t)\right]
\nabla_w\hat v(S_t,w_t)
.
$$

常用目标为：

| 方法 | $U_t$ |
| :--- | :--- |
| MC 近似 | $G_t$ |
| TD 近似 | $R_{t+1}+\gamma\hat v(S_{t+1},w_t)$ |

TD 目标也含参数，但求导时将目标视为常数，只对当前预测 $\hat v(S_t,w_t)$ 求导，因此称为半梯度。

</details>

<details class="kt-derivation" id="rl-q-approximation">
<summary>动作价值近似与 DQN</summary>

使用 $\hat q(s,a,w)$ 近似动作价值。通用更新为：

$$

w_{t+1}
=w_t+\alpha
\left[U_t-\hat q(S_t,A_t,w_t)\right]
\nabla_w\hat q(S_t,A_t,w_t)
.
$$

| 方法 | $U_t$ | 估计目标 |
| :--- | :--- | :--- |
| Sarsa + 函数近似 | $R_{t+1}+\gamma\hat q(S_{t+1},A_{t+1},w_t)$ | $q_\pi$ |
| Q-Learning + 函数近似 | $R_{t+1}+\gamma\max_{a'}\hat q(S_{t+1},a',w_t)$ | $q_*$ |

非线性函数近似、自举和离轨学习同时出现时容易导致训练不稳定，DQN 使用额外机制缓解这一问题。

DQN 使用在线网络 $Q(s,a;w)$ 近似 $q_*(s,a)$，使用目标网络 $Q(s,a;w^-)$ 构造相对稳定的拟合目标。

对经验样本 $(S_i,A_i,R_{i+1},S_{i+1})$：

$$
y_i
=R_{i+1}
+\gamma\max_{a'}Q(S_{i+1},a';w^-).
$$

若 $S_{i+1}$ 为终止状态，约定最大后继价值为 $0$。

大小为 $B$ 的小批量损失为：

$$
L(w)
=\frac{1}{2B}
\sum_{i=1}^{B}
\left[
y_i-Q(S_i,A_i;w)
\right]^2.
$$

定义 $\delta_i=y_i-Q(S_i,A_i;w)$，对目标停止梯度：

$$
\nabla_wL(w)
=-\frac1B
\sum_{i=1}^{B}
\delta_i\nabla_wQ(S_i,A_i;w).
$$

因此在线网络执行：

$$
w
\leftarrow
w+\frac{\alpha}{B}
\sum_{i=1}^{B}
\delta_i\nabla_wQ(S_i,A_i;w).
$$

DQN 的两个关键稳定机制：

1. 经验回放：随机采样历史转移，降低连续数据相关性并提高数据复用率。
2. 目标网络：在若干步内固定 $w^-$，周期性执行 $w^-\leftarrow w$。

</details>

### 3.6 策略梯度

直接参数化策略 $\pi_\theta$，梯度将动作价值转化为动作概率的调整。对于有限回合目标 $J=\mathbb E[G_0]$：

$$
\nabla_\theta J
=\mathbb E_\tau\left[\sum_{t=0}^{T-1}\gamma^t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)q_{\pi_\theta}(S_t,A_t)\right].
$$

REINFORCE 用采样回报 $G_t$ 估计动作价值。减去动作无关的状态基线 $b(S_t)$，得到信号 $G_t-b(S_t)$；合适的基线可降低梯度方差。

<details class="kt-derivation" id="rl-policy-gradient">
<summary>策略梯度推导</summary>

采用初始状态期望回报：

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta(\tau)}[G_0].
$$

对轨迹概率，只需写出与策略参数有关的部分：

$$
p_\theta(\tau)
=c(\tau)
\prod_{t=0}^{T-1}
\pi_\theta(A_t\mid S_t),
$$

其中 $c(\tau)$ 汇总所有与 $\theta$ 无关的初始状态、奖励和状态转移概率。于是：

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=0}^{T-1}
\nabla_\theta\log
\pi_\theta(A_t\mid S_t).
$$

使用对数导数恒等式 $\nabla p=p\nabla\log p$：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\mathbb E_\tau
\left[
G_0\nabla_\theta\log p_\theta(\tau)
\right]\\
&=\mathbb E_\tau
\left[
\sum_{t=0}^{T-1}
\gamma^tG_t
\nabla_\theta\log
\pi_\theta(A_t\mid S_t)
\right].
\end{aligned}
$$

第二个等号使用因果关系：动作 $A_t$ 之前的奖励在给定历史后与当前动作采样无关，其与策略得分的乘积期望为零。$G_0$ 中从时刻 $t$ 开始的部分为 $\gamma^tG_t$。

根据 $q_{\pi_\theta}(s,a)=\mathbb E[G_t\mid s,a]$：

$$

\nabla_\theta J(\theta)
=\mathbb E_\tau
\left[
\sum_{t=0}^{T-1}
\gamma^t
\nabla_\theta\log
\pi_\theta(A_t\mid S_t)
q_{\pi_\theta}(S_t,A_t)
\right]
.
$$

在 $\gamma<1$ 时定义归一化折扣状态分布：

$$
d_\gamma^{\pi_\theta}(s)
=(1-\gamma)
\sum_{t=0}^{\infty}
\gamma^t\Pr_{\pi_\theta}(S_t=s).
$$

标准策略梯度公式为：

$$

\nabla_\theta J(\theta)
=\frac{1}{1-\gamma}
\mathbb E_{\substack{
S\sim d_\gamma^{\pi_\theta}\\
A\sim\pi_\theta(\cdot\mid S)
}}
\left[
\nabla_\theta\log\pi_\theta(A\mid S)
q_{\pi_\theta}(S,A)
\right]
.
$$

正常数 $1/(1-\gamma)$ 不改变优化方向，算法中通常将其吸收到学习率。

</details>

<details class="kt-derivation" id="rl-reinforce">
<summary>蒙特卡洛策略梯度</summary>

REINFORCE 使用蒙特卡洛回报替换未知动作价值。令：

$$
Z_t
=\nabla_\theta\log
\pi_\theta(A_t\mid S_t).
$$

由全期望公式：

$$
\begin{aligned}
\mathbb E[Z_tq_{\pi_\theta}(S_t,A_t)]
&=\mathbb E
\left[
Z_t\mathbb E[G_t\mid S_t,A_t]
\right]\\
&=\mathbb E[Z_tG_t].
\end{aligned}
$$

因此一条轨迹给出的无偏梯度估计为：

$$
\widehat g(\tau)
=\sum_{t=0}^{T-1}
\gamma^tG_t
\nabla_\theta\log
\pi_\theta(A_t\mid S_t).
$$

REINFORCE 执行梯度上升：

$$

\theta
\leftarrow
\theta+\alpha\widehat g(\tau)
.
$$

逐时间步写为：

$$
\theta
\leftarrow
\theta
+\alpha\gamma^tG_t
\nabla_\theta\log
\pi_\theta(A_t\mid S_t).
$$

</details>

<details class="kt-derivation" id="rl-baseline">
<summary>状态基线与梯度期望</summary>

固定状态 $s$，动作无关的基线满足：

$$
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}[b(s)\nabla_\theta\log\pi_\theta(a\mid s)]
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
$$

因此 $q_\pi(s,a)-b(s)$ 与 $q_\pi(s,a)$ 给出相同的精确策略梯度期望。使用 $b(s)=v_\pi(s)$ 时得到优势函数；其样本估计的误差仍需单独分析。

</details>

### 3.7 Actor-Critic

Actor 参数化策略，Critic 估计价值。优势定义为 $A^\pi(s,a)=q_\pi(s,a)-v_\pi(s)$。状态价值准确时，一步 TD 残差满足：

$$
\mathbb E[\delta_t\mid S_t=s,A_t=a]=A^\pi(s,a).
$$

因此可以用 TD 或多步优势估计 $\hat A_t$ 构造 Actor 损失，并用回报目标训练 Critic：

$$
L_\pi=-\hat{\mathbb E}_t[\log\pi_\theta(A_t\mid S_t)\operatorname{sg}(\hat A_t)],
\qquad
L_V=\frac12\hat{\mathbb E}_t[(V_\phi(S_t)-\operatorname{sg}(\hat R_t))^2].
$$

$\operatorname{sg}$ 表示停止梯度。此处经验平均表示所选训练样本的局部损失；严格估计初始状态折扣目标时，需使用对应的 $\gamma^t$ 权重或折扣访问分布。PPO 在此基础上使用概率比率和截断目标。

<details class="kt-derivation" id="rl-actor-critic-gradient">
<summary>Actor-Critic 的梯度</summary>

由动作价值的贝尔曼期望关系：

$$
q_{\pi_\theta}(s,a)
=\mathbb E_{\pi_\theta}
\left[
R_{t+1}
+\gamma v_{\pi_\theta}(S_{t+1})
\mid S_t=s,A_t=a
\right].
$$

代入优势函数：

$$

A_{\pi_\theta}(s,a)
=\mathbb E_{\pi_\theta}
\left[
R_{t+1}
+\gamma v_{\pi_\theta}(S_{t+1})
-v_{\pi_\theta}(S_t)
\mid S_t=s,A_t=a
\right]
.
$$

使用 Critic $\hat v(S,w)$ 替代真实价值，定义：

$$
y_t^v
=R_{t+1}+\gamma\hat v(S_{t+1},w_t),
$$

$$

\delta_t
=y_t^v-\hat v(S_t,w_t)
=R_{t+1}
+\gamma\hat v(S_{t+1},w_t)
-\hat v(S_t,w_t)
.
$$

当 Critic 精确时，$\delta_t$ 是优势函数的条件无偏样本；Critic 尚未收敛时，它是优势函数的近似。

记 $\operatorname{sg}(\cdot)$ 为停止梯度算子。Actor 的单样本目标与梯度方向为：

$$
L_{\mathrm{actor}}(\theta)
=-\operatorname{sg}(\delta_t)
\log\pi_\theta(A_t\mid S_t),
$$

$$
-\nabla_\theta L_{\mathrm{actor}}
=\delta_t
\nabla_\theta\log
\pi_\theta(A_t\mid S_t).
$$

Critic 的半梯度平方误差为：

$$
L_{\mathrm{critic}}(w)
=\frac12
\left[
\operatorname{sg}(y_t^v)-\hat v(S_t,w)
\right]^2,
$$

$$
\nabla_wL_{\mathrm{critic}}(w_t)
=-\delta_t\nabla_w\hat v(S_t,w_t).
$$

最终同步更新公式为：

$$

\begin{aligned}
\text{Actor:}\quad
\theta_{t+1}
&=\theta_t
+\alpha_\theta\delta_t
\nabla_{\theta_t}
\log\pi_{\theta_t}(A_t\mid S_t),\\
\text{Critic:}\quad
w_{t+1}
&=w_t
+\alpha_w\delta_t
\nabla_w\hat v(S_t,w_t).
\end{aligned}

$$

若严格按有限回合目标 $J(\theta)=\mathbb E[G_0]$ 构造逐时间步梯度，Actor 项还应乘以 $\gamma^t$；采用折扣状态访问分布记号时通常省略这一显式权重。

</details>

<details class="kt-derivation" id="rl-deterministic-gradient">
<summary>确定性策略梯度</summary>

若行为策略为 $\beta$、目标策略为 $\pi_\theta$，重要性采样比率为：

$$
\rho_t
=\frac{\pi_\theta(A_t\mid S_t)}
{\beta(A_t\mid S_t)}.
$$

只修正动作采样分布时，基本的离轨 Actor 更新方向可写为：

$$
\theta
\leftarrow
\theta
+\alpha_\theta\rho_t\delta_t
\nabla_\theta
\log\pi_\theta(A_t\mid S_t).
$$

要求 $\pi_\theta(a\mid s)>0$ 的动作在行为策略下也满足 $\beta(a\mid s)>0$。
严格修正一般离轨目标时还需考虑行为策略与目标策略的状态访问分布差异；上式是最基础的逐动作重要性采样形式。

连续动作空间可使用确定性策略 $a=\mu_\theta(s)$。对初始状态折扣目标，满足可微条件时：

$$
\nabla_\theta J(\theta)
=\frac{1}{1-\gamma}\mathbb E_{s\sim d_\gamma^\mu}
\left[(\nabla_\theta\mu_\theta(s))^\top
\nabla_aQ^\mu(s,a)\big|_{a=\mu_\theta(s)}\right].
$$

$d_\gamma^\mu$ 是确定性策略的归一化折扣状态访问分布。DDPG 用经验池状态与近似 Critic 构造实际训练损失，其具体梯度见 [DDPG](../02_model-free-rl/04_ddpg.md)。

</details>

## 4. 实现与应用

### 4.1 数据与更新流程

1. 指定状态、动作、奖励和回合边界，确定策略的可用信息。
2. 用模型求期望或用交互收集样本，构造价值目标。
3. 通过表格更新或函数回归估计价值。
4. 从动作价值提取策略，或用优势加权的策略梯度更新 Actor。
5. 在更新后的策略下继续采样，检查回报与估计误差。

### 4.2 数值算例

<details class="kt-derivation" id="rl-td-example">
<summary>TD 目标与价值更新</summary>

给定普通转移 $R_{t+1}=2,\gamma=0.9,V(S_t)=3,V(S_{t+1})=4$，有：

$$
Y_t=2+0.9\times4=5.6,\qquad\delta_t=5.6-3=2.6.
$$

学习率 $\alpha=0.1$ 时，表格价值更新为 $3+0.1\times2.6=3.26$。若该转移真正终止，则 $Y_t=2$，更新为 $2.9$。外部时间限制截断仍从实际末状态自举。

</details>

## 5. 分析与总结

### 5.1 方法关系

| 方法族 | 更新依据 | 主要误差来源 |
| --- | --- | --- |
| 动态规划 | 模型期望与价值递推 | 模型与数值近似 |
| 蒙特卡洛 | 完整采样回报 | 采样方差 |
| 时序差分 | 奖励与后继估计 | 采样噪声和自举误差 |
| 函数近似 | 样本回归目标 | 表示能力、优化与目标误差 |
| 策略梯度 | 回报或优势加权的概率梯度 | 信号方差与估计偏差 |

### 5.2 核心结论

回报定义优化目标，贝尔曼方程分解长期评价，价值估计与策略改进形成学习循环。深度算法通过网络、数据复用方式和目标构造，将这些关系转化为具体的训练损失。

## 6. 关联与参考

### 6.1 相关知识

- [算法演进](./02_rl-algorithm-evolution.md)：价值学习与策略优化的方法关系。
- [有模型与无模型强化学习](./03_model-free-vs-model-based.md)：环境模型在算法中的作用。
- [POMDP 与循环策略](./04_pomdp-recurrent-policies.md)：历史信息与部分可观测决策。
- [PPO](../02_model-free-rl/03_ppo.md) 与 [多价值网络](./05_multi-critic.md)：优势估计、损失和梯度的具体组织。

### 6.2 参考文献

1. Sutton & Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 第二版，2018。
2. Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696), 1992。
3. Silver et al., [Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.html), 2014。
4. Mnih et al., [Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236), 2015。
