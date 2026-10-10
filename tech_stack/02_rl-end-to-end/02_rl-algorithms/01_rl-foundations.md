# 强化学习核心公式手册

## 知识主线

$$
\text{MDP 建模}
\longrightarrow
\text{贝尔曼方程}
\longrightarrow
\text{动态规划}
\longrightarrow
\text{MC/TD}
\longrightarrow
\text{值函数近似}
\longrightarrow
\text{策略梯度}
\longrightarrow
\text{Actor-Critic}
$$

## 目录

* [第一部分：基本概念与 MDP](#第一部分基本概念与-mdp)
* [第二部分：贝尔曼方程与动态规划](#第二部分贝尔曼方程与动态规划)
* [第三部分：无模型表格型方法](#第三部分无模型表格型方法)
* [第四部分：值函数近似与 DQN](#第四部分值函数近似与-dqn)
* [第五部分：策略梯度与 REINFORCE](#第五部分策略梯度与-reinforce)
* [第六部分：Actor-Critic](#第六部分actor-critic)
* [附录：符号与算法速查](#附录符号与算法速查)

---

## 第一部分：基本概念与 MDP

### 1.1 状态、动作、奖励与概率模型

* 状态空间：$\mathcal S=\{s_1,s_2,\dots\}$。
* 动作空间：状态 $s$ 下的合法动作集合记为 $\mathcal A(s)$。
* 奖励空间：所有可能即时奖励的集合记为 $\mathcal R$。
* 策略：智能体在状态 $s$ 选择动作 $a$ 的概率分布：
  $$
  \pi(a\mid s)=\Pr(A_t=a\mid S_t=s),
  \qquad
  \sum_{a\in\mathcal A(s)}\pi(a\mid s)=1.
  $$
* 状态转移概率：
  $$
  p(s'\mid s,a)
  =\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).
  $$
* 奖励概率：
  $$
  p(r\mid s,a)
  =\Pr(R_{t+1}=r\mid S_t=s,A_t=a).
  $$

状态—动作对的期望即时奖励为：

$$
r(s,a)
=\mathbb E[R_{t+1}\mid S_t=s,A_t=a]
=\sum_{r\in\mathcal R}r\,p(r\mid s,a).
$$

本文始终分别使用 $p(r\mid s,a)$ 和 $p(s'\mid s,a)$。在贝尔曼方程中：

$$
\begin{aligned}
&\mathbb E\!\left[
R_{t+1}+\gamma V(S_{t+1})
\mid S_t=s,A_t=a
\right]\\
&=
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)V(s').
\end{aligned}
$$


### 1.2 轨迹、回合与马尔可夫性质

一条轨迹为：

$$
\tau=(S_0,A_0,R_1,S_1,A_1,R_2,\dots).
$$

从初始状态运行至终止状态构成一个回合。若当前状态已经包含预测未来所需的全部信息，则满足马尔可夫性质：

$$
\Pr(S_{t+1}\mid S_0,A_0,\dots,S_t,A_t)
=\Pr(S_{t+1}\mid S_t,A_t).
$$

本文将环境表示为：

$$
(\mathcal S,\mathcal A,\mathcal R,p_s,p_r,\gamma),
$$

其中 $p_s$ 对应 $p(s'\mid s,a)$，$p_r$ 对应 $p(r\mid s,a)$，策略 $\pi$ 是智能体的决策规则，不属于环境动力学。

### 1.3 回报与优化目标

时刻 $t$ 之后的折扣回报定义为：

$$
G_t
=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}.
$$

其递归形式为：

$$
\boxed{
G_t=R_{t+1}+\gamma G_{t+1}
}.
$$

折扣因子满足 $\gamma\in[0,1]$。回合任务可取 $\gamma=1$；持续任务通常取 $\gamma<1$。后文统一约定终止状态的价值为 $0$。

---

## 第二部分：贝尔曼方程与动态规划

### 2.1 状态价值与动作价值

给定策略 $\pi$，状态价值函数为：

$$
v_\pi(s)
=\mathbb E_\pi[G_t\mid S_t=s].
$$

动作价值函数为：

$$
q_\pi(s,a)
=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

二者的直接关系为：

$$
\boxed{
v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a)
}.
$$

### 2.2 贝尔曼期望方程

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
\boxed{
v_\pi(s)
=\sum_a\pi(a\mid s)
\left[
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')
\right]
}.
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
\boxed{
q_\pi(s,a)
=\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)
\sum_{a'}\pi(a'\mid s')q_\pi(s',a')
}.
$$

也可以先通过状态价值表示动作价值：

$$
\boxed{
q_\pi(s,a)
=r(s,a)
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')
}.
$$

贝尔曼方程体现自举：使用后继价值推算当前价值。

### 2.3 矩阵形式与策略评估

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
\boxed{
v_\pi=r_\pi+\gamma P_\pi v_\pi
}.
$$

当逆矩阵存在时，可闭式求解：

$$
v_\pi=(I-\gamma P_\pi)^{-1}r_\pi.
$$

更常用的是迭代求解：

$$
v_{k+1}=r_\pi+\gamma P_\pi v_k.
$$

### 2.4 贝尔曼最优方程

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
\boxed{
v_*(s)
=\max_a
\left[
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_*(s')
\right]
}.
$$

最优动作价值方程为：

$$
\boxed{
q_*(s,a)
=\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)
\max_{a'}q_*(s',a')
}.
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

### 2.5 策略迭代、值迭代与截断策略迭代

#### 策略迭代

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

#### 值迭代

值迭代直接反复应用贝尔曼最优算子：

$$
\boxed{
V_{k+1}(s)
=\max_a
\left[
r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_k(s')
\right]
}.
$$

当

$$
\max_s|V_{k+1}(s)-V_k(s)|<\varepsilon
$$

时停止，并提取：

$$
\pi_*(s)
\in\arg\max_a
\left[
r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V(s')
\right].
$$

#### 截断策略迭代

截断策略迭代又称修正策略迭代。每次策略改进后只执行有限次策略评估：

$$
\pi_{k+1}=\operatorname{Greedy}(V_k),
\qquad
V_{k+1}=(\mathcal T_{\pi_{k+1}})^mV_k.
$$

* $m=1$ 时退化为值迭代；
* $m\to\infty$ 时趋近完整策略迭代；
* 有限 $m$ 在二者之间平衡评估成本与迭代次数。

---

## 第三部分：无模型表格型方法

当 $p(r\mid s,a)$ 和 $p(s'\mid s,a)$ 未知时，无法直接展开贝尔曼期望，只能使用交互样本估计价值。

### 3.1 蒙特卡洛方法

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
\boxed{
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+\alpha
\left[G_t-Q(S_t,A_t)\right]
}.
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

### 3.2 随机近似与随机梯度

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

常用收敛步长条件为：

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

GD 使用总体梯度，BGD 使用完整批量，SGD 使用单样本，Mini-batch SGD 使用小批量样本。

### 3.3 时序差分评估

蒙特卡洛需要等待完整回合，而 TD(0) 使用一步自举目标：

$$
Y_t^{\mathrm{TD}}
=R_{t+1}+\gamma V(S_{t+1}).
$$

时序差分误差为：

$$
\delta_t
=R_{t+1}
+\gamma V(S_{t+1})
-V(S_t).
$$

TD(0) 更新为：

$$
\boxed{
V(S_t)
\leftarrow
V(S_t)+\alpha\delta_t
}.
$$

| 方法 | 更新时机 | 目标 | 自举 | 典型特点 |
| :--- | :---: | :--- | :---: | :--- |
| 蒙特卡洛 | 回合结束 | $G_t$ | 否 | 无偏、方差较大 |
| TD(0) | 每一步 | $R_{t+1}+\gamma V(S_{t+1})$ | 是 | 可在线更新、方差较小 |

### 3.4 Sarsa 系列与 Q-Learning

动作价值算法统一写为：

$$
\boxed{
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+\alpha
\left[Y_t-Q(S_t,A_t)\right]
}.
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

---

## 第四部分：值函数近似与 DQN

表格型方法为每个状态或状态—动作对单独保存数值。状态空间巨大或连续时，改用参数化函数进行泛化。

### 4.1 状态价值函数近似

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
\boxed{
w_{t+1}
=w_t+\alpha
\left[U_t-\hat v(S_t,w_t)\right]
\nabla_w\hat v(S_t,w_t)
}.
$$

常用目标为：

| 方法 | $U_t$ |
| :--- | :--- |
| MC 近似 | $G_t$ |
| TD 近似 | $R_{t+1}+\gamma\hat v(S_{t+1},w_t)$ |

TD 目标也含参数，但求导时将目标视为常数，只对当前预测 $\hat v(S_t,w_t)$ 求导，因此称为半梯度。

### 4.2 动作价值函数近似

使用 $\hat q(s,a,w)$ 近似动作价值。通用更新为：

$$
\boxed{
w_{t+1}
=w_t+\alpha
\left[U_t-\hat q(S_t,A_t,w_t)\right]
\nabla_w\hat q(S_t,A_t,w_t)
}.
$$

| 方法 | $U_t$ | 估计目标 |
| :--- | :--- | :--- |
| Sarsa + 函数近似 | $R_{t+1}+\gamma\hat q(S_{t+1},A_{t+1},w_t)$ | $q_\pi$ |
| Q-Learning + 函数近似 | $R_{t+1}+\gamma\max_{a'}\hat q(S_{t+1},a',w_t)$ | $q_*$ |

非线性函数近似、自举和离轨学习同时出现时容易导致训练不稳定，DQN 使用额外机制缓解这一问题。

### 4.3 深度 Q 网络

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

---

## 第五部分：策略梯度与 REINFORCE

值函数方法先估计价值再提取策略；策略梯度直接参数化 $\pi_\theta(a\mid s)$ 并最大化性能指标。

### 5.1 从目标函数到策略梯度

采用初始状态期望回报：

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta(\tau)}[G_0].
$$

西湖笔记中常见的其他等价或相关指标包括平均状态价值与平均奖励：

$$
J_V(\theta)=\sum_s d(s)v_{\pi_\theta}(s),
\qquad
J_R(\theta)=\sum_s d_{\pi_\theta}(s)
\sum_a\pi_\theta(a\mid s)r(s,a).
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

根据 $q_{\pi_\theta}(s,a)=\mathbb E[G_t\mid s,a]$：

$$
\boxed{
\nabla_\theta J(\theta)
=\mathbb E_\tau
\left[
\sum_{t=0}^{T-1}
\gamma^t
\nabla_\theta\log
\pi_\theta(A_t\mid S_t)
q_{\pi_\theta}(S_t,A_t)
\right]
}.
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
\boxed{
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
}.
$$

正常数 $1/(1-\gamma)$ 不改变优化方向，算法中通常将其吸收到学习率。

### 5.2 REINFORCE

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
\boxed{
\theta
\leftarrow
\theta+\alpha\widehat g(\tau)
}.
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

### 5.3 基线与方差降低

对于任何与动作无关的基线 $b(s)$：

$$
\mathbb E_{A\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(A\mid s)b(s)
\right]=0.
$$

因此可用 $G_t-b(S_t)$ 替换 $G_t$：

$$
\widehat g_b(\tau)
=\sum_{t=0}^{T-1}
\gamma^t
\left[G_t-b(S_t)\right]
\nabla_\theta\log
\pi_\theta(A_t\mid S_t).
$$

常用选择为 $b(s)\approx v_{\pi_\theta}(s)$。基线不改变梯度期望，但可以显著降低方差。

---

## 第六部分：Actor-Critic

Actor 负责更新策略，Critic 负责评估策略并为 Actor 提供低方差的学习信号。

### 6.1 优势函数与优势策略梯度

优势函数定义为：

$$
A_{\pi_\theta}(s,a)
=q_{\pi_\theta}(s,a)
-v_{\pi_\theta}(s).
$$

取基线 $b(s)=v_{\pi_\theta}(s)$，得到优势策略梯度：

$$
\boxed{
\nabla_\theta J(\theta)
\propto
\mathbb E_{\pi_\theta}
\left[
\nabla_\theta\log
\pi_\theta(A_t\mid S_t)
A_{\pi_\theta}(S_t,A_t)
\right]
}.
$$

### 6.2 从 TD 误差到 Actor-Critic 更新

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
\boxed{
A_{\pi_\theta}(s,a)
=\mathbb E_{\pi_\theta}
\left[
R_{t+1}
+\gamma v_{\pi_\theta}(S_{t+1})
-v_{\pi_\theta}(S_t)
\mid S_t=s,A_t=a
\right]
}.
$$

使用 Critic $\hat v(S,w)$ 替代真实价值，定义：

$$
y_t^v
=R_{t+1}+\gamma\hat v(S_{t+1},w_t),
$$

$$
\boxed{
\delta_t
=y_t^v-\hat v(S_t,w_t)
=R_{t+1}
+\gamma\hat v(S_{t+1},w_t)
-\hat v(S_t,w_t)
}.
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
\boxed{
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
}
$$

若严格按有限回合目标 $J(\theta)=\mathbb E[G_0]$ 构造逐时间步梯度，Actor 项还应乘以 $\gamma^t$；采用折扣状态访问分布记号时通常省略这一显式权重。

### 6.3 离轨策略与确定性策略梯度

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

连续动作空间可使用确定性策略 $a=\mu_\theta(s)$。确定性策略梯度为：

$$
\boxed{
\nabla_\theta J(\theta)
\propto
\mathbb E_s
\left[
\left.
\nabla_\theta\mu_\theta(s)
\nabla_aQ^\mu(s,a)
\right|_{a=\mu_\theta(s)}
\right]
}.
$$

---

## 附录：符号与算法速查

### 符号表

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
| $\theta,w$ | 策略参数与价值参数 |
| $\delta_t$ | 时序差分误差 |

### 核心更新目标

| 方法 | 模型依赖 | 策略属性 | 核心目标 |
| :--- | :---: | :---: | :--- |
| 策略评估 | 是 | on-policy | $\sum_a\pi(a\mid s)[r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V(s')]$ |
| 值迭代 | 是 | 最优控制 | $\max_a[r+\gamma\sum_{s'}p(s'\mid s,a)V(s')]$ |
| MC | 否 | on-policy | $G_t$ |
| TD(0) | 否 | on-policy | $R_{t+1}+\gamma V(S_{t+1})$ |
| Sarsa | 否 | on-policy | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ |
| Expected Sarsa | 否 | on-policy | $R_{t+1}+\gamma\sum_{a'}\pi(a'\mid S_{t+1})Q(S_{t+1},a')$ |
| Q-Learning | 否 | off-policy | $R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')$ |
| DQN | 否 | off-policy | 固定目标网络给出的贝尔曼最优目标 |
| REINFORCE | 否 | on-policy | $G_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)$ |
| Actor-Critic | 否 | 通常 on-policy | $\delta_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)$ |

### 最短复习路径

1. 用 MDP 描述状态、动作、奖励及两个条件概率分布。
2. 用回报 $G_t$ 定义长期目标。
3. 用贝尔曼方程连接当前价值和后继价值。
4. 模型已知时使用动态规划；模型未知时使用 MC 或 TD。
5. 状态空间过大时用参数化函数近似价值。
6. 直接优化策略时使用策略梯度。
7. Actor-Critic 用 Critic 的 TD 误差指导 Actor 更新。
