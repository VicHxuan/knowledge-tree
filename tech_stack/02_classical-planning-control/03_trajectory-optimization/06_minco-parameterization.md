# MINCO 参数化

## 1. 概述

MINCO 用中间位置与段时间决定一条最小控制能量分段多项式。内层线性系统消去多项式系数，外层优化空间、时间与运动约束，从而保留较少的自由变量和可计算的梯度。

## 2. 基础与定义

### 2.1 内层最小能量问题

给定段时间 $T_i>0$、中间位置 $q_i$，固定首尾零至 $s-1$ 阶导数，求解

$$
\min_p\sum_{i=1}^M\int_0^{T_i}\|p_i^{(s)}(t)\|^2dt,
\qquad p_i(T_i)=q_i.
$$

轨迹属于具有平方可积 $s$ 阶导数的空间，并满足段间低阶连续性。$s=3$ 对应最小 jerk，$s=4$ 对应最小 Snap。

### 2.2 系数映射

内层解在每段为至多 $2s-1$ 次多项式。将所有系数堆叠为 $c$，可写为

$$
M(T)c=b(q),\qquad c=\mathcal M(q,T).
$$

边界条件消除能量零空间后，该映射在有效时间域内唯一确定。

## 3. 核心原理

### 3.1 连续性与稀疏结构

仅固定中间位置、允许中间高阶导数自由变化时，最优轨迹在内部连接点达到 $C^{2s-2}$ 连续。边界、路径点和连续性方程仅耦合相邻段，形成固定带宽系统。固定 $s$ 时，可用带状分解按段数线性增长的计算量求解。
<details class="kt-derivation" id="minco-variation">
<summary>变分与连接条件</summary>

对单轴能量取变分并分部积分 $s$ 次：

$$
\frac{\delta J}{2}=\sum_i\left[
\sum_{k=0}^{s-1}(-1)^kp_i^{(s+k)}\delta p_i^{(s-1-k)}
\right]_0^{T_i}
+(-1)^s\sum_i\int_0^{T_i}p_i^{(2s)}\delta p_i\,dt.
$$

段内任意变分给出 $p_i^{(2s)}=0$，故次数至多 $2s-1$。连接点位置固定，$\delta p=0$；共享的一至 $s-1$ 阶导数变分任意，因此 $s$ 至 $2s-2$ 阶导数的左右值相等。结合函数空间要求的零至 $s-1$ 阶连续性，得到 $C^{2s-2}$。

最高的 $2s-1$ 阶导数可在路径点跳变，对应位置约束的乘子。

</details>

### 3.2 外层目标与梯度

外层代价 $J(q,T)=F(c(q,T),T)$ 可以包含时间、碰撞、推力与角速度。先计算 $g_c=\nabla_cF$，再解伴随系统

$$
M^\top\lambda=g_c.
$$

梯度由

$$
\frac{dJ}{dq_j}=\lambda^\top\frac{\partial b}{\partial q_j},\qquad
\frac{dJ}{dT_i}=\frac{\partial F}{\partial T_i}
-\lambda^\top\frac{\partial M}{\partial T_i}c
$$

给出。这里固定物理时间下的边界导数，$b$ 对 $T$ 无显式依赖；采用其他变量定义时，应加入 $\lambda^\top\partial b/\partial T_i$。
<details class="kt-derivation" id="minco-adjoint">
<summary>隐式梯度推导</summary>

对 $Mc=b$ 微分：

$$
M\,dc=db-(dM)c.
$$

由 $dF=g_c^\top dc+F_T^\top dT$ 与 $M^\top\lambda=g_c$，得到

$$
dF=\lambda^\top db-\lambda^\top(dM)c+F_T^\top dT.
$$

一次伴随求解同时支持全部几何与时间参数的梯度，避免逐参数构造完整系数雅可比。

</details>

### 3.3 时间与空间参数

可用 $T_i=\exp(\tau_i)$ 保证时间为正，并通过链式法则得到 $dJ/d\tau_i=T_i\,dJ/dT_i$。中间位置可直接优化，也可由走廊内坐标映射生成。整段避障仍取决于曲线约束的具体实施。

## 4. 实现与应用

根据阶数与段时间组装带状系统，求解系数，计算积分代价和约束，累加系数梯度，再用伴随系统回传到位置与时间。

验证包括路径点残差、零至 $2s-2$ 阶拼接残差、能量积分与解析值，以及位置和时间梯度的中心差分。极短段时间会放大高阶导数与数值条件问题。

<details class="kt-derivation" id="minco-dimension">
<summary>变量维数</summary>

三维最小 jerk 的每段有 18 个多项式系数。首尾位置、速度和加速度固定后，$M$ 段 MINCO 只优化 $3(M-1)$ 个中间位置坐标与 $M$ 个时间，共 $4M-3$ 个变量。中间速度、加速度由内层最小能量条件确定。

</details>

## 5. 分析与总结

MINCO 将平滑结构编码进参数化，同时为一般外层目标提供梯度。它适用于边界和连续性明确的分段轨迹；外层避障与时间优化仍然具有非凸性。

## 6. 关联与参考

- [Wang 等：Geometrically Constrained Trajectory Optimization](https://arxiv.org/abs/2103.00190)
- [GCOPTER 的 MINCO 实现](https://github.com/ZJU-FAST-Lab/GCOPTER/blob/main/include/gcopter/minco.hpp)
- [GCOPTER](09_gcopter-minco.md)
