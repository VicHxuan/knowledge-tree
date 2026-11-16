# 最优控制问题建模

## 1. 概述

最优控制同时选择一段时间上的状态与控制，使系统满足动力学并优化累计代价。轨迹优化、LQR 与 MPC 都可以由这一问题结构描述。

## 2. 基础与定义

### 2.1 状态与控制

状态 $x(t)$、控制 $u(t)$ 满足 $\dot x=f(x,u,t)$，给定初值 $x(0)=x_0$。终止时间 $T$ 可固定，也可作为变量。

## 3. 核心原理

### 3.1 目标与约束

$$
\min_{x,u}\ \Phi(x(T))+\int_0^T\ell(x(t),u(t),t)dt
$$

满足动力学、路径约束 $g(x,u,t)\le0$ 与终端条件。运行代价、终端代价和约束分别表达过程偏好、末端目标与可行范围。

### 3.2 直接离散化

选离散步长后，以 $x_k,u_k$ 为变量，并加入 $x_{k+1}=F_h(x_k,u_k)$。单次打靶由控制递推全部状态，多重打靶引入节点状态与连续性约束，配点通过局部多项式约束动力学。

### 3.3 动态规划与变分

动态规划以最优剩余代价连接相邻时刻；变分法对轨迹扰动构造必要条件，形成协态与哈密顿系统。

<details class="kt-derivation" id="ocp-euler-lagrange">
<summary>欧拉—拉格朗日条件</summary>

对固定两端的 $J=\int L(x,\dot x,t)dt$，扰动 $x+\epsilon\eta$ 且 $\eta(0)=\eta(T)=0$。一阶变分经分部积分为 $\delta J=\int[L_x-\frac d{dt}L_{\dot x}]^\top\eta\,dt$。任意扰动下驻值要求 $L_x-\frac d{dt}L_{\dot x}=0$。

</details>

## 4. 实现与应用

### 4.1 建模流程

选状态与输入 → 固定时间和单位 → 建立目标与边界 → 选择离散化 → 检查动力学缺陷与约束 → 在更细时间网格上复核。

## 5. 分析与总结

模型、目标和数值离散化分别引入假设。优化后的离散节点满足条件时，仍需验证节点之间的连续轨迹。

## 6. 关联与参考

[MIT：Trajectory Optimization](https://underactuated.mit.edu/trajopt.html)。

关联：[Pontryagin 原理](./02_hamiltonian-pontryagin.md)、[LQR](../../02_classical-planning-control/04_tracking-control/02_lqr.md)。
