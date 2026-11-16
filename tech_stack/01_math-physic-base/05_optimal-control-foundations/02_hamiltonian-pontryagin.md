# 哈密顿函数与 Pontryagin 原理

## 1. 概述

Pontryagin 原理将连续时间最优控制的驻值条件写成状态、协态与控制的耦合方程。协态描述状态扰动对目标的边际影响。

## 2. 基础与定义

### 2.1 问题与约定

考虑固定终止时间、固定初值、自由终态的正常情形：$J=\Phi(x(T))+\int_0^T\ell(x,u,t)dt$，$\dot x=f(x,u,t)$。定义极小化约定的哈密顿函数 $H=\ell+\lambda^\top f$。

## 3. 核心原理

### 3.1 状态与协态

必要条件为

$$
\dot x=H_\lambda=f,\qquad\dot\lambda=-H_x,\qquad\lambda(T)=\Phi_x(x(T)).
$$

控制满足 $u_*(t)\in\arg\min_{u\in\mathcal U}H(x_*,u,\lambda,t)$；内部光滑解满足 $H_u=0$。终态约束会改变横截条件。

<details class="kt-derivation" id="pmp-variation">
<summary>协态与终端条件推导</summary>

增广泛函取 $\Phi(x(T))+\int[\ell+\lambda^\top(f-\dot x)]dt$。对状态求变分并对 $-\lambda^\top\delta\dot x$ 分部积分，得到内部系数 $H_x+\dot\lambda$ 和终端系数 $\Phi_x-\lambda(T)$；令它们为零得到正文条件。

</details>

### 3.2 有界控制

若 $H$ 对标量控制仿射，写为 $H_0+\sigma(t)u$，且 $|u|\le u_{\max}$，则 $\sigma>0$ 时取 $-u_{\max}$，$\sigma<0$ 时取 $u_{\max}$。$\sigma=0$ 的区间需进一步分析奇异控制。

### 3.3 线性二次型

对 $\ell=\tfrac12(x^\top Qx+u^\top Ru)$、$\dot x=Ax+Bu$，有 $u=-R^{-1}B^\top\lambda$。令 $\lambda=P(t)x$ 可导出 Riccati 方程。

## 4. 实现与应用

### 4.1 边值问题

从状态初值与协态终值构成两点边值问题。打靶法调整未知协态初值；验收时同时检查动力学、协态、控制驻点及终端残差。

## 5. 分析与总结

Pontryagin 原理给出必要条件。凸性、二阶条件与候选轨迹比较进一步支持最优性判断。

## 6. 关联与参考

[MIT：LQR](https://underactuated.mit.edu/lqr.html)、[Trajectory Optimization](https://underactuated.mit.edu/trajopt.html)。

关联：[最优控制建模](./01_optimal-control-formulation.md)。
