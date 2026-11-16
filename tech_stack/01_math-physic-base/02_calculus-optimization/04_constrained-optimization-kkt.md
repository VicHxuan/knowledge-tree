# 约束优化与 KKT 条件

## 1. 概述

约束优化在可行域内寻找最优点。乘子将边界的作用写入驻点条件，数值方法通过可行性与驻点残差的联合校正逼近解。

## 2. 基础与定义

### 2.1 可行域与拉格朗日函数

$$
\min_x f(x)\quad\text{s.t.}\quad g(x)\le0,\ h(x)=0,\qquad
\mathcal L(x,\lambda,\nu)=f(x)+\lambda^\top g(x)+\nu^\top h(x).
$$

$g_i(x)=0$ 的不等式称为激活约束，$\lambda_i\ge0$。等式乘子 $\nu$ 可取任意实数。

## 3. 核心原理

### 3.1 KKT 与对偶

在约束资格条件下，局部极小点满足

$$
\nabla_x\mathcal L=0,\quad g\le0,\quad h=0,\quad\lambda\ge0,\quad\lambda_i g_i=0.
$$

对偶函数 $q(\lambda,\nu)=\inf_x\mathcal L(x,\lambda,\nu)$ 给出原问题的下界。凸问题中 KKT 条件充分；Slater 条件可保证强对偶与相应乘子的存在。

### 3.2 等式二次规划

对于 $\min\tfrac12x^\top Qx+c^\top x$、$Ax=b$：

$$
\begin{bmatrix}Q&A^\top\\A&0\end{bmatrix}\begin{bmatrix}x\\\nu\end{bmatrix}=\begin{bmatrix}-c\\b\end{bmatrix}.
$$

$A$ 满行秩且 $Q$ 在 $\ker A$ 上正定时，该系统非奇异。一般不等式 QP 通过有效集或内点方法确定激活约束。

<details class="kt-derivation" id="kkt-example">
<summary>边界最优解算例</summary>

对 $\min_x\tfrac12(x-2)^2$、$x\le1$，有 $\mathcal L=\tfrac12(x-2)^2+\lambda(x-1)$。驻点条件 $x-2+\lambda=0$ 与边界 $x=1$ 给出 $\lambda=1$，满足全部 KKT 条件。

</details>

### 3.3 非线性约束求解

SQP 用拉格朗日函数的局部二次模型与线性化约束构造 QP。等式增广拉格朗日采用

$$
\mathcal L_\rho=f+\nu^\top h+\frac\rho2\|h\|^2,\qquad
\nu_{k+1}=\nu_k+\rho h(x_{k+1}).
$$

乘子迭代与内层求解共同减少约束误差；罚参数和内层精度影响数值表现。

## 4. 实现与应用

### 4.1 残差验收

分别检查原始可行残差、拉格朗日梯度、负乘子与互补残差。对约束进行单位缩放，并报告求解器停止状态及容差。

## 5. 分析与总结

KKT 将目标梯度与边界法向量联系起来。数值求解应同时满足可行性和驻点条件；凸性决定局部条件能否支持全局结论。

## 6. 关联与参考

[Boyd 与 Vandenberghe：Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)，第 5、10、11 章。

关联：[轨迹优化建模](../../02_classical-planning-control/03_trajectory-optimization/05_trajectory-optimization-formulation.md)。
