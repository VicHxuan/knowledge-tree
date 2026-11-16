# 数学优化基础

## 1. 概述

优化问题由目标、变量和可行域构成。最优性条件说明解应满足什么关系，问题结构决定可采用的求解方法，数值算法通过迭代逼近这些条件。

## 2. 基础与定义

### 2.1 优化问题

$$
\min_x f(x)\quad\text{s.t.}\quad g_i(x)\le0,\quad h_j(x)=0.
$$

局部最优比较邻域内可行点，全局最优比较整个可行域。凸问题要求目标和不等式约束凸、等式约束仿射。

### 2.2 最优性条件

无约束内部局部极小点满足 $\nabla f(x_*)=0$。二阶连续可微时还需 $\nabla^2f(x_*)\succeq0$；梯度为零且海塞正定是严格局部极小的充分条件。

## 3. 核心原理

### 3.1 约束与乘子

拉格朗日函数为 $\mathcal L=f+\lambda^\top g+\nu^\top h$。在适当约束资格条件下，局部最优点满足 KKT 条件：

$$
\nabla_x\mathcal L=0,\quad g\le0,\quad h=0,\quad\lambda\ge0,\quad\lambda_i g_i=0.
$$

凸问题的 KKT 解是全局最优解；Slater 等条件可保证乘子存在及强对偶。约束激活时乘子可为零，严格正乘子意味着该约束激活。

### 3.2 二次规划

$$
\min_x\frac12x^\top Qx+c^\top x\quad\text{s.t.}\quad Ax=b,\ Gx\le h.
$$

$Q\succeq0$ 给出凸 QP；目标在可行方向上严格凸时，存在的最优解唯一。可行性、有界性与唯一性分别检查。

### 3.3 算法分类

| 结构 | 主要方法 | 专项入口 |
| --- | --- | --- |
| 无约束平滑目标 | 梯度、牛顿、拟牛顿 | [无约束优化](./03_unconstrained-optimization.md) |
| 约束目标 | 有效集、内点、SQP、增广拉格朗日 | [约束优化](./04_constrained-optimization-kkt.md) |
| 数值实现 | 缩放、线搜索、信赖域、停止准则 | [数值优化](./05_numerical-optimization.md) |

<details class="kt-derivation" id="optimization-regularization">
<summary>范数约束与正则化</summary>

对 $\|x\|^2\le C$，最优乘子 $\lambda_*$ 对应拉格朗日目标 $f(x)+\lambda_*\|x\|^2-\lambda_*C$。满足强对偶等条件时，可通过相应乘子联系约束问题和惩罚问题；任意指定的 $\lambda$ 对应其自身的权衡，需重新确定对应约束半径。

</details>

## 4. 实现与应用

### 4.1 建模与求解

1. 明确变量单位、目标和硬约束。
2. 判定凸性、稀疏性与可微性。
3. 选择方法并检查梯度、可行性与停止残差。
4. 将解代回原始模型验证。

## 5. 分析与总结

目标下降、约束满足与最优性是三个独立的检查量。问题结构提供可证明的性质，数值结果还需报告容差与残差。

## 6. 关联与参考

[Boyd 与 Vandenberghe：Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)。

关联：[轨迹优化建模](../../02_classical-planning-control/03_trajectory-optimization/05_trajectory-optimization-formulation.md)。
