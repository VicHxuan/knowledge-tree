# 数值优化与求解验证

## 1. 概述

数值优化把理论迭代转为有限精度计算。变量尺度、线性求解、步长控制和停止条件共同影响最终解的可靠性。

## 2. 基础与定义

### 2.1 尺度与误差

以 $x=Sz$ 对变量缩放，得到 $\nabla_zf=S^\top\nabla_xf$。线性系统的条件数衡量输入扰动对解的放大程度；残差衡量计算解对方程的满足程度。

## 3. 核心原理

### 3.1 线搜索

给定下降方向 $d$，Armijo 条件为

$$
f(x+\alpha d)\le f(x)+c_1\alpha\nabla f(x)^\top d,\qquad0<c_1<1.
$$

回溯从候选步长逐步减小，直至充分下降成立。Wolfe 条件进一步约束新点的方向导数。

### 3.2 信赖域

局部模型 $m(d)=f+g^\top d+\tfrac12d^\top Bd$ 在 $\|d\|\le\Delta$ 内求解。实际下降与预测下降的比值

$$
\rho=\frac{f(x)-f(x+d)}{m(0)-m(d)}
$$

用于接受或拒绝候选，并调整半径。

<details class="kt-derivation" id="numerics-scaling">
<summary>缩放后的二次目标</summary>

若 $f(x)=\tfrac12x^\top Qx+c^\top x$，则 $f(Sz)=\tfrac12z^\top S^\top QSz+(S^\top c)^\top z$。缩放改变数值曲率，而原变量中的目标和可行解通过 $x=Sz$ 对应。

</details>

## 4. 实现与应用

### 4.1 求解流程

检查输入单位与尺度 → 验证导数 → 分解线性系统 → 控制步长 → 检查原问题残差。

### 4.2 求解器接口

| 求解器 | 主要问题结构 | 应检查的量 |
| --- | --- | --- |
| OSQP | 凸二次规划 | 原始与对偶残差 |
| Ceres | 非线性最小二乘 | 残差块、雅可比、鲁棒核 |
| Ipopt | 平滑非线性约束优化 | 可行性、互补性与驻点残差 |

初值、容差、迭代上限和失败返回均属于复现配置。

## 5. 分析与总结

数值精度与模型精度分别评价。较小迭代增量可能来自收敛，也可能来自病态尺度或步长受限，应结合梯度和约束残差判断。

## 6. 关联与参考

[Boyd 与 Vandenberghe：Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)。

官方资料：[OSQP](https://osqp.org/docs/)、[Ceres](https://ceres-solver.org/nnls_tutorial.html)、[Ipopt](https://coin-or.github.io/Ipopt/)。
