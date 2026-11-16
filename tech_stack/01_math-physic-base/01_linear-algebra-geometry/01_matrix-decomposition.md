# 矩阵分解与最小二乘

## 1. 概述

矩阵分解将线性映射拆成便于解释和计算的结构。特征分解描述不变方向，奇异值分解描述伸缩，QR 与 Cholesky 分解支持线性方程和最小二乘求解。

## 2. 基础与定义

### 2.1 线性映射

设 $A\in\mathbb R^{m\times n}$，列向量 $x\in\mathbb R^n$，$Ax$ 为输出。秩描述独立输出方向的数量；正交矩阵满足 $Q^\top Q=I$，保持欧氏长度。

### 2.2 最小二乘目标

含噪线性观测 $b\approx Ax$ 的估计问题为

$$
\min_x\frac12\|Ax-b\|_2^2,\qquad \nabla f=A^\top(Ax-b).
$$

## 3. 核心原理

### 3.1 特征分解与奇异值分解

实对称矩阵满足 $A=Q\Lambda Q^\top$，$\Lambda$ 的对角元素为特征值。一般矩阵采用

$$
A=U\Sigma V^\top.
$$

$V$ 的列给出输入方向，奇异值给出伸缩比例，$U$ 的列给出输出方向。保留最大的 $r$ 个奇异值可得到 Frobenius 范数下的最佳秩 $r$ 近似。

### 3.2 QR 与 Cholesky 分解

满列秩矩阵的薄 QR 分解为 $A=QR$，最小二乘解满足 $Rx=Q^\top b$。对称正定矩阵可分解为 $A=LL^\top$，通过两次三角求解计算 $Ax=b$。

### 3.3 伪逆与秩亏

对非零奇异值取倒数得到 $\Sigma^+$，伪逆为 $A^+=V\Sigma^+U^\top$。$x=A^+b$ 是最小范数最小二乘解。数值秩由奇异值相对于尺度和精度的阈值判定。

<details class="kt-derivation" id="matrix-normal-equations">
<summary>正规方程与条件数</summary>

令梯度为零，得到 $A^\top Ax=A^\top b$。满列秩时解唯一；若采用二范数，则 $\kappa_2(A^\top A)=\kappa_2(A)^2$。QR 直接在 $A$ 上计算，可避免形成正规方程造成的条件数平方；SVD 还可显式处理秩亏。

</details>

## 4. 实现与应用

### 4.1 求解流程

1. 根据矩阵结构与秩选择 QR、SVD 或 Cholesky。
2. 通过分解求解方程，复用同一矩阵的分解处理多个右端项。
3. 检查相对残差、最小奇异值以及输入单位。

### 4.2 数值验证

构造已知解 $x_*$，计算 $b=Ax_*$；分别比较方程残差与解误差。对病态矩阵，小残差可能对应明显的解误差。

## 5. 分析与总结

分解方法的选择由矩阵结构、秩和误差敏感度决定。最小二乘提供统计拟合目标，数值分解决定该目标如何稳定求解。

## 6. 关联与参考

[Boyd 与 Vandenberghe：Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)，附录中的线性代数与数值求解。

关联：[高斯估计](../03_probability-statistics/02_gaussian-estimation.md)、[数值优化](../02_calculus-optimization/05_numerical-optimization.md)。
