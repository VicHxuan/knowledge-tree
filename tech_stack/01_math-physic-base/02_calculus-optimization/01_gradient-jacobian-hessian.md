# 梯度、雅可比与海塞矩阵

## 1. 概述

导数把输入变化转换为输出变化。梯度描述标量目标的局部变化，雅可比连接向量映射，海塞矩阵描述曲率；三者构成优化与反向传播的基础。

## 2. 基础与定义

### 2.1 维度约定

$x\in\mathbb R^n$、$f(x)\in\mathbb R$、$y=g(x)\in\mathbb R^m$。梯度为列向量，雅可比 $J_g\in\mathbb R^{m\times n}$ 的第 $i,j$ 项为 $\partial g_i/\partial x_j$。

### 2.2 局部近似

$$
df=\nabla f^\top dx,\qquad dy=J_gdx.
$$

## 3. 核心原理

### 3.1 链式法则

对 $L(g(x))$，有

$$
\nabla_xL=J_g(x)^\top\nabla_yL.
$$

反向传播逐层计算雅可比转置与向量的乘积，以输出损失的梯度更新上游参数。

### 3.2 二阶展开

二阶连续可微时，$H=\nabla^2f$ 对称，且

$$
f(x+d)=f(x)+\nabla f^\top d+\frac12d^\top Hd+o(\|d\|^2).
$$

### 3.3 最小二乘导数

令 $f(x)=\tfrac12\|r(x)\|^2$，则

$$
\nabla f=J_r^\top r,\qquad H=J_r^\top J_r+\sum_i r_i\nabla^2r_i.
$$

Gauss–Newton 用 $J_r^\top J_r$ 近似曲率；残差较小或近线性时，该近似更准确。

<details class="kt-derivation" id="derivatives-least-squares">
<summary>最小二乘梯度推导</summary>

从 $df=r^\top dr=r^\top J_rdx$ 读出 $\nabla f=J_r^\top r$。再次求导时，导数分别作用于 $J_r$ 与 $r$，产生残差曲率项与 $J_r^\top J_r$ 项。

</details>

## 4. 实现与应用

### 4.1 方向导数检验

选单位方向 $v$，比较 $
abla f(x)^	op v$ 与 $[f(x+hv)-f(x-hv)]/(2h)$。在多个步长上检查误差；过小步长会放大浮点消减，过大步长会增加截断误差。

## 5. 分析与总结

推导先固定变量维度和梯度方向，再应用微分与链式法则。方向导数检验可直接验证优化所使用的梯度。

## 6. 关联与参考

[Boyd 与 Vandenberghe：Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)，附录中的微分公式。

关联：[无约束优化](./03_unconstrained-optimization.md)。
