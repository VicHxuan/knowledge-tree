# 轨迹优化问题

## 1. 概述

轨迹优化在参数化曲线空间中选择运动路径与时间，使任务代价最小，并满足边界、碰撞和动力学约束。参数化定义可搜索的轨迹族，目标函数表达偏好，约束定义允许的运动。

## 2. 基础与定义

### 2.1 决策变量

用 $p=\Phi(q,T)$ 表示轨迹，其中 $q$ 为几何或系数参数，$T_i>0$ 为段时间。状态与控制通过动力学积分或平坦性映射得到。

### 2.2 优化目标

常用目标由平滑、耗时与任务项组成：

$$
J(q,T)=\sum_i\int_0^{T_i}\ell(p_i,\dot p_i,\ldots)\,dt
+\rho\sum_iT_i+\ell_f(p(T_{\rm total})).
$$

各项权重将不同单位的量转换到统一的代价尺度。

## 3. 核心原理

### 3.1 约束结构

边界与连续性写为 $h(q,T)=0$。运动与空间限制写为

$$
g(p(t),\dot p(t),\ddot p(t),\ldots)\le0,\qquad t\in[0,T_{\rm total}].
$$

凸包或解析极值可把部分连续约束转成有限条件；数值采样则需要额外的间隔控制与可行性验证。

### 3.2 梯度传播

令多项式系数 $c=\mathcal M(q,T)$，外层代价为 $F(c,T)$，链式法则给出

$$
\nabla_qJ=\left(\frac{\partial c}{\partial q}\right)^\top\nabla_cF,
\qquad
\nabla_TJ=\nabla_TF+\left(\frac{\partial c}{\partial T}\right)^\top\nabla_cF.
$$

时间梯度同时包含积分区间、曲线导数和系数映射的变化。

### 3.3 约束处理

硬约束由可行性条件或约束保持参数化直接实施。软惩罚将违反量加入目标，例如 $\mu\sum_j\max(0,g_j)^2$。有限惩罚系数下，最终解应按原始约束计算残差。
<details class="kt-derivation" id="quadrature-gradient">
<summary>积分离散与时间梯度</summary>

令 $t=T_iu$，积分写为 $T_i\int_0^1\ell_i(T_iu)du$。用固定归一化节点 $u_k$ 和权重 $w_k$ 得

$$
J_i\approx T_i\sum_kw_k\ell_i(T_iu_k),\qquad
\frac{dJ_i}{dT_i}\approx\sum_kw_k\ell_i+
T_i\sum_kw_k\frac{d\ell_i}{dT_i}.
$$

第二项包含采样时刻与轨迹参数的全部依赖。以归一化时间实现可保持求积节点数量和索引固定。

</details>

## 4. 实现与应用

执行流程为：建立初值与时间 → 计算轨迹和约束 → 计算目标及梯度 → 更新参数 → 检查原始可行性与执行接口。记录目标各分量、最大约束违反量、求解耗时和初值敏感性。

## 5. 分析与总结

凸二次问题可获得全局最优解；非凸避障与联合时间优化通常依赖初值。比较算法时应统一轨迹边界、可行性容差、计算预算及失败处理。

## 6. 关联与参考

- [Boyd 与 Vandenberghe：Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
- [MIT：Trajectory Optimization](https://underactuated.mit.edu/trajopt.html)
- [MINCO 参数化](06_minco-parameterization.md)
