# 模型预测控制

## 1. 概述

MPC 在每个控制时刻求解有限时域优化，仅执行首个控制动作，再利用新状态滚动求解。预测模型连接未来动作与状态，约束和参考目标直接进入同一个优化问题。

## 2. 基础与定义

### 2.1 预测模型

线性离散系统为 $x_{k+1}=Ax_k+Bu_k$。在当前状态 $x_0$ 下优化 $N$ 个控制量，预测状态为 $x_1,\ldots,x_N$。

### 2.2 优化目标

调节问题的目标为

$$
J=\sum_{i=0}^{N-1}(x_i^\top Qx_i+u_i^\top Ru_i)+x_N^\top P_fx_N.
$$

跟踪问题使用状态与控制相对参考的偏差，并可加入控制增量代价。

## 3. 核心原理

### 3.1 预测矩阵

堆叠 $X=[x_1;\ldots;x_N]$、$U=[u_0;\ldots;u_{N-1}]$，得到

$$
X=Mx_0+CU,\qquad M_i=A^i,\quad
C_{ij}=\begin{cases}A^{i-1-j}B,&j<i,\\0,&j\ge i,\end{cases}
$$

其中 $i=1,\ldots,N$、$j=0,\ldots,N-1$。

### 3.2 二次规划

令 $\bar Q=\operatorname{diag}(Q,\ldots,Q,P_f)$，$\bar R=\operatorname{diag}(R,\ldots,R)$，省略与 $U$ 无关的项：

$$
\min_U\frac12U^\top HU+f^\top U,
\quad H=2(C^\top\bar QC+\bar R),\quad
f=2C^\top\bar QMx_0.
$$

输入边界直接约束 $U$；状态边界转换为

$$
x_{\min}-Mx_0\le CU\le x_{\max}-Mx_0.
$$

这里上下界均按预测时域堆叠。
<details class="kt-derivation" id="mpc-prediction">
<summary>两步预测矩阵</summary>

当 $N=2$，

$$
\begin{bmatrix}x_1\\x_2\end{bmatrix}
=\begin{bmatrix}A\\A^2\end{bmatrix}x_0+
\begin{bmatrix}B&0\\AB&B\end{bmatrix}
\begin{bmatrix}u_0\\u_1\end{bmatrix}.
$$

同一输入对较远状态的作用需要经过更多次 $A$，因此预测矩阵的块顺序必须与时间下标一致。

</details>

### 3.3 递归可行性与稳定性

终端代价、终端集合和局部控制律可共同建立递归可行性与稳定性。模型误差与扰动存在时，可使用约束收紧或鲁棒 MPC 保留裕量。求解状态、残差与截止时间均属于控制输出的有效性条件。

## 4. 实现与应用

每个周期读取状态，更新参考与约束，使用上轮移位解热启动 QP，检查求解状态及原始残差，再执行 $U$ 的第一段。

```text
X = predict(x0, U)
H, f, A_qp, lower, upper = assemble_problem(x0, reference)
solution = qp_solve(H, f, A_qp, lower, upper, warm_start)
if solution passes status, residual and deadline checks:
    apply first input
else:
    apply the predefined feasible fallback
```

可使用 OSQP 官方 MPC 示例作为完整求解器集成参考。验证预测矩阵、状态约束和终端权重后，再测试闭环扰动与不可行情形。

## 5. 分析与总结

MPC 能在预测时域内协调约束与目标，代价是在线计算和模型依赖。预测时域、离散精度与求解预算应匹配系统动态和控制周期。

## 6. 关联与参考

- [OSQP：MPC 示例](https://osqp.org/docs/examples/mpc.html)
- [Rawlings、Mayne 与 Diehl：Model Predictive Control](https://sites.engineering.ucsb.edu/~jbraw/mpc/)
- [线性二次调节器](02_lqr.md)
