# 微分平坦性与最小 Snap

## 1. 概述

四旋翼的微分平坦性将位置与偏航轨迹连接到姿态、推力和力矩。最小 Snap 通过惩罚位置的四阶导数生成平滑参考，再利用平坦性映射恢复控制所需的状态与输入。

## 2. 基础与定义

### 2.1 平坦输出

采用世界坐标竖直向上、机体第三轴沿推力方向的约定。理想刚体模型为

$$
m\ddot p=fRe_3-mge_3.
$$

平坦输出取 $y=(p_x,p_y,p_z,\psi)$，其中 $\psi$ 为偏航角。位置及其高阶导数与偏航导数共同决定姿态和输入。

### 2.2 平滑目标

对固定段时间，最小 Snap 问题为

$$
\min_p\sum_{i=1}^M\int_0^{T_i}\|p_i^{(4)}(\tau)\|^2d\tau,
$$

并满足路径点、首尾导数与段间连续性条件。最小 jerk 使用三阶导数，对应另一种平滑目标。

## 3. 核心原理

### 3.1 平坦性映射

令 $a_f=\ddot p+ge_3$，则

$$
f=m\|a_f\|,\qquad b_3=\frac{a_f}{\|a_f\|}.
$$

用偏航构造 $b_{1c}=(\cos\psi,\sin\psi,0)^\top$，进而取

$$
b_2=\frac{b_3\times b_{1c}}{\|b_3\times b_{1c}\|},\quad
b_1=b_2\times b_3,\quad R=[b_1\ b_2\ b_3].
$$

该映射要求两处分母非零。对 $R$ 求导可得 $\omega=(R^\top\dot R)^\vee$；jerk 进入角速度，Snap 进入角加速度与力矩。

### 3.2 二次规划

取七次多项式 $p_i(\tau)=\sum_{r=0}^7 c_{ir}\tau^r$。单轴代价写为 $c^\top Qc$，每段矩阵元素为

$$
Q_{rs}=\begin{cases}
\dfrac{r!s!}{(r-4)!(s-4)!}\dfrac{T_i^{r+s-7}}{r+s-7},&r,s\ge4,\\
0,&\text{其余情况}.
\end{cases}
$$

位置与导数约束形成 $Ac=b$，等式约束问题的一阶条件为

$$
\begin{bmatrix}2Q&A^\top\\A&0\end{bmatrix}
\begin{bmatrix}c\\\nu\end{bmatrix}
=\begin{bmatrix}0\\b\end{bmatrix}.
$$

约束独立且 $Q$ 在 $A$ 的零空间上正定时，系数解唯一。

<details class="kt-derivation" id="snap-integral">
<summary>平滑矩阵推导</summary>

对 $r\ge4$，$d^4\tau^r/d\tau^4=r!\tau^{r-4}/(r-4)!$。将导数展开后逐项积分：

$$
\int_0^T\left(\sum_{r=4}^7\frac{r!}{(r-4)!}c_r\tau^{r-4}\right)^2d\tau
=\sum_{r,s=4}^7c_rc_s\frac{r!s!}{(r-4)!(s-4)!}
\frac{T^{r+s-7}}{r+s-7}.
$$

这给出 $Q$。三维各轴代价相加；涉及三维范数的约束会重新耦合各轴。

</details>

### 3.3 边界导数参数化

将分段端点导数写为 $d$，满足 $Ac=d$。利用选择矩阵 $C$ 合并共享端点与导数，有 $d=C[d_F;d_P]$，其中 $d_F$ 固定、$d_P$ 自由。代价矩阵变为

$$
R=C^\top A^{-\top}QA^{-1}C,\qquad
R_{PP}d_P=-R_{PF}d_F.
$$

无额外不等式且 $R_{PP}$ 正定时，这一线性系统给出自由导数。加入走廊或动力学约束后，应在同一参数空间继续求解约束优化。
<details class="kt-derivation" id="endpoint-jerk">
<summary>最小 Jerk 的端点映射</summary>

五次多项式采用端点次序 $d=[p(0),p(T),v(0),v(T),a(0),a(T)]^\top$，其映射为

$$
A(T)=\begin{bmatrix}
1&0&0&0&0&0\\
1&T&T^2&T^3&T^4&T^5\\
0&1&0&0&0&0\\
0&1&2T&3T^2&4T^3&5T^4\\
0&0&2&0&0&0\\
0&0&2&6T&12T^2&20T^3
\end{bmatrix}.
$$

三阶导数积分对应 $Q_{rs}=r!s!T^{r+s-5}/[(r-3)!(s-3)!(r+s-5)]$，$r,s\ge3$。这解释了部分工程代码中五次多项式与三阶导数矩阵的实际目标。求解时使用矩阵分解实现 $A^{-1}$ 的作用。

</details>

## 4. 实现与应用

按段时间构造 $Q$，再组装边界与连续性约束，求解系数并恢复姿态和输入。验收检查等式残差、轨迹导数极值及推力和角速度需求。

<details class="kt-derivation" id="snap-eigen">
<summary>最小 Snap 数值实现</summary>

下面的单段算例取 $T=1$、位置从 0 到 1、两端一至三阶导数为零。系数应为 $[0,0,0,0,35,-84,70,-20]$。代码依赖 Eigen。

```cpp
#include <Eigen/Dense>
#include <iostream>

int main()
{
  constexpr int n_coef = 8;
  constexpr int n_cons = 8;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n_coef, n_coef);

  Q(4, 4) = 576;
  Q(4, 5) = 1440;
  Q(4, 6) = 2880;
  Q(4, 7) = 5040;

  Q(5, 4) = 1440;
  Q(5, 5) = 4800;
  Q(5, 6) = 10800;
  Q(5, 7) = 20160;

  Q(6, 4) = 2880;
  Q(6, 5) = 10800;
  Q(6, 6) = 25920;
  Q(6, 7) = 50400;

  Q(7, 4) = 5040;
  Q(7, 5) = 20160;
  Q(7, 6) = 50400;
  Q(7, 7) = 100800;

  Eigen::MatrixXd Aeq(n_cons, n_coef);
  Aeq << 1, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 0, 6, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1,
         0, 1, 2, 3, 4, 5, 6, 7,
         0, 0, 2, 6, 12, 20, 30, 42,
         0, 0, 0, 6, 24, 60, 120, 210;

  Eigen::VectorXd beq(n_cons);
  beq << 0, 0, 0, 0, 1, 0, 0, 0;

  Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n_coef + n_cons, n_coef + n_cons);

  KKT.block(0, 0, n_coef, n_coef) = 2.0 * Q;
  KKT.block(0, n_coef, n_coef, n_cons) = Aeq.transpose();
  KKT.block(n_coef, 0, n_cons, n_coef) = Aeq;

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_coef + n_cons);
  rhs.segment(n_coef, n_cons) = beq;

  Eigen::VectorXd solution = KKT.fullPivLu().solve(rhs);

  Eigen::VectorXd p = solution.segment(0, n_coef);

  std::cout << "Polynomial coefficients:" << std::endl;
  std::cout << p.transpose() << std::endl;

  return 0;
}
```

</details>

## 5. 分析与总结

固定时间与线性边界约束给出二次规划结构；联合优化时间、姿态约束或避障会引入非线性。平滑代价控制高阶导数，完整可行性由显式约束与最终轨迹检查共同确定。

## 6. 关联与参考

- [Mellinger 与 Kumar：Minimum Snap Trajectory Generation and Control for Quadrotors](https://doi.org/10.1109/ICRA.2011.5980409)
- [Fast-Planner 官方实现](https://github.com/HKUST-Aerial-Robotics/Fast-Planner)
- [EGO-Planner 多项式初始化](https://github.com/ZJU-FAST-Lab/ego-planner/blob/master/src/planner/traj_utils/src/polynomial_traj.cpp)
- [MINCO 参数化](06_minco-parameterization.md)
