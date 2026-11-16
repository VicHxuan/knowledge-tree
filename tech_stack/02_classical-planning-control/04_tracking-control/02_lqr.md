# 线性二次调节器

## 1. 概述

LQR 在给定线性动力学下最小化状态与控制的二次代价。价值函数的二次结构将最优控制转化为 Riccati 方程，得到直接作用于状态误差的线性反馈。

## 2. 基础与定义

### 2.1 系统与目标

连续系统 $\dot x=Ax+Bu$ 的无限时域目标为

$$
J=\int_0^\infty(x^\top Qx+u^\top Ru)dt,
\qquad Q\succeq0,\ R\succ0.
$$

跟踪平衡点时，$x,u$ 表示相对平衡状态与输入的偏差。

### 2.2 价值函数

最优剩余代价具有形式 $V(x)=x^\top Px$，其中 $P$ 为对称半正定矩阵。其梯度为 $\nabla_xV=2Px$。

## 3. 核心原理

### 3.1 连续时间控制律

最优控制和代数 Riccati 方程为

$$
u=-Kx,\quad K=R^{-1}B^\top P,
\qquad A^\top P+PA-PBR^{-1}B^\top P+Q=0.
$$

若 $(A,B)$ 可稳定、$(Q^{1/2},A)$ 可检测，则存在唯一的稳定化半正定解，$A-BK$ 为 Hurwitz 矩阵。
<details class="kt-derivation" id="lqr-hjb">
<summary>Riccati 方程推导</summary>

HJB 方程为

$$
0=\min_u\{x^\top Qx+u^\top Ru+2x^\top P(Ax+Bu)\}.
$$

对 $u$ 求导得 $2Ru+2B^\top Px=0$。代回并对所有 $x$ 比较二次型系数，即得到 Riccati 方程。沿最优闭环有 $\dot V=-x^\top(Q+K^\top RK)x$。

</details>

### 3.2 离散时间控制律

对 $x_{k+1}=Ax_k+Bu_k$，有

$$
K=(R+B^\top PB)^{-1}B^\top PA,
$$

$$
P=Q+A^\top PA-A^\top PB(R+B^\top PB)^{-1}B^\top PA.
$$

有限时域问题从终端权重反向递推 $P_k$；无限时域使用稳定化不动点。

### 3.3 轨迹跟踪

沿参考轨迹线性化可得时变 $A_k,B_k$，反馈写为 $u_k=u_k^{\rm ref}-K_k(x_k-x_k^{\rm ref})$。状态估计提供反馈所需状态，控制输入的饱和需另行纳入设计。

## 4. 实现与应用

根据工作点建立并离散化模型，按物理尺度设置 $Q,R$，求解 Riccati 方程并检查残差和闭环特征值，再评估模型偏差与饱和工况。

<details class="kt-derivation" id="lqr-eigen">
<summary>离散 Riccati 迭代</summary>

以下 Eigen 示例使用定点迭代。输入矩阵须满足维数、对称性及稳定化条件；未在迭代预算内收敛时抛出异常。

```cpp
#include <Eigen/Dense>
#include <stdexcept>

class LQRController {
private:
    Eigen::MatrixXd K;

    bool solveDARE(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                   const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                   Eigen::MatrixXd& P) {
        P = Q;
        Eigen::MatrixXd P_next;
        double tolerance = 1e-4;
        int max_iter = 1000;

        for (int i = 0; i < max_iter; ++i) {
            Eigen::MatrixXd temp = R + B.transpose() * P * B;
            P_next = Q + A.transpose() * P * A -
                     A.transpose() * P * B * temp.ldlt().solve(B.transpose() * P * A);

            if ((P_next - P).cwiseAbs().maxCoeff() < tolerance) {
                P = P_next;
                return true;
            }
            P = P_next;
        }
        return false;
    }

public:
    LQRController(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                  const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
        Eigen::MatrixXd P;
        if (solveDARE(A, B, Q, R, P)) {
            Eigen::MatrixXd temp = R + B.transpose() * P * B;
            K = temp.ldlt().solve(B.transpose() * P * A);
        } else {
            throw std::runtime_error("DARE solution did not converge");
        }
    }

    Eigen::VectorXd calculate(const Eigen::VectorXd& current_state) {
        return -K * current_state;
    }
};
```

</details>

## 5. 分析与总结

LQR 给出二次目标下的状态反馈，并可作为 MPC 的终端控制器。权重选择体现状态偏差与控制强度的权衡；线性化范围与执行器能力决定实际适用区间。

## 6. 关联与参考

- [MIT：Linear Quadratic Regulators](https://underactuated.mit.edu/lqr.html)
- [模型预测控制](03_mpc.md)
- [Hamilton 函数与 Pontryagin 原理](../../01_math-physic-base/05_optimal-control-foundations/02_hamiltonian-pontryagin.md)
