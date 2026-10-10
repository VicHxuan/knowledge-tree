---
title: "LQR 控制器"
sidebar:
  order: 2
---

## 2.1 数学公式

连续时间线性系统状态空间方程：
$$\dot{x}(t) = Ax(t) + Bu(t)$$

二次型代价函数：
$$J = \int_{0}^{\infty} \left( x(t)^T Q x(t) + u(t)^T R u(t) \right) dt$$
其中 $Q$ 为半正定状态权重矩阵，$R$ 为正定控制权重矩阵。

最优控制律为全状态反馈：
$$u(t) = -K x(t)$$
反馈增益矩阵 $K = R^{-1} B^T P$，其中 $P$ 是代数黎卡提方程（ARE）的唯一正定解：
$$A^T P + P A - P B R^{-1} B^T P + Q = 0$$

## 2.2 使用条件与性质

* **使用条件**：需要被控对象的精确线性数学模型。要求系统的所有状态变量均可观测。
* **性质**：

  * 属于最优控制，保证线性系统在无限时域内的渐近稳定性。
  * 适用于多输入多输出系统。
  * 无法在算法层面处理状态约束和控制约束。

## 2.3 C++ 实现代码

数字控制系统中通常使用离散LQR。以下代码使用Eigen库通过迭代法求解离散代数黎卡提方程（DARE）并计算反馈增益 $K$。

```cpp
#include <Eigen/Dense>
#include <iostream>

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
                     A.transpose() * P * B * temp.inverse() * B.transpose() * P * A;
            
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
            K = temp.inverse() * B.transpose() * P * A;
        } else {
            std::cerr << "DARE solution did not converge." << std::endl;
        }
    }

    Eigen::VectorXd calculate(const Eigen::VectorXd& current_state) {
        return -K * current_state;
    }
};
```