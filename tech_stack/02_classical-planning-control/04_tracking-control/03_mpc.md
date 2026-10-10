# 3. MPC 控制器

## 3.1 数学公式

离散时间系统模型：
$$x_{k+1} = A x_k + B u_k$$

在 $k$ 时刻，求解有限预测步长 $N$ 内的二次规划（QP）问题：
$$\min_{u_{k|k}, \dots, u_{k+N-1|k}} \sum_{i=0}^{N-1} \left( x_{k+i|k}^T Q x_{k+i|k} + u_{k+i|k}^T R u_{k+i|k} \right) + x_{k+N|k}^T P_f x_{k+N|k}$$

受限于以下约束条件：
$$x_{k+i+1|k} = A x_{k+i|k} + B u_{k+i|k} \quad (系统动力学约束)$$
$$x_{min} \le x_{k+i|k} \le x_{max} \quad (状态边界约束)$$
$$u_{min} \le u_{k+i|k} \le u_{max} \quad (控制边界约束)$$

## 3.2 使用条件与性质

* **使用条件**：需要已知系统的线形或非线性模型。需要较高的计算资源以满足实时在线求解优化问题的要求。
* **性质**：

  * 基于滚动时域机制，仅执行优化序列的第一个控制输入。
  * 显式处理硬约束，保证系统在安全边界内运行。
  * 具备前瞻性，能够针对未来设定值的变化提前做出响应。

## 3.3 C++ 实现代码

完整的MPC通常需要依赖第三方QP求解器（如OSQP、qpOASES）。以下代码展示MPC的结构框架及将控制问题转化为标准QP形式的过程。

```cpp
#include <Eigen/Dense>
#include <vector>

// 假设存在一个外部QP求解器接口
// 求解形式: min 0.5 * U^T * H * U + f^T * U, s.t. lb <= U <= ub
class QPSolver {
public:
    Eigen::VectorXd solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& f,
                          const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
        // 调用外部库求解二次规划，此处返回占位结果
        return Eigen::VectorXd::Zero(f.size()); 
    }
};

class MPCController {
private:
    Eigen::MatrixXd A, B, Q, R;
    int N; // 预测步长
    Eigen::VectorXd u_min, u_max;
    QPSolver solver;

public:
    MPCController(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, 
                  const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, 
                  int N, const Eigen::VectorXd& u_min, const Eigen::VectorXd& u_max) 
        : A(A), B(B), Q(Q), R(R), N(N), u_min(u_min), u_max(u_max) {}

    Eigen::VectorXd calculate(const Eigen::VectorXd& current_state) {
        int nx = A.rows();
        int nu = B.cols();

        // 构造增广矩阵 M 和 C (使 X = M * x0 + C * U)
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N * nx, nx);
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(N * nx, N * nu);
        
        Eigen::MatrixXd A_pow = Eigen::MatrixXd::Identity(nx, nx);
        for (int i = 0; i < N; ++i) {
            A_pow = A_pow * A;
            M.block(i * nx, 0, nx, nx) = A_pow;
            
            Eigen::MatrixXd A_pow_B = B;
            for (int j = i; j >= 0; --j) {
                C.block(i * nx, (i - j) * nu, nx, nu) = A_pow_B;
                A_pow_B = A * A_pow_B;
            }
        }

        // 构造增广权重矩阵 Q_bar 和 R_bar
        Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(N * nx, N * nx);
        Eigen::MatrixXd R_bar = Eigen::MatrixXd::Zero(N * nu, N * nu);
        for (int i = 0; i < N; ++i) {
            Q_bar.block(i * nx, i * nx, nx, nx) = Q;
            R_bar.block(i * nu, i * nu, nu, nu) = R;
        }

        // 转换为标准QP形式的 H 和 f
        Eigen::MatrixXd H = 2 * (C.transpose() * Q_bar * C + R_bar);
        Eigen::VectorXd f = 2 * C.transpose() * Q_bar * M * current_state;

        // 构造控制约束边界
        Eigen::VectorXd lb = u_min.replicate(N, 1);
        Eigen::VectorXd ub = u_max.replicate(N, 1);

        // 求解QP问题
        Eigen::VectorXd U = solver.solve(H, f, lb, ub);

        // 仅返回预测序列的第一个控制动作 (滚动优化)
        return U.head(nu);
    }
};
```
