# GCOPTER-MINCO 框架

## 1. 概述（Overview）

> 待补充。

## 2. 基础与定义（Foundations）

> 待补充。

## 3. 核心原理（Core Principles）

### 1. GCOPTER-MINCO 轨迹规划框架

#### 1.1 数学定义与系统构造

GCOPTER 提出了一种能够将严格的几何约束与连续时间动力学约束转化为无约束优化的多旋翼轨迹生成框架。该框架的底层基石是 MINCO 轨迹类。

MINCO 定义为在给定中间航路点序列 $q$ 与对应时间分配 $T$ 下，控制代价积分严格最小的多段多项式轨迹。通过推导多段最优控制的必要与充分条件，MINCO 的多项式系数矩阵 $c$ 由空间参数 $q$ 和时间参数 $T$ 唯一确定。

系统将轨迹规划问题重构为对 $q$ 和 $T$ 的联合优化，并通过以下机制实现降维与无约束化：

1. **时间约束消除**：利用指数映射等微分同胚函数，将定义域受限的时间变量 $T > 0$ 映射为全空间无约束变量 $\tau$。
2. **空间几何约束消除**：利用逆立体投影与正交投影的复合平滑满射，将安全飞行走廊的凸多面体或球体几何约束彻底消除。空间点 $q$ 被等价映射为全空间无约束变量 $\xi$。
3. **时间积分惩罚泛函**：对于连续时间的动力学约束（如最大速度、最大加速度、推力极限），系统将其转化为沿轨迹的积分惩罚项 $I_G[p]$。通过数值求积，在有限控制点数量下实现高保真的连续时间约束。

最终的优化问题转化为在无约束欧氏空间中寻找最优的 $\xi$ 和 $\tau$：
$$ \min_{\xi, \tau} J(q(\xi), T(\tau)) + I_G(q(\xi), T(\tau)) $$

#### 1.2 算法性质与证明

1. **时空变形的线性计算复杂度**
   求解 MINCO 系数矩阵以及计算目标函数关于 $q$ 和 $T$ 的梯度，其时间与空间复杂度均为 $O(M)$，其中 $M$ 为轨迹段数。
   证明：MINCO 的最优化条件构成了一个结构特殊的带状稀疏线性方程组 $Mc = b$。矩阵 $M$ 的非奇异性由最优解的唯一性保证。利用带状 PLU 分解，前向求解系数 $c$ 与后向传播伴随梯度 $M^T G = \partial K / \partial c$ 均可在严格的线性时间内完成，彻底避免了稠密矩阵求逆。

2. **约束消除的局部极小值保真性**
   通过微分同胚或平滑满射引入的代理变量 $\xi$ 和 $\tau$，不会在无约束空间中凭空引入虚假的局部极小值，也不会消除原约束空间中已有的局部极小值。
   证明：设原函数为 $F(x)$，经微分同胚映射 $G$ 后的新函数为 $H(y) = F(G(y))$。求导可知 $\nabla H(y) = J(y)^T \nabla F(x)$。由于雅可比矩阵 $J$ 在映射域内处处非奇异，$\nabla F(x) = 0$ 与 $\nabla H(y) = 0$ 严格等价。同理，其海森矩阵在极值点处合同，严格保留了一阶和二阶最优性条件。

### 2. 算法伪代码

```text
Algorithm GCOPTER_MINCO(Start, Goal, Safe_Flight_Corridors):
    // 1. 初始化无约束变量
    T_init = Estimate_Time()
    q_init = Extract_Path_Waypoints(Safe_Flight_Corridors)
    tau = Inverse_Time_Diffeomorphism(T_init)
    xi = Inverse_Spatial_Surjection(q_init, Safe_Flight_Corridors)
    
    // 2. L-BFGS 无约束迭代优化
    While not Converged:
        // 正向映射
        T = Time_Diffeomorphism(tau)
        q = Spatial_Surjection(xi, Safe_Flight_Corridors)
        
        // 求解 MINCO 轨迹系数 (带状 PLU 分解, O(M) 复杂度)
        c = Solve_Banded_System(q, T)
        
        // 评估目标函数与连续时间惩罚
        Cost_effort = Evaluate_Control_Effort(c, T)
        Cost_penalty = Evaluate_Integral_Penalty(c, T)
        Total_Cost = Cost_effort + Cost_penalty
        
        // 反向梯度传播
        Grad_c = Compute_Gradient_wrt_c()
        Grad_T_direct = Compute_Gradient_wrt_T_direct()
        
        // MINCO 核心 O(M) 伴随梯度计算
        G_matrix = Solve_Banded_System_Transpose(Grad_c)
        Grad_q = Extract_Spatial_Gradient(G_matrix)
        Grad_T_implicit = Extract_Temporal_Gradient(G_matrix, c, T)
        
        Total_Grad_T = Grad_T_direct + Grad_T_implicit
        
        // 映射层链式求导
        Grad_xi = Apply_Spatial_Jacobian(Grad_q, xi)
        Grad_tau = Apply_Temporal_Jacobian(Total_Grad_T, tau)
        
        // 更新无约束变量
        xi, tau = L_BFGS_Update(Total_Cost, Grad_xi, Grad_tau)
        
    Return Construct_Trajectory(xi, tau)
```

## 4. 实现与应用（Implementation & Applications）

### 3. C++代码实现

以下代码展示 GCOPTER 中最核心的 MINCO 时空变形梯度反向传播逻辑。为了突出 $O(M)$ 复杂度的带状矩阵求解思想，忽略了外部 L-BFGS 和微分同胚映射的封包细节。

```cpp
#include <Eigen/Dense>
#include <vector>

class MincoTrajectory {
private:
    int num_segments;
    int state_dim;
    int poly_order;
    
    // 预分解的带状矩阵抽象表示，用于 O(M) 求解
    // 实际实现中依赖专门的 BandedMatrix 类进行 PLU 分解
    Eigen::MatrixXd banded_M; 

public:
    MincoTrajectory(int M, int dim, int order) 
        : num_segments(M), state_dim(dim), poly_order(order) {}

    // 给定 q 和 T 求解轨迹系数 c
    Eigen::MatrixXd computeCoefficients(const std::vector<Eigen::VectorXd>& q, 
                                        const std::vector<double>& T) {
        Eigen::MatrixXd b = constructBoundaryVector(q);
        banded_M = constructBandedMatrix(T);
        
        // 前向带状求解: M * c = b
        // 复杂度 O(M)
        Eigen::MatrixXd c = banded_M.lu().solve(b); 
        return c;
    }

    // 根据高层目标函数对 c 和 T 的梯度，计算对 q 和 T 的全梯度
    void propagateGradients(const Eigen::MatrixXd& c,
                            const std::vector<double>& T,
                            const Eigen::MatrixXd& grad_c,     // ∂J / ∂c
                            const std::vector<double>& grad_T, // ∂J / ∂T (直接部分)
                            std::vector<Eigen::VectorXd>& final_grad_q,
                            std::vector<double>& final_grad_T) {
        
        // 反向带状求解: M^T * G = grad_c
        // 复杂度 O(M)，复用正向求解时的 LU 分解
        Eigen::MatrixXd G = banded_M.transpose().lu().solve(grad_c);
        
        final_grad_q.assign(num_segments - 1, Eigen::VectorXd::Zero(state_dim));
        final_grad_T = grad_T;

        for (int i = 0; i < num_segments; ++i) {
            // 提取空间梯度 ∂J / ∂qi
            if (i < num_segments - 1) {
                // 依据论文 (64) 式，从 G 矩阵的特定列抽取对 q 的梯度
                int q_idx = getSpatialIndex(i); 
                final_grad_q[i] = G.row(q_idx).transpose();
            }

            // 计算时间梯度的隐式部分 ∂J / ∂Ti
            // 依据论文 (68) 式: -Tr(G_i^T * (∂E_i / ∂T_i) * c_i)
            Eigen::MatrixXd dE_dT = computeDerivativeE_wrt_T(T[i]);
            Eigen::MatrixXd c_i = extractSegmentCoeffs(c, i);
            Eigen::MatrixXd G_i = extractSegmentG(G, i);
            
            double implicit_grad_T = -(G_i.transpose() * dE_dT * c_i).trace();
            final_grad_T[i] += implicit_grad_T;
        }
    }

private:
    // 以下为辅助矩阵构造抽象函数
    Eigen::MatrixXd constructBoundaryVector(const std::vector<Eigen::VectorXd>& q) { return Eigen::MatrixXd(); }
    Eigen::MatrixXd constructBandedMatrix(const std::vector<double>& T) { return Eigen::MatrixXd(); }
    int getSpatialIndex(int i) { return 0; }
    Eigen::MatrixXd computeDerivativeE_wrt_T(double t) { return Eigen::MatrixXd(); }
    Eigen::MatrixXd extractSegmentCoeffs(const Eigen::MatrixXd& c, int i) { return Eigen::MatrixXd(); }
    Eigen::MatrixXd extractSegmentG(const Eigen::MatrixXd& G, int i) { return Eigen::MatrixXd(); }
};
```

## 5. 分析与总结（Analysis & Summary）

> 待补充。

## 6. 关联与参考（Related Topics & References）

> 待补充。
