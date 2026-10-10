---
title: "FAST-LIO 系列"
sidebar:
  order: 2
---

# 1. FAST-LIO系列主框架 (基于 FAST-LIO2)

## 1.1 数学定义与系统构造

FAST-LIO 系列的核心是一个基于流形（Manifold）的紧耦合迭代扩展卡尔曼滤波（iEKF）框架。FAST-LIO2 在此基础上引入了直接法（无需特征提取）与增量式空间数据结构。

系统状态定义在流形 $\mathcal{M} = SO(3) \times \mathbb{R}^{15} \times SO(3) \times \mathbb{R}^3$ 上，包含IMU位姿、速度、零偏以及LiDAR-IMU外参：
$$ x \triangleq \begin{bmatrix} ^G R_I^T & ^G p_I^T & ^G v_I^T & b_\omega^T & b_a^T & ^G g^T & ^I R_L^T & ^I p_L^T \end{bmatrix}^T $$

系统运行包含以下核心步骤：

1. **前向传播 (Forward Propagation)**：利用高频 IMU 离散运动学模型，计算先验状态估计 $\hat{x}_k$ 和协方差 $\hat{P}_k$。
2. **反向传播 (Backward Propagation)**：根据一帧 Scan 中每个点的时间戳，利用 IMU 积分推算局部相对位姿，将所有点投影到 Scan 结束时刻，消除运动畸变。
3. **直接法残差构建 (Measurement Model)**：将去畸变后的点 $^L p_j$ 投影至全局坐标系，并在地图中搜索最近邻点拟合局部平面，构建点到面的距离残差：
   $$ z_j = u_j^T \left( ^G\hat{T}_{I_k} ^I\hat{T}_L ^L p_j - ^G q_j \right) $$
   其中 $u_j$ 为局部平面法向量， $q_j$ 为平面上一点。
4. **状态更新 (Iterated Update)**：利用一阶泰勒展开线性化残差方程，结合状态维度的卡尔曼增益公式迭代求解最大后验估计（MAP）。

## 1.2 算法基本性质

1. **状态维度的卡尔曼增益**
   传统 EKF 的卡尔曼增益计算需对观测维度矩阵求逆 $\mathcal{O}(m^3)$。由于 LiDAR 点数 $m$ 极大，传统方法必须降采样。FAST-LIO 利用矩阵求逆引理，将公式等价转换为对状态维度求逆 $\mathcal{O}(n^3)$：
   $$ K = (H^T R^{-1} H + P^{-1})^{-1} H^T R^{-1} $$
   彻底解除了计算量对测量点数的依赖。
2. **无特征提取的直接映射**
   FAST-LIO2 直接将原始密集点云配准至地图，保留了环境中的微小特征，显著提升了在缺乏几何特征（如狭长走廊）或小视场角（固态雷达）环境中的鲁棒性。

---

# 2. 算法演进与改进

## 2.1 FAST-LIO：理论奠基

* **核心贡献**：确立了紧耦合 iEKF 架构；提出了反向传播运动补偿机制；从数学上推导并证明了状态维度卡尔曼增益公式。
* **局限性**：仍依赖传统的边缘点/平面点特征提取，环境适应性受限；且在每次更新地图时计算开销较大。

## 2.2 FAST-LIO2：数据结构与直接法突破

* **核心贡献**：移除特征提取模块，直接使用原始点云。
* **关键改进 (ikd-Tree)**：提出了增量式 k-d 树 `ikd-Tree`。支持点级别插入、树内降采样和基于惰性标签（Lazy Label）的包围盒删除。通过双线程并行重建（Parallel Re-build）维持树的平衡性，将建图时间压缩至极致，支持 100Hz 的里程计输出。

## 2.3 Faster-LIO：体素化极速检索

* **核心贡献**：用增量稀疏体素（iVox）替代了复杂的树状结构，追求极致的运行速度。
* **关键改进 (iVox / iVox-PHC)**：抛弃了严格的 k-NN 搜索，采用稀疏哈希体素管理点云。体素内部提供两种存储方式：线性数组（Linear iVox）和伪希尔伯特曲线（iVox-PHC）。利用 LRU 缓存策略维护局部地图。由于体素查询的时间复杂度接近 $\mathcal{O}(1)$ 且极易并行化，算法在主流 CPU 上达到了 1000-2000Hz 的惊人处理速度。

---

# 3. 主框架伪代码

```text
Algorithm FAST_LIO2_Process(x_prev, P_prev, IMU_data, LiDAR_scan):
    // 1. IMU 前向传播
    x_pred, P_pred = IMU_Integrate(x_prev, P_prev, IMU_data)
    
    // 2. 运动畸变补偿
    For each point p_j in LiDAR_scan:
        p_j_undistorted = Backward_Propagation(p_j, IMU_data)
        
    // 3. 迭代卡尔曼滤波更新
    x_curr = x_pred
    For iteration = 1 to max_iter:
        H_matrix = [], z_vector = []
        
        For each point p_j in LiDAR_scan:
            p_global = Transform(p_j_undistorted, x_curr)
            
            // k-NN 搜索 (ikd-Tree 或 iVox)
            neighbors = Spatial_Structure.Search_kNN(p_global, k=5)
            plane_normal, plane_point = Fit_Plane(neighbors)
            
            // 计算残差与雅可比
            z_j = plane_normal^T * (p_global - plane_point)
            H_j = Compute_Jacobian(plane_normal, p_j_undistorted, x_curr)
            
            H_matrix.append(H_j)
            z_vector.append(z_j)
            
        // FAST-LIO 专属低维卡尔曼增益
        P_inv = Inverse(P_pred)
        R_inv = Inverse(Measurement_Noise_Covariance)
        K = Inverse(H_matrix^T * R_inv * H_matrix + P_inv) * H_matrix^T * R_inv
        
        // 状态更新 (流形加法)
        delta_x = -K * z_vector - (I - K * H_matrix) * (x_curr (-) x_pred)
        x_curr = x_curr (+) delta_x
        
        If norm(delta_x) < threshold: 
            Break
            
    // 4. 更新协方差与地图
    P_curr = (I - K * H_matrix) * P_pred
    Registered_Points = Transform(LiDAR_scan, x_curr)
    Spatial_Structure.Insert(Registered_Points) // ikd-Tree 或 iVox 更新
    Spatial_Structure.Delete_Out_of_FOV()       // LRU 缓存或包围盒删除
    
    Return x_curr, P_curr
```

---

# 4. 主框架 C++ 代码实现

以下代码提炼了 FAST-LIO 框架中最核心的 iEKF 状态更新逻辑，展示了维度降低后的卡尔曼增益求解过程。为保持清晰，省略了流形库的依赖定义。

```cpp
#include <Eigen/Dense>
#include <vector>

// 状态维度的极简抽象 (实际系统中为24维流形)
const int STATE_DIM = 18; 

struct Point3D { Eigen::Vector3d pt; };

class FastLioIEKF {
private:
    Eigen::VectorXd x; // 系统状态
    Eigen::MatrixXd P; // 状态协方差
    Eigen::MatrixXd Q; // 过程噪声
    double R_noise;    // 测量噪声标量

public:
    FastLioIEKF(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, double R) 
        : x(x0), P(P0), R_noise(R) {}

    // IMU 前向传播抽象
    void predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& W) {
        // x = f(x, u) 流形上的离散积分 (省略具体物理公式)
        P = F * P * F.transpose() + W * Q * W.transpose();
    }

    // iEKF 核心更新逻辑
    void update(const std::vector<Point3D>& lidar_points, 
                const std::vector<Eigen::Vector3d>& plane_normals, 
                const std::vector<Eigen::Matrix<double, 1, STATE_DIM>>& jacobians,
                const std::vector<double>& residuals) {
        
        int m = lidar_points.size();
        if (m == 0) return;

        Eigen::VectorXd x_pred = x;
        Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(STATE_DIM);

        // 迭代求解 MAP
        for (int iter = 0; iter < 5; ++iter) {
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, STATE_DIM);
            Eigen::VectorXd Z = Eigen::VectorXd::Zero(m);

            // 聚合当前迭代下的观测矩阵
            for (int i = 0; i < m; ++i) {
                H.row(i) = jacobians[i];
                Z(i) = residuals[i];
            }

            // FAST-LIO 核心：计算维度降低的卡尔曼增益
            // K = (H^T * R^{-1} * H + P^{-1})^{-1} * H^T * R^{-1}
            Eigen::MatrixXd P_inv = P.inverse();
            Eigen::MatrixXd H_T_R_inv = H.transpose() / R_noise;
            Eigen::MatrixXd S_state_dim = H_T_R_inv * H + P_inv; 
            
            Eigen::MatrixXd K = S_state_dim.inverse() * H_T_R_inv;

            // 计算状态更新量 (此处的减法在实际中应为流形 BoxMinus)
            Eigen::VectorXd x_diff = x - x_pred; 
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
            
            delta_x = -K * Z - (I - K * H) * x_diff;
            
            // 状态流形更新 (此处的加法在实际中应为流形 BoxPlus)
            x = x + delta_x;

            if (delta_x.norm() < 1e-4) {
                break;
            }
        }

        // 迭代收敛后更新后验协方差
        // 需重新计算最终状态下的 H 矩阵，此处简化表示
        Eigen::MatrixXd H_final = Eigen::MatrixXd::Zero(m, STATE_DIM);
        for (int i = 0; i < m; ++i) H_final.row(i) = jacobians[i];
        
        Eigen::MatrixXd H_T_R_inv = H_final.transpose() / R_noise;
        Eigen::MatrixXd K = (H_T_R_inv * H_final + P.inverse()).inverse() * H_T_R_inv;
        
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        P = (I - K * H_final) * P;
    }
};
```
