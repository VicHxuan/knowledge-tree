---
title: "卡尔曼滤波系列（BF/KF/EKF/PF）"
sidebar:
  order: 1
---

# 1. 贝叶斯滤波

## 1.1 数学定义构造

给定系统状态空间，贝叶斯滤波旨在通过已知的控制输入序列 $u_{1:k}$ 和观测序列 $z_{1:k}$，估计当前时刻状态 $x_k$ 的后验概率密度 $p(x_k | z_{1:k}, u_{1:k})$。
贝叶斯滤波基于马尔可夫假设，推导分为预测和更新两个阶段：

预测阶段（先验概率）：
$$p(x_k | z_{1:k-1}, u_{1:k}) = \int p(x_k | x_{k-1}, u_k) p(x_{k-1} | z_{1:k-1}, u_{1:k-1}) dx_{k-1}$$

更新阶段（后验概率）：
$$p(x_k | z_{1:k}, u_{1:k}) = \frac{p(z_k | x_k) p(x_k | z_{1:k-1}, u_{1:k})}{\int p(z_k | x_k) p(x_k | z_{1:k-1}, u_{1:k}) dx_k}$$

## 1.2 算法性质与证明

1. 递推完备性
   贝叶斯滤波只需当前时刻的观测与上一时刻的后验概率即可完成状态更新，无需保存历史观测序列。
   证明：依据马尔可夫假设，当前状态 $x_k$ 仅依赖于前一状态 $x_{k-1}$，观测 $z_k$ 仅依赖于当前状态 $x_k$。利用全概率公式和贝叶斯定理，联合概率可严格分解为预测积分与观测似然乘积的形式，从而实现递推闭环。

2. 解析计算的难解性
   在连续状态空间中，分母的边缘似然积分通常不存在解析解。
   证明：分母积分 $\int p(z_k | x_k) p(x_k | \dots) dx_k$ 需要在整个高维状态空间上进行计算。除非状态转移概率 $p(x_k | x_{k-1})$ 和观测似然 $p(z_k | x_k)$ 均为线性高斯模型或离散有限状态空间，否则该高维非线性积分无法求出闭式解，必须依赖近似算法。

## 1.3 算法伪代码

```text
Algorithm Bayesian_Filter(Belief_{k-1}, u_k, z_k):
    // 1. 预测阶段
    For each state x_k in StateSpace:
        Belief_prior(x_k) = 0
        For each state x_k_1 in StateSpace:
            Belief_prior(x_k) += P(x_k | x_k_1, u_k) * Belief_{k-1}(x_k_1)
            
    // 2. 更新阶段
    eta = 0
    For each state x_k in StateSpace:
        Belief_{k}(x_k) = P(z_k | x_k) * Belief_prior(x_k)
        eta += Belief_{k}(x_k)
        
    // 3. 归一化
    For each state x_k in StateSpace:
        Belief_{k}(x_k) = Belief_{k}(x_k) / eta
        
    Return Belief_{k}
```

## 1.4 C++代码实现

为便于实现，此处展示离散状态空间的贝叶斯滤波一维直方图滤波。

```cpp
#include <vector>
#include <numeric>

class DiscreteBayesFilter {
private:
    std::vector<double> belief;

public:
    DiscreteBayesFilter(int state_space_size) {
        belief.assign(state_space_size, 1.0 / state_space_size);
    }

    void predict(int u, double transition_prob) {
        int n = belief.size();
        std::vector<double> prior(n, 0.0);
        for (int i = 0; i < n; ++i) {
            int next_state = (i + u) % n; 
            if (next_state < 0) next_state += n;
            prior[next_state] += belief[i] * transition_prob;
            prior[i] += belief[i] * (1.0 - transition_prob); 
        }
        belief = prior;
    }

    void update(int measurement, std::vector<double>& likelihood) {
        double sum = 0.0;
        for (size_t i = 0; i < belief.size(); ++i) {
            belief[i] *= likelihood[i];
            sum += belief[i];
        }
        for (size_t i = 0; i < belief.size(); ++i) {
            belief[i] /= sum;
        }
    }
};
```

# 2. 卡尔曼滤波 (KF)

## 2.1 数学定义构造

卡尔曼滤波是贝叶斯滤波在线性高斯假设下的精确解析解。系统建模为线性方程：
状态方程：$x_k = F x_{k-1} + B u_k + w_k$
观测方程：$z_k = H x_k + v_k$
其中 $w_k \sim N(0, Q)$，$v_k \sim N(0, R)$。

预测方程组：
$$\hat{x}*{k|k-1} = F \hat{x}*{k-1|k-1} + B u_k$$
$$P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$

更新方程组：
$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$
$$\hat{x}*{k|k} = \hat{x}*{k|k-1} + K_k (z_k - H \hat{x}*{k|k-1})$$
$$P*{k|k} = (I - K_k H) P_{k|k-1}$$

## 2.2 算法性质与证明

1. 最小均方误差最优性
   卡尔曼滤波的增益 $K_k$ 使得后验状态估计的误差协方差迹最小。
   证明：构建后验协方差矩阵 $P_{k|k} = E[(x_k - \hat{x}*{k|k})(x_k - \hat{x}*{k|k})^T]$。将更新方程代入展开，得到 $P_{k|k}$ 关于 $K_k$ 的二次型函数。对矩阵的迹 $Tr(P_{k|k})$ 关于 $K_k$ 求矩阵导数并令其为 0：
   $$\frac{\partial Tr(P_{k|k})}{\partial K_k} = -2 P_{k|k-1} H^T + 2 K_k (H P_{k|k-1} H^T + R) = 0$$
   解此线性方程即可严格推导出卡尔曼增益公式。

2. 高斯封闭性
   若初始状态服从高斯分布，系统状态在任何时刻均保持高斯分布。
   证明：已知独立高斯随机变量的线性组合仍服从高斯分布。状态转移 $F x + w$ 和观测 $H x + v$ 均为线性变换。因此概率密度推导过程中的所有分布形式均严格保持为高斯分布，仅需维护均值向量和协方差矩阵两个一二阶矩。

## 2.3 算法伪代码

```text
Algorithm Kalman_Filter(x_prev, P_prev, u, z, F, B, H, Q, R):
    // 1. 预测
    x_pred = F * x_prev + B * u
    P_pred = F * P_prev * F^T + Q
    
    // 2. 计算卡尔曼增益
    S = H * P_pred * H^T + R
    K = P_pred * H^T * inv(S)
    
    // 3. 更新
    y = z - H * x_pred
    x_curr = x_pred + K * y
    P_curr = (I - K * H) * P_pred
    
    Return x_curr, P_curr
```

## 2.4 C++代码实现

基于 Eigen3 矩阵库实现。

```cpp
#include <Eigen/Dense>

class KalmanFilter {
private:
    Eigen::VectorXd x; 
    Eigen::MatrixXd P; 
    Eigen::MatrixXd F, B, H, Q, R, I;

public:
    KalmanFilter(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0) : x(x0), P(P0) {}

    void setMatrices(const Eigen::MatrixXd& F_, const Eigen::MatrixXd& B_, 
                     const Eigen::MatrixXd& H_, const Eigen::MatrixXd& Q_, const Eigen::MatrixXd& R_) {
        F = F_; B = B_; H = H_; Q = Q_; R = R_;
        I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
    }

    void predict(const Eigen::VectorXd& u) {
        x = F * x + B * u;
        P = F * P * F.transpose() + Q;
    }

    void update(const Eigen::VectorXd& z) {
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();
        
        Eigen::VectorXd y = z - H * x;
        x = x + K * y;
        P = (I - K * H) * P;
    }
};
```

# 3. 扩展卡尔曼滤波 (EKF)

## 3.1 数学定义构造

由于实际物理系统存在非线性行为，构建非线性状态模型和观测模型：
状态方程：$x_k = f(x_{k-1}, u_k) + w_k$
观测方程：$z_k = h(x_k) + v_k$

EKF通过在估计点处使用一阶泰勒展开进行局部线性化，计算雅可比矩阵：
$$F_k = \left. \frac{\partial f}{\partial x} \right|*{\hat{x}*{k-1|k-1}, u_k}, \quad H_k = \left. \frac{\partial h}{\partial x} \right|*{\hat{x}*{k|k-1}}$$

预测更新公式与KF基本一致，仅均值传递采用非线性函数：
$\hat{x}*{k|k-1} = f(\hat{x}*{k-1|k-1}, u_k)$
$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q$
$K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R)^{-1}$
$\hat{x}*{k|k} = \hat{x}*{k|k-1} + K_k (z_k - h(\hat{x}_{k|k-1}))$

## 3.2 算法性质与证明

1. 局部近似截断误差
   EKF 的估计结果并非全局最优解，强非线性下存在发散风险。
   证明：一阶泰勒展开为 $f(x) \approx f(\hat{x}) + F(x-\hat{x}) + O((x-\hat{x})^2)$。在协方差更新中直接舍弃了二阶及以上的高阶项 $O$。若函数局部曲率过大或初始估计偏离真实值较远，高阶截断误差将累积至协方差矩阵中，使得滤波增益严重失真。

2. 雅可比导数依赖性
   算法执行过程与数学模型紧耦合。
   证明：计算 $F_k$ 和 $H_k$ 需事先获取非线性物理方程 $f$ 和 $h$ 的偏导数解析式。若系统映射关系仅为隐函数或存在黑盒模型，则无法直接求导，通常需采用数值差分近似，进一步增加计算误差。

## 3.3 算法伪代码

```text
Algorithm Extended_Kalman_Filter(x_prev, P_prev, u, z, Q, R):
    // 1. 均值非线性预测
    x_pred = f(x_prev, u)
    
    // 2. 计算雅可比矩阵
    F_k = compute_jacobian_f(x_prev, u)
    
    // 3. 协方差线性预测
    P_pred = F_k * P_prev * F_k^T + Q
    
    // 4. 观测雅可比与增益
    H_k = compute_jacobian_h(x_pred)
    S = H_k * P_pred * H_k^T + R
    K = P_pred * H_k^T * inv(S)
    
    // 5. 均值非线性更新与协方差更新
    y = z - h(x_pred)
    x_curr = x_pred + K * y
    P_curr = (I - K * H_k) * P_pred
    
    Return x_curr, P_curr
```

## 3.4 C++代码实现

需要提供计算非线性函数和雅可比矩阵的回调机制。

```cpp
#include <Eigen/Dense>
#include <functional>

class ExtendedKalmanFilter {
private:
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Q, R, I;

public:
    ExtendedKalmanFilter(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, 
                         const Eigen::MatrixXd& Q_, const Eigen::MatrixXd& R_) 
        : x(x0), P(P0), Q(Q_), R(R_) {
        I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
    }

    void predict(const Eigen::VectorXd& u, 
                 std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f,
                 std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> jacobian_f) {
        Eigen::MatrixXd F = jacobian_f(x, u);
        x = f(x, u);
        P = F * P * F.transpose() + Q;
    }

    void update(const Eigen::VectorXd& z,
                std::function<Eigen::VectorXd(Eigen::VectorXd)> h,
                std::function<Eigen::MatrixXd(Eigen::VectorXd)> jacobian_h) {
        Eigen::MatrixXd H = jacobian_h(x);
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();
        
        Eigen::VectorXd y = z - h(x);
        x = x + K * y;
        P = (I - K * H) * P;
    }
};
```

# 4. 粒子滤波 (PF)

## 4.1 数学定义构造

粒子滤波摒弃线性高斯约束，基于蒙特卡洛随机采样逼近任意非线性非高斯的后验概率分布。通过一组带权重的随机样本集合（粒子）表示概率分布：
$$S = { (x_k^{(i)}, w_k^{(i)}) | i = 1, \dots, N }$$
后验概率密度近似为狄拉克冲击函数的加权和：
$$p(x_k | z_{1:k}) \approx \sum_{i=1}^{N} w_k^{(i)} \delta(x_k - x_k^{(i)})$$
权重通过重要性采样与观测似然更新：
$$w_k^{(i)} \propto w_{k-1}^{(i)} \frac{p(z_k | x_k^{(i)}) p(x_k^{(i)} | x_{k-1}^{(i)})}{q(x_k^{(i)} | x_{k-1}^{(i)}, z_k)}$$
当提议分布取状态转移模型 $p(x_k|x_{k-1})$ 时，权重更新简化为 $w_k^{(i)} \propto w_{k-1}^{(i)} p(z_k | x_k^{(i)})$。

## 4.2 算法性质与证明

1. 粒子耗散悖论
   随着递推时间增加，除少数粒子外，绝大多数粒子的权重将趋近于 0，导致极大的计算资源浪费。
   证明：可以证明权重方差 $\text{Var}(w_k)$ 随着时间 $k$ 是单调非递减的。由于方差无界增长，最终分布必定表现为某单一粒子权重趋近于 1，其余所有粒子权重趋近于 0。这种现象使得对后验分布的有效表示彻底崩溃。

2. 重采样修复机制
   引入序列重要性重采样（SIR）以解决耗散问题，其原则为消除低权重粒子并复制高权重粒子。
   证明：设定有效粒子数阈值 $N_{eff} = 1 / \sum (w_k^{(i)})^2$。当 $N_{eff}$ 过低时，根据当前权重分布作为概率重新抽取 $N$ 个新粒子，并强制将所有新粒子权重重置为 $1/N$。虽然引入了统计噪声，但在物理层面上成功将计算资源集中至高概率区域。

## 4.3 算法伪代码

```text
Algorithm Particle_Filter(S_prev, u, z, N):
    S_curr = empty_set
    weight_sum = 0
    
    // 1. 采样与权重更新
    For i = 1 to N:
        x_sample = sample_from_transition_model(S_prev[i].x, u)
        weight = S_prev[i].w * compute_measurement_likelihood(z, x_sample)
        S_curr.append(Particle(x_sample, weight))
        weight_sum += weight
        
    // 2. 权重归一化
    For i = 1 to N:
        S_curr[i].w /= weight_sum
        
    // 3. 有效粒子数判定与重采样
    N_eff = 1 / sum(S_curr[i].w ^ 2)
    If N_eff < Threshold:
        S_curr = Resample(S_curr, N)
        
    Return S_curr
```

## 4.4 C++代码实现

提供带系统重采样的基础一维粒子滤波实现框架。

```cpp
#include <vector>
#include <random>
#include <numeric>

struct Particle {
    double x;
    double w;
};

class ParticleFilter {
private:
    std::vector<Particle> particles;
    int num_particles;
    std::default_random_engine gen;

public:
    ParticleFilter(int N, double init_x) : num_particles(N) {
        double init_weight = 1.0 / N;
        for (int i = 0; i < N; ++i) {
            particles.push_back({init_x, init_weight});
        }
    }

    void predict(double u, double noise_std) {
        std::normal_distribution<double> dist(0, noise_std);
        for (auto& p : particles) {
            p.x += u + dist(gen); 
        }
    }

    void update(double z, double measure_std) {
        double weight_sum = 0.0;
        for (auto& p : particles) {
            double error = z - p.x;
            p.w *= exp(-(error * error) / (2 * measure_std * measure_std));
            weight_sum += p.w;
        }
        for (auto& p : particles) {
            p.w /= weight_sum;
        }
    }

    void resample() {
        std::vector<double> weights;
        for (const auto& p : particles) weights.push_back(p.w);
        
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        std::vector<Particle> new_particles;
        
        double new_weight = 1.0 / num_particles;
        for (int i = 0; i < num_particles; ++i) {
            int idx = dist(gen);
            new_particles.push_back({particles[idx].x, new_weight});
        }
        particles = new_particles;
    }
};
```
