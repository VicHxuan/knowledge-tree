# 贝叶斯滤波与卡尔曼滤波

## 1. 概述

递归状态估计先用运动模型预测状态分布，再用新观测更新后验。KF、EKF 与粒子滤波采用不同的分布表示和近似方式。

## 2. 基础与定义

### 2.1 状态与观测

状态 $x_k$、输入 $u_k$、观测 $z_k$ 由转移分布 $p(x_k\mid x_{k-1},u_k)$ 与观测分布 $p(z_k\mid x_k)$ 连接。估计目标是 $p(x_k\mid z_{1:k},u_{1:k})$。

### 2.2 预测与更新

$$
p^-(x_k)=\int p(x_k\mid x_{k-1},u_k)p^+(x_{k-1})dx_{k-1},
$$

$$
p^+(x_k)\propto p(z_k\mid x_k)p^-(x_k).
$$

## 3. 核心原理

### 3.1 线性高斯滤波

对 $x_k=Fx_{k-1}+Bu_k+w_k$、$z_k=Hx_k+v_k$，过程与观测噪声独立，协方差为 $Q,R$。预测为

$$
\hat x^-=F\hat x^++Bu,\quad P^-=FP^+F^\top+Q.
$$

创新 $e=z-H\hat x^-$、创新协方差 $S=HP^-H^\top+R$，更新为

$$
K=P^-H^\top S^{-1},\quad\hat x^+=\hat x^-+Ke,\quad
P^+=(I-KH)P^-(I-KH)^\top+KRK^\top.
$$

最后一式是 Joseph 协方差更新。线性高斯条件下，均值与协方差完整描述精确后验。

<details class="kt-derivation" id="kf-gain">
<summary>卡尔曼增益推导</summary>

误差更新为 $\tilde x^+=(I-KH)\tilde x^- -Kv$。独立零均值误差给出 Joseph 形式。对 $\operatorname{tr}P^+$ 关于 $K$ 求导并令零：$KS=P^-H^\top$，因此 $K=P^-H^\top S^{-1}$。实现中求解线性系统即可。

</details>

### 3.2 非线性线性化

EKF 对 $x_k=f(x_{k-1},u_k)+w_k$、$z_k=h(x_k)+v_k$ 在当前估计处计算雅可比 $F,H$。均值使用非线性函数传播，协方差使用局部线性模型。迭代 EKF 在固定预测先验下重新线性化观测。

### 3.3 粒子近似

粒子滤波用 $\{x_k^{(i)},w_k^{(i)}\}$ 表示后验。以转移先验为提议分布时，先采样状态，再更新

$$
\widetilde w_k^{(i)}=w_{k-1}^{(i)}p(z_k\mid x_k^{(i)}),\qquad
w_k^{(i)}=\frac{\widetilde w_k^{(i)}}{\sum_j\widetilde w_k^{(j)}}.
$$

有效样本数 $N_{\rm eff}=1/\sum_i(w_k^{(i)})^2$ 可用于触发重采样。对数权重与归一化可减少数值下溢。

<details class="kt-derivation" id="bayes-discrete">
<summary>离散状态递推</summary>

有限状态下，转移矩阵 $T_{ij}=p(x_k=i\mid x_{k-1}=j,u_k)$ 每列和为一。令先验概率向量为 $b$，观测似然向量为 $\ell$，则 $b^-=Tb$，$b^+=(\ell\odot b^-)/(\mathbf1^\top(\ell\odot b^-))$。归一化分母为零表示观测与当前模型没有共同支持。

</details>

## 4. 实现与应用

### 4.1 估计流程

时间同步 → 预测均值与协方差 → 构造创新 → 更新状态 → 检查创新统计及协方差。

<details class="kt-derivation" id="kf-eigen">
<summary>线性 KF 的 Eigen 实现</summary>

输入矩阵维度匹配，$P,Q$ 半正定、$R$ 正定，观测噪声与先验独立。

```cpp
#include <Eigen/Dense>

class KalmanFilter {
private:
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd F, B, H, Q, R, I;

public:
    const Eigen::VectorXd& state() const { return x; }
    const Eigen::MatrixXd& covariance() const { return P; }

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
        Eigen::MatrixXd K = S.ldlt().solve(H * P).transpose();

        Eigen::VectorXd y = z - H * x;
        x = x + K * y;
        Eigen::MatrixXd A = I - K * H;
        P = A * P * A.transpose() + K * R * K.transpose();
    }
};
```

</details>

<details class="kt-derivation" id="ekf-eigen">
<summary>EKF 的 Eigen 实现</summary>

该示例采用欧氏状态和加性噪声，调用方提供函数及其雅可比。

```cpp
#include <Eigen/Dense>
#include <functional>

class ExtendedKalmanFilter {
private:
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Q, R, I;

public:
    const Eigen::VectorXd& state() const { return x; }
    const Eigen::MatrixXd& covariance() const { return P; }

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
        Eigen::MatrixXd K = S.ldlt().solve(H * P).transpose();

        Eigen::VectorXd y = z - h(x);
        x = x + K * y;
        Eigen::MatrixXd A = I - K * H;
        P = A * P * A.transpose() + K * R * K.transpose();
    }
};
```

</details>

## 5. 分析与总结

| 方法 | 后验表示 | 主要近似 |
| --- | --- | --- |
| 离散 Bayes | 有限状态概率 | 状态离散化 |
| KF | 高斯 | 线性高斯下闭式精确 |
| EKF | 高斯 | 非线性一阶线性化 |
| 粒子滤波 | 加权样本 | 有限采样与重采样 |

噪声模型、时间同步和可观测性共同决定估计质量。

## 6. 关联与参考

[Särkkä 与 Svensson：Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf)。

关联：[高斯估计](../../01_math-physic-base/03_probability-statistics/02_gaussian-estimation.md)、[FAST-LIO](./02_fast-lio.md)。
