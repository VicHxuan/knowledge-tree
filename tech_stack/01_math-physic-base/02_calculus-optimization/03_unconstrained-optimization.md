# 无约束优化方法

## 1. 概述

无约束优化从目标梯度构造搜索方向，再决定步长。随机一阶方法适合小批量目标，牛顿与拟牛顿方法利用曲率改善确定性平滑目标的局部求解。

## 2. 基础与定义

### 2.1 目标与方向

设 $f:\mathbb R^n\to\mathbb R$，$g_k=\nabla f(x_k)$。迭代写为 $x_{k+1}=x_k+\alpha_kd_k$；$g_k^\top d_k<0$ 表示局部下降方向。

## 3. 核心原理

### 3.1 梯度与动量

梯度下降取 $d_k=-g_k$。动量累积历史方向：

$$
v_k=\beta v_{k-1}+\eta g_k,\qquad x_{k+1}=x_k-v_k.
$$

对 $L$ 光滑目标，$0<\eta<2/L$ 可使精确梯度步满足相应下降界；随机梯度还涉及估计噪声。

### 3.2 自适应梯度

Adam 维护一阶矩与二阶矩：

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2.
$$

偏差修正为 $\hat m_k=m_k/(1-\beta_1^k)$、$\hat v_k=v_k/(1-\beta_2^k)$，逐元素更新

$$
x_{k+1}=x_k-\eta\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.
$$

AdamW 额外施加独立衰减 $-\eta\lambda x_k$，使衰减项与自适应梯度缩放分开。

### 3.3 牛顿与拟牛顿

牛顿方向解 $H_kd_k=-g_k$；海塞正定时为下降方向。BFGS 用 $s_k=x_{k+1}-x_k$ 与 $y_k=g_{k+1}-g_k$ 更新曲率近似。L-BFGS 保存有限历史，通过双循环计算近似逆海塞与梯度的乘积。

<details class="kt-derivation" id="unconstrained-descent">
<summary>梯度下降界</summary>

$L$ 光滑性给出 $f(x+d)\le f(x)+g^\top d+\tfrac L2\|d\|^2$。代入 $d=-\eta g$ 得 $f(x-\eta g)\le f(x)-\eta(1-L\eta/2)\|g\|^2$。强凸性进一步把梯度范数与最优误差联系起来，得到线性收敛界。

</details>

## 4. 实现与应用

### 4.1 更新实现

<details class="kt-derivation" id="optimization-update-kernels">
<summary>一阶更新核</summary>

输入向量尺寸一致，矩估计初始化为零，AdamW 的步数从一开始。以下核函数由调用方提供梯度与学习率。

```cpp
#include <vector>
void update_SGD(std::vector<double>& theta, const std::vector<double>& grad, double lr) {
    for (std::size_t i = 0; i < theta.size(); ++i) theta[i] -= lr * grad[i];
}
```

```cpp
#include <vector>
#include <cmath>
void update_Momentum(std::vector<double>& theta, const std::vector<double>& grad,
                     std::vector<double>& velocity, double lr, double gamma = 0.9) {
    for (std::size_t i = 0; i < theta.size(); ++i) {
        velocity[i] = gamma * velocity[i] + lr * grad[i];
        theta[i] -= velocity[i];
    }
}
```

```cpp
#include <vector>
#include <cmath>
void update_RMSProp(std::vector<double>& theta, const std::vector<double>& grad,
                    std::vector<double>& v, double lr, double beta = 0.9, double epsilon = 1e-8) {
    for (std::size_t i = 0; i < theta.size(); ++i) {
        v[i] = beta * v[i] + (1.0 - beta) * grad[i] * grad[i];
        theta[i] -= (lr / (std::sqrt(v[i]) + epsilon)) * grad[i];
    }
}
```

```cpp
#include <vector>
#include <cmath>
void update_AdamW(std::vector<double>& theta, const std::vector<double>& grad,
                  std::vector<double>& m, std::vector<double>& v,
                  int t, double lr, double weight_decay,
                  double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) {
    for (std::size_t i = 0; i < theta.size(); ++i) {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        double m_hat = m[i] / (1.0 - std::pow(beta1, t));
        double v_hat = v[i] / (1.0 - std::pow(beta2, t));
        theta[i] -= lr * weight_decay * theta[i];
        theta[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}
```

</details>

<details class="kt-derivation" id="optimization-lbfgs">
<summary>L-BFGS 双循环</summary>

历史按时间顺序存储，保留正曲率样本。返回近似逆海塞与梯度的乘积，搜索方向取其负值；步长由线搜索确定。

```cpp
#include <vector>
#include <cmath>
#include <stdexcept>
double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("dimension mismatch");
    double result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

std::vector<double> two_loop_recursion(
    std::vector<double> q,
    const std::vector<std::vector<double>>& s_hist,
    const std::vector<std::vector<double>>& y_hist
) {
    if (s_hist.size() != y_hist.size()) throw std::invalid_argument("history mismatch");
    int m = s_hist.size();
    if (m == 0) return q;
    for (int i = 0; i < m; ++i) {
        if (s_hist[i].size() != q.size() || y_hist[i].size() != q.size())
            throw std::invalid_argument("dimension mismatch");
        double curvature = dot_product(y_hist[i], s_hist[i]);
        if (!std::isfinite(curvature) || curvature <= 0)
            throw std::invalid_argument("nonpositive curvature");
    }
    std::vector<double> alpha(m, 0.0), rho(m, 0.0);

    for (int i = m - 1; i >= 0; --i) {
        rho[i] = 1.0 / dot_product(y_hist[i], s_hist[i]);
        alpha[i] = rho[i] * dot_product(s_hist[i], q);
        for (std::size_t j = 0; j < q.size(); ++j) q[j] -= alpha[i] * y_hist[i][j];
    }

    double gamma = dot_product(s_hist.back(), y_hist.back()) / dot_product(y_hist.back(), y_hist.back());
    std::vector<double> r = q;
    for (std::size_t j = 0; j < r.size(); ++j) r[j] *= gamma;

    for (int i = 0; i < m; ++i) {
        double beta = rho[i] * dot_product(y_hist[i], r);
        for (std::size_t j = 0; j < r.size(); ++j) r[j] += s_hist[i][j] * (alpha[i] - beta);
    }
    return r;
}
```

</details>

## 5. 分析与总结

条件数影响不同方向的收敛速度。曲率近似改善尺度，但需要可靠的梯度差；随机梯度、裁剪和正则化都会改变实际更新。

## 6. 关联与参考

[Boyd 与 Vandenberghe：Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)，第 9 章。

[Adam](https://arxiv.org/abs/1412.6980)、[AdamW](https://arxiv.org/abs/1711.05101)。

关联：[数值优化](./05_numerical-optimization.md)。
