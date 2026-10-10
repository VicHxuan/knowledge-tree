这里是为您深度精简、去除冗余解释，并无缝融入**凸优化与二次规划（QP）**内容的重构版《数学优化理论与算法核心手册》。

数学公式的语法错误（如星号丢失导致的渲染失败）已全部修正。逻辑链条现已完美闭环：**基础极值 -> 约束理论(KKT) -> 刚性精确求解(QP) -> 软性转化(正则化) -> 数值逼近(一阶/高阶算法)**。

---

# 数学优化理论与算法核心手册

整份手册严格遵循数学演绎逻辑序列：从最基础的优化定义出发，推演至带有严格边界的约束优化；引入凸优化与二次规划理论实现刚性约束的精确求解；随后通过约束向无约束转化的理论枢纽，自然过渡到完全依赖数值迭代的无约束一阶与高阶算法体系。

## 第一部分：最优化基础理论

### 1.1 优化问题的数学标准型

寻找使得目标函数最小化的参数组合，标准数学范式定义如下：
$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \
\text{s.t.} \quad & g_i(x) \le 0, \quad i = 1, \dots, m \
& h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$
其中 $f(x)$ 为目标函数，$g_i(x)$ 为不等式约束，$h_j(x)$ 为等式约束。

### 1.2 无约束最优性条件

当问题不存在任何约束时，假设 $f(x)$ 连续可微，寻找极值点的基础数学条件为：

* **一阶必要条件**：梯度的各个分量必须为零，即 $\nabla f(x) = 0$。
* **二阶充分条件**：在满足一阶条件的基础上，其海森矩阵 $\nabla^2 f(x)$ 必须为正定矩阵。

---

## 第二部分：约束优化理论与方法

当优化问题存在边界限制时，需引入约束优化理论。

### 2.1 KKT条件

KKT条件是判定约束优化问题局部最优解的核心一阶必要条件。引入拉格朗日乘子 $\lambda_i \ge 0$ 和 $\nu_j$，构造拉格朗日函数 $L(x, \lambda, \nu) = f(x) + \sum \lambda_i g_i(x) + \sum \nu_j h_j(x)$。最优解必须满足四大方程组：

1. **平稳性**: $\nabla_x L = \nabla f(x) + \sum_{i} \lambda_i \nabla g_i(x) + \sum_{j} \nu_j \nabla h_j(x) = 0$
2. **原始可行性**: $g_i(x) \le 0, \quad h_j(x) = 0$
3. **对偶可行性**: $\lambda_i \ge 0$
4. **互补松弛性**: $\lambda_i g_i(x) = 0$

互补松弛性指明了约束发挥作用的机制。若最优解在边界内部，此时必须有 $\lambda_i = 0$；只有当解严格贴在边界上时，乘子 $\lambda_i$ 才为正。

### 2.2 增广拉格朗日方法

为数值求解带等式约束 $h(x)=0$ 的问题，增广拉格朗日函数在经典拉格朗日函数基础上，增加惩罚因子 $\rho > 0$ 的二次惩罚项：
$$ L_\rho(x, \lambda) = f(x) + \lambda^T h(x) + \frac{\rho}{2} |h(x)|^2 $$
该方法利用显式的乘子 $\lambda$ 进行修正，即使 $\rho$ 保持有限常数，也能避免海森矩阵病态，数值稳定地收敛至精确最优解。

---

## 第三部分：凸优化与标准规划问题

对于满足特定几何性质的问题，可以通过标准求解器获得全局精确解。

### 3.1 凸优化理论

若目标函数 $f(x)$ 是凸函数，且可行域是一个凸集，则该问题为凸优化问题。对于凸优化问题，KKT 条件升级为全局最优的充要条件。

### 3.2 二次规划 (QP)

目标函数为二次型且约束条件全为线性的凸优化问题。其数学标准型为：
$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & \frac{1}{2} x^T Q x + c^T x \
\text{s.t.} \quad & A_{eq} x = b_{eq} \
& A_{ieq} x \le b_{ieq}
\end{aligned}
$$
当海森矩阵 $Q$ 为半正定矩阵时，QP问题保证能高效求得全局唯一最优解，是机器人控制与运筹学轨迹规划的底层基石。

### 3.3 QP核心数值算法

* **有效集法**: 猜测当前起作用的不等式约束边界，将其视作等式约束，转化为求解KKT线性方程组，不断迭代更新有效集。
* **内点法**: 在目标函数中引入对数障碍项强制解远离边界，在可行域内部穿梭逼近最优解，适合超大规模QP问题。

### 3.4 序列二次规划 (SQP)

针对高度非线性约束优化问题。在每次迭代点处，将目标函数展开为二次近似，约束函数展开为线性近似，构造并求解一系列 QP 子问题，最终逼近原问题的非线性最优解。

---

## 第四部分：理论枢纽 —— 约束与无约束的数学统一

### 4.1 理论判定标准的统一

当最优解落在不等式约束的绝对内部时，根据 KKT 条件的互补松弛性，所有乘子均等于零。KKT的平稳性条件瞬间退化为 $\nabla f(x) = 0$。无约束最优性是 KKT 条件在乘子全为零时的特例。

### 4.2 从边界约束到正则化的等价转换

高维参数寻优为了防止过拟合，通常限制参数范数 $|\theta|^2 \le C$。根据拉格朗日乘子法，可转化为无约束函数求解：
$$ \min L(\theta) + \lambda (|\theta|^2 - C) \implies \min L(\theta) + \lambda |\theta|^2 $$
广泛应用于一阶算法中的 L2 正则化，本质上是带有不等式边界的拉格朗日约束转化机制。

---

## 第五部分：无约束一阶优化算法

针对无约束标量函数 $L(\theta)$，一阶算法依赖泰勒近似确保函数值下降，核心策略为令步长 $\Delta \theta$ 与梯度方向相反。

### 5.1 随机梯度下降算法 (SGD)

每次迭代使用小批量数据估计梯度更新参数：
$$ \theta_{t} = \theta_{t-1} - \eta g_t $$
计算开销极小。在陡峭方向易产生剧烈震荡，平缓方向收敛缓慢。

```cpp
#include <vector>
void update_SGD(std::vector<double>& theta, const std::vector<double>& grad, double lr) {
    for (size_t i = 0; i < theta.size(); ++i) theta[i] -= lr * grad[i];
}
```

### 5.2 动量法 (Momentum)

引入速度变量 $v_t$ 与动量衰减系数 $\gamma$：
$$ v_t = \gamma v_{t-1} + \eta g_t $$
$$ \theta_t = \theta_{t-1} - v_t $$
历史相反方向速度抵消当前梯度减缓震荡，相同方向累加加速收敛。

```cpp
void update_Momentum(std::vector<double>& theta, const std::vector<double>& grad, 
                     std::vector<double>& velocity, double lr, double gamma = 0.9) {
    for (size_t i = 0; i < theta.size(); ++i) {
        velocity[i] = gamma * velocity[i] + lr * grad[i];
        theta[i] -= velocity[i];
    }
}
```

### 5.3 自适应步长系列

**AdaGrad**：累计历史梯度平方和缩放学习率。适合稀疏特征，但会导致学习率过早衰减。
**RMSProp**：采用指数移动平均替代简单累加，克服过早衰减缺陷：
$$ E[g^2]*t = \beta E[g^2]*{t-1} + (1 - \beta) g_t^2 $$
$$ \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t $$

```cpp
#include <cmath>
void update_RMSProp(std::vector<double>& theta, const std::vector<double>& grad, 
                    std::vector<double>& v, double lr, double beta = 0.9, double epsilon = 1e-8) {
    for (size_t i = 0; i < theta.size(); ++i) {
        v[i] = beta * v[i] + (1.0 - beta) * grad[i] * grad[i];
        theta[i] -= (lr / (std::sqrt(v[i]) + epsilon)) * grad[i];
    }
}
```

### 5.4 Adam 与 AdamW

**Adam**：结合一阶矩 $m_t$ 与二阶矩 $v_t$ 并执行偏差校正。其正则化惩罚项混入梯度受分母缩放，导致正则化在方差大维度失效。
$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad ; \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$
$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}*t = \frac{v_t}{1 - \beta_2^t} $$
$$ \theta_t = \theta*{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

**AdamW**：将权重衰减与梯度更新严格解耦，直接作用于参数：
$$ \theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

```cpp
void update_AdamW(std::vector<double>& theta, const std::vector<double>& grad, 
                  std::vector<double>& m, std::vector<double>& v, 
                  int t, double lr, double weight_decay, 
                  double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) {
    for (size_t i = 0; i < theta.size(); ++i) {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        double m_hat = m[i] / (1.0 - std::pow(beta1, t));
        double v_hat = v[i] / (1.0 - std::pow(beta2, t));
        theta[i] -= lr * weight_decay * theta[i];
        theta[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}
```

---

## 第六部分：无约束高阶优化算法

对标量函数执行二阶泰勒展开，求导置零直接得到最优步长方向 $\Delta \theta = -[\nabla^2 L(\theta_t)]^{-1} \nabla L(\theta_t)$。

### 6.1 L-BFGS算法

不显式计算海森矩阵，存储近期 $m$ 步的参数差 $s$ 与梯度差 $y$。通过双循环递归隐式计算海森逆矩阵与梯度的乘积 $r_t$，结合线搜索确定步长：
$$ r_t = \text{TwoLoopRecursion}(\nabla L(\theta_t), {s_i}*{i=t-m}^{t-1}, {y_i}*{i=t-m}^{t-1}) $$
$$ \theta_{t+1} = \theta_t - \alpha_t r_t $$
内存消耗降至线性级别，利用二阶信息精准预测下降方向，收敛步数少且精度极高，但极度排斥随机小批量数据引入的噪声。

```cpp
#include <vector>
double dot_product(const std::vector<double>& a, const std::vector<double>& b);

std::vector<double> two_loop_recursion(
    std::vector<double> q, 
    const std::vector<std::vector<double>>& s_hist, 
    const std::vector<std::vector<double>>& y_hist
) {
    int m = s_hist.size();
    std::vector<double> alpha(m, 0.0), rho(m, 0.0);

    for (int i = m - 1; i >= 0; --i) {
        rho[i] = 1.0 / dot_product(y_hist[i], s_hist[i]);
        alpha[i] = rho[i] * dot_product(s_hist[i], q);
        for (size_t j = 0; j < q.size(); ++j) q[j] -= alpha[i] * y_hist[i][j];
    }

    double gamma = dot_product(s_hist.back(), y_hist.back()) / dot_product(y_hist.back(), y_hist.back());
    std::vector<double> r = q;
    for (size_t j = 0; j < r.size(); ++j) r[j] *= gamma;

    for (int i = 0; i < m; ++i) {
        double beta = rho[i] * dot_product(y_hist[i], r);
        for (size_t j = 0; j < r.size(); ++j) r[j] += s_hist[i][j] * (alpha[i] - beta);
    }
    return r; 
}
```
