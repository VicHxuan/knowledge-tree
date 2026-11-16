# Bézier 曲线与 B 样条

## 1. 概述

Bézier 曲线与 B 样条使用控制点和基函数表示轨迹。基函数的非负性与单位分解带来凸包性质，B 样条的局部支撑进一步降低局部修改和约束计算的范围。

## 2. 基础与定义

### 2.1 Bézier 曲线

$n$ 次 Bézier 曲线由 $n+1$ 个控制点 $P_i$ 定义：

$$
p(u)=\sum_{i=0}^n B_i^n(u)P_i,\qquad
B_i^n(u)=\binom ni u^i(1-u)^{n-i},\quad u\in[0,1].
$$

### 2.2 B 样条

给定次数 $p$、控制点 $P_0,\ldots,P_n$ 与非递减节点向量 $t_0,\ldots,t_{n+p+1}$，曲线为

$$
C(t)=\sum_{i=0}^nN_{i,p}(t)P_i,\qquad t\in[t_p,t_{n+1}].
$$

节点与次数共同决定基函数；端点采用区间内的单侧极限。

## 3. 核心原理

### 3.1 递推与导数

B 样条基函数满足 Cox–de Boor 递推：\n

$$
N_{i,p}(t)=\frac{t-t_i}{t_{i+p}-t_i}N_{i,p-1}(t)
+\frac{t_{i+p+1}-t}{t_{i+p+1}-t_{i+1}}N_{i+1,p-1}(t).
$$

零分母项按零处理，零次基函数在 $[t_i,t_{i+1})$ 上为 1。导数曲线的控制点为

$$
D_i=\frac{p(P_{i+1}-P_i)}{t_{i+p+1}-t_{i+1}},
$$

其次数降低一阶，节点向量去掉首尾节点。

<details class="kt-derivation" id="bezier-derivative">
<summary>Bézier 导数与时间尺度</summary>

利用 $dB_i^n/du=n(B_{i-1}^{n-1}-B_i^{n-1})$，整理得到

$$
\frac{dp}{du}=n\sum_{i=0}^{n-1}B_i^{n-1}(u)(P_{i+1}-P_i).
$$

若真实时间为 $\tau=Tu$，速度控制点为 $n(P_{i+1}-P_i)/T$，加速度控制点为 $n(n-1)(P_{i+2}-2P_{i+1}+P_i)/T^2$。

</details>

### 3.2 凸包与连续性

有效区间内基函数非负且和为 1，曲线位于激活控制点的凸包内。若这些控制点共同位于一个凸安全集合中，整段曲线也位于该集合中。

次数为 $p$、内部节点重数为 $r$ 时，通常具有 $C^{p-r}$ 连续性；简单内部节点对应 $C^{p-1}$。端点插值可通过 $p+1$ 重夹持节点实现。

### 3.3 曲线拟合

采样位置与边界导数可统一写成 $AP\approx b$。软拟合采用加权最小二乘；需要精确满足的边界条件使用等式约束。

### 3.4 参数化方法的关系

Bézier 与 B 样条通过控制点确定曲线。[MINCO](06_minco-parameterization.md)通过中间位置和段时间求解内层最小能量问题，得到分段多项式系数。两者可通过基变换连接，但自由变量与内层条件不同。

## 4. 实现与应用

固定节点后，位置和导数对控制点均为线性映射。实现时可预计算基函数矩阵及其导数，将拟合、平滑与约束写成矩阵运算。

均匀三次 B 样条在节点处的局部关系为

$$
p_i=\frac{P_i+4P_{i+1}+P_{i+2}}6,\quad
v_i=\frac{-P_i+P_{i+2}}{2\Delta t},\quad
a_i=\frac{P_i-2P_{i+1}+P_{i+2}}{\Delta t^2}.
$$

EGO-Planner 的初始化将采样位置与首尾导数共同放入超定线性系统，通过 QR 求最小二乘解。采样条件与边界条件可能具有非零残差，验收时应分别检查。

<details class="kt-derivation" id="bezier-evaluation">
<summary>Bézier 曲线求值</summary>

以下 Eigen 实现接受至少一个控制点，参数取 $u\in[0,1]$。

```cpp
#include <vector>
#include <cmath>
#include <Eigen/Dense>

class BezierCurveEigen {
private:
    std::vector<Eigen::Vector3d> control_points;

    double binomial_coefficient(int n, int k) {
        if (k < 0 || k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        double res = 1.0;
        for (int i = 1; i <= k; ++i) {
            res = res * (n - i + 1) / i;
        }
        return res;
    }

public:
    BezierCurveEigen(const std::vector<Eigen::Vector3d>& points) : control_points(points) {}

    Eigen::Vector3d evaluate(double t) {
        int n = control_points.size() - 1;
        Eigen::Vector3d result = Eigen::Vector3d::Zero();
        for (int i = 0; i <= n; ++i) {
            double b = binomial_coefficient(n, i) * std::pow(t, i) * std::pow(1 - t, n - i);
            result += b * control_points[i];
        }
        return result;
    }
};
```

</details>

<details class="kt-derivation" id="uniform-spline-fit">
<summary>均匀三次 B 样条拟合</summary>

输入依次给出位置样本，以及起点速度、终点速度、起点加速度和终点加速度。矩阵有 $K+4$ 行、$K+2$ 列，输出控制点按列存储；各行按原实现采用未加权最小二乘，实际任务可按单位与置信度设置行权重。

```cpp
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
  void parameterizeToBspline(const double &ts, const std::vector<Eigen::Vector3d> &point_set,
                                             const std::vector<Eigen::Vector3d> &start_end_derivative,
                                             Eigen::MatrixXd &ctrl_pts)
  {
    if (ts <= 0)
    {
      throw std::invalid_argument("time step must be positive");
    }

    if (point_set.size() <= 3)
    {
      throw std::invalid_argument("at least four samples are required");
    }

    if (start_end_derivative.size() != 4)
    {
      throw std::invalid_argument("four endpoint derivatives are required");
    }

    int K = point_set.size();

    // write A
    Eigen::Vector3d prow(3), vrow(3), arow(3);
    prow << 1, 4, 1;
    vrow << -1, 0, 1;
    arow << 1, -2, 1;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 4, K + 2);

    for (int i = 0; i < K; ++i)
      A.block(i, i, 1, 3) = (1 / 6.0) * prow.transpose();

    A.block(K, 0, 1, 3) = (1 / 2.0 / ts) * vrow.transpose();
    A.block(K + 1, K - 1, 1, 3) = (1 / 2.0 / ts) * vrow.transpose();

    A.block(K + 2, 0, 1, 3) = (1 / ts / ts) * arow.transpose();
    A.block(K + 3, K - 1, 1, 3) = (1 / ts / ts) * arow.transpose();

    //cout << "A" << endl << A << endl << endl;

    // write b
    Eigen::VectorXd bx(K + 4), by(K + 4), bz(K + 4);
    for (int i = 0; i < K; ++i)
    {
      bx(i) = point_set[i](0);
      by(i) = point_set[i](1);
      bz(i) = point_set[i](2);
    }

    for (int i = 0; i < 4; ++i)
    {
      bx(K + i) = start_end_derivative[i](0);
      by(K + i) = start_end_derivative[i](1);
      bz(K + i) = start_end_derivative[i](2);
    }

    // solve Ax = b
    Eigen::VectorXd px = A.colPivHouseholderQr().solve(bx);
    Eigen::VectorXd py = A.colPivHouseholderQr().solve(by);
    Eigen::VectorXd pz = A.colPivHouseholderQr().solve(bz);

    // convert to control pts
    ctrl_pts.resize(3, K + 2);
    ctrl_pts.row(0) = px.transpose();
    ctrl_pts.row(1) = py.transpose();
    ctrl_pts.row(2) = pz.transpose();

    // cout << "[B-spline]: parameterization ok." << endl;
  }
```

</details>

<details class="kt-derivation" id="bspline-evaluation">
<summary>de Boor 曲线求值</summary>

输入有限控制点与有限非递减节点，节点重数至多为次数加一。有效区间右端使用左极限，兼容夹持与非夹持节点。

```cpp
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <stdexcept>

Eigen::Vector3d evaluateBSpline(const std::vector<Eigen::Vector3d>& points,
                               const std::vector<double>& knots,
                               int degree, double t) {
    const int n = static_cast<int>(points.size()) - 1;
    if (degree < 0 || n < degree || knots.size() != points.size() + degree + 1 ||
        !std::is_sorted(knots.begin(), knots.end()))
        throw std::invalid_argument("invalid spline structure");
    if (!(knots[degree] < knots[n + 1]) || !(t >= knots[degree] && t <= knots[n + 1]))
        throw std::invalid_argument("parameter outside spline domain");
    int span;
    if (t == knots[n + 1]) {
        // Select the final nonempty span for the left endpoint limit.
        span = static_cast<int>(std::lower_bound(knots.begin(), knots.end(), t) - knots.begin()) - 1;
    } else {
        span = static_cast<int>(std::upper_bound(knots.begin(), knots.end(), t) - knots.begin()) - 1;
    }
    std::vector<Eigen::Vector3d> d(degree + 1);
    for (int j = 0; j <= degree; ++j) d[j] = points[span - degree + j];
    for (int r = 1; r <= degree; ++r) {
        for (int j = degree; j >= r; --j) {
            const int i = span - degree + j;
            const double denominator = knots[i + degree - r + 1] - knots[i];
            const double alpha = denominator > 0 ? (t - knots[i]) / denominator : 0.0;
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }
    return d[degree];
}
```

</details>

## 5. 分析与总结

Bézier 适合分段凸约束，B 样条适合局部调整。控制点距离惩罚通常只是避障代理；可靠的几何约束来自共同凸走廊、曲线分割检查或其他覆盖整段轨迹的验证。

## 6. 关联与参考

- [SciPy：B 样条定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
- [EGO-Planner 的 B 样条实现](https://github.com/ZJU-FAST-Lab/ego-planner/blob/master/src/planner/bspline_opt/src/uniform_bspline.cpp)
- [安全飞行走廊](04_safe-flight-corridor.md)
