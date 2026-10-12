# 贝塞尔曲线与 B 样条

## 1. 概述（Overview）

> 待补充。

## 2. 基础与定义（Foundations）

> 待补充。

## 3. 核心原理（Core Principles）

### 1. 贝塞尔曲线

#### 1.1 数学定义构造

给定 $n+1$ 个控制点 $P_0, P_1, \dots, P_n$，$n$ 阶贝塞尔曲线 $C(t)$ 的数学定义由伯恩斯坦多项式构造：

$$C(t) = \sum_{i=0}^{n} B_{i,n}(t) P_i , \quad t \in [0, 1]$$

其中 $B_{i,n}(t)$ 为伯恩斯坦基函数，其解析表达式为：

$$B_{i,n}(t) = \binom{n}{i} t^i (1-t)^{n-i} = \frac{n!}{i!(n-i)!} t^i (1-t)^{n-i}$$

#### 1.2 曲线性质与证明

1. **端点插值性**
   贝塞尔曲线严格经过起始控制点 $P_0$ 和终止控制点 $P_n$。
   **证明**：当 $t=0$ 时，对于 $i > 0$，$t^i = 0$。只有当 $i=0$ 时，$t^0 = 1$。
   $$C(0) = B_{0,n}(0)P_0 = \binom{n}{0} 0^0 (1-0)^n P_0 = P_0$$
   同理，当 $t=1$ 时，对于 $i < n$，$(1-t)^{n-i} = 0$。只有当 $i=n$ 时，$(1-t)^0 = 1$。
   $$C(1) = B_{n,n}(1)P_n = \binom{n}{n} 1^n (1-1)^0 P_n = P_n$$

2. **规范性**
   伯恩斯坦基函数在任意参数 $t$ 处之和为 1。
   **证明**：根据二项式定理，对于任意实数 $x$ 和 $y$，有 $(x+y)^n = \sum_{i=0}^{n} \binom{n}{i} x^i y^{n-i}$。
   令 $x=t, y=1-t$，代入二项式展开式得：
   $$\sum_{i=0}^{n} B_{i,n}(t) = \sum_{i=0}^{n} \binom{n}{i} t^i (1-t)^{n-i} = (t + 1 - t)^n = 1^n = 1$$

3. **非负性**
   在区间 $t \in [0,1]$ 内，所有的伯恩斯坦基函数均非负。
   **证明**：由于 $t \ge 0$ 且 $(1-t) \ge 0$，因此 $t^i \ge 0$ 且 $(1-t)^{n-i} \ge 0$。
   组合数 $\binom{n}{i}$ 为正整数，三项非负数相乘必然大于等于 0，即 $B_{i,n}(t) \ge 0$。

4. **凸包性**
   曲线 $C(t)$ 在 $t \in [0,1]$ 上的任何一点都落在控制点构成的凸包内部。
   **证明**：根据规范性 $\sum B_{i,n}(t) = 1$ 和非负性 $B_{i,n}(t) \ge 0$，曲线上的任意一点 $C(t)$ 都是控制点 $P_i$ 的凸组合。根据凸包的数学定义，任意点集的凸组合必属于该点集的凸包，因此曲线完全位于控制多边形的凸包内。

5. **端点切线方向**
   曲线起点的切线方向等同于 $P_1 - P_0$ 的方向，终点的切线方向等同于 $P_n - P_{n-1}$ 的方向。
   **证明**：对 $C(t)$ 求关于 $t$ 的一阶导数：
   $$C'(t) = n \sum_{i=0}^{n-1} B_{i,n-1}(t) (P_{i+1} - P_i)$$
   代入 $t=0$，只有 $i=0$ 项不为零，得到 $C'(0) = n(P_1 - P_0)$。
   代入 $t=1$，只有 $i=n-1$ 项不为零，得到 $C'(1) = n(P_n - P_{n-1})$。

#### 1.3 C++代码定义

##### Eigen库实现

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

### 2. B样条曲线

#### 2.1 数学定义构造

给定 $n+1$ 个控制点 $P_0, P_1, \dots, P_n$，$p$ 阶B样条曲线 $C(t)$ 由控制点和B样条基函数组成：

$$C(t) = \sum_{i=0}^{n} N_{i,p}(t) P_i$$

##### 2.1.1 节点向量 (Knot Vector)

B样条的定义依赖于一个非递减的实数序列，称为**节点向量** $U = {u_0, u_1, \dots, u_{m}}$，即满足 $u_i \le u_{i+1}$。
控制点数量 $n+1$、曲线阶数 $p$ 与节点数量 $m+1$ 之间必须满足严格的等式关系：
$$m = n + p + 1$$

节点 $u_i$ 将参数 $t$ 的定义域划分成了多个**节点区间** $[u_i, u_{i+1})$。B样条曲线的有效参数 $t$ 的取值范围通常定义在 $[u_p, u_{n+1}]$ 之间。

根据节点分布的不同，节点向量通常分为以下几种：

1. **均匀节点向量** (Uniform)：节点等距分布，如 $U = {0, 1, 2, 3, 4, 5, 6}$。曲线不一定经过首尾控制点。
2. **准均匀/钳位节点向量** (Clamped / Open Uniform)：实际工程中最常用的类型。为了让曲线像贝塞尔曲线一样准确穿过第一个和最后一个控制点，要求首尾两端的节点具有 $p+1$ 的**重复度** (Multiplicity)。
   例如，归一化到 $[0, 1]$ 区间的三阶 $p=3$ 曲线，其首尾需各有 4 个重复节点：
   $$U = {\underbrace{0, 0, 0, 0}*{p+1}, u_4, \dots, u*{m-4}, \underbrace{1, 1, 1, 1}_{p+1}}$$

##### 2.1.2 Cox-de Boor 递推公式

B样条基函数 $N_{i,p}(t)$ 建立在节点向量 $U$ 之上，由 Cox-de Boor 递推公式定义：

当 $p=0$ 时（即阶跃函数）：
$$N_{i,0}(t) = \begin{cases} 1, & u_i \le t < u_{i+1} \ 0, & \text{otherwise} \end{cases}$$

当 $p > 0$ 时：
$$N_{i,p}(t) = \frac{t - u_i}{u_{i+p} - u_i} N_{i,p-1}(t) + \frac{u_{i+p+1} - t}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(t)$$

**特别规定**：如果出现分母为零的情况（通常因存在重复节点导致 $u_{i+p} = u_i$ 或 $u_{i+p+1} = u_{i+1}$），则该项商的整体计算结果强制记为零（即规定 $\frac{0}{0} = 0$）。

#### 2.2 曲线性质与证明

1. **局部支撑性**
   基函数 $N_{i,p}(t)$ 仅在区间 $[u_i, u_{i+p+1})$ 上非零。
   **证明**：采用数学归纳法。对于 $p=0$，根据定义 $N_{i,0}(t)$ 仅在 $[u_i, u_{i+1})$ 上非零，结论成立。
   假设对阶数 $p-1$ 成立，即 $N_{i,p-1}(t)$ 在 $[u_i, u_{i+p})$ 非零，$N_{i+1,p-1}(t)$ 在 $[u_{i+1}, u_{i+p+1})$ 非零。
   根据 Cox-de Boor 递推公式，$N_{i,p}(t)$ 由 $N_{i,p-1}(t)$ 和 $N_{i+1,p-1}(t)$ 线性组合而成。其非零区间即为二者非零区间的并集 $[u_i, u_{i+p}) \cup [u_{i+1}, u_{i+p+1}) = [u_i, u_{i+p+1})$。结论得证。

2. **规范性**
   对于区间 $[u_k, u_{k+1})$ 内的任意参数 $t$，所有非零基函数之和等于 1。
   **证明**：由局部支撑性可知，在此区间内至多只有 $p+1$ 个非零基函数：$N_{k-p,p}, \dots, N_{k,p}$。
   采用归纳法。对于 $p=0$，区间 $[u_k, u_{k+1})$ 只有 $N_{k,0}=1$，和为1成立。
   假设对于 $p-1$ 阶的非零基函数之和为1。根据递推公式代入求和：
   $$\sum_{i=k-p}^{k} N_{i,p}(t) = \sum_{i=k-p}^{k} \left( \frac{t - u_i}{u_{i+p} - u_i} N_{i,p-1}(t) + \frac{u_{i+p+1} - t}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(t) \right)$$
   对索引进行位移合并同类项，所有 $t$ 的系数抵消，最终等价于 $\sum N_{j,p-1}(t) = 1$。结论得证。

3. **非负性**
   对于所有 $i$, $p$, 和 $t$，有 $N_{i,p}(t) \ge 0$。
   **证明**：采用归纳法。$p=0$ 时非 0 即 1，成立。
   假设 $p-1$ 时非负。递推公式中 $N_{i,p-1}$ 与 $N_{i+1,p-1}$ 前的乘法系数 $\frac{t - u_i}{u_{i+p} - u_i}$ 和 $\frac{u_{i+p+1} - t}{u_{i+p+1} - u_{i+1}}$，在 $t \in [u_i, u_{i+p+1})$ 的支撑区间内均大于等于0。非负数之和必非负，结论得证。

4. **强凸包性**
   曲线落在由对应控制点构成的局部凸包内。
   **证明**：已知对于任意 $t \in [u_k, u_{k+1})$，$C(t) = \sum_{i=k-p}^{k} N_{i,p}(t) P_i$。
   结合已证的规范性 $\sum_{i=k-p}^{k} N_{i,p}(t) = 1$ 和非负性 $N_{i,p}(t) \ge 0$，$C(t)$ 是局部 $p+1$ 个控制点 $P_{k-p}, \dots, P_k$ 的凸组合。因此该段曲线完全位于这 $p+1$ 个点构成的凸包内部。

#### 2.3 C++代码定义

##### Eigen库实现

```cpp
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

class BSplineCurveEigen {
private:
    std::vector<Eigen::Vector3d> control_points;
    std::vector<double> knots;
    int degree;

    double basis_function(int i, int p, double t) const {
        if (p == 0) {
            if (t >= knots[i] && t < knots[i + 1]) return 1.0;
            if (t == knots[i + 1] && t == knots.back()) return 1.0; 
            return 0.0;
        }

        double left = 0.0, right = 0.0;
        
        double denom1 = knots[i + p] - knots[i];
        if (denom1 != 0.0) {
            left = ((t - knots[i]) / denom1) * basis_function(i, p - 1, t);
        }

        double denom2 = knots[i + p + 1] - knots[i + 1];
        if (denom2 != 0.0) {
            right = ((knots[i + p + 1] - t) / denom2) * basis_function(i + 1, p - 1, t);
        }

        return left + right;
    }

public:
    BSplineCurveEigen(const std::vector<Eigen::Vector3d>& points, const std::vector<double>& knot_vector, int p)
        : control_points(points), knots(knot_vector), degree(p) {
        
        int n = control_points.size() - 1;
        int m = knots.size() - 1;
        if (m != n + p + 1) {
            throw std::invalid_argument("Knot vector size must be equal to number of control points + degree + 1");
        }
    }

    Eigen::Vector3d evaluate(double t) const {
        Eigen::Vector3d result = Eigen::Vector3d::Zero();
        int n = control_points.size() - 1;
        
        for (int i = 0; i <= n; ++i) {
            double b = basis_function(i, degree, t);
            result += b * control_points[i];
        }
        
        return result;
    }
};
```

#### 2.4 ego-planner内部的3次均匀B样条曲线实现

ego-planner 中的 `parameterizeToBspline` 函数用于将离散路径点和边界导数条件转换为三次均匀 B 样条控制点。其核心思想是：三次均匀 B 样条在节点处的位置、速度和加速度均可表示为相邻控制点的线性组合，因此可以构造线性方程组反求控制点。

##### 2.4.1 完整代码

```cpp
  void UniformBspline::parameterizeToBspline(const double &ts, const vector<Eigen::Vector3d> &point_set,
                                             const vector<Eigen::Vector3d> &start_end_derivative,
                                             Eigen::MatrixXd &ctrl_pts)
  {
    if (ts <= 0)
    {
      cout << "[B-spline]:time step error." << endl;
      return;
    }

    if (point_set.size() <= 3)
    {
      cout << "[B-spline]:point set have only " << point_set.size() << " points." << endl;
      return;
    }

    if (start_end_derivative.size() != 4)
    {
      cout << "[B-spline]:derivatives error." << endl;
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

##### 2.4.2 函数输入与输出

```cpp
  void UniformBspline::parameterizeToBspline(const double &ts, const vector<Eigen::Vector3d> &point_set,
                                             const vector<Eigen::Vector3d> &start_end_derivative,
                                             Eigen::MatrixXd &ctrl_pts)
```

其中 `ts` 表示均匀 B 样条的时间间隔。若相邻节点时间为：

$$
t_{i+1} - t_i = t_s
$$

则所有节点间隔相同，曲线属于均匀 B 样条。

`point_set` 表示输入路径点序列：

$$
q_{0}, q_{1}, \dots, q_{K-1}
$$

其中：

$$
K = \mathrm{point\_set.size()}
$$

`start_end_derivative` 表示起点和终点的边界导数：

$$
\dot{q}_{0},\ \dot{q}_{K-1},\ \ddot{q}_{0},\ \ddot{q}_{K-1}
$$

`ctrl_pts` 表示输出控制点矩阵。若输入路径点数量为 $K$，则需要求解的控制点数量为：

$$
K + 2
$$

即：

$$
P_{0}, P_{1}, \dots, P_{K}, P_{K+1}
$$

##### 2.4.3 输入合法性检查

```cpp
    if (ts <= 0)
    {
      cout << "[B-spline]:time step error." << endl;
      return;
    }
```

时间间隔必须满足：

$$
t_s > 0
$$

因为速度和加速度计算分别包含：

$$
\frac{1}{t_s},\quad \frac{1}{t_s^2}
$$

当 $t_s \le 0$ 时，时间参数无效。

```cpp
    if (point_set.size() <= 3)
    {
      cout << "[B-spline]:point set have only " << point_set.size() << " points." << endl;
      return;
    }
```

路径点数量需要大于 3：

$$
K > 3
$$

三次均匀 B 样条需要足够数量的路径点来构造稳定的控制点序列。

```cpp
    if (start_end_derivative.size() != 4)
    {
      cout << "[B-spline]:derivatives error." << endl;
    }
```

边界导数数量应为 4：

$$
\left[
\dot{q}_{0},\
\dot{q}_{K-1},\
\ddot{q}_{0},\
\ddot{q}_{K-1}
\right]
$$

分别对应起点速度、终点速度、起点加速度、终点加速度。

##### 2.4.4 控制点数量定义

```cpp
    int K = point_set.size();
```

令路径点数量为：

$$
K = \left|\mathrm{point\_set}\right|
$$

对应输入路径点：

$$
q_{0}, q_{1}, \dots, q_{K-1}
$$

三次均匀 B 样条在节点处的位置由连续三个控制点决定：

$$
q_i = \frac{1}{6}P_i + \frac{4}{6}P_{i+1} + \frac{1}{6}P_{i+2}
$$

最后一个路径点 $q_{K-1}$ 需要使用控制点：

$$
P_{K-1},\ P_K,\ P_{K+1}
$$

因此控制点总数为：

$$
K + 2
$$

##### 2.4.5 三次均匀B样条的节点公式

```cpp
    Eigen::Vector3d prow(3), vrow(3), arow(3);
    prow << 1, 4, 1;
    vrow << -1, 0, 1;
    arow << 1, -2, 1;
```

`prow` 对应节点位置公式。三次均匀 B 样条在节点处的位置为：

$$
q_i = \frac{P_i + 4P_{i+1} + P_{i+2}}{6}
$$

因此位置系数为：

$$
\frac{1}{6}[1,\ 4,\ 1]
$$

`vrow` 对应节点速度公式。对位置公式求一阶导数可得：

$$
\dot{q}_i = \frac{P_{i+2} - P_i}{2t_s}
$$

写成矩阵系数形式为：

$$
\dot{q}_i =
\frac{1}{2t_s}
[-1,\ 0,\ 1]
\begin{bmatrix}
P_i \\
P_{i+1} \\
P_{i+2}
\end{bmatrix}
$$

因此速度系数为：

$$
\frac{1}{2t_s}[-1,\ 0,\ 1]
$$

`arow` 对应节点加速度公式。对位置公式求二阶导数可得：

$$
\ddot{q}_i = \frac{P_i - 2P_{i+1} + P_{i+2}}{t_s^2}
$$

写成矩阵系数形式为：

$$
\ddot{q}_i =
\frac{1}{t_s^2}
[1,\ -2,\ 1]
\begin{bmatrix}
P_i \\
P_{i+1} \\
P_{i+2}
\end{bmatrix}
$$

因此加速度系数为：

$$
\frac{1}{t_s^2}[1,\ -2,\ 1]
$$

##### 2.4.6 构造线性方程矩阵

```cpp
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 4, K + 2);
```

该函数构造线性系统：

$$
AP = b
$$

其中 $P$ 为控制点坐标向量。每个维度独立求解：

$$
A P_x = b_x,\quad
A P_y = b_y,\quad
A P_z = b_z
$$

矩阵 $A$ 的行数为 $K+4$，其中 $K$ 行来自路径点位置约束，4 行来自边界导数约束。矩阵 $A$ 的列数为 $K+2$，对应 $K+2$ 个控制点。

$$
A \in \mathbb{R}^{(K+4)\times(K+2)}
$$

##### 2.4.7 写入位置约束

```cpp
    for (int i = 0; i < K; ++i)
      A.block(i, i, 1, 3) = (1 / 6.0) * prow.transpose();
```

第 $i$ 个路径点满足：

$$
q_i = \frac{P_i + 4P_{i+1} + P_{i+2}}{6}
$$

因此矩阵 $A$ 的第 $i$ 行只在第 $i$ 到第 $i+2$ 列非零：

$$
A_{i,i} = \frac{1}{6},\quad
A_{i,i+1} = \frac{4}{6},\quad
A_{i,i+2} = \frac{1}{6}
$$

该部分将所有路径点写成控制点的线性组合：

$$
\begin{aligned}
q_{0} &= \frac{P_{0} + 4P_{1} + P_{2}}{6} \\
q_{1} &= \frac{P_{1} + 4P_{2} + P_{3}}{6} \\
&\vdots \\
q_{K-1} &= \frac{P_{K-1} + 4P_{K} + P_{K+1}}{6}
\end{aligned}
$$

##### 2.4.8 写入速度边界约束

```cpp
    A.block(K, 0, 1, 3) = (1 / 2.0 / ts) * vrow.transpose();
    A.block(K + 1, K - 1, 1, 3) = (1 / 2.0 / ts) * vrow.transpose();
```

第一行速度约束对应起点速度：

$$
\dot{q}_{0} = \frac{-P_{0} + P_{2}}{2t_s}
$$

其矩阵系数为：

$$
A_{K,0} = -\frac{1}{2t_s},\quad
A_{K,1} = 0,\quad
A_{K,2} = \frac{1}{2t_s}
$$

第二行速度约束对应终点速度：

$$
\dot{q}_{K-1} = \frac{-P_{K-1} + P_{K+1}}{2t_s}
$$

其矩阵系数为：

$$
A_{K+1,K-1} = -\frac{1}{2t_s},\quad
A_{K+1,K} = 0,\quad
A_{K+1,K+1} = \frac{1}{2t_s}
$$

##### 2.4.9 写入加速度边界约束

```cpp
    A.block(K + 2, 0, 1, 3) = (1 / ts / ts) * arow.transpose();
    A.block(K + 3, K - 1, 1, 3) = (1 / ts / ts) * arow.transpose();
```

第一行加速度约束对应起点加速度：

$$
\ddot{q}_{0} = \frac{P_{0} - 2P_{1} + P_{2}}{t_s^2}
$$

其矩阵系数为：

$$
A_{K+2,0} = \frac{1}{t_s^2},\quad
A_{K+2,1} = -\frac{2}{t_s^2},\quad
A_{K+2,2} = \frac{1}{t_s^2}
$$

第二行加速度约束对应终点加速度：

$$
\ddot{q}_{K-1} = \frac{P_{K-1} - 2P_K + P_{K+1}}{t_s^2}
$$

其矩阵系数为：

$$
A_{K+3,K-1} = \frac{1}{t_s^2},\quad
A_{K+3,K} = -\frac{2}{t_s^2},\quad
A_{K+3,K+1} = \frac{1}{t_s^2}
$$

##### 2.4.10 构造右端向量

```cpp
    Eigen::VectorXd bx(K + 4), by(K + 4), bz(K + 4);
```

由于轨迹为三维空间曲线，需要分别构造三个方向的右端向量：

$$
b_x,\quad b_y,\quad b_z
$$

每个向量长度均为：

$$
K + 4
$$

前 $K$ 项对应路径点位置，后 4 项对应边界导数。

```cpp
    for (int i = 0; i < K; ++i)
    {
      bx(i) = point_set[i](0);
      by(i) = point_set[i](1);
      bz(i) = point_set[i](2);
    }
```

该部分将路径点写入右端项：

$$
b_x(i) = q_i^x,\quad
b_y(i) = q_i^y,\quad
b_z(i) = q_i^z
$$

其中：

$$
i = 0,1,\dots,K-1
$$

```cpp
    for (int i = 0; i < 4; ++i)
    {
      bx(K + i) = start_end_derivative[i](0);
      by(K + i) = start_end_derivative[i](1);
      bz(K + i) = start_end_derivative[i](2);
    }
```

该部分将边界导数写入右端项：

$$
\begin{aligned}
b(K) &= \dot{q}_{0} \\
b(K+1) &= \dot{q}_{K-1} \\
b(K+2) &= \ddot{q}_{0} \\
b(K+3) &= \ddot{q}_{K-1}
\end{aligned}
$$

在 $x,y,z$ 三个方向上分别写入对应分量。

##### 2.4.11 求解控制点

```cpp
    Eigen::VectorXd px = A.colPivHouseholderQr().solve(bx);
    Eigen::VectorXd py = A.colPivHouseholderQr().solve(by);
    Eigen::VectorXd pz = A.colPivHouseholderQr().solve(bz);
```

该部分分别求解三个线性系统：

$$
A P_x = b_x
$$

$$
A P_y = b_y
$$

$$
A P_z = b_z
$$

其中：

$$
P_x =
\begin{bmatrix}
P_{0}^{x} & P_{1}^{x} & \cdots & P_{K+1}^{x}
\end{bmatrix}^{T}
$$

$$
P_y =
\begin{bmatrix}
P_{0}^{y} & P_{1}^{y} & \cdots & P_{K+1}^{y}
\end{bmatrix}^{T}
$$

$$
P_z =
\begin{bmatrix}
P_{0}^{z} & P_{1}^{z} & \cdots & P_{K+1}^{z}
\end{bmatrix}^{T}
$$

由于 $A$ 的行数大于列数：

$$
K + 4 > K + 2
$$

因此该系统通常为超定方程组。`colPivHouseholderQr().solve()` 求解的是最小二乘意义下的控制点：

$$
P^{\star} = \arg\min_P \left\|AP - b\right\|^2
$$

##### 2.4.12 转换为控制点矩阵

```cpp
    ctrl_pts.resize(3, K + 2);
    ctrl_pts.row(0) = px.transpose();
    ctrl_pts.row(1) = py.transpose();
    ctrl_pts.row(2) = pz.transpose();
```

控制点矩阵尺寸为：

$$
3 \times (K+2)
$$

矩阵每一列对应一个三维控制点：

$$
\mathrm{ctrl\_pts.col}(i) =
\begin{bmatrix}
P_i^x \\
P_i^y \\
P_i^z
\end{bmatrix}
$$

三行分别存储所有控制点的 $x,y,z$ 坐标：

$$
\mathrm{ctrl\_pts.row}(0) = P_x^T
$$

$$
\mathrm{ctrl\_pts.row}(1) = P_y^T
$$

$$
\mathrm{ctrl\_pts.row}(2) = P_z^T
$$

##### 2.4.13 数学流程总结

该函数实现的整体流程为：

$$
\{q_i\}_{i=0}^{K-1},
\dot{q}_{0},
\dot{q}_{K-1},
\ddot{q}_{0},
\ddot{q}_{K-1}
\quad
\Longrightarrow
\quad
AP=b
\quad
\Longrightarrow
\quad
\{P_i\}_{i=0}^{K+1}
$$

其核心约束为：

$$
q_i = \frac{P_i + 4P_{i+1} + P_{i+2}}{6}
$$

$$
\dot{q}_{0} = \frac{-P_{0} + P_{2}}{2t_s}
$$

$$
\dot{q}_{K-1} = \frac{-P_{K-1} + P_{K+1}}{2t_s}
$$

$$
\ddot{q}_{0} = \frac{P_{0} - 2P_{1} + P_{2}}{t_s^2}
$$

$$
\ddot{q}_{K-1} = \frac{P_{K-1} - 2P_K + P_{K+1}}{t_s^2}
$$

最终得到的控制点可进一步代入三次均匀 B 样条曲线：

$$
C(t) = \sum_i N_{i,3}(t)P_i
$$

从而生成连续、平滑、具有局部控制特性的无人机初始轨迹。

### 3. MINCO曲线

#### 3.1 数学定义构造

给定 $M$ 个时间段 $T = (T_1, \dots, T_M)$ 及其对应的时间节点 $t_0, t_1, \dots, t_M$，以及序列路径点 $q_0, q_1, \dots, q_M$。
MINCO 定义为能够最小化 $s$ 阶导数平方积分控制代价的分段轨迹：

$$J = \min_{p(t)} \int_{t_0}^{t_M} || p^{(s)}(t) ||^2 dt$$

服从航路点插值约束：
$$p(t_0) = q_0, p(t_M) = q_M$$
$$p(t_i) = q_i, \quad i = 1, \dots, M-1$$

基于变分法推导，求解该最优控制问题得到的 $p(t)$ 必然由 $M$ 段 $2s-1$ 阶多项式构成。第 $i$ 段多项式参数化表示为：
$$p_i(t) = c_{i,0} + c_{i,1} t + c_{i,2} t^2 + \dots + c_{i,N} t^N , \quad t \in [0, T_i]$$
其中 $N = 2s - 1$，多项式系数矩阵为 $C$。

#### 3.2 曲线性质与证明

1. **最优轨迹的多项式本质**
   使得代价函数最小的轨迹必须是 $2s-1$ 阶的分段多项式。
   **证明**：构造泛函 $F(t, p, p^{(s)}) = (p^{(s)})^2$。根据变分法求极值，欧拉-拉格朗日方程必须满足：
   $$\sum_{k=0}^{s} (-1)^k \frac{d^k}{dt^k} \left( \frac{\partial F}{\partial p^{(k)}} \right) = 0$$
   对于 $F = (p^{(s)})^2$，关于 $p^{(s)}$ 的偏导数为 $2p^{(s)}$，代入方程得到：
   $$(-1)^s \frac{d^s}{dt^s} (2 p^{(s)}) = 0 \implies p^{(2s)}(t) = 0$$
   既然轨迹的第 $2s$ 阶导数恒为 0，对 0 进行 $2s$ 次积分，轨迹解析式必然是一个最高次为 $2s-1$ 的多项式。

2. **内部节点连续性**
   在没有额外约束的中间航路点处，曲线自动满足 $C^{2s-2}$ 连续。
   **证明**：由变分法中的横截条件。目标函数为最小化分段区间上的 $\sum \int_{t_{i-1}}^{t_i} (p_i^{(s)})^2 dt$。
   加入插值约束并在节点处取一阶变分 $\delta J = 0$。分部积分后，节点边界处的变分项要求：
   $$\sum_{i=1}^{M-1} \sum_{k=0}^{s-1} (-1)^{s-k-1} \left( p_i^{(2s-k-1)}(t_i) - p_{i+1}^{(2s-k-1)}(t_i) \right) \delta p^{(k)}(t_i) = 0$$
   由于 $\delta p^{(k)}(t_i)$ 在无约束下可取任意值，其系数必须全为 0。因此对于所有的连续阶数 $d \in [s, 2s-2]$，必然有：
   $$p_i^{(d)}(t_i) = p_{i+1}^{(d)}(t_i)$$
   即证明了轨迹自动具有 $2s-2$ 阶连续性。

3. **时空联合求解非奇异性**
   给定固定时间分配 $T$ 和航路点 $q$，多项式系数矩阵 $C$ 存在唯一的闭式解。
   **证明**：基于插值约束、导数连续性约束以及端点边界条件，可以构建一个维数为 $M \times 2s$ 的线性方程组 $M(T) C = B(q)$。
   由于上述变分方程推导证明了该二次泛函优化问题存在唯一全局极小值，矩阵 $M(T)$ 必然是非奇异的满秩矩阵。
   因此可逆向映射 $C = M(T)^{-1} B(q)$。该性质使得 MINCO 能实现基于时间 $T$ 和点 $q$ 的降维优化。

#### 3.3 C++代码定义

##### 3.3.1 标准库实现

```cpp
#include <vector>
#include <cmath>
#include <algorithm>

class MincoPolynomial {
private:
    std::vector<double> coefficients; 
    double duration;
    int degree;

public:
    MincoPolynomial(const std::vector<double>& coeff, double T) 
        : coefficients(coeff), duration(T), degree(coeff.size() - 1) {}

    double getDuration() const {
        return duration;
    }

    double evaluate(double t) const {
        t = std::max(0.0, std::min(t, duration));
        double result = 0;
        double t_pow = 1;
        for (int i = 0; i <= degree; ++i) {
            result += coefficients[i] * t_pow;
            t_pow *= t;
        }
        return result;
    }
};

class MincoTrajectory {
private:
    std::vector<MincoPolynomial> segments;

public:
    void addSegment(const std::vector<double>& coeff, double T) {
        segments.emplace_back(coeff, T);
    }

    double evaluate(double t) const {
        if (segments.empty()) return 0.0;
        for (const auto& segment : segments) {
            double T = segment.getDuration(); 
            if (t <= T) {
                return segment.evaluate(t);
            }
            t -= T; 
        }
        return segments.back().evaluate(segments.back().getDuration());
    }
};
```

##### 3.3.2 Eigen库实现

```cpp
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

class MincoPolynomialEigen {
private:
    Eigen::VectorXd coefficients; 
    double duration;

public:
    MincoPolynomialEigen(const Eigen::VectorXd& coeff, double T) 
        : coefficients(coeff), duration(T) {}

    double getDuration() const {
        return duration;
    }

    double evaluate(double t) const {
        t = std::max(0.0, std::min(t, duration));
        int degree = coefficients.size() - 1;
        
        Eigen::VectorXd t_vec(degree + 1);
        double t_pow = 1.0;
        for (int i = 0; i <= degree; ++i) {
            t_vec(i) = t_pow;
            t_pow *= t;
        }
        
        // 利用内积计算多项式求和
        return coefficients.dot(t_vec); 
    }
};

class MincoTrajectoryEigen {
private:
    std::vector<MincoPolynomialEigen> segments;

public:
    void addSegment(const Eigen::VectorXd& coeff, double T) {
        segments.emplace_back(coeff, T);
    }

    double evaluate(double t) const {
        if (segments.empty()) return 0.0;
        for (const auto& segment : segments) {
            double T = segment.getDuration(); 
            if (t <= T) {
                return segment.evaluate(t);
            }
            t -= T; 
        }
        return segments.back().evaluate(segments.back().getDuration());
    }
};
```

## 4. 实现与应用（Implementation & Applications）

> 待补充。

## 5. 分析与总结（Analysis & Summary）

### 4. 跨界轨迹规划与应用拓展

#### 4.1 硬件物理约束决定轨迹形态

1. **自动驾驶系统**
   物理约束体现在汽车只能顺着轮胎方向行进，且转向角直接对应轨迹曲率。若使用常规多项式进行规划，曲率公式极为复杂且难以保证不突变。曲率突变在物理上要求方向盘瞬间打死，无法实现。因此自动驾驶发展出回旋曲线，该曲线曲率随弧长线性变化，允许车辆通过匀速转动方向盘完美贴合轨迹。

2. **工业机械臂**
   物理约束体现在机械臂各关节减速机昂贵且脆弱。加速度瞬间突变会产生巨大机械冲击并损坏齿轮。因此机械臂重点关注时间轴上的导数连续性，发展出关节空间的S型速度曲线。其核心为将加加速度限制为常数梯形，以物理寿命优先替代几何路径优先。

3. **旋翼无人机**
   物理约束体现在全向空间内的欠驱动特性，由于没有地面摩擦限制，飞行轨迹衍生出严格追求高阶绝对平滑的多项式轨迹群。

#### 4.2 无人机关注高阶平滑度的原因

无人机运动学具有微分平坦性质。四旋翼无人机仅有四个推力控制输入，却需控制六个自由度。其空间位置导数与底层物理指令严格绑定：

1. **速度**：位置的一阶导数。
2. **加速度**：位置的二阶导数。直接决定无人机总推力大小与机身倾斜角。
3. **加加速度**：位置的三阶导数。改变倾斜角所需角速度，直接对应机身三轴角速度指令。
4. **加加加速度**：位置的四阶导数。改变角速度所需角加速度，直接对应四个电机的电流控制指令。

轨迹的高阶导数即为控制器指令。最小化四阶导数的平方积分，其物理本质是生成一条使电机转速变化平顺、能耗低且符合硬件物理极限的运动轨迹。

#### 4.3 技术栈跨界迁移指南

1. **核心优势转换**
   在无人机中常用的优化理论与目标函数构建能力，在自动驾驶领域可直接转换为模型预测控制与序列二次规划。前端的飞行走廊寻路经验可升级为带有车辆运动学约束的混合A星等图搜索算法。

2. **非完整运动学建模补齐**
   需完成从欠驱动模型向阿克曼转向模型的认知转换。理解笛卡尔坐标系规划向基于横纵参考的Frenét坐标系规划的思维过渡。

3. **动态预测与决策补齐**
   由静态避障转向博弈环境避障。需补充行为预测模型与部分可观测马尔可夫决策过程理论，将时间维度与动态障碍物意图纳入碰撞检测框架。

4. **离散化与图优化思维补齐**
   针对算力受限的地面底盘设备，需掌握时间弹性带算法与动态窗口法，利用弹簧质点力学模型结合离散节点进行轨迹优化替代连续解析函数优化。

## 6. 关联与参考（Related Topics & References）

> 待补充。
