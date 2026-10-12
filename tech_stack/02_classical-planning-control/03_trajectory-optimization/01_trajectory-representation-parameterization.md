# Trajectory Representation and Parameterization

## 1. 概述（Overview）

### 1. Trajectory Generation Strategy

轨迹生成描述的是轨迹从哪里来，以及轨迹参数如何获得。在当前讨论范围内，可以归纳为三类。

#### 1.1 Path + Time Parameterization

首先获得一条几何路径：

$$
\mathbf{p}(s)
$$

再定义时间参数化：

$$
s=s(t)
$$

从而得到真正的时间轨迹：

$$
\mathbf{p}(t)=\mathbf{p}(s(t))
$$

其速度为：

$$
\mathbf{v}(t)
=
\frac{d\mathbf{p}}{ds}\dot{s}
$$

其加速度为：

$$
\mathbf{a}(t)
=
\frac{d^2\mathbf{p}}{ds^2}\dot{s}^2
+
\frac{d\mathbf{p}}{ds}\ddot{s}
$$

该方法将几何路径规划和时间分配分开完成。

#### 1.2 Direct Trajectory Parameter Generation

直接生成某种轨迹参数：

$$
\boldsymbol{\theta}
$$

再通过确定的参数化映射得到完整轨迹：

$$
\boldsymbol{\theta}
\rightarrow
\mathbf{p}(t)
$$

例如直接生成：

* Polynomial coefficients
* Bezier control points
* B-spline control points
* MINCO waypoint 和 segment duration

参数可以由优化器、搜索算法或神经网络直接产生。

#### 1.3 Motion Primitive Selection / Composition

首先构造有限或参数化的候选轨迹集合：

$$
\mathcal{T}
=
\{
\tau_1(t),
\tau_2(t),
\ldots,
\tau_N(t)
\}
$$

规划器从中选择一条轨迹：

$$
\tau^\star
=
\operatorname{Select}
(
\tau_1,
\ldots,
\tau_N
)
$$

也可以进一步将多个 motion primitive 拼接成完整轨迹。

Motion Primitive 本身不是一种独立的轨迹数学表示方式。每一个 primitive 内部仍然可以采用：

* Polynomial
* Bezier
* B-spline
* MINCO

等轨迹表示或参数化形式。

## 2. 基础与定义（Foundations）

### 2. Trajectory Mathematical Representation

当前讨论统一采用：

Piecewise Polynomial Trajectory

Single Polynomial 可以视为 Piecewise Polynomial 在段数 $M=1$ 时的特殊情况，因此不再单独分类。

对于三维位置轨迹：

$$
\mathbf{p}(t)
=
\begin{bmatrix}
x(t) \\
y(t) \\
z(t)
\end{bmatrix}
$$

将总轨迹划分为 $M$ 段。

第 $i$ 段轨迹表示为：

$$
\mathbf{p}(t)
=
\mathbf{p}_i(\tau_i)
$$

其中局部时间定义为：

$$
\tau_i=t-t_{i-1}
$$

并满足：

$$
t\in[t_{i-1},t_i]
$$

第 $i$ 段的 $n$ 次多项式可以写为：

$$
\mathbf{p}_i(\tau)
=
\mathbf{C}_i^T
\boldsymbol{\beta}_n(\tau)
$$

其中多项式基为：

$$
\boldsymbol{\beta}_n(\tau)
=
\begin{bmatrix}
1 &
\tau &
\tau^2 &
\cdots &
\tau^n
\end{bmatrix}^T
$$

三维 polynomial coefficient matrix 为：

$$
\mathbf{C}_i
\in
\mathbb{R}^{(n+1)\times3}
$$

对于采用位置、速度、加速度作为边界状态的情况，即 PVA boundary conditions：

$$
\mathbf{p},
\quad
\mathbf{v},
\quad
\mathbf{a}
$$

单个 segment 常采用五次多项式，因此：

$$
n=5
$$

此时：

$$
\boldsymbol{\beta}_5(\tau)
\in
\mathbb{R}^{6}
$$

并且：

$$
\mathbf{C}_i
\in
\mathbb{R}^{6\times3}
$$

因此每一段三维五次多项式包含：

$$
6\times3=18
$$

个 polynomial coefficients。

对于 $M$ 段轨迹，其完整 coefficient 集合为：

$$
\mathbf{C}
=
\{
\mathbf{C}_1,
\mathbf{C}_2,
\ldots,
\mathbf{C}_M
\}
$$

总共包含：

$$
18M
$$

个 polynomial coefficients。

不同的 Polynomial Parameterization，本质上是在回答如何通过不同类型的参数得到：

$$
\mathbf{C}_1,
\mathbf{C}_2,
\ldots,
\mathbf{C}_M
$$

后续比较默认各段持续时间 $T_i$ 已知。

如果时间本身也是自由变量，则对于 $M$ 段轨迹还需要额外引入：

$$
T_1,
T_2,
\ldots,
T_M
$$

共 $M$ 个时间参数。

## 3. 核心原理（Core Principles）

### 3. Polynomial Parameterization

#### 3.1 Direct Polynomial Coefficients

##### 参数化方法

直接使用每一段的 polynomial coefficients 作为轨迹参数：

$$
\boldsymbol{\theta}_i
=
\mathbf{C}_i
$$

对于三维五次多项式：

$$
\mathbf{C}_i
=
\begin{bmatrix}
c_{0,x} & c_{0,y} & c_{0,z} \\
c_{1,x} & c_{1,y} & c_{1,z} \\
c_{2,x} & c_{2,y} & c_{2,z} \\
c_{3,x} & c_{3,y} & c_{3,z} \\
c_{4,x} & c_{4,y} & c_{4,z} \\
c_{5,x} & c_{5,y} & c_{5,z}
\end{bmatrix}
$$

对应轨迹为：

$$
\mathbf{p}_i(t)
=
\mathbf{C}_i^T
\boldsymbol{\beta}_5(t)
$$

##### 3D PVA 实例

对于一个三维五次 polynomial segment：

$$
\mathbf{C}_i
\in
\mathbb{R}^{6\times3}
$$

因此每段参数维度为：

$$
6\times3=18
$$

对于 $M$ 段轨迹，总 polynomial coefficient 数量为：

$$
18M
$$

如果轨迹需要满足给定的 PVA 边界条件，则需要对 coefficients 施加相应的边界约束。

对于多段轨迹，还需要进一步约束不同 segment 连接位置的连续性。

##### 特性

* 最直接的 polynomial 表示方式
* 参数就是最终的 polynomial coefficients
* 参数本身的物理意义较弱
* 修改某个 coefficient 会影响整个 segment
* PVA 边界条件需要显式施加
* 多段轨迹的连续性需要显式约束

#### 3.2 Hermite Parameterization

##### 参数化方法

Hermite Parameterization 不直接给定 polynomial coefficients，而使用一个 segment 两端的状态描述轨迹。

对于 PVA 边界，单个 segment 的参数为：

$$
\boldsymbol{\theta}_i
=
[
\mathbf{p}_i,
\mathbf{v}_i,
\mathbf{a}_i,
\mathbf{p}_{i+1},
\mathbf{v}_{i+1},
\mathbf{a}_{i+1}
]
$$

通过 Quintic Hermite Interpolation，可以得到对应的 polynomial coefficients：

$$
\boldsymbol{\theta}_i
\rightarrow
\mathbf{C}_i
$$

最终轨迹仍然表示为：

$$
\mathbf{p}_i(t)
=
\mathbf{C}_i^T
\boldsymbol{\beta}_5(t)
$$

##### 3D PVA 实例

每个位置、速度和加速度均为三维向量：

$$
\mathbf{p}\in\mathbb{R}^3
$$

$$
\mathbf{v}\in\mathbb{R}^3
$$

$$
\mathbf{a}\in\mathbb{R}^3
$$

单个 segment 包含起点和终点两组 PVA：

$$
6\times3=18
$$

因此单段参数维度为：

$$
18
$$

对于 $M$ 段轨迹，可以定义 $M+1$ 个 knot states：

$$
\{
\mathbf{p}_k,
\mathbf{v}_k,
\mathbf{a}_k
\}_{k=0}^{M}
$$

每一个 knot state 包含：

$$
3+3+3=9
$$

个标量。

因此完整参数维度为：

$$
9(M+1)
$$

由于相邻 segment 共享相同的 knot state，因此可以直接保证位置、速度和加速度在连接位置连续。

##### 特性

* 参数具有明确的物理意义
* PVA 边界条件可以直接定义
* 每个 segment 由两端 PVA 状态唯一确定
* 相邻 segment 共享 knot state 时可以自然保证 PVA 连续
* 中间 knot 的速度和加速度必须显式给定或通过其他方法计算
* 适合已知边界状态下的轨迹插值

#### 3.3 Bezier Parameterization

##### 参数化方法

五次 Bezier trajectory 使用 6 个三维 control points：

$$
\mathbf{P}_0,
\mathbf{P}_1,
\mathbf{P}_2,
\mathbf{P}_3,
\mathbf{P}_4,
\mathbf{P}_5
$$

定义归一化时间：

$$
u=\frac{t}{T}
$$

其中：

$$
u\in[0,1]
$$

Bezier trajectory 为：

$$
\mathbf{p}(u)
=
\sum_{j=0}^{5}
B_j^5(u)\mathbf{P}_j
$$

其中 Bernstein basis function 为：

$$
B_j^5(u)
=
\binom{5}{j}
u^j
(1-u)^{5-j}
$$

##### 3D PVA 实例

6 个三维 control points 可以组成参数矩阵：

$$
\mathbf{P}
=
[
\mathbf{P}_0,
\mathbf{P}_1,
\mathbf{P}_2,
\mathbf{P}_3,
\mathbf{P}_4,
\mathbf{P}_5
]
$$

其维度为：

$$
\mathbf{P}
\in
\mathbb{R}^{3\times6}
$$

因此共有：

$$
3\times6=18
$$

个 control-point scalars。

对于给定 segment duration $T$ 和起点 PVA：

$$
\mathbf{p}_0,
\quad
\mathbf{v}_0,
\quad
\mathbf{a}_0
$$

起始侧 control points 可以写为：

$$
\mathbf{P}_0
=
\mathbf{p}_0
$$

$$
\mathbf{P}_1
=
\mathbf{p}_0
+
\frac{T}{5}\mathbf{v}_0
$$

$$
\mathbf{P}_2
=
\mathbf{p}_0
+
\frac{2T}{5}\mathbf{v}_0
+
\frac{T^2}{20}\mathbf{a}_0
$$

对于终点 PVA：

$$
\mathbf{p}_1,
\quad
\mathbf{v}_1,
\quad
\mathbf{a}_1
$$

终止侧 control points 可以写为：

$$
\mathbf{P}_5
=
\mathbf{p}_1
$$

$$
\mathbf{P}_4
=
\mathbf{p}_1
-
\frac{T}{5}\mathbf{v}_1
$$

$$
\mathbf{P}_3
=
\mathbf{p}_1
-
\frac{2T}{5}\mathbf{v}_1
+
\frac{T^2}{20}\mathbf{a}_1
$$

因此给定两端 PVA 和 segment duration 后，可以唯一确定该五次 Bezier segment。

##### 特性

* 使用 control points 代替 monomial coefficients
* control points 具有较直观的几何意义
* 具有 convex hull property
* 单个 control point 通常影响整个 Bezier segment
* PVA 边界可以转换为对应 control points
* 多段 Bezier 需要进一步保证 segment 连接处的连续性

#### 3.4 B-Spline Parameterization

##### 参数化方法

B-spline 主要由以下参数定义：

* Degree
* Control Points
* Knot Vector

B-spline trajectory 表示为：

$$
\mathbf{p}(t)
=
\sum_{i=0}^{N}
N_{i,d}(t)
\mathbf{P}_i
$$

其中：

$d$ 表示 B-spline degree。

$\mathbf{P}_i$ 表示第 $i$ 个 control point。

$N_{i,d}(t)$ 表示第 $i$ 个 degree-$d$ B-spline basis function。

Knot vector 表示为：

$$
\mathbf{U}
=
[
u_0,
u_1,
\ldots,
u_K
]
$$

##### 3D PVA 实例

采用五次 B-spline：

$$
d=5
$$

假设存在 8 个三维 control points：

$$
\mathbf{P}_0,
\mathbf{P}_1,
\ldots,
\mathbf{P}_7
$$

则 control-point matrix 可以表示为：

$$
\mathbf{P}
\in
\mathbb{R}^{8\times3}
$$

因此 control-point 参数共有：

$$
8\times3=24
$$

个标量。

对于 $N+1$ 个 control points 和 degree $d$，knot vector 的长度为：

$$
N+d+2
$$

当前例子中：

$$
N=7
$$

$$
d=5
$$

因此 knot vector 长度为：

$$
7+5+2=14
$$

如果使用固定的 clamped knot vector，并给定起点和终点 PVA，则边界附近的 control points 会受到 PVA 条件约束。

例如在某些五次 clamped B-spline 构造中，起始侧前三个和终止侧后三个 control points 可以由边界 PVA 决定。

此时剩余内部 control points 作为主要自由参数。

例如只有：

$$
\mathbf{P}_3
$$

和

$$
\mathbf{P}_4
$$

作为内部自由 control points 时，几何自由参数维度为：

$$
2\times3=6
$$

##### 特性

* B-spline 本身直接定义完整的 Piecewise Polynomial
* 使用 control points 和 knot vector 进行参数化
* control point 具有 local support 特性
* 单个 control point 只影响有限数量的相邻 knot spans
* 连续性由 degree 和 knot multiplicity 共同决定
* 适合局部轨迹形状调整
* 适合局部避障优化

#### 3.5 MINCO Parameterization

##### 参数化方法

MINCO 不直接将所有 polynomial coefficients 作为自由参数，而是使用：

$$
\mathbf{Q},
\quad
\mathbf{T},
\quad
\mathbf{z}_0,
\quad
\mathbf{z}_M
$$

描述轨迹。

起始边界状态为：

$$
\mathbf{z}_0
=
[
\mathbf{p}_0,
\mathbf{v}_0,
\mathbf{a}_0
]
$$

终止边界状态为：

$$
\mathbf{z}_M
=
[
\mathbf{p}_M,
\mathbf{v}_M,
\mathbf{a}_M
]
$$

内部 waypoints 为：

$$
\mathbf{Q}
=
[
\mathbf{q}_1,
\mathbf{q}_2,
\ldots,
\mathbf{q}_{M-1}
]
$$

segment duration 为：

$$
\mathbf{T}
=
[
T_1,
T_2,
\ldots,
T_M
]
$$

对于 minimum-jerk MINCO，其内部最小化：

$$
J
=
\int
\left\|
\mathbf{p}^{(3)}(t)
\right\|^2
dt
$$

对应每个 segment 为五次 polynomial：

$$
\mathbf{p}_i(t)
=
\mathbf{C}_i^T
\boldsymbol{\beta}_5(t)
$$

MINCO 建立从低维参数到完整 polynomial coefficients 的映射：

$$
(
\mathbf{z}_0,
\mathbf{Q},
\mathbf{z}_M,
\mathbf{T}
)
\rightarrow
(
\mathbf{C}_1,
\mathbf{C}_2,
\ldots,
\mathbf{C}_M
)
$$

内部 waypoint 只需要定义位置：

$$
\mathbf{q}_i
\in
\mathbb{R}^3
$$

其对应的速度和加速度不需要显式定义，而是由 minimum-control optimality condition 自动确定。

##### 3D PVA 实例

考虑两段 MINCO trajectory：

$$
M=2
$$

起始 PVA 为：

$$
\mathbf{p}_0,
\quad
\mathbf{v}_0,
\quad
\mathbf{a}_0
$$

总维度为：

$$
3+3+3=9
$$

存在一个中间 waypoint：

$$
\mathbf{q}_1
\in
\mathbb{R}^3
$$

维度为：

$$
3
$$

终点 PVA 为：

$$
\mathbf{p}_2,
\quad
\mathbf{v}_2,
\quad
\mathbf{a}_2
$$

维度为：

$$
9
$$

两个 segment durations 为：

$$
T_1,
\quad
T_2
$$

维度为：

$$
2
$$

因此完整 MINCO 构造输入维度为：

$$
9+3+9+2=23
$$

如果起点和终点 PVA 均已经固定，则自由参数只剩：

$$
\mathbf{q}_1,
\quad
T_1,
\quad
T_2
$$

因此自由参数维度为：

$$
3+1+1=5
$$

对于一般的 $M$ 段三维 MINCO trajectory，在两端 PVA 固定的情况下：

$$
\mathbf{Q}
\in
\mathbb{R}^{3(M-1)}
$$

时间参数为：

$$
\mathbf{T}
\in
\mathbb{R}^{M}
$$

因此总自由参数维度为：

$$
3(M-1)+M
$$

即：

$$
4M-3
$$

##### 特性

* 最终生成完整的 Piecewise Polynomial
* 不直接优化所有 polynomial coefficients
* 主要使用 waypoint、segment duration 和 boundary state 参数化
* 中间 waypoint 的速度和加速度由最优性条件自动确定
* 参数维度较低
* 可以内嵌 minimum jerk、minimum snap 等高阶导数代价
* 多个 segment 之间存在耦合
* 不具有 B-spline control point 那样严格的局部支撑特性

## 4. 实现与应用（Implementation & Applications）

> 待补充。

## 5. 分析与总结（Analysis & Summary）

### 4. Parameterization Summary

| Parameterization  | 核心参数                            | 3D Quintic 示例               | 参数影响范围              | 是否内嵌优化目标 |
| ----------------- | ------------------------------- | --------------------------- | ------------------- | -------- |
| Direct Polynomial | Polynomial coefficients         | 每段 18 个参数                   | 单个 segment          | 否        |
| Hermite           | Endpoint PVA states             | 每段两端共 18 个 PVA 参数           | 单个 segment          | 否        |
| Bezier            | 6 个 control points              | 每段 18 个参数                   | 单个 segment          | 否        |
| B-spline          | Control points + knot vector    | $3(N+1)$ 个 control-point 参数 | Local knot spans    | 否        |
| MINCO             | Waypoints + time + boundary PVA | 两端固定时为 $4M-3$ 个自由参数         | 多个 coupled segments | 是        |

当前讨论可以统一为：

$$
\text{Trajectory Generation Strategy}
\rightarrow
\text{Piecewise Polynomial}
\rightarrow
\text{Polynomial Parameterization}
$$

不同 Polynomial Parameterization 的主要区别包括：

* 参数语义
* 参数维度
* 参数影响范围
* 连续性构造方式
* 边界条件表达方式
* 是否内嵌优化目标

NURBS，即 Non-Uniform Rational B-Spline，属于 Piecewise Rational Polynomial，而不是严格意义上的 Piecewise Polynomial，因此不纳入当前分类主体。

## 6. 关联与参考（Related Topics & References）

> 待补充。
