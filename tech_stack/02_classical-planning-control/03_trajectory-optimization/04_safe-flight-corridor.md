# 安全飞行走廊（SFC）

## 1. 概述（Overview）

> 待补充。

## 2. 基础与定义（Foundations）

> 待补充。

## 3. 核心原理（Core Principles）

### 1. 飞行安全走廊 (SFC) 核心概念

#### 1.1 数学定义构造

给定三维环境中的自由空间 $\mathcal{F}$ 与分段线性路径骨架 $P = p_0 \rightarrow p_1 \rightarrow \dots \rightarrow p_N$。飞行安全走廊（Safe Flight Corridor, SFC）旨在为每条路径线段 $L_i$ 构造一个凸区域 $C_i$，将原始非凸的避障问题转化为一系列连通的凸空间组合：

$$\mathcal{S} = {C_0, C_1, \dots, C_{N-1}}$$

每个凸区域通常由一组超半空间的交集构成，定义为凸多面体：

$$C_i = {p \in \mathbb{R}^3 \mid A_i p \le b_i}$$

其中 $A_i \in \mathbb{R}^{m \times 3}$ 为多面体的法向量矩阵，$b_i \in \mathbb{R}^m$ 为偏置向量。飞行器轨迹 $\Phi_i(t)$ 在第 $i$ 段必须完全位于 $C_i$ 内，从而将避障约束转化为线性不等式：

$$A_i \Phi_i(t) \le b_i, \quad t \in [0, T_i]$$

#### 1.2 走廊性质与推导

1. 几何包容性
   走廊的第 $i$ 个多面体 $C_i$ 必须完全包含对应的路径骨架线段 $L_i$。
   证明：线段 $L_i$ 可参数化为 $p(\lambda) = p_i + \lambda(p_{i+1} - p_i)$，$\lambda \in [0,1]$。由于 $C_i$ 是通过围绕 $L_i$ 膨胀并生成的半空间交集，对于所有的切割超平面 $a_j^\top p \le b_j$，在线段生成阶段已约束 $a_j^\top p(\lambda) \le b_j$ 恒成立。因此 $L_i \subset C_i$。

2. 绝对安全性
   走廊 $C_i$ 与环境中的障碍物点集 $\mathcal{O}$ 的交集为空集。
   证明：对于任意障碍物点 $o \in \mathcal{O}$，在构造 $C_i$ 时，每个超平面 $H_j = {p \mid a_j^\top p \le b_j}$ 均被设计为分离平面，使得至少存在一个 $j$ 满足 $a_j^\top o > b_j$。因此，没有障碍物点能同时满足 $C_i$ 的所有半空间不等式，即 $C_i \cap \mathcal{O} = \varnothing$。

3. 节点连通性
   相邻多面体 $C_i$ 和 $C_{i+1}$ 必然存在重叠，且路径节点 $p_{i+1}$ 位于重叠区域内。
   证明：根据包容性定理，$L_i \subset C_i$ 且 $L_{i+1} \subset C_{i+1}$。路径节点 $p_{i+1}$ 是 $L_i$ 的终点，同时是 $L_{i+1}$ 的起点，因此 $p_{i+1} \in C_i$ 且 $p_{i+1} \in C_{i+1}$。取交集即得 $p_{i+1} \in C_i \cap C_{i+1} \ne \varnothing$。

#### 1.3 C++代码定义

```cpp
#include <vector>

struct Point {
    double x, y, z;
    Point(double x=0, double y=0, double z=0) : x(x), y(y), z(z) {}
};

struct Hyperplane {
    Point normal;
    double d; 
    // 定义平面: normal.x * p.x + normal.y * p.y + normal.z * p.z <= d
};

class ConvexPolyhedron {
private:
    std::vector<Hyperplane> faces;

public:
    ConvexPolyhedron(const std::vector<Hyperplane>& h_planes) : faces(h_planes) {}

    void add_face(const Point& normal, double d) {
        faces.push_back({normal, d});
    }

    // 验证点是否在多面体安全走廊内
    bool is_inside(const Point& p) const {
        for (const auto& face : faces) {
            double dot_product = face.normal.x * p.x + 
                                 face.normal.y * p.y + 
                                 face.normal.z * p.z;
            if (dot_product > face.d) {
                return false; // 违反任何一个半空间约束则在外部
            }
        }
        return true;
    }
};
```

### 2. 椭球引导凸分解

#### 2.1 数学定义构造

为了生成贴合障碍物环境的 $C_i$，通常使用椭球膨胀法（如 Liu et al. 2017）。围绕线段中心 $d_i$ 构建椭球 $\xi(E, d_i)$：

$$\xi(E, d_i) = {p \in \mathbb{R}^3 \mid (p - d_i)^\top Q (p - d_i) \le 1}$$

其中形状矩阵 $Q = E^{-\top}E^{-1}$ 为正定矩阵。
算法通过不断膨胀椭球，直到触碰最近的局部障碍物点 $p_j^c$，并在该触点处生成切平面（超平面）：

$$H_j = {p \in \mathbb{R}^3 \mid a_j^\top p \le b_j}$$

切平面的法向量为 $a_j = 2Q(p_j^c - d_i)$，偏置为 $b_j = a_j^\top p_j^c$。

#### 2.2 分解性质与推导

1. 障碍物隔离性
   生成的切平面 $H_j$ 必然将障碍物点 $p_j^c$ 及部分周围障碍物排斥在椭球与多面体之外。
   证明：该椭球函数 $f(p) = (p - d_i)^\top Q (p - d_i)$ 为严格凸函数。切平面法向量 $a_j$ 正是 $f(p)$ 在边界点 $p_j^c$ 处的梯度 $\nabla f(p_j^c)$。根据凸函数的一阶条件，对于椭球内任意一点 $p_{in}$，均有 $a_j^\top (p_{in} - p_j^c) < 0$，即 $a_j^\top p_{in} < b_j$。因此切平面成功将包含椭球的一侧作为安全半空间，将障碍物分离。

2. 物理半径收缩安全保证
   若在质点模型下生成了多面体 $C_i = {p \mid a_j^\top p \le b_j}$，将其向内收缩无人机半径 $r_r$ 后，体积缩小但物理绝对安全。
   证明：定义无人机为半径 $r_r$ 的球体。要使机身完全处于原多面体内，机体中心 $p$ 到各平面的距离必须大于 $r_r$。点到平面 $a_j^\top p = b_j$ 的距离公式为 $d = \frac{b_j - a_j^\top p}{|a_j|_2} \ge r_r$。移项变形可得 $a_j^\top p \le b_j - r_r |a_j|_2$。这构成了一个向内收缩了 $r_r$ 偏置量的新多面体约束。

#### 2.3 C++代码定义

```cpp
#include <vector>

struct Ellipsoid {
    // 为保持代码简洁无依赖，此处不引入矩阵库，用简化表达表示正定矩阵Q及其中心
    double Q[3][3];
    Point center;

    // 获取触点处的切平面 (梯度法向)
    Hyperplane get_tangent_plane(const Point& obstacle_pt) const {
        Point diff(obstacle_pt.x - center.x, 
                   obstacle_pt.y - center.y, 
                   obstacle_pt.z - center.z);
        Point normal;
        // 梯度 a = 2 * Q * (p - d)
        normal.x = 2 * (Q[0][0]*diff.x + Q[0][1]*diff.y + Q[0][2]*diff.z);
        normal.y = 2 * (Q[1][0]*diff.x + Q[1][1]*diff.y + Q[1][2]*diff.z);
        normal.z = 2 * (Q[2][0]*diff.x + Q[2][1]*diff.y + Q[2][2]*diff.z);

        // 偏置 b = a^T * p
        double d = normal.x * obstacle_pt.x + 
                   normal.y * obstacle_pt.y + 
                   normal.z * obstacle_pt.z;
                   
        return {normal, d};
    }
};
```

### 3. SFC 约束下的轨迹优化

#### 3.1 数学定义构造

将微分平坦轨迹表示为分段伯恩斯坦（Bézier）多项式 $\Phi_i(s)$，$s \in [0, 1]$：

$$\Phi_i(s) = \sum_{k=0}^{K} P_{i,k} B_{k,K}(s)$$

目标函数为最小化高阶导数（如 Minimum Snap），即：

$$J = \min \sum_{i=0}^{N-1} \int_{0}^{T_i} \left| \frac{d^4}{d\tau^4}\Phi_i(\tau) \right|^2 d\tau$$

将其嵌入 SFC 约束中，只需约束所有的 Bézier 控制点 $P_{i,k}$ 位于多面体 $C_i$ 内部：

$$A_i P_{i,k} \le b_i, \quad \forall k \in [0, K]$$

该形式构成了一个标准的二次规划（QP）问题。

#### 3.2 优化性质与推导

1. 连续轨迹的安全凸包保证
   只需约束有限的离散控制点 $P_{i,k}$，即可绝对保证连续时间上的轨迹 $\Phi_i(s)$ 全程不出界。
   证明：伯恩斯坦基函数满足规范性和非负性（$\sum B_{k,K}(s) = 1$ 且 $B_{k,K}(s) \ge 0$）。因此 $\Phi_i(s)$ 是其控制点集 ${P_{i,k}}$ 的凸组合。根据凸多面体定义，若所有的顶点（此处对应控制点 $P_{i,k}$）均满足线性约束 $A_i p \le b_i$，则其凸组合必然也满足该线性不等式。即 $A_i \Phi_i(s) \le b_i$ 对于任意 $s \in [0,1]$ 恒成立，轨迹被严格封印在走廊内。

2. 时间缩放适应硬件极限
   若优化出的轨迹其速度、加速度等物理量超出无人机硬件极值，可通过时间缩放因子 $\gamma$ 进行等比例修正而不改变空间几何形态。
   证明：设原轨迹时长为 $T_i$，引入参数 $t = \gamma \tau$。根据链式法则，轨迹的 $d$ 阶导数为 $\frac{d^d p}{dt^d} = \frac{1}{\gamma^d} \frac{d^d p}{d\tau^d}$。若原轨迹最大速度超标量为 $\frac{v_{max}}{\bar{v}}$，最大加速度超标量为 $\sqrt{\frac{a_{max}}{\bar{a}}}$，只需取最大超标比率作为 $\gamma$，即可使新时长 $T_i' = \gamma T_i$ 下的所有动力学导数均衰减至硬件容许的上限域内。

#### 3.3 C++代码定义

```cpp
#include <vector>

class BezierTrajectorySFC {
private:
    std::vector<Point> control_points;
    ConvexPolyhedron safe_corridor;

public:
    BezierTrajectorySFC(const std::vector<Point>& pts, const ConvexPolyhedron& corridor)
        : control_points(pts), safe_corridor(corridor) {}

    // 验证整个轨迹段是否严格满足SFC安全要求
    bool verify_safety() const {
        for (const auto& cp : control_points) {
            if (!safe_corridor.is_inside(cp)) {
                // 只要有任意控制点出界，根据凸包性质，无法保证轨迹绝对安全
                return false; 
            }
        }
        return true; // 所有控制点在多面体内，轨迹100%安全
    }
};
```

## 4. 实现与应用（Implementation & Applications）

> 待补充。

## 5. 分析与总结（Analysis & Summary）

### 4. 走廊形态演进与跨界拓展

#### 4.1 走廊形态的代际演进

1. **轴对齐栅格走廊 (Axis-Aligned Grid Corridor)**
   早期 SFC 由八叉树的三维长方体叠加构成。计算极其简单，但受坐标轴对齐限制极大，产生的自由空间保守，不利于斜向飞行。
2. **椭球引导多面体 (Ellipsoid-driven Polyhedron)**
   以 Liu 等 2017 年工作为代表，利用椭球法向生成切平面，摆脱了栅格束缚。多面体紧贴障碍物，极大地提升了轨迹优化空间的机动上限。
3. **时空语义走廊 (Spatio-temporal Semantic Corridor)**
   在自动驾驶和复杂城市中，SFC 将纯几何空间 $A_i p \le b_i$ 拓展为时空域 $A_i p + c_i t \le b_i$。此多面体同时封禁了空间障碍物与动态交通规则（如红绿灯时序），统一为 QP 约束。
4. **多机相对走廊 (Relative Safe Flight Corridor)**
   通过定义相对 Bernstein 多项式的控制点 $P_{ab,k} = P_{a,k} - P_{b,k}$，将机器人防互碰的高维非凸问题，映射为相对坐标系下的安全凸多面体，为机群编队提供了封闭数学解。

#### 4.2 SFC 核心依赖凸约束的原因

复杂的三维世界避障本身是一个 $p(t) \notin \mathcal{O}$ 的非凸、高度非线性的逻辑判断。数值优化器（特别是处理高阶导数的高频控制器）面对非凸惩罚项时极易陷入局部死锁甚至发散。SFC 运用“分而治之”的思想，将不可导的避障空间置换为多个“凸半空间交集”。它为二次规划 (QP) 直接提供了干净、可求导的线性不等式域（Linear Inequalities），将求解耗时从非线性规划的数百毫秒级压缩至几毫秒的实时级别。

#### 4.3 技术栈跨界迁移指南

1. **从空中到地面的降维映射**
   无人机领域的 SFC 多面体生成策略，可被降维打击至地面的自动驾驶与移动底盘规划。空中的三维椭球膨胀退化为地面的二维凸多边形或圆形气泡（Bubble）。
2. **从质点运动学到非完整约束**
   四旋翼基于微分平坦视为质点，而地面车辆须补齐阿克曼转向或差速模型。轨迹不可随意在多面体内左右平移，需将 SFC 叠加在 Frenét 坐标系下，结合曲率极值约束求解。
3. **离散图优化思维的补充**
   对于算力孱弱的边缘设备，可能无法负担 Bernstein 凸包的密集 QP 求解。可采用类似时间弹性带（TEB）的弹簧力学模型，将 SFC 退化为离散点到障碍物边缘的安全排斥力场，以局部次优解换取工程绝对可用性。

## 6. 关联与参考（Related Topics & References）

> 待补充。
