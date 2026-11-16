# 安全飞行走廊

## 1. 概述

安全飞行走廊将地图中的自由空间表示为一系列相交凸集，使连续轨迹避障转化为可计算的几何约束。走廊生成负责自由空间认证，轨迹优化负责在这些集合内生成满足运动条件的曲线。

## 2. 基础与定义

### 2.1 凸区域

第 $i$ 个走廊单元表示为半空间交集

$$
\mathcal C_i=\{p\mid A_ip\le b_i\}.
$$

相邻单元交集用于安排段间连接。地图分辨率、未知区域处理和机器人尺寸共同决定可认证的自由区域。

### 2.2 机器人包络

若机器人由半径 $r$ 的球包络，法向为 $a_j$ 的半空间可收缩为

$$
a_j^\top p\le b_j-r\|a_j\|.
$$

这一约束保证以 $p$ 为球心的机器人包络位于原半空间内。

## 3. 核心原理

### 3.1 走廊生成

以无碰撞路径为种子，在局部地图中构造分离障碍的半空间，再求其交集。椭球膨胀可为分离平面提供几何方向；生成后检查种子包含性、相邻交集与障碍隔离。

对椭球 $(p-d)^\top Q(p-d)\le1$，边界点 $p_*$ 的外法向与 $Q(p_*-d)$ 平行。切平面提供支撑方向，实际分离还需结合障碍体积和安全裕量。

### 3.2 曲线约束

若 Bézier 段的全部控制点 $P_k$ 满足

$$
A_iP_k\le b_i\quad\text{对所有 }k,
$$

则整段曲线包含于 $\mathcal C_i$。B 样条可对每个节点区间的激活控制点施加同类约束。
<details class="kt-derivation" id="corridor-convexity">
<summary>凸包包含性证明</summary>

曲线写为 $p(u)=\sum_k B_k(u)P_k$，其中 $B_k\ge0$、$\sum_kB_k=1$。于是

$$
A_ip(u)=\sum_kB_k(u)A_iP_k\le\sum_kB_k(u)b_i=b_i.
$$

证明依赖同一个凸集合包含全部激活控制点。各控制点单独处于自由空间，只能提供离散位置检查。

</details>

### 3.3 动态可行性

走廊约束限制空间位置，速度、加速度与推力约束由轨迹导数给出。统一放慢时间 $t'=\beta t$ 会使 $k$ 阶导数缩放为 $\beta^{-k}$，并保留几何路径。局部节点时间调整需要重新检查曲线与约束。

## 4. 实现与应用

构建机器人膨胀后的地图，生成并验证凸走廊；将曲线段分配给单元，约束控制点并优化边界、平滑度与时间；最后检查整条轨迹的空间覆盖、连接连续性和动力学需求。

<details class="kt-derivation" id="corridor-membership">
<summary>半空间集合实现</summary>

以下代码仅负责点的半空间成员检查，输入平面应已完成地图与包络处理。

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

</details>

## 5. 分析与总结

凸走廊把复杂障碍约束转化为有限线性条件，代价是对自由空间的保守表示。狭窄区域需要同时检查走廊是否足够宽、机器人包络是否一致以及动力学是否允许通过。

## 6. 关联与参考

- [Wang 等：Geometrically Constrained Trajectory Optimization](https://arxiv.org/abs/2103.00190)
- [Bézier 曲线与 B 样条](03_bezier-bspline.md)
- [轨迹优化问题](05_trajectory-optimization-formulation.md)
