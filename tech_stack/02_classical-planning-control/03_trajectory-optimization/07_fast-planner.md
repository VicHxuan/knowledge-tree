# Fast-Planner 框架

## 1. 概述（Overview）

> 待补充。

## 2. 基础与定义（Foundations）

> 待补充。

## 3. 核心原理（Core Principles）

### 1. FAST-Planner 轨迹规划框架

#### 1.1 数学定义与系统构造

FAST-Planner 旨在解决无人机在复杂未知环境中的高速自主飞行问题。系统由前端动力学路径搜索、后端B样条轨迹优化以及迭代时间分配三个核心模块构成。

前端采用基于混合状态A*的动力学路径搜索算法。系统状态定义为位置与速度，控制输入为加速度。在离散的控制空间中，通过求解线性二次最小时间控制问题生成运动基元。目标函数为控制代价与时间代价的积分：
$$ J(T) = \int_0^T ||u(t)||^2 dt + \rho T $$

后端采用基于B样条的轨迹优化。给定由前端生成的初始轨迹，将其参数化为3阶均匀B样条曲线。控制点集合为 $Q_0, Q_1, \dots, Q_N$。优化的目标函数由平滑度、安全性以及动力学可行性惩罚项联合构成：
$$ f_{total} = \lambda_1 f_s + \lambda_2 f_c + \lambda_3 (f_v + f_a) $$
其中平滑度 $f_s$ 基于弹性带理论约束相邻控制点的位移；碰撞代价 $f_c$ 通过查询欧几里得符号距离场（ESDF）对靠近障碍物的控制点施加排斥力；$f_v$ 和 $f_a$ 分别为速度和加速度超出物理极值时的二次惩罚项。

在优化后，将轨迹转换为非均匀B样条。计算所有速度与加速度控制点，若存在超出物理极限的情况，计算最大超限比率 $\mu_v$ 和 $\mu_a$，并局部拉伸对应节点的时间跨度（Knot Span），以确保最终轨迹的动力学可行性。

#### 1.2 算法性质与证明

1. 基于凸包性质的安全性与可行性保障
   B样条曲线严格包络在其控制点构成的凸包内部。
   证明：B样条的基函数非负且和为1。通过约束所有位置控制点 $Q_i$ 距离障碍物大于一定安全阈值，其凸包必然不与障碍物相交，从而保证整条连续轨迹的安全。同理，B样条的导数（速度和加速度）依然是B样条。只需约束速度控制点 $V_i$ 和加速度控制点 $A_i$ 在最大物理极限 $[ -v_{max}, v_{max} ]$ 内部，整条连续速度和加速度曲线就严格不会超限。

2. 时间调整策略的非保守性
   对局部时间跨度进行放缩，可严格使不可行轨迹变为可行，且不破坏轨迹几何形状。
   证明：非均匀B样条的速度控制点与其所依赖的时间跨度成反比关系。若某维度最大超限速度为 $v_m$，将对应时间跨度乘以放缩系数 $\mu_v = v_m / v_{max}$。由于时间处于分母位置，新的速度控制点计算结果将精准变为原结果的 $1/\mu_v$。此时最大速度恰好等于 $v_{max}$。该方法精确压制了超限导数，避免了传统优化中对全局时间进行统一拉伸所导致的保守飞行行为。

### 2. 算法伪代码

```text
Algorithm FAST_Planner(start_state, goal_state, ESDF_map):
    // 1. 前端动力学路径搜索
    open_set.push(start_state)
    While open_set is not empty:
        curr_state = open_set.pop()
        If Analytic_Expand(curr_state, goal_state) is feasible:
            initial_path = Construct_Path(curr_state, goal_state)
            Break
            
        For each control_input in Discretized_Space:
            next_state = Simulate_Dynamics(curr_state, control_input)
            If Is_Safe(next_state, ESDF_map) and Is_Feasible(next_state):
                Calculate_Cost(curr_state, next_state)
                open_set.push(next_state)
                
    // 2. B样条轨迹优化
    control_points = Initialize_BSpline(initial_path)
    For iteration = 1 to Max_Optimization_Iter:
        gradient = 0
        For i = 3 to N - 3:
            gradient += Smoothness_Gradient(control_points, i)
            gradient += ESDF_Collision_Gradient(control_points[i], ESDF_map)
            gradient += Dynamics_Penalty_Gradient(control_points, i)
        
        control_points -= learning_rate * gradient
        If gradient < threshold: Break
        
    // 3. 迭代时间调整
    knot_vector = Uniform_Knots(num_points)
    While True:
        max_v, max_a = Check_Derivatives(control_points, knot_vector)
        If max_v <= V_MAX and max_a <= A_MAX:
            Break
            
        If max_v > V_MAX:
            ratio_v = max_v / V_MAX
            Scale_Knot_Span(knot_vector, violation_index, ratio_v)
            
        If max_a > A_MAX:
            ratio_a = sqrt(max_a / A_MAX)
            Scale_Knot_Span(knot_vector, violation_index, ratio_a)
            
    Return Non_Uniform_BSpline(control_points, knot_vector)
```

## 4. 实现与应用（Implementation & Applications）

### 3. C++代码实现

以下代码展示 FAST-Planner 后端最核心的非均匀B样条时间调整算法逻辑。

```cpp
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

const int PB = 3; 

class FastPlannerBackend {
private:
    std::vector<Eigen::Vector3d> control_points;
    std::vector<double> knot_vector;
    double vmax;
    double amax;

    Eigen::Vector3d calcVelocityControlPoint(int i) {
        double dt = knot_vector[i + PB + 1] - knot_vector[i + 1];
        return PB * (control_points[i + 1] - control_points[i]) / dt;
    }

    Eigen::Vector3d calcAccelerationControlPoint(int i) {
        double dt1 = knot_vector[i + PB + 1] - knot_vector[i + 1];
        double dt2 = knot_vector[i + PB + 2] - knot_vector[i + 2];
        double dt_acc = knot_vector[i + PB + 1] - knot_vector[i + 2];
        
        Eigen::Vector3d V_i = PB * (control_points[i + 1] - control_points[i]) / dt1;
        Eigen::Vector3d V_next = PB * (control_points[i + 2] - control_points[i + 1]) / dt2;
        
        return (PB - 1) * (V_next - V_i) / dt_acc;
    }

public:
    FastPlannerBackend(const std::vector<Eigen::Vector3d>& pts, double initial_dt, double v_max, double a_max) 
        : control_points(pts), vmax(v_max), amax(a_max) {
        
        int m = pts.size() + PB;
        knot_vector.resize(m + 1);
        for (int i = 0; i <= m; ++i) {
            knot_vector[i] = i * initial_dt;
        }
    }

    void iterativeTimeAdjustment() {
        bool feasible = false;
        int max_iter = 10;
        int iter = 0;

        while (!feasible && iter < max_iter) {
            feasible = true;
            int num_v_ctrl = control_points.size() - 1;
            int num_a_ctrl = control_points.size() - 2;

            for (int i = 0; i < num_v_ctrl; ++i) {
                Eigen::Vector3d V = calcVelocityControlPoint(i);
                double max_v_axis = V.cwiseAbs().maxCoeff();
                
                if (max_v_axis > vmax) {
                    feasible = false;
                    double ratio = std::min(max_v_axis / vmax, 1.1); 
                    double original_span = knot_vector[i + PB + 1] - knot_vector[i + 1];
                    double added_time = original_span * (ratio - 1.0);
                    
                    for (size_t j = i + PB + 1; j < knot_vector.size(); ++j) {
                        knot_vector[j] += added_time;
                    }
                }
            }

            for (int i = 0; i < num_a_ctrl; ++i) {
                Eigen::Vector3d A = calcAccelerationControlPoint(i);
                double max_a_axis = A.cwiseAbs().maxCoeff();
                
                if (max_a_axis > amax) {
                    feasible = false;
                    double ratio = std::min(std::sqrt(max_a_axis / amax), 1.1);
                    
                    for (int k = i + 1; k <= i + PB + 1; ++k) {
                        double span = knot_vector[k + 1] - knot_vector[k];
                        double added_time = span * (ratio - 1.0);
                        for (size_t j = k + 1; j < knot_vector.size(); ++j) {
                            knot_vector[j] += added_time;
                        }
                    }
                }
            }
            iter++;
        }
    }
};
```

## 5. 分析与总结（Analysis & Summary）

> 待补充。

## 6. 关联与参考（Related Topics & References）

> 待补充。
