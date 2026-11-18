# YOPO 轨迹规划

## 1. 概述

YOPO 用一次网络前向预测多个运动基元的终端状态修正与评分，再生成所选轨迹。原始方法通过环境代价对轨迹参数的梯度训练网络，深度观测与机体运动状态用于在线推理。本文以原论文的单段多项式模型为基准。[原论文](https://tju-air-lab.github.io/projects/YOPO/YOPO.pdf)

## 2. 基础与定义

### 2.1 网络输入与输出

输入包含深度图、速度、加速度和目标方向。每个运动基元对应视场中的一个方向，输出终点偏移、终端速度、终端加速度及评分。不同坐标系之间的转换在编码和解码两端成对出现。

### 2.2 轨迹参数

固定时长 $T$，一维五次多项式为 $p(t)=\sum_{k=0}^5c_kt^k$。给定起终点位置、速度和加速度组成的边界向量 $b$，有

$$
M(T)c=b,\qquad c=M(T)^{-1}b.
$$

三轴分别求解。起始边界来自状态估计，自由终端边界由网络产生。固定时长下，轨迹系数与边界状态呈线性关系。

## 3. 核心原理

### 3.1 运动基元与候选评分

空间分散的基元给多条候选提供不同起始方向。网络在各基元附近预测连续修正，同时估计对应轨迹的质量。原论文将评分目标定义为代价的相反数，并用 Smooth L1 训练评分分支；部署选择较高评分的候选。[评分定义](https://tju-air-lab.github.io/projects/YOPO/YOPO.pdf)

后续实现可能使用预测代价并取最小值，接入时应核对评分方向、单位和归一化。多候选描述的是不同局部解，不自动构成覆盖全部可行路径的集合。

### 3.2 轨迹代价与网络梯度

以平滑、避障和目标三类代价说明训练链路：

$$
J=\lambda_sJ_s+\lambda_oJ_o+\lambda_gJ_g,\qquad
\nabla_\theta J=\left(\frac{\partial b_P}{\partial\theta}\right)^{\mathsf T}\nabla_{b_P}J.
$$

$b_P$ 为自由终端状态。训练时的环境地图提供距离和梯度，代价梯度经轨迹解码传给网络；推理只使用约定的传感器输入。

<details class="kt-derivation" id="trajectory-cost-gradient">
<summary>边界状态的代价梯度</summary>

将系数写为 $c=L_Fb_F+L_Pb_P$。若平滑代价为 $J_s=c^{\mathsf T}Qc$，其中 $Q$ 对称，则

$$
\nabla_{b_P}J_s=2L_P^{\mathsf T}Qc.
$$

对采样得到的避障代价 $J_o=\sum_j w_j\phi(d(p(t_j)))$，有

$$
\nabla_{b_P}J_o=\sum_jw_j
\left(\frac{\partial p(t_j)}{\partial b_P}\right)^{\mathsf T}
\phi'(d_j)\nabla_pd_j.
$$

这里 $d$ 是环境距离查询，$w_j$ 是数值积分权重。可用中心差分核对梯度方向；距离场插值和采样精度决定数值误差。固定时长是上述线性映射的条件，学习时长时还需对 $M(T)$ 和积分边界求导。

</details>

### 3.3 训练数据与推理流程

训练数据关联深度观测、运动状态、目标和用于代价评价的环境信息。轨迹分支学习降低代价，评分分支学习候选比较。原论文还对参与终端状态优化的候选设定质量筛选，复现时应保留该规则。

在线流程为深度预处理、状态变换、候选预测、评分选择、轨迹求解和世界系参考输出。控制器跟踪参考轨迹，新观测到来后重新规划。

## 4. 实现与应用

### 4.1 模型接入

| 接口 | 核查内容 |
|---|---|
| 深度输入 | 分辨率、量纲、近远裁剪、无效值填充 |
| 状态输入 | 坐标系、速度与加速度尺度、目标方向 |
| 基元解码 | 基元排列、终点偏移范围、导数坐标系 |
| 轨迹求解 | 时间、边界顺序、系数排列 |
| 评分选择 | 分数方向、无效候选与独立碰撞检查 |

原论文、官方不同分支以及本地多视角、MINCO 或选择器变体分别记录版本。不能仅凭相同的权重文件名确定输入输出语义。[官方实现](https://github.com/TJU-Aerial-Robotics/YOPO)

### 4.2 验证方法

用固定边界检查轨迹端点与导数；用固定输入比较训练框架和部署引擎输出；用记录场景对照预测评分与实际代价；最后运行闭环导航，统计重规划失败、交接误差和控制跟踪误差。

## 5. 分析与总结

YOPO 将部分在线优化工作转移到离线网络训练。其表现取决于训练场景、代价设计、感知覆盖和动作执行条件。动态障碍适配需要明确运动表示与未来碰撞评价；静态距离场代价本身只描述空间几何。

## 6. 关联与参考

- [轨迹表示](/knowledge-tree/classical-planning-control/trajectory-optimization/trajectory-representation-parameterization/)。
- [轨迹候选生成与选择](./04_trajectory-candidate-selection.md)。
- [YOPO 项目](/knowledge-tree/projects/yopo/)。
- [Lu 等：You Only Plan Once](https://tju-air-lab.github.io/projects/YOPO/YOPO.pdf)。
- [TJU-Aerial-Robotics/YOPO](https://github.com/TJU-Aerial-Robotics/YOPO)。
