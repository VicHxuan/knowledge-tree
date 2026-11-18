# PX4 与 MAVROS 控制接口

## 1. 概述

PX4 运行状态估计、控制与执行器输出，MAVROS 将 ROS 消息与 MAVLink 数据连接起来。上层规划器首先确定输出位置、速度、姿态还是角速度，再选择对应接口和控制层级。

## 2. 基础与定义

### 2.1 控制链

典型多旋翼控制链为位置与速度参考、加速度或推力方向、姿态、角速度、控制分配和执行器。外部控制器可接入其中某一层，接管层级决定下游仍由 PX4 负责的环节。

### 2.2 坐标与模式

ROS 常用 ENU 世界坐标和 FLU 机体坐标，PX4 常用 NED 与 FRD。MAVROS 的相关插件执行约定的转换；应用端必须核对所用消息与插件，避免重复转换。

Offboard 模式持续接收外部控制有效信号。按 PX4 v1.16 文档，进入前需要持续发送超过一秒，信号频率应高于 2 Hz；具体指令频率还要满足跟踪需求。信号丢失后的超时和行为由对应参数及模式条件确定。[Offboard 文档](https://docs.px4.io/v1.16/en/flight_modes/offboard)

## 3. 核心原理

### 3.1 指令接口

| ROS1 MAVROS 接口 | 控制含义 | 关键字段 |
|---|---|---|
| `setpoint_position/local` | 位置与姿态参考 | 位置、姿态四元数 |
| `setpoint_velocity/cmd_vel` | 速度与偏航角速度参考 | 线速度、角速度 |
| `setpoint_raw/local` | 按掩码选择位置、速度或加速度字段 | `coordinate_frame`、`type_mask` |
| `setpoint_raw/attitude` | 按掩码选择姿态或角速度，并给推力 | 姿态、机体角速度、归一化推力 |

表中是常见默认命名，实际名称受命名空间与重映射影响。`type_mask` 决定字段参与控制的方式；未启用字段的数值不能用于推断最终控制目标。[MAVROS 示例](https://docs.px4.io/v1.16/en/ros/mavros_offboard_python)

### 3.2 轨迹与低层控制

规划器给出 $p^{\mathrm{ref}},v^{\mathrm{ref}},a^{\mathrm{ref}}$ 后，可以交给 PX4 位置控制链，也可以由机载外部控制器求姿态与推力，再接入更低层。两种接入分别记录参考、控制误差和饱和状态。

### 3.3 仿真与真实飞控

SITL 在仿真中运行飞控软件，HITL 使用实体飞控连接仿真输入。验证沿“指令正确发布 → 飞控接收且模式有效 → 目标字段生效 → 执行响应符合预期”逐层推进。

## 4. 实现与应用

### 4.1 接入验证

先在 SITL 中验证坐标与模式，再进行拆桨地面检查。使用单轴位置或速度输入检查符号，核对姿态四元数顺序、推力范围和消息时间。记录飞控状态、参考值和实测值，观察失联与超时后的预期行为。

<details class="kt-derivation" id="enu-ned">
<summary>ENU 与 NED 变换</summary>

世界系向量转换为

$$
\begin{bmatrix}x_N\\y_E\\z_D\end{bmatrix}
=\begin{bmatrix}0&1&0\\1&0&0\\0&0&-1\end{bmatrix}
\begin{bmatrix}x_E\\y_N\\z_U\end{bmatrix}.
$$

机体 FLU 到 FRD 的轴向转换为 $\operatorname{diag}(1,-1,-1)$。姿态表示同时涉及世界系与机体系变换，应按完整旋转关系处理，并用前、左、上三个基向量验证。MAVROS 已执行转换的字段在应用端沿 ROS 约定使用。

</details>

### 4.2 排错顺序

先检查定位有效性与连接状态，再检查模式、指令频率和掩码，最后检查坐标、跟踪误差和控制器饱和。通信成功与模式切换成功分别记录，避免把发布话题等同于飞控正在执行。

## 5. 分析与总结

接入层级决定外部算法承担的控制责任。位置和速度策略依赖下游控制动态，姿态与推力接口还要求外部控制器正确处理重力补偿和推力标定。所有实验记录 PX4、MAVROS、机型配置与参数版本。

## 6. 关联与参考

- [PX4 控制原理](/knowledge-tree/classical-planning-control/tracking-control/px4-controller-attitude-rate/)。
- [ROS1 系统集成](./01_ros1.md)。
- [PX4 v1.16 Offboard](https://docs.px4.io/v1.16/en/flight_modes/offboard)。
- [MAVROS 官方实现](https://github.com/mavlink/mavros)。
