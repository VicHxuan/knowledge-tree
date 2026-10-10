---
title: "PX4 / MAVROS 面试知识点整理"
sidebar:
  order: 1
---



````markdown
# PX4 / MAVROS 面试知识点整理

# 1. PX4 内部处理流程

PX4 内部控制可以理解为一条级联控制链路。上层算法可以根据自身输出形式，选择从位置、速度、加速度、姿态、角速度等不同层级接入。

```text
位置 setpoint
Xsp = [x, y, z, yaw]
        ↓
位置控制器
        ↓
速度 setpoint
Vsp = [vx, vy, vz, yaw]
        ↓
速度控制器
        ↓
加速度 setpoint
Asp = [ax, ay, az, yaw]
        ↓
加速度 + yaw 转换为姿态与推力
        ↓
姿态 setpoint
qsp = [roll, pitch, yaw] + thrust
        ↓
姿态控制器
        ↓
角速度 setpoint
Ωsp = [wx, wy, wz] + thrust
        ↓
角速度控制器
        ↓
三轴力矩 + 总推力
        ↓
Mixer / Control Allocation
        ↓
四个电机推力
T1, T2, T3, T4
````

核心理解：

```text
位置 setpoint
    PX4 负责位置环、速度环、姿态环、角速度环。

速度 setpoint
    上层直接给速度，PX4 负责速度环之后的控制。

加速度 setpoint
    上层给期望加速度，PX4 将其转换为姿态和推力。

姿态 setpoint
    上层给姿态和推力，PX4 主要负责姿态环和角速度环。

角速度 setpoint
    上层给机体系角速度和推力，PX4 主要负责角速度环和混控。

电机推力
    最底层输出，直接对应各电机控制量。
```

面试表达：

```text
PX4 内部不是简单接收一个控制指令后直接驱动电机，而是通过位置、速度、姿态、角速度等多层级联控制逐步生成电机输出。MAVROS 的不同 setpoint 接口，本质上对应 PX4 控制链路中不同深度的接入点。
```

# 2. PX4-Autopilot、Pixhawk、MAVROS 和 MAVLink 的关系

```text
PX4-Autopilot
    飞控软件 / 固件 / 飞控算法栈。

Pixhawk
    飞控硬件平台。

MAVROS
    ROS1 和 PX4 之间的 MAVLink 通信桥。

MAVLink
    上位机、飞控、地面站之间的通信协议。
```

关系可以理解为：

```text
ROS1 上层算法
        ↓
MAVROS
        ↓
MAVLink
        ↓
PX4-Autopilot
        ↓
Pixhawk / PX4 SITL
        ↓
无人机执行
```

简要说明：

```text
PX4-Autopilot 是运行在飞控板或 SITL 中的软件系统。

Pixhawk 是承载 PX4 的硬件平台。

MAVROS 是 ROS1 与 PX4 通信的桥梁，将 ROS topic / service 转换为 MAVLink 消息。

MAVLink 是 PX4、MAVROS、QGroundControl 等模块之间使用的通信协议。
```

结合项目经历：

```text
在仿真中：
    PX4-Autopilot 以 SITL 形式运行，
    Gazebo / AirSim 提供物理环境，
    MAVROS 负责 ROS1 与 PX4 通信。

在实机中：
    PX4-Autopilot 运行在 Pixhawk 上，
    机载电脑通过 MAVROS 与 Pixhawk 通信，
    上层算法通过 /mavros/setpoint_* 向 PX4 发布控制指令。
```

# 3. MAVROS setpoint 系列

我目前主要使用 `/mavros/setpoint_*` 系列接口，应当按 PX4 控制层级理解这些 topic。

## 3.1 `/mavros/setpoint_position/local`

```text
作用：
    发布局部位置目标。

典型输入：
    x, y, z, yaw

适合场景：
    上层只给目标点或轨迹采样点；
    希望 PX4 内部完成位置控制和后续控制。
```

对应控制链路：

```text
位置 setpoint
    ↓
PX4 位置控制
    ↓
速度控制
    ↓
姿态控制
    ↓
角速度控制
    ↓
电机输出
```

## 3.2 `/mavros/setpoint_velocity/cmd_vel`

```text
作用：
    发布速度目标。

典型输入：
    vx, vy, vz, yaw 或 yaw_rate

适合场景：
    局部规划器输出速度；
    避障策略输出速度；
    RL planner 输出速度指令。
```

对应控制链路：

```text
速度 setpoint
    ↓
PX4 速度控制
    ↓
姿态控制
    ↓
角速度控制
    ↓
电机输出
```

## 3.3 `/mavros/setpoint_raw/local`

```text
作用：
    发布 raw position / velocity / acceleration / yaw / yaw_rate。

典型输入：
    position
    velocity
    acceleration
    yaw
    yaw_rate

适合场景：
    自定义规划器；
    复杂轨迹跟踪；
    需要同时控制位置、速度、加速度等多种 setpoint。
```

特点：

```text
通过 type_mask 选择启用或忽略某些字段。

可以只使用位置；
可以只使用速度；
可以使用加速度；
也可以组合使用 position、velocity、acceleration、yaw。
```

对应控制链路：

```text
raw local setpoint
    ↓
PX4 根据 type_mask 解析有效字段
    ↓
位置 / 速度 / 加速度控制
    ↓
姿态控制
    ↓
角速度控制
    ↓
电机输出
```

## 3.4 `/mavros/setpoint_raw/attitude`

```text
作用：
    发布姿态、机体系角速度和推力。

典型输入：
    orientation
    body_rate
    thrust

适合场景：
    外部控制器已经计算出姿态和推力；
    几何控制器；
    RL controller；
    更接近底层控制的自定义控制器。
```

对应控制链路：

```text
姿态 / 角速度 / 推力 setpoint
    ↓
PX4 姿态控制 / 角速度控制
    ↓
Mixer / Control Allocation
    ↓
电机输出
```

注意：

```text
该接口比位置、速度 setpoint 更底层；
上层需要保证姿态、角速度和推力合理；
实机中必须做好限幅、failsafe 和遥控器接管。
```

# 4. SITL、Gazebo、AirSim、MAVROS 和 QGroundControl

```text
SITL
    Software In The Loop。
    PX4 作为软件进程运行，不依赖真实飞控硬件。

Gazebo / AirSim
    提供仿真环境、传感器模型、无人机动力学和物理交互。

MAVROS
    ROS1 和 PX4 SITL 之间的通信桥。
    上层 ROS 节点通过 MAVROS 向 PX4 发送 setpoint，并读取 PX4 状态。

QGroundControl
    地面站。
    用于查看飞控状态、参数、模式、电池、传感器、日志等信息。
```

仿真通信链路：

```text
ROS1 planner / controller
        ↓
MAVROS
        ↓
PX4 SITL
        ↓
Gazebo / AirSim
        ↓
仿真无人机运动
```

实机通信链路：

```text
ROS1 planner / controller
        ↓
MAVROS
        ↓
Pixhawk
        ↓
PX4-Autopilot
        ↓
真实无人机运动
```

仿真与实机的区别：

```text
仿真中：
    主要验证算法逻辑、通信接口、setpoint 格式和控制链路。

实机中：
    除了验证接口，还必须关注解锁、模式切换、遥控器接管、failsafe、定位质量和安全边界。
```

# 5. 结合个人经历的面试表达

```text
我在项目中使用过 PX4 系列飞控。仿真阶段使用 PX4-Autopilot 配合 Gazebo / AirSim，通过 MAVROS 与 ROS1 上层算法通信；实机阶段使用 Pixhawk 硬件飞控。

我的主要工作是完成 ROS1 上层算法与 PX4 Offboard 控制接口的对接。上层算法根据 Fast-LIO / Odometry 提供的位姿信息，生成位置、速度、raw local 或 attitude setpoint，并通过 /mavros/setpoint_* 系列 topic 发送给 PX4。

我对 PX4 的理解主要集中在上层接口和控制链路上：不同 setpoint 对应 PX4 内部不同控制层级。位置 setpoint 依赖 PX4 完成更多外环控制；速度和 raw local setpoint 适合局部规划和自定义轨迹跟踪；attitude setpoint 更接近底层控制，需要更严格的限幅和安全保护。
```

```
```
