---
title: "PX4 Controller 代码讲解清单"
sidebar:
  order: 4
---

## 一. 代码功能简介

该代码实现一套基于 PX4 控制结构的无人机姿态与角速度控制器，支持批量并行环境计算。

整体控制流程为：

```text
期望姿态 / 期望角速度
        ↓
姿态控制环
        ↓
期望机体系角速度
        ↓
角速度控制环
        ↓
归一化三轴力矩
        ↓
归一化总推力 + 三轴力矩
        ↓
Control Allocation
        ↓
各电机控制指令
```

控制器提供两种主要输入接口。

| 控制接口               | 输入                               | 控制流程                    |
| ------------------ | -------------------------------- | ----------------------- |
| `compute_attitude` | `[roll, pitch, yaw, collective]` | 姿态环 → 角速度环 → 控制分配       |
| `compute_rate`     | `[p, q, r, collective]`          | 角速度环 → 控制分配             |
| `compute`          | `[roll, pitch, yaw, collective]` | 默认调用 `compute_attitude` |

其中：

* `roll`、`pitch`、`yaw` 为期望姿态角
* `p`、`q`、`r` 为期望机体系角速度
* `collective` 为归一化总推力指令
* 姿态和角速度相关计算采用四元数完成
* 姿态环输出期望机体系角速度
* 角速度环输出归一化三轴力矩
* 控制分配模块将总推力和三轴力矩转换为具体电机指令

控制器主要依赖两类参数。

第一类为控制器参数，由 `ControllerConfig` 提供，主要包括：

```text
attitude_p
yaw_weight

rate_p
rate_i
rate_d
rate_k

rate_limit_rad_s
integrator_limit
angular_accel_filter_alpha
```

第二类为无人机动力学参数，由 `P300ProDynamics` 提供，用于完成：

```text
归一化总推力
+
归一化三轴力矩
        ↓
各电机控制量
        ↓
实际可实现的推力和力矩
```

控制器最终输出：

```text
actuator_control
```

即对应无人机各执行器的控制指令。

同时内部保存：

```text
last_rate_setpoint
last_normalized_torque
last_achieved_normalized_wrench
last_achieved_wrench
```

用于记录期望角速度、期望归一化力矩、实际实现的归一化控制量以及实际实现的物理控制量。

---

# 二. 模块拆分讲解

## 二.1. 依赖模块

```python
from __future__ import annotations

import math

import torch

from .config import ControllerConfig
from .p300pro_dynamics import P300ProDynamics
```

主要依赖：

```text
torch
```

用于批量 Tensor 运算以及 GPU 并行计算。

```text
ControllerConfig
```

用于提供姿态环、角速度环以及滤波和限幅相关控制参数。

```text
P300ProDynamics
```

用于描述具体无人机机型的动力学参数和电机控制分配。

---

## 二.2. 四元数与旋转处理方法

无人机姿态主要使用四元数表示，因此首先定义一系列四元数和坐标变换相关方法。

### 二.2.1. 四元数乘法

```python
def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * w2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )
```

用于计算两个四元数的乘积。

在姿态控制过程中，四元数乘法主要用于：

```text
组合旋转
计算姿态误差
进行坐标系旋转
```

---

### 二.2.2. 四元数共轭

```python
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[:, :1], -q[:, 1:]), dim=-1)
```

对于四元数：

```text
q = [w, x, y, z]
```

其共轭为：

```text
q* = [w, -x, -y, -z]
```

单位四元数的逆等于其共轭，因此该方法主要用于姿态逆旋转以及姿态误差计算。

---

### 二.2.3. 世界系向量旋转到机体系

```python
def quat_rotate_inverse(q: torch.Tensor, vector_world: torch.Tensor) -> torch.Tensor:
    vector_quat = torch.cat((torch.zeros_like(vector_world[:, :1]), vector_world), dim=-1)
    return quat_multiply(quat_multiply(quat_conjugate(q), vector_quat), q)[:, 1:]
```

该方法将世界坐标系中的向量转换到当前姿态对应的机体坐标系。

主要用于：

```text
World Frame
    ↓
Body Frame
```

代码中主要用于将世界系下的 yaw 角速度前馈转换到机体系。

---

### 二.2.4. 计算机体系 Z 轴在世界系中的方向

```python
def body_z_world(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (2.0 * (w * y + x * z), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x.square() + y.square())),
        dim=-1,
    )
```

根据无人机当前姿态四元数，计算机体系 Z 轴在世界坐标系中的方向。

由于多旋翼总推力沿机体系 Z 轴方向作用，因此：

```text
机体系 Z 轴方向
≈
无人机总推力方向
```

该方向用于姿态环中优先计算当前推力方向与期望推力方向之间的偏差。

---

### 二.2.5. 计算两个向量之间的旋转四元数

```python
def quat_from_two_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = source / source.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    cross = torch.cross(source, target, dim=-1)
    dot = (source * target).sum(dim=-1).clamp(-1.0, 1.0)
    scalar = torch.sqrt(((1.0 + dot) * 0.5).clamp_min(1.0e-12))
    vector = cross / (2.0 * scalar[:, None]).clamp_min(1.0e-8)
    return torch.cat((scalar[:, None], vector), dim=-1)
```

输入：

```text
source
target
```

计算将 `source` 方向旋转到 `target` 方向所需要的四元数。

姿态环中使用该方法计算：

```text
当前机体 Z 轴
        ↓
期望机体 Z 轴
```

所需要的旋转。

---

### 二.2.6. 欧拉角转换为四元数

```python
def quat_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )
```

将：

```text
roll
pitch
yaw
```

转换为：

```text
[w, x, y, z]
```

形式的姿态四元数。

主要用于 `compute_attitude` 接口，将上层输入的欧拉角期望转换成姿态环使用的四元数期望。

---

# 二.3. PX4Controller 本体类

## 二.3.1. 控制器初始化

```python
def __init__(self, cfg: ControllerConfig, dynamics: P300ProDynamics):
    self.cfg = cfg
    self.dynamics = dynamics
    self.num_envs = dynamics.num_envs
    self.device = dynamics.device
    tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=self.device)
    self.attitude_p = tensor(cfg.attitude_p)
    self.attitude_p[2] /= cfg.yaw_weight
    self.rate_p = tensor(cfg.rate_p) * tensor(cfg.rate_k)
    self.rate_i = tensor(cfg.rate_i) * tensor(cfg.rate_k)
    self.rate_d = tensor(cfg.rate_d) * tensor(cfg.rate_k)
    self.rate_limit = tensor(cfg.rate_limit_rad_s)
    self.integrator_limit = tensor(cfg.integrator_limit)
    self.rate_integral = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
    self.previous_rate = torch.zeros_like(self.rate_integral)
    self.filtered_angular_accel = torch.zeros_like(self.rate_integral)
    self.saturation_positive = torch.zeros_like(self.rate_integral, dtype=torch.bool)
    self.saturation_negative = torch.zeros_like(self.rate_integral, dtype=torch.bool)
    self.last_rate_setpoint = torch.zeros_like(self.rate_integral)
    self.last_normalized_torque = torch.zeros_like(self.rate_integral)
    # Compatibility alias. PX4 rate gains produce normalized control, not N*m.
    self.last_desired_torque = self.last_normalized_torque
    self.last_achieved_normalized_wrench = torch.zeros(
        (self.num_envs, 4), dtype=torch.float32, device=self.device
    )
    self.last_achieved_wrench = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
```

初始化阶段主要完成三部分工作。

控制器参数初始化：

```text
attitude_p

rate_p
rate_i
rate_d

rate_limit
integrator_limit
```

控制器状态初始化：

```text
rate_integral
previous_rate
filtered_angular_accel
```

控制分配状态初始化：

```text
saturation_positive
saturation_negative

last_achieved_normalized_wrench
last_achieved_wrench
```

所有状态均按照：

```text
num_envs
```

批量创建，用于多个 Isaac Sim 环境并行计算。

---

# 二.4. 姿态控制环

```python
def attitude_control(
    self,
    attitude_wxyz: torch.Tensor,
    attitude_setpoint_wxyz: torch.Tensor,
    yaw_rate_feedforward: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run only the attitude loop and return body rates in rad/s."""
    expected_quaternion_shape = (self.num_envs, 4)
    if attitude_wxyz.shape != expected_quaternion_shape:
        raise ValueError(
            f"attitude must have shape {expected_quaternion_shape}, got {tuple(attitude_wxyz.shape)}"
        )
    if attitude_setpoint_wxyz.shape != expected_quaternion_shape:
        raise ValueError(
            "attitude setpoint must have shape "
            f"{expected_quaternion_shape}, got {tuple(attitude_setpoint_wxyz.shape)}"
        )
    desired_q = attitude_setpoint_wxyz / attitude_setpoint_wxyz.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    current_q = attitude_wxyz / attitude_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    # PX4 first aligns thrust directions, then adds the weighted yaw part.
    current_z = body_z_world(current_q)
    desired_z = body_z_world(desired_q)
    reduced_delta = quat_from_two_vectors(current_z, desired_z)
    reduced_desired = quat_multiply(reduced_delta, current_q)
    opposite_thrust = (current_z * desired_z).sum(dim=-1) < -0.99999
    reduced_desired = torch.where(opposite_thrust[:, None], desired_q, reduced_desired)
    yaw_mix = quat_multiply(quat_conjugate(reduced_desired), desired_q)
    yaw_mix = torch.where(yaw_mix[:, :1] < 0.0, -yaw_mix, yaw_mix)
    yaw_mix_w = yaw_mix[:, 0].clamp(-1.0, 1.0)
    yaw_mix_z = yaw_mix[:, 3].clamp(-1.0, 1.0)
    weighted_yaw = torch.stack(
        (
            torch.cos(self.cfg.yaw_weight * torch.acos(yaw_mix_w)),
            torch.zeros_like(yaw_mix_w),
            torch.zeros_like(yaw_mix_w),
            torch.sin(self.cfg.yaw_weight * torch.asin(yaw_mix_z)),
        ),
        dim=-1,
    )
    mixed_desired = quat_multiply(reduced_desired, weighted_yaw)
    error_q = quat_multiply(quat_conjugate(current_q), mixed_desired)
    error_q = torch.where(error_q[:, :1] < 0.0, -error_q, error_q)
    attitude_error = 2.0 * error_q[:, 1:]
    rate_setpoint = attitude_error * self.attitude_p.unsqueeze(0)
    if yaw_rate_feedforward is not None:
        if yaw_rate_feedforward.shape != (self.num_envs,):
            raise ValueError(
                f"yaw-rate feed-forward must have shape {(self.num_envs,)}, "
                f"got {tuple(yaw_rate_feedforward.shape)}"
            )
        yaw_rate_world = torch.zeros_like(rate_setpoint)
        yaw_rate_world[:, 2] = yaw_rate_feedforward
        rate_setpoint += quat_rotate_inverse(current_q, yaw_rate_world)
    rate_setpoint.clamp_(-self.rate_limit, self.rate_limit)
    self.last_rate_setpoint.copy_(rate_setpoint)
    return rate_setpoint
```

输入：

```text
当前姿态四元数
期望姿态四元数
yaw 角速度前馈
```

输出：

```text
期望机体系角速度
[p_sp, q_sp, r_sp]
```

姿态环首先对当前姿态和期望姿态进行四元数归一化。

随后分别计算：

```text
current_z
desired_z
```

即当前机体系 Z 轴和期望机体系 Z 轴在世界系中的方向。

再计算：

```text
current_z
    ↓
desired_z
```

对应的旋转四元数，得到优先完成推力方向对齐后的期望姿态。

当当前推力方向和期望推力方向接近完全相反时：

```python
opposite_thrust = (current_z * desired_z).sum(dim=-1) < -0.99999
```

直接采用完整期望姿态处理该特殊情况。

推力方向处理完成后，再计算剩余 yaw 旋转：

```text
yaw_mix
```

并通过：

```text
yaw_weight
```

调整 yaw 的控制权重。

最终得到：

```text
mixed_desired
```

作为姿态环实际使用的期望姿态。

姿态误差为：

```text
current_q
    ↓
mixed_desired
```

对应的误差四元数。

根据误差四元数的向量部分得到：

```text
attitude_error
```

再通过姿态 P 控制器得到：

```text
rate_setpoint
```

即期望机体系角速度。

姿态环整体逻辑为：

```text
当前姿态
+
期望姿态
    ↓
优先修正推力方向
    ↓
加入 yaw 权重
    ↓
新的期望姿态
    ↓
姿态误差
    ↓
P 控制
    ↓
期望角速度
```

---

# 二.5. 角速度控制环

```python
def rate_control(
    self,
    angular_velocity_body: torch.Tensor,
    rate_setpoint: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Run only the PX4 rate PID and return normalized torque control.

    The returned values have PX4's dimensionless torque-setpoint semantics.
    They are deliberately not clipped here; the allocator clips motor
    commands and feeds the unallocated control back into anti-windup.
    """
    expected_rate_shape = (self.num_envs, 3)
    if angular_velocity_body.shape != expected_rate_shape:
        raise ValueError(
            f"angular velocity must have shape {expected_rate_shape}, got {tuple(angular_velocity_body.shape)}"
        )
    if rate_setpoint.shape != expected_rate_shape:
        raise ValueError(
            f"rate setpoint must have shape {expected_rate_shape}, got {tuple(rate_setpoint.shape)}"
        )
    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError(f"controller dt must be finite and positive, got {dt!r}")
    rate_setpoint = rate_setpoint.clamp(-self.rate_limit, self.rate_limit)
    self.last_rate_setpoint.copy_(rate_setpoint)

    raw_accel = (angular_velocity_body - self.previous_rate) / float(dt)
    alpha = self.cfg.angular_accel_filter_alpha
    self.filtered_angular_accel.mul_(alpha).add_(raw_accel, alpha=1.0 - alpha)
    self.previous_rate.copy_(angular_velocity_body)

    rate_error = rate_setpoint - angular_velocity_body
    normalized_torque = (
        self.rate_p.unsqueeze(0) * rate_error
        + self.rate_integral
        - self.rate_d.unsqueeze(0) * self.filtered_angular_accel
    )
    self.last_normalized_torque.copy_(normalized_torque)

    limited_error = torch.where(
        self.saturation_positive, torch.minimum(rate_error, torch.zeros_like(rate_error)), rate_error
    )
    limited_error = torch.where(
        self.saturation_negative, torch.maximum(limited_error, torch.zeros_like(limited_error)), limited_error
    )
    i_factor = (1.0 - (limited_error / math.radians(400.0)).square()).clamp_min(0.0)
    self.rate_integral.add_(i_factor * self.rate_i.unsqueeze(0) * limited_error * float(dt))
    self.rate_integral.clamp_(-self.integrator_limit, self.integrator_limit)
    return normalized_torque
```

输入：

```text
当前机体系角速度
期望机体系角速度
控制周期 dt
```

输出：

```text
归一化三轴力矩
[τx, τy, τz]
```

首先计算角加速度：

```text
当前角速度 - 上一时刻角速度
────────────────────────
            dt
```

随后通过一阶低通滤波处理角加速度：

```text
raw_accel
    ↓
Low Pass Filter
    ↓
filtered_angular_accel
```

角速度误差为：

```text
rate_setpoint - angular_velocity_body
```

控制器最终使用：

```text
P 项
+
I 项
-
D 项
```

得到归一化力矩：

```text
normalized_torque
```

其中 P 项使用角速度误差：

```text
Kp × rate_error
```

D 项使用经过滤波的实际角加速度：

```text
-Kd × angular_acceleration
```

积分项根据角速度误差持续更新。

控制器通过：

```text
saturation_positive
saturation_negative
```

限制已经发生执行器饱和方向上的积分累积。

同时使用：

```python
i_factor = (1.0 - (limited_error / math.radians(400.0)).square()).clamp_min(0.0)
```

根据角速度误差大小动态调整积分作用。

误差越小时积分作用越完整，误差增大时积分作用逐渐降低。

积分值最终限制在：

```text
[-integrator_limit, integrator_limit]
```

范围内。

角速度环整体逻辑为：

```text
当前角速度
+
期望角速度
    ↓
角速度误差
    ↓
P
+
积分状态 I
-
滤波角加速度 D
    ↓
归一化三轴力矩
```

---

# 二.6. 控制分配

```python
def control_allocation(
    self,
    collective_control: torch.Tensor,
    normalized_torque: torch.Tensor,
) -> torch.Tensor:
    """Allocate normalized PX4 control to P300Pro motor commands."""
    if collective_control.shape != (self.num_envs,):
        raise ValueError(
            f"collective control must have shape {(self.num_envs,)}, got {tuple(collective_control.shape)}"
        )
    collective_control = collective_control.clamp(0.0, 1.0)
    actuator_control, achieved_normalized, achieved_physical = self.dynamics.allocate_normalized(
        collective_control,
        normalized_torque,
    )
    self.last_achieved_normalized_wrench.copy_(achieved_normalized)
    self.last_achieved_wrench.copy_(achieved_physical)
    unallocated_torque = normalized_torque - achieved_normalized[:, 1:]
    self.saturation_positive.copy_(unallocated_torque > 1.0e-5)
    self.saturation_negative.copy_(unallocated_torque < -1.0e-5)
    return actuator_control
```

输入：

```text
collective_control
normalized_torque
```

对应：

```text
归一化总推力
+
归一化 roll/pitch/yaw 力矩
```

归一化总推力首先限制在：

```text
[0, 1]
```

随后调用：

```python
self.dynamics.allocate_normalized()
```

将总推力和三轴力矩交给具体无人机动力学模型进行控制分配。

动力学模型返回三个结果：

```text
actuator_control
achieved_normalized
achieved_physical
```

其中：

```text
actuator_control
```

表示各执行器最终接收到的控制指令。

```text
achieved_normalized
```

表示这些执行器实际实现的归一化总推力和三轴力矩。

```text
achieved_physical
```

表示对应的实际物理推力和力矩。

通过：

```python
unallocated_torque = normalized_torque - achieved_normalized[:, 1:]
```

计算期望三轴力矩与实际可实现三轴力矩之间的差值。

根据差值方向更新：

```text
saturation_positive
saturation_negative
```

并将饱和状态反馈给下一次角速度控制器计算，用于积分 anti-windup。

整体流程为：

```text
归一化总推力
+
归一化三轴力矩
        ↓
P300ProDynamics
        ↓
控制分配
        ↓
各电机控制指令
        ↓
实际可实现控制量
        ↓
计算未实现力矩
        ↓
更新饱和状态
```

---

# 二.7. 角速度命令接口

```python
def compute_rate(
    self,
    angular_velocity_body: torch.Tensor,
    command_rates_thrust: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Direct rate-command interface: ``[p, q, r, collective]``."""
    if command_rates_thrust.shape != (self.num_envs, 4):
        raise ValueError(
            f"rate/thrust command must have shape {(self.num_envs, 4)}, "
            f"got {tuple(command_rates_thrust.shape)}"
        )
    normalized_torque = self.rate_control(
        angular_velocity_body,
        command_rates_thrust[:, :3],
        dt,
    )
    return self.control_allocation(command_rates_thrust[:, 3], normalized_torque)
```

输入：

```text
[p, q, r, collective]
```

其中：

```text
[p, q, r]
```

作为期望机体系角速度输入角速度控制环。

```text
collective
```

作为归一化总推力直接输入控制分配模块。

完整流程为：

```text
[p_sp, q_sp, r_sp]
        ↓
rate_control
        ↓
normalized_torque
        │
        ├──────── collective
        ↓
control_allocation
        ↓
actuator_control
```

该接口适合上层直接提供角速度控制命令。

---

# 二.8. 姿态命令接口

```python
def compute_attitude(
    self,
    attitude_wxyz: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    command_rpy_thrust: torch.Tensor,
    dt: float,
    yaw_rate_feedforward: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cascaded attitude-command interface: ``[roll, pitch, yaw, collective]``."""
    if command_rpy_thrust.shape != (self.num_envs, 4):
        raise ValueError(
            f"attitude/thrust command must have shape {(self.num_envs, 4)}, "
            f"got {tuple(command_rpy_thrust.shape)}"
        )
    desired_q = quat_from_euler(
        command_rpy_thrust[:, 0], command_rpy_thrust[:, 1], command_rpy_thrust[:, 2]
    )
    rate_setpoint = self.attitude_control(
        attitude_wxyz,
        desired_q,
        yaw_rate_feedforward=yaw_rate_feedforward,
    )
    normalized_torque = self.rate_control(angular_velocity_body, rate_setpoint, dt)
    return self.control_allocation(command_rpy_thrust[:, 3], normalized_torque)
```

输入：

```text
[roll, pitch, yaw, collective]
```

首先将：

```text
roll
pitch
yaw
```

转换为期望姿态四元数：

```text
desired_q
```

随后依次调用：

```text
attitude_control
        ↓
rate_setpoint

rate_control
        ↓
normalized_torque

control_allocation
        ↓
actuator_control
```

完整控制链路为：

```text
roll / pitch / yaw
        ↓
quat_from_euler
        ↓
desired_q
        ↓
姿态控制环
        ↓
期望角速度
        ↓
角速度控制环
        ↓
归一化三轴力矩
        │
        ├──────── collective
        ↓
控制分配
        ↓
电机指令
```

---

# 二.9. 默认控制接口

```python
def compute(
    self,
    attitude_wxyz: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    command_rpy_thrust: torch.Tensor,
    dt: float,
    yaw_rate_feedforward: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for :meth:`compute_attitude`."""
    return self.compute_attitude(
        attitude_wxyz,
        angular_velocity_body,
        command_rpy_thrust,
        dt,
        yaw_rate_feedforward=yaw_rate_feedforward,
    )
```

`compute` 为默认控制接口。

该方法直接调用：

```text
compute_attitude
```

因此默认控制链路为：

```text
期望姿态
+
归一化总推力
        ↓
姿态控制环
        ↓
角速度控制环
        ↓
控制分配
        ↓
电机控制指令
```

---

# 二.10. 控制器状态复位

```python
def reset(self, env_ids: torch.Tensor, angular_velocity_body: torch.Tensor | None = None) -> None:
    self.rate_integral[env_ids] = 0.0
    self.filtered_angular_accel[env_ids] = 0.0
    self.saturation_positive[env_ids] = False
    self.saturation_negative[env_ids] = False
    self.last_rate_setpoint[env_ids] = 0.0
    self.last_normalized_torque[env_ids] = 0.0
    self.last_achieved_normalized_wrench[env_ids] = 0.0
    self.last_achieved_wrench[env_ids] = 0.0
    if angular_velocity_body is None:
        self.previous_rate[env_ids] = 0.0
    else:
        self.previous_rate[env_ids] = angular_velocity_body
```

用于指定环境的控制器状态复位。

主要清除：

```text
角速度积分状态
角加速度滤波状态
执行器饱和状态
上一周期控制输出
上一周期实际控制量
```

同时初始化：

```text
previous_rate
```

如果提供当前角速度：

```text
previous_rate = angular_velocity_body
```

下一周期差分计算角加速度时直接以当前状态作为初始值。

---

# 三. 控制流程总结

姿态命令接口：

```text
[roll, pitch, yaw, collective]
                 │
                 ▼
          quat_from_euler
                 │
                 ▼
         attitude_control
                 │
                 ▼
          [p_sp, q_sp, r_sp]
                 │
                 ▼
           rate_control
                 │
                 ▼
       normalized_torque
                 │
        collective
             │   │
             └─┬─┘
               ▼
       control_allocation
               │
               ▼
        actuator_control
```

角速度命令接口：

```text
[p_sp, q_sp, r_sp, collective]
                 │
                 ▼
           rate_control
                 │
                 ▼
       normalized_torque
                 │
        collective
             │   │
             └─┬─┘
               ▼
       control_allocation
               │
               ▼
        actuator_control
```

控制器核心模块可以归纳为：

```text
四元数与坐标变换
        ↓
姿态控制环
        ↓
角速度控制环
        ↓
控制分配
        ↓
电机控制
```

# 四. 源代码

```python
# Copyright (c) 2026 DR Sim contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Batched PX4-style quaternion attitude and angular-rate controller.

The control structure follows the BSD-3-Clause PX4 attitude/rate controller:
quaternion attitude error, rate PID with I-factor anti-windup, and geometry-based
control allocation. The algorithm reference is the PX4 Development Team's
``AttitudeControl.cpp`` and ``RateControl.cpp`` in this repository's locked
PX4-Autopilot source tree. This implementation uses the local P300Pro parameters
instead of values from any external training project.
"""

from __future__ import annotations

import math

import torch

from .config import ControllerConfig
from .p300pro_dynamics import P300ProDynamics


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[:, :1], -q[:, 1:]), dim=-1)


def quat_rotate_inverse(q: torch.Tensor, vector_world: torch.Tensor) -> torch.Tensor:
    vector_quat = torch.cat((torch.zeros_like(vector_world[:, :1]), vector_world), dim=-1)
    return quat_multiply(quat_multiply(quat_conjugate(q), vector_quat), q)[:, 1:]


def body_z_world(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (2.0 * (w * y + x * z), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x.square() + y.square())),
        dim=-1,
    )


def quat_from_two_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = source / source.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    cross = torch.cross(source, target, dim=-1)
    dot = (source * target).sum(dim=-1).clamp(-1.0, 1.0)
    scalar = torch.sqrt(((1.0 + dot) * 0.5).clamp_min(1.0e-12))
    vector = cross / (2.0 * scalar[:, None]).clamp_min(1.0e-8)
    return torch.cat((scalar[:, None], vector), dim=-1)


def quat_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


class PX4Controller:
    """Batched PX4 attitude/rate loops and normalized motor allocation.

    The public loop boundaries intentionally mirror PX4:

    - :meth:`attitude_control`: attitude quaternion -> body-rate setpoint.
    - :meth:`rate_control`: body-rate setpoint -> normalized torque setpoint.
    - :meth:`control_allocation`: normalized thrust/torque -> motor commands.

    :meth:`compute_attitude` and :meth:`compute_rate` are command-level
    convenience interfaces. :meth:`compute` remains an alias of the attitude
    cascade so existing trajectory environments keep their API.
    """

    def __init__(self, cfg: ControllerConfig, dynamics: P300ProDynamics):
        self.cfg = cfg
        self.dynamics = dynamics
        self.num_envs = dynamics.num_envs
        self.device = dynamics.device
        tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=self.device)
        self.attitude_p = tensor(cfg.attitude_p)
        self.attitude_p[2] /= cfg.yaw_weight
        self.rate_p = tensor(cfg.rate_p) * tensor(cfg.rate_k)
        self.rate_i = tensor(cfg.rate_i) * tensor(cfg.rate_k)
        self.rate_d = tensor(cfg.rate_d) * tensor(cfg.rate_k)
        self.rate_limit = tensor(cfg.rate_limit_rad_s)
        self.integrator_limit = tensor(cfg.integrator_limit)
        self.rate_integral = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.previous_rate = torch.zeros_like(self.rate_integral)
        self.filtered_angular_accel = torch.zeros_like(self.rate_integral)
        self.saturation_positive = torch.zeros_like(self.rate_integral, dtype=torch.bool)
        self.saturation_negative = torch.zeros_like(self.rate_integral, dtype=torch.bool)
        self.last_rate_setpoint = torch.zeros_like(self.rate_integral)
        self.last_normalized_torque = torch.zeros_like(self.rate_integral)
        # Compatibility alias. PX4 rate gains produce normalized control, not N*m.
        self.last_desired_torque = self.last_normalized_torque
        self.last_achieved_normalized_wrench = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.last_achieved_wrench = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

    def attitude_control(
        self,
        attitude_wxyz: torch.Tensor,
        attitude_setpoint_wxyz: torch.Tensor,
        yaw_rate_feedforward: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run only the attitude loop and return body rates in rad/s."""
        expected_quaternion_shape = (self.num_envs, 4)
        if attitude_wxyz.shape != expected_quaternion_shape:
            raise ValueError(
                f"attitude must have shape {expected_quaternion_shape}, got {tuple(attitude_wxyz.shape)}"
            )
        if attitude_setpoint_wxyz.shape != expected_quaternion_shape:
            raise ValueError(
                "attitude setpoint must have shape "
                f"{expected_quaternion_shape}, got {tuple(attitude_setpoint_wxyz.shape)}"
            )
        desired_q = attitude_setpoint_wxyz / attitude_setpoint_wxyz.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        current_q = attitude_wxyz / attitude_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        # PX4 first aligns thrust directions, then adds the weighted yaw part.
        current_z = body_z_world(current_q)
        desired_z = body_z_world(desired_q)
        reduced_delta = quat_from_two_vectors(current_z, desired_z)
        reduced_desired = quat_multiply(reduced_delta, current_q)
        opposite_thrust = (current_z * desired_z).sum(dim=-1) < -0.99999
        reduced_desired = torch.where(opposite_thrust[:, None], desired_q, reduced_desired)
        yaw_mix = quat_multiply(quat_conjugate(reduced_desired), desired_q)
        yaw_mix = torch.where(yaw_mix[:, :1] < 0.0, -yaw_mix, yaw_mix)
        yaw_mix_w = yaw_mix[:, 0].clamp(-1.0, 1.0)
        yaw_mix_z = yaw_mix[:, 3].clamp(-1.0, 1.0)
        weighted_yaw = torch.stack(
            (
                torch.cos(self.cfg.yaw_weight * torch.acos(yaw_mix_w)),
                torch.zeros_like(yaw_mix_w),
                torch.zeros_like(yaw_mix_w),
                torch.sin(self.cfg.yaw_weight * torch.asin(yaw_mix_z)),
            ),
            dim=-1,
        )
        mixed_desired = quat_multiply(reduced_desired, weighted_yaw)
        error_q = quat_multiply(quat_conjugate(current_q), mixed_desired)
        error_q = torch.where(error_q[:, :1] < 0.0, -error_q, error_q)
        attitude_error = 2.0 * error_q[:, 1:]
        rate_setpoint = attitude_error * self.attitude_p.unsqueeze(0)
        if yaw_rate_feedforward is not None:
            if yaw_rate_feedforward.shape != (self.num_envs,):
                raise ValueError(
                    f"yaw-rate feed-forward must have shape {(self.num_envs,)}, "
                    f"got {tuple(yaw_rate_feedforward.shape)}"
                )
            yaw_rate_world = torch.zeros_like(rate_setpoint)
            yaw_rate_world[:, 2] = yaw_rate_feedforward
            rate_setpoint += quat_rotate_inverse(current_q, yaw_rate_world)
        rate_setpoint.clamp_(-self.rate_limit, self.rate_limit)
        self.last_rate_setpoint.copy_(rate_setpoint)
        return rate_setpoint

    def rate_control(
        self,
        angular_velocity_body: torch.Tensor,
        rate_setpoint: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Run only the PX4 rate PID and return normalized torque control.

        The returned values have PX4's dimensionless torque-setpoint semantics.
        They are deliberately not clipped here; the allocator clips motor
        commands and feeds the unallocated control back into anti-windup.
        """
        expected_rate_shape = (self.num_envs, 3)
        if angular_velocity_body.shape != expected_rate_shape:
            raise ValueError(
                f"angular velocity must have shape {expected_rate_shape}, got {tuple(angular_velocity_body.shape)}"
            )
        if rate_setpoint.shape != expected_rate_shape:
            raise ValueError(
                f"rate setpoint must have shape {expected_rate_shape}, got {tuple(rate_setpoint.shape)}"
            )
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError(f"controller dt must be finite and positive, got {dt!r}")
        rate_setpoint = rate_setpoint.clamp(-self.rate_limit, self.rate_limit)
        self.last_rate_setpoint.copy_(rate_setpoint)

        raw_accel = (angular_velocity_body - self.previous_rate) / float(dt)
        alpha = self.cfg.angular_accel_filter_alpha
        self.filtered_angular_accel.mul_(alpha).add_(raw_accel, alpha=1.0 - alpha)
        self.previous_rate.copy_(angular_velocity_body)

        rate_error = rate_setpoint - angular_velocity_body
        normalized_torque = (
            self.rate_p.unsqueeze(0) * rate_error
            + self.rate_integral
            - self.rate_d.unsqueeze(0) * self.filtered_angular_accel
        )
        self.last_normalized_torque.copy_(normalized_torque)

        limited_error = torch.where(
            self.saturation_positive, torch.minimum(rate_error, torch.zeros_like(rate_error)), rate_error
        )
        limited_error = torch.where(
            self.saturation_negative, torch.maximum(limited_error, torch.zeros_like(limited_error)), limited_error
        )
        i_factor = (1.0 - (limited_error / math.radians(400.0)).square()).clamp_min(0.0)
        self.rate_integral.add_(i_factor * self.rate_i.unsqueeze(0) * limited_error * float(dt))
        self.rate_integral.clamp_(-self.integrator_limit, self.integrator_limit)
        return normalized_torque

    def control_allocation(
        self,
        collective_control: torch.Tensor,
        normalized_torque: torch.Tensor,
    ) -> torch.Tensor:
        """Allocate normalized PX4 control to P300Pro motor commands."""
        if collective_control.shape != (self.num_envs,):
            raise ValueError(
                f"collective control must have shape {(self.num_envs,)}, got {tuple(collective_control.shape)}"
            )
        collective_control = collective_control.clamp(0.0, 1.0)
        actuator_control, achieved_normalized, achieved_physical = self.dynamics.allocate_normalized(
            collective_control,
            normalized_torque,
        )
        self.last_achieved_normalized_wrench.copy_(achieved_normalized)
        self.last_achieved_wrench.copy_(achieved_physical)
        unallocated_torque = normalized_torque - achieved_normalized[:, 1:]
        self.saturation_positive.copy_(unallocated_torque > 1.0e-5)
        self.saturation_negative.copy_(unallocated_torque < -1.0e-5)
        return actuator_control

    def compute_rate(
        self,
        angular_velocity_body: torch.Tensor,
        command_rates_thrust: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Direct rate-command interface: ``[p, q, r, collective]``."""
        if command_rates_thrust.shape != (self.num_envs, 4):
            raise ValueError(
                f"rate/thrust command must have shape {(self.num_envs, 4)}, "
                f"got {tuple(command_rates_thrust.shape)}"
            )
        normalized_torque = self.rate_control(
            angular_velocity_body,
            command_rates_thrust[:, :3],
            dt,
        )
        return self.control_allocation(command_rates_thrust[:, 3], normalized_torque)

    def compute_attitude(
        self,
        attitude_wxyz: torch.Tensor,
        angular_velocity_body: torch.Tensor,
        command_rpy_thrust: torch.Tensor,
        dt: float,
        yaw_rate_feedforward: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cascaded attitude-command interface: ``[roll, pitch, yaw, collective]``."""
        if command_rpy_thrust.shape != (self.num_envs, 4):
            raise ValueError(
                f"attitude/thrust command must have shape {(self.num_envs, 4)}, "
                f"got {tuple(command_rpy_thrust.shape)}"
            )
        desired_q = quat_from_euler(
            command_rpy_thrust[:, 0], command_rpy_thrust[:, 1], command_rpy_thrust[:, 2]
        )
        rate_setpoint = self.attitude_control(
            attitude_wxyz,
            desired_q,
            yaw_rate_feedforward=yaw_rate_feedforward,
        )
        normalized_torque = self.rate_control(angular_velocity_body, rate_setpoint, dt)
        return self.control_allocation(command_rpy_thrust[:, 3], normalized_torque)

    def compute(
        self,
        attitude_wxyz: torch.Tensor,
        angular_velocity_body: torch.Tensor,
        command_rpy_thrust: torch.Tensor,
        dt: float,
        yaw_rate_feedforward: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible alias for :meth:`compute_attitude`."""
        return self.compute_attitude(
            attitude_wxyz,
            angular_velocity_body,
            command_rpy_thrust,
            dt,
            yaw_rate_feedforward=yaw_rate_feedforward,
        )

    def reset(self, env_ids: torch.Tensor, angular_velocity_body: torch.Tensor | None = None) -> None:
        self.rate_integral[env_ids] = 0.0
        self.filtered_angular_accel[env_ids] = 0.0
        self.saturation_positive[env_ids] = False
        self.saturation_negative[env_ids] = False
        self.last_rate_setpoint[env_ids] = 0.0
        self.last_normalized_torque[env_ids] = 0.0
        self.last_achieved_normalized_wrench[env_ids] = 0.0
        self.last_achieved_wrench[env_ids] = 0.0
        if angular_velocity_body is None:
            self.previous_rate[env_ids] = 0.0
        else:
            self.previous_rate[env_ids] = angular_velocity_body
```
