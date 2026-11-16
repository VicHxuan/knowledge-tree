# 级联姿态与角速度控制

## 1. 概述

级联控制由姿态环生成角速度目标，由角速度环生成力矩控制量，再由控制分配器转换为电机命令。以下以 PX4 风格的四元数控制结构为主线，并保留原有批量仿真实现用于接口与状态更新分析。

## 2. 基础与定义

### 2.1 姿态与坐标

单位四元数 $q=(w,x,y,z)$ 表示机体系到世界系的旋转，采用 Hamilton 乘法。目标姿态为 $q_d$，机体系角速度为 $\omega$。世界向量到机体系的变换为 $R(q)^\top v$。

### 2.2 信号与状态

姿态环输出 $\omega_{\rm sp}$，角速度环输出归一化力矩控制 $u_\tau$，分配器结合归一化总推力生成电机命令。积分项、上一拍角速度和滤波状态需要跨周期保存。

## 3. 核心原理

### 3.1 姿态误差

对目标姿态取机体系相对误差

$$
q_e=q^{-1}\otimes q_d,\qquad e_R=2\operatorname{sgn}_+(q_{e,w})q_{e,xyz},
$$

其中 $\operatorname{sgn}_+(0)=1$，用四元数标量部选择等价表示。角速度目标为

$$
\omega_{\rm sp}=K_Re_R+R(q)^\top e_3\dot\psi_{\rm ff},
$$

再按各轴能力限幅。实现中先对齐推力方向，再按偏航权重混合剩余旋转。
<details class="kt-derivation" id="attitude-reduced">
<summary>推力方向与偏航分解</summary>

当前与目标推力方向分别为 $b_3=R(q)e_3$、$b_{3d}=R(q_d)e_3$。构造将 $b_3$ 转至 $b_{3d}$ 的最短旋转 $q_\Delta$，得到 $q_{\rm red}=q_\Delta\otimes q$。

剩余旋转为 $q_{\rm mix}=q_{\rm red}^{-1}\otimes q_d$。缩放其偏航角后与 $q_{\rm red}$ 组合，得到用于误差计算的目标。两推力方向接近反向时，最短旋转轴不唯一；保留实现使用完整目标姿态作为回退。

</details>

### 3.2 角速度反馈

令 $e_\omega=\omega_{\rm sp}-\omega$，角速度控制为

$$
u_\tau=K_Pe_\omega+I-K_D\widehat{\dot\omega}.
$$

微分作用于测量角速度，滤波后用于抑制高频噪声。积分更新采用饱和方向限制和随误差减小的积分因子，并对积分状态限幅。

### 3.3 控制分配与反馈

物理分配模型写为 $w=Bf$，其中 $w$ 为总推力与三轴力矩，$f$ 为各旋翼推力。归一化控制需经过对应的尺度映射。

分配器返回实际可实现控制 $u_{\tau,\rm achieved}$。残差 $u_\tau-u_{\tau,\rm achieved}$ 给出各轴饱和方向，下一周期在继续加深饱和的方向上限制积分。

## 4. 实现与应用

### 4.1 接口与更新顺序

姿态命令接口依次执行姿态环、角速度环和分配；角速度命令接口直接进入角速度环。仿真批次中各环境独立保存控制状态，环境复位时同步清空积分、滤波和饱和标记，并用当前角速度初始化差分历史。

### 4.2 仿真实现

保留代码依赖 PyTorch、`ControllerConfig` 与 `P300ProDynamics.allocate_normalized`。配置提供增益、限幅、滤波参数与正偏航权重，动力学对象提供批次大小、设备和控制分配接口。它实现上述控制结构，物理缩放由本地动力学对象约定。

<details class="kt-derivation" id="px4-style-source">
<summary>批量控制器实现</summary>

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

</details>

## 5. 分析与总结

姿态环与角速度环的带宽应形成层级，控制分配结果应反馈给积分环节。评估包括四元数符号一致性、参考阶跃、饱和恢复、延迟和批次复位；物理力矩与归一化控制分别记录。

## 6. 关联与参考

- [PX4 v1.15 控制器结构](https://docs.px4.io/v1.15/en/flight_stack/controller_diagrams)
- [PX4 v1.15 姿态控制源码](https://github.com/PX4/PX4-Autopilot/blob/v1.15.0/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp)
- [PX4 v1.15 角速度控制源码](https://github.com/PX4/PX4-Autopilot/blob/v1.15.0/src/lib/rate_control/rate_control.cpp)
- [PID 控制](01_pid.md)
- [几何控制](05_geometric-control-se3.md)
