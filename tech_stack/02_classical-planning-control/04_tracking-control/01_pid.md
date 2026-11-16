# PID 控制

## 1. 概述

PID 将跟踪误差、累计误差和误差变化率组合成控制输入。比例项提供即时反馈，积分项补偿持续偏差，微分项调节瞬态响应；实际闭环还需要处理噪声、采样和执行器饱和。

## 2. 基础与定义

### 2.1 跟踪误差

设参考值为 $r(t)$、测量值为 $y(t)$，误差 $e=r-y$。连续控制律为

$$
u=K_pe+K_i\int_0^te(\tau)d\tau+K_d\dot e.
$$

增益的单位分别由输入、误差和时间单位决定。

### 2.2 闭环结构

对被控对象 $G(s)$ 和控制器 $C(s)$，单位负反馈下参考到输出的传递函数为 $G C/(1+GC)$。稳定性由闭环特征方程决定。

## 3. 核心原理

### 3.1 离散更新

采样周期为 $\Delta t$，积分状态可按 $I_{k+1}=I_k+K_ie_k\Delta t$ 更新，微分项由差分和低通滤波获得。

参考阶跃较多时，对测量求微分可取

$$
u_k=K_pe_k+I_k-K_d\widehat{\dot y}_k.
$$

### 3.2 饱和与积分

执行器输出 $u_{\rm sat}=\operatorname{clip}(u,u_{\min},u_{\max})$。反算抗饱和使用

$$
\dot I=K_ie+K_{\rm aw}(u_{\rm sat}-u).
$$

也可在误差继续推动饱和的方向上暂停积分，并允许反向误差释放积分。
<details class="kt-derivation" id="pid-windup">
<summary>积分饱和机制</summary>

当 $u>u_{\max}$ 且 $e>0$ 时，执行器输出保持在上限，普通积分仍以 $K_ie$ 增长。误差反向后需要先抵消这部分累计量，导致恢复迟缓。反算项在此时为负，将内部输出拉回可实现范围。

</details>

### 3.3 滤波与带宽

微分放大高频噪声，滤波引入相位滞后。采样周期、滤波截止频率、执行器响应和闭环带宽需要共同设计。

## 4. 实现与应用

先确认输入输出方向、采样周期与执行器范围，再选择反馈增益，最后加入积分与微分并检查饱和恢复。评价阶跃响应、扰动抑制、噪声放大和积分复位。

<details class="kt-derivation" id="pid-basic-code">
<summary>离散 PID 实现</summary>

以下为固定正采样周期、对误差微分的基础实现，适用于无饱和的数值示例。接入受限执行器时使用上述抗饱和机制。

```cpp
class PIDController {
private:
    double kp, ki, kd;
    double dt;
    double integral;
    double prev_error;

public:
    PIDController(double p, double i, double d, double dt)
        : kp(p), ki(i), kd(d), dt(dt), integral(0.0), prev_error(0.0) {}

    double calculate(double setpoint, double current_value) {
        double error = setpoint - current_value;

        integral += error * dt;
        double derivative = (error - prev_error) / dt;

        double output = kp * error + ki * integral + kd * derivative;

        prev_error = error;
        return output;
    }

    void reset() {
        integral = 0.0;
        prev_error = 0.0;
    }
};
```

</details>

## 5. 分析与总结

PID 以较少状态实现反馈，适合结构清楚的单回路与级联系统。模型不确定性、强耦合和硬约束较强时，可结合前馈、解耦、状态反馈或预测控制。

## 6. 关联与参考

- [PX4 v1.15 控制器结构](https://docs.px4.io/v1.15/en/flight_stack/controller_diagrams)
- [级联姿态与角速度控制](04_px4-controller-attitude-rate.md)
- [跟踪延迟与执行器动力学](06_tracking-latency-and-actuator-dynamics.md)
