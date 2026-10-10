# 1. PID 控制器

## 1.1 数学公式

PID控制器通过计算设定值与实际输出值之间的误差，利用比例、积分、微分的线性组合计算控制输入。

连续时间形式：
$$u(t) = K_p e(t) + K_i \int_{0}^{t} e(\tau) d\tau + K_d \frac{de(t)}{dt}$$

离散时间形式（位置式）：
$$u(k) = K_p e(k) + K_i \sum_{i=0}^{k} e(i)\Delta t + K_d \frac{e(k) - e(k-1)}{\Delta t}$$
其中 $e(t)$ 为误差，$K_p, K_i, K_d$ 分别为比例、积分、微分增益。

## 1.2 使用条件与性质

* **使用条件**：适用于单输入单输出系统，系统呈现线性或弱非线性特征。不需要被控对象的精确数学模型。
* **性质**：

  * 属于无模型反馈控制，依赖误差驱动。
  * 不具备直接处理系统状态约束和控制量约束的能力。
  * 计算复杂度极低，易于在底层嵌入式硬件中实现。

## 1.3 C++ 实现代码

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

