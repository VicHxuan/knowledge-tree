# SAC 算法完整解析与代码实现

## 1. 概述（Overview）

SAC（Soft Actor-Critic，软演员—评论家）是一种面向连续动作空间的离策略（Off-Policy）Actor-Critic 算法。它在标准累计奖励目标中加入策略熵，将“获得高回报”和“保持充分探索”统一为最大熵强化学习目标；再结合经验回放、双 Q 网络、目标网络和重参数化采样，实现稳定且样本效率较高的策略学习。

本文的符号和基础结论承接 [01_rl-foundations.md](./01_rl-foundations.md)，原始算法依据 [SAC v1 论文](../../references/papers/SAC_v1.pdf)：策略网络记为 $\pi_\phi$，两个动作价值网络记为 $Q_{\theta_1},Q_{\theta_2}$，目标网络参数记为 $\bar\theta_1,\bar\theta_2$。本文件下方代码实现的是当前常用的现代 SAC，而 SAC v1 使用独立状态价值网络；两者的衔接将在 2.2 节单独说明。

---

## 2. 基础与定义（Foundations）

> 待补充。

## 3. 核心原理（Core Principles）

### 1. 从标准 Actor-Critic 到最大熵目标

#### 1.1 基础方法的局限

基础 Actor-Critic 用 Critic 估计价值，再用价值信号更新 Actor。对连续动作使用确定性策略时，可进一步得到 DPG/DDPG 一类方法：Actor 直接输出单个动作，并通过 $Q(s,a)$ 对动作的梯度更新策略。

这一路线存在三个直接问题：

- 确定性策略自身不表达探索分布，探索通常依赖额外噪声。
- 单个 Q 估计的误差可能被 Actor 利用并放大，造成价值高估。
- 只追求当前估计下的最高价值，策略可能过早集中到局部最优动作。

SAC 保留“Actor 产生动作、Critic 评价动作”和“经验回放提高样本利用率”的框架，但把随机策略的熵直接写入优化目标。

#### 1.2 从标准回报到最大熵回报

标准强化学习最大化折扣累计奖励：

$$
J(\pi)
=
\mathbb{E}_{\tau\sim\pi}
\left[
\sum_{t=0}^{\infty}\gamma^t r_{t+1}
\right].
$$

状态 $s$ 下策略的熵定义为

$$
\mathcal H\!\left(\pi(\cdot\mid s)\right)
=
-\mathbb{E}_{a\sim\pi}
\left[\log\pi(a\mid s)\right].
$$

SAC 将目标改为

$$
\begin{aligned}
J_{\mathrm{soft}}(\pi)
&=
\mathbb{E}_{\tau\sim\pi}
\left[
\sum_{t=0}^{\infty}\gamma^t
\left(
r_{t+1}
+\alpha\mathcal H\!\left(\pi(\cdot\mid s_t)\right)
\right)
\right]\\
&=
\mathbb{E}_{\tau\sim\pi}
\left[
\sum_{t=0}^{\infty}\gamma^t
\left(
r_{t+1}-\alpha\log\pi(a_t\mid s_t)
\right)
\right].
\end{aligned}
$$

其中 $\alpha>0$ 是温度系数：$\alpha$ 越大，越重视策略随机性；$\alpha$ 越小，越重视任务奖励；当 $\alpha\to0$ 时，软目标退化为标准累计奖励目标。

### 2. 软贝尔曼方程与 SAC 版本演化

#### 2.1 软价值函数与软策略改进

在策略 $\pi$ 下，软状态价值定义为

$$
V^\pi(s)
=
\mathbb{E}_{a\sim\pi(\cdot\mid s)}
\left[
Q^\pi(s,a)-\alpha\log\pi(a\mid s)
\right].
$$

软动作价值满足贝尔曼期望方程

$$
Q^\pi(s,a)
=
\mathbb{E}_{r,s'\mid s,a}
\left[
r+\gamma(1-d)V^\pi(s')
\right].
$$

将 $V^\pi(s')$ 代入，得到 SAC 的核心软贝尔曼备份：

$$
Q^\pi(s,a)
=
\mathbb{E}_{\substack{r,s'\mid s,a\\a'\sim\pi(\cdot\mid s')}}
\left[
r+\gamma(1-d)
\left(
Q^\pi(s',a')-\alpha\log\pi(a'\mid s')
\right)
\right].
$$

对固定的 $Q^{\pi_{\mathrm{old}}}$，最大熵策略改进可写成对玻尔兹曼分布的 KL 投影：

$$
\pi_{\mathrm{new}}
=
\arg\min_{\pi'}
D_{\mathrm{KL}}
\left(
\pi'(\cdot\mid s)
\,\middle\|\,
\frac{
\exp\!\left(Q^{\pi_{\mathrm{old}}}(s,\cdot)/\alpha\right)
}{Z^{\pi_{\mathrm{old}}}(s)}
\right),
$$

其中 $Z(s)$ 是与动作无关的归一化常数。展开 KL 散度：

$$
\begin{aligned}
D_{\mathrm{KL}}
&=
\mathbb{E}_{a\sim\pi'}
\left[
\log\pi'(a\mid s)
-\frac{1}{\alpha}Q^{\pi_{\mathrm{old}}}(s,a)
+\log Z(s)
\right].
\end{aligned}
$$

去掉与策略参数无关的 $\log Z(s)$，再乘以正数 $\alpha$，即可得到 Actor 需要最小化的基本形式

$$
L_\pi
=
\mathbb{E}_{s,a\sim\pi}
\left[
\alpha\log\pi(a\mid s)-Q(s,a)
\right].
$$

因此，SAC 的 Actor 更新不是额外拼接“熵奖励”，而是软策略改进的直接结果。

#### 2.2 SAC v1 与当前代码的衔接

SAC v1 使用四类函数：策略 $\pi_\phi$、软 Q 网络 $Q_{\theta_i}$、独立状态价值网络 $V_\psi$ 以及目标状态价值网络 $V_{\bar\psi}$。结合双 Q 技巧后，状态价值的拟合目标可写为

$$
y_V(s)
=
\mathbb{E}_{a\sim\pi_\phi}
\left[
\min_{i=1,2}Q_{\theta_i}(s,a)
-\alpha\log\pi_\phi(a\mid s)
\right],
$$

$$
L_V(\psi)
=
\frac{1}{2}
\mathbb{E}_{s\sim\mathcal B}
\left[
\left(V_\psi(s)-y_V(s)\right)^2
\right].
$$

相应的 Q 目标为

$$
y_Q^{\mathrm{v1}}
=
r+\gamma(1-d)V_{\bar\psi}(s'),
$$

$$
L_Q(\theta_i)
=
\frac{1}{2}
\mathbb{E}_{(s,a,r,s',d)\sim\mathcal B}
\left[
\left(Q_{\theta_i}(s,a)-y_Q^{\mathrm{v1}}\right)^2
\right].
$$

现代 SAC 观察到 $V(s')$ 本来就是动作上的软价值期望，因此删除独立的 $V_\psi$ 和 $V_{\bar\psi}$，直接用一次当前策略采样近似该期望：

$$
a'\sim\pi_\phi(\cdot\mid s'),
$$

$$
V(s')
\approx
\min_{i=1,2}Q_{\bar\theta_i}(s',a')
-\alpha\log\pi_\phi(a'\mid s').
$$

代回软贝尔曼方程，得到当前代码使用的目标：

$$
\boxed{
y_Q
=
r+\gamma(1-d)
\left[
\min_{i=1,2}Q_{\bar\theta_i}(s',a')
-\alpha\log\pi_\phi(a'\mid s')
\right]
}.
$$

这里的两个目标 Q 网络负责稳定自举目标，取二者最小值用于抑制 Q 值高估。删除独立 V 网络减少了一组网络及其拟合误差，但最大熵目标和软贝尔曼关系没有改变。

### 3. 现代 SAC 的各项损失

#### 3.1 双 Critic 损失

从经验回放池 $\mathcal B$ 采样 $(s,a,r,s',d)$，并将 $y_Q$ 停止梯度。两个 Critic 分别最小化

$$
L_Q(\theta_i)
=
\mathbb{E}_{\mathcal B}
\left[
\left(Q_{\theta_i}(s,a)-y_Q\right)^2
\right],
\qquad i\in\{1,2\}.
$$

经验回放使旧策略收集的数据可以被重复使用；Q-learning 式的贝尔曼目标不要求样本动作由当前策略产生，因此 SAC 是离策略算法。

#### 3.2 重参数化与 Actor 损失

Actor 输出高斯分布参数 $\mu_\phi(s)$ 和 $\sigma_\phi(s)$。为了让采样动作对 $\phi$ 可微，先把随机性移到独立噪声中：

$$
\varepsilon\sim\mathcal N(0,I),
\qquad
u=\mu_\phi(s)+\sigma_\phi(s)\odot\varepsilon.
$$

再用 $\tanh$ 和仿射变换把动作映射到环境范围：

$$
a=f_\phi(\varepsilon;s)
=
c\odot\tanh(u)+b,
$$

其中 $c=(a_{\max}-a_{\min})/2$，$b=(a_{\max}+a_{\min})/2$。变量变换后的对数概率必须包含雅可比修正：

$$
\log\pi_\phi(a\mid s)
=
\sum_j
\left[
\log\mathcal N\!\left(u_j;\mu_{\phi,j}(s),\sigma_{\phi,j}^2(s)\right)
-\log|c_j|
-\log\!\left(1-\tanh^2(u_j)\right)
\right].
$$

实现时在最后一项内加入很小的常数以避免 $\log 0$。代入软策略改进目标，并用双 Q 的较小值评价新动作，Actor 损失为

$$
\boxed{
L_\pi(\phi)
=
\mathbb{E}_{\substack{s\sim\mathcal B\\
\varepsilon\sim\mathcal N(0,I)}}
\left[
\alpha\log\pi_\phi(f_\phi(\varepsilon;s)\mid s)
-\min_{i=1,2}Q_{\theta_i}(s,f_\phi(\varepsilon;s))
\right]
}.
$$

第一项通过最小化 $\log\pi$ 提高熵，第二项通过最小化 $-Q$ 提高动作价值。重参数化使梯度可以沿“动作 $\rightarrow$ Q 值”和“动作 $\rightarrow$ 对数概率”两条路径传回 Actor。

#### 3.3 温度系数自动调节

固定 $\alpha$ 需要手工平衡奖励与熵。现代实现通常给定目标熵 $\mathcal H_{\mathrm{target}}$，连续动作任务常用经验设置

$$
\mathcal H_{\mathrm{target}}=-\dim(\mathcal A).
$$

令 $\beta=\log\alpha$、$\alpha=e^\beta>0$。与当前代码完全对应的温度损失为

$$
\boxed{
L_\alpha(\beta)
=
\mathbb{E}_{\substack{s\sim\mathcal B\\a\sim\pi_\phi}}
\left[
-e^\beta
\left(
\log\pi_\phi(a\mid s)
+\mathcal H_{\mathrm{target}}
\right)
\right]
}.
$$

更新 $\alpha$ 时将 $\log\pi_\phi$ 视为常数，只更新 $\beta$。当策略熵低于目标水平时，梯度推动 $\alpha$ 增大，从而加强熵项；反之则减小 $\alpha$。自动温度调节是后续现代 SAC 的常用扩展，不是 v1 论文最初算法结构中的必需部分。

#### 3.4 目标网络软更新

每次梯度更新后，对两个目标 Q 网络执行 Polyak 平均：

$$
\bar\theta_i
\leftarrow
\tau\theta_i+(1-\tau)\bar\theta_i,
\qquad i\in\{1,2\},
$$

其中 $\tau\ll1$。缓慢变化的目标网络可以降低贝尔曼自举目标随在线 Critic 剧烈振荡的风险。

### 4. 现代 SAC 的完整训练流程

1. 初始化随机策略 $\pi_\phi$、两个 Critic $Q_{\theta_1},Q_{\theta_2}$ 和经验回放池 $\mathcal B$。
2. 将在线 Critic 参数复制到目标网络：$\bar\theta_i\leftarrow\theta_i$；初始化 $\beta=\log\alpha$。
3. 与环境交互。训练时从策略分布采样动作，将 $(s,a,r,s',d)$ 存入 $\mathcal B$；评估时可使用变换后的均值动作。
4. 从 $\mathcal B$ 随机采样小批量数据，在每个 $s'$ 上由当前 Actor 采样 $a'$ 并计算修正后的 $\log\pi_\phi(a'\mid s')$。
5. 使用目标双 Q 的最小值和熵项构造停止梯度的软贝尔曼目标 $y_Q$，更新两个在线 Critic。
6. 在批次状态 $s$ 上重新采样可微动作，最小化 $L_\pi$ 更新 Actor。
7. 最小化 $L_\alpha$ 更新温度系数，使策略熵趋近目标熵。
8. 用 Polyak 平均更新两个目标 Q 网络，并继续环境交互与训练。

## 4. 实现与应用（Implementation & Applications）

### 5. 公式与下方代码的对应关系

| 数学量或步骤 | 代码中的对应实现 | 作用 |
| --- | --- | --- |
| $\mu_\phi(s),\log\sigma_\phi(s)$ | `Actor.forward` | 参数化对角高斯策略 |
| $u=\mu+\sigma\odot\varepsilon$ | `Normal(...).rsample()` | 重参数化采样 |
| $a=c\odot\tanh(u)+b$ | `action_scale`、`action_bias` | 将动作映射到环境范围 |
| $\log\pi_\phi(a\mid s)$ | `normal.log_prob` 与 tanh/缩放修正 | 为软价值、Actor 和温度目标提供熵项 |
| $Q_{\theta_1},Q_{\theta_2}$ | `qf1`、`qf2` | 双 Critic 估计动作价值 |
| $Q_{\bar\theta_1},Q_{\bar\theta_2}$ | `qf1_target`、`qf2_target` | 构造稳定的自举目标 |
| $y_Q$ | `min_qf_next_target`、`next_q_value` | 现代 SAC 的软贝尔曼目标 |
| $L_Q$ | `qf_loss` | 拟合两个 Critic |
| $L_\pi$ | `actor_loss` | 同时提高软 Q 值和策略熵 |
| $L_\alpha$ | `alpha_loss`、`log_alpha` | 自动调节熵权重 |
| $\mathcal B$ | `ReplayBuffer` | 复用离策略经验 |
| $\bar\theta_i\leftarrow\tau\theta_i+(1-\tau)\bar\theta_i$ | 两个目标网络的参数复制循环 | 平滑更新目标 Critic |

下方代码没有独立的状态价值网络，这与现代 SAC 公式一致；SAC v1 中的 $V_\psi$ 已被“目标双 Q 的最小值减去熵项”显式替代。

---

### 6. 核心代码架构实现

以下代码剔除了辅助功能，高度抽象了 SAC 的核心算法逻辑。完整保留了重参数化技巧、对数概率的变量代换修正以及温度系数自动调节机制，完全符合 Python 与 PyTorch 规范。

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =======================================
# 1. 网络架构定义
# =======================================

class SoftQNetwork(nn.Module):
    """Critic 网络：评估状态-动作对的软 Q 值"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    """Actor 网络：输出动作的高斯分布参数并进行重参数化采样"""
    def __init__(self, obs_dim, action_dim, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))
        
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        """步骤2：动作生成与对数概率计算"""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # 重参数化采样
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # 变量代换公式修正对数概率 (Enforcing Action Bound)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

# =======================================
# 2. 核心算法类与训练逻辑
# =======================================

class SACAgent:
    def __init__(self, obs_dim, action_dim, action_low, action_high, device):
        self.device = device
        
        # 初始化 Actor 和两个 Critic
        self.actor = Actor(obs_dim, action_dim, action_low, action_high).to(device)
        self.qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
        self.qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
        
        # 初始化 Target Critic
        self.qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
        self.qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        # 优化器
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=1e-3)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=3e-4)
        
        # 自动调整温度系数 Alpha 
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=1e-3)
        
        # 超参数
        self.gamma = 0.99
        self.tau = 0.005

    def get_action(self, obs, deterministic=False):
        """与环境交互时获取动作"""
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action, _, mean_action = self.actor.get_action(obs_tensor)
            if deterministic:
                return mean_action.cpu().numpy()[0]
            return action.cpu().numpy()[0]

    def train_step(self, replay_buffer, batch_size):
        """执行一个批次的完整训练更新"""
        observations, actions, rewards, next_observations, dones = replay_buffer.sample(batch_size)
        
        # -----------------------------
        # 步骤 3 & 4: 计算并更新 Q 网络
        # -----------------------------
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_observations)
            qf1_next_target = self.qf1_target(next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(next_observations, next_state_actions)
            
            # 软目标 Q 值计算
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + (1 - dones) * self.gamma * min_qf_next_target.view(-1)

        qf1_a_values = self.qf1(observations, actions).view(-1)
        qf2_a_values = self.qf2(observations, actions).view(-1)
        qf_loss = F.mse_loss(qf1_a_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # -----------------------------
        # 步骤 5: 计算并更新 Actor 网络
        # -----------------------------
        pi, log_pi, _ = self.actor.get_action(observations)
        qf1_pi = self.qf1(observations, pi)
        qf2_pi = self.qf2(observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # 最大化 Q 且最大化熵
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -----------------------------
        # 步骤 6: 自动调节温度系数 Alpha
        # -----------------------------
        with torch.no_grad():
            _, log_pi, _ = self.actor.get_action(observations)
            
        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
        
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # -----------------------------
        # 步骤 7: Target 网络软更新
        # -----------------------------
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# =======================================
# 3. 主循环抽象调用演示
# =======================================
def main_training_loop(env, agent, replay_buffer, max_steps, batch_size, learning_starts):
    obs = env.reset()
    for global_step in range(max_steps):
        
        # 预填充回放池或正常采样
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs)
            
        next_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        
        if done:
            obs = env.reset()

        # 满足条件后开始训练
        if global_step > learning_starts:
            agent.train_step(replay_buffer, batch_size)
```

## 5. 分析与总结（Analysis & Summary）

> 待补充。

## 6. 关联与参考（Related Topics & References）

> 待补充。
