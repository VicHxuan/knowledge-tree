# SAC 算法完整解析与代码实现

## 1. 概述

SAC 在累计奖励目标中加入策略熵，用随机 Actor 与双 Critic 学习连续控制策略。Actor 同时优化动作价值与熵，Critic 拟合包含未来熵的软价值目标；经验回放支持离策略学习。

## 2. 基础与定义

### 2.1 网络结构

采用一个随机策略网络 $\pi_\theta$、两个动作价值网络 $Q_{\phi_1},Q_{\phi_2}$ 以及两个目标价值网络 $Q_{\bar\phi_1},Q_{\bar\phi_2}$。Actor 输出高斯均值与标准差，经动作变换得到有界随机动作。温度 $\alpha>0$ 控制熵的权重。

### 2.2 最大熵目标

策略在状态 $s$ 下的熵为 $\mathcal H(\pi(\cdot\mid s))=-\mathbb E_{a\sim\pi}[\log\pi(a\mid s)]$。最大熵目标为

$$
J_{\mathrm{soft}}(\pi)=\mathbb E_{\tau\sim\pi}\!\left[
\sum_{t=0}^{\infty}\gamma^t\left(r_{t+1}-\alpha\log\pi(a_t\mid s_t)\right)\right].
$$

经验池 $\mathcal B$ 存储 $(s,a,r,s',d)$，$d$ 表示真正终止。外部截断时保留末状态价值，自举使用重置前的 $s'$。

## 3. 核心原理

### 3.1 损失与梯度

令 $Q_{\min}(s,a)=\min_iQ_{\phi_i}(s,a)$。双 Critic 和 Actor 分别最小化

$$
L_{Q_i}(\phi_i)=\mathbb E_{\mathcal B}[(Q_{\phi_i}(s,a)-y)^2],
$$

$$
L_\pi(\theta)=\mathbb E_{s\sim\mathcal B,\,\epsilon}\!\left[
\alpha\log\pi_\theta(f_\theta(\epsilon;s)\mid s)-Q_{\min}(s,f_\theta(\epsilon;s))\right].
$$

这里 $a=f_\theta(\epsilon;s)$ 是可微的策略采样，$y$ 是固定的软价值目标。Critic 梯度为

$$
\nabla_{\phi_i}L_{Q_i}=2\mathbb E_{\mathcal B}[(Q_{\phi_i}-y)\nabla_{\phi_i}Q_{\phi_i}].
$$

Actor 更新固定 Critic 参数和温度，通过动作及其对数概率求导：

$$
\nabla_\theta L_\pi
=\mathbb E\!\left[
\alpha\frac{d}{d\theta}\log\pi_\theta(f_\theta(\epsilon;s)\mid s)
-(\nabla_\theta f_\theta)^\top\nabla_a Q_{\min}(s,a)\big|_{a=f_\theta(\epsilon;s)}\right].
$$

总导数包含策略密度的显式参数依赖与采样动作的参数依赖。$Q_{\min}$ 的动作梯度来自当前取值较小的 Critic；相等处使用次梯度。

<details class="kt-derivation" id="sac-policy-improvement">
<summary>软策略改进与 Actor 损失</summary>

对固定的 $Q$，软策略改进将策略投影到玻尔兹曼分布：

$$
\pi_{\mathrm{new}}
=\arg\min_{\pi'}D_{\mathrm{KL}}\!\left(
\pi'(\cdot\mid s)\,\middle\|\,\frac{\exp(Q(s,\cdot)/\alpha)}{Z(s)}\right).
$$

展开并乘以 $\alpha$：

$$
\alpha D_{\mathrm{KL}}
=\mathbb E_{a\sim\pi'}[\alpha\log\pi'(a\mid s)-Q(s,a)]+\alpha\log Z(s).
$$

$Z(s)$ 与候选策略参数无关，因此最小化 KL 等价于最小化熵正则化 Actor 损失。有限参数网络通过梯度近似完成这一投影，双 Q 最小值为实际更新提供保守的价值估计。

</details>

### 3.2 软价值目标

软状态价值与动作价值满足

$$
V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)-\alpha\log\pi(a\mid s)],
\qquad
Q^\pi(s,a)=\mathbb E[r+\gamma(1-d)V^\pi(s')].
$$

在下一状态用当前 Actor 采样 $a'\sim\pi_\theta(\cdot\mid s')$，以目标双 Q 近似下一状态价值：

$$
y=r+\gamma(1-d)\left[\min_iQ_{\bar\phi_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right].
$$

$y$ 中的全部量停止梯度；在线 Critic 拟合当前动作价值。两个目标网络通过滑动平均更新：

$$
\bar\phi_i\leftarrow\tau\phi_i+(1-\tau)\bar\phi_i.
$$

<details class="kt-derivation" id="sac-value-variants">
<summary>状态价值网络与直接自举</summary>

早期 SAC 显式学习状态价值网络 $V_\psi$，目标为

$$
y_V(s)=\mathbb E_{a\sim\pi_\theta}[\min_iQ_{\phi_i}(s,a)-\alpha\log\pi_\theta(a\mid s)],
\qquad
L_V(\psi)=\frac12\mathbb E[(V_\psi-y_V)^2].
$$

对应的动作价值目标为 $y_Q=r+\gamma(1-d)V_{\bar\psi}(s')$。后续 SAC 直接从当前策略采样下一动作，用目标双 Q 与熵项估计 $V(s')$，得到正文的 $y$。两种结构对应同一软贝尔曼关系，区别在于状态价值期望的近似方式。

</details>

### 3.3 重参数化采样

将随机性写为独立标准高斯噪声：

$$
\epsilon\sim\mathcal N(0,I),\quad
u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,\quad
 a=f_\theta(\epsilon;s)=c\odot\tanh(u)+b,
$$

其中 $c=(a_{\max}-a_{\min})/2>0$，$b=(a_{\max}+a_{\min})/2$。动作密度包含变量变换的雅可比：

$$
\log\pi_\theta(a\mid s)=\sum_j\!\left[
\log\mathcal N(u_j;\mu_{\theta,j},\sigma_{\theta,j}^2)
-\log c_j-\log(1-\tanh^2u_j)\right].
$$

该密度同时用于 Actor、软价值目标与温度更新。

<details class="kt-derivation" id="sac-action-density">
<summary>动作密度与稳定计算</summary>

对每个动作维度，$da_j/du_j=c_j(1-\tanh^2u_j)$。由变量变换公式，$\log\pi(a)=\log p(u)-\log|\det(\partial a/\partial u)|$，得到正文的修正项。

在 $|u|$ 很大时，直接计算 $1-\tanh^2u$ 容易出现数值消减。可使用等价形式：

$$
\log(1-\tanh^2u)=2\left[\log2-u-\operatorname{softplus}(-2u)\right].
$$

动作缩放还会改变微分熵：$\mathcal H(a)=\mathcal H(\tanh u)+\sum_j\log c_j$。目标熵与对数概率需使用同一动作单位。

</details>

### 3.4 温度调节

给定目标熵 $\mathcal H_{\mathrm{target}}$，令 $\beta=\log\alpha$。温度损失与梯度为

$$
L_\alpha(\beta)
=\mathbb E[-e^\beta(\log\pi_\theta(a\mid s)+\mathcal H_{\mathrm{target}})],
$$

$$
\frac{\partial L_\alpha}{\partial\beta}
=\alpha\left(\mathcal H(\pi)-\mathcal H_{\mathrm{target}}\right).
$$

更新时固定策略样本及其对数概率。熵低于目标时，梯度下降提高 $\beta$ 和 $\alpha$，增加 Actor 中的熵权重；熵高于目标时则降低该权重。

<details class="kt-derivation" id="sac-temperature-gradient">
<summary>温度参数化与梯度尺度</summary>

直接优化 $\alpha$ 的损失为 $L(\alpha)=\alpha(\mathcal H-\mathcal H_{\mathrm{target}})$。以 $\alpha=e^\beta$ 参数化后，链式法则给出额外因子 $\alpha$。

部分实现采用 $\widetilde L(\beta)=-\beta\,\mathbb E[\log\pi+\mathcal H_{\mathrm{target}}]$，其梯度为 $\mathcal H-\mathcal H_{\mathrm{target}}$。两种形式有相同的平衡条件与更新方向，梯度尺度相差 $\alpha$；给定学习率时，温度调整轨迹随之改变。

连续动作的微分熵可为负。常用 $\mathcal H_{\mathrm{target}}=-\dim(\mathcal A)$ 是经验设定，需与动作尺度约定一起理解。

</details>

## 4. 实现与应用

### 4.1 训练流程

1. 用随机 Actor 采集转移并存入经验池。
2. 从经验池采样，用下一状态动作、目标双 Q 与熵构造固定目标 $y$。
3. 更新两个在线 Critic。
4. 在批次状态上重新采样可微动作，更新 Actor。
5. 用停止梯度的对数概率更新温度，再滑动更新目标 Critic。

### 4.2 实现示例

<details class="kt-derivation" id="sac-pytorch-update">
<summary>SAC 的 PyTorch 实现</summary>

经验池 `sample` 返回同设备浮点张量：状态 `[B, obs_dim]`、动作 `[B, action_dim]`、奖励和终止标志 `[B]`。`dones` 只存真正终止标志；动作上下界为有限的一维 NumPy 数组。

主循环使用 Gymnasium 单环境接口并显式重置。例中温度损失采用 $e^\beta$ 形式，动作概率使用环境动作单位；Actor 更新时固定两个 Critic 的参数，保留其动作梯度。

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

        # tanh 与动作尺度的雅可比修正
        log_prob = normal.log_prob(x_t)
        log_tanh_jacobian = 2 * (np.log(2.0) - x_t - F.softplus(-2 * x_t))
        log_prob -= self.action_scale.log() + log_tanh_jacobian
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
        self.qf1.requires_grad_(False)
        self.qf2.requires_grad_(False)
        pi, log_pi, _ = self.actor.get_action(observations)
        qf1_pi = self.qf1(observations, pi)
        qf2_pi = self.qf2(observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # 最大化 Q 且最大化熵
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.qf1.requires_grad_(True)
        self.qf2.requires_grad_(True)

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
        with torch.no_grad():
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.lerp_(param, self.tau)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.lerp_(param, self.tau)


# =======================================
# 3. 主循环抽象调用演示
# =======================================
def main_training_loop(env, agent, replay_buffer, max_steps, batch_size, learning_starts):
    obs, _ = env.reset()
    for global_step in range(max_steps):

        # 预填充回放池或正常采样
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()

        # 满足条件后开始训练
        if global_step > learning_starts:
            agent.train_step(replay_buffer, batch_size)
```

</details>

## 5. 分析与总结

### 5.1 梯度路径比较

| 方法 | Actor 的直接训练信号 | Critic 的目标 |
| --- | --- | --- |
| PPO | 固定优势加权的对数概率梯度 | 轨迹构造的回报 |
| TD3 | 第一个 Q 对确定性动作的导数 | 平滑目标动作上的双 Q 最小值 |
| SAC | 熵项与较小 Q 对采样动作的导数 | 下一动作的双 Q 最小值与熵项 |

SAC 的双 Critic 共同估计同一软回报。奖励分解式 Multi-Critic 则为不同回报建立独立估计目标，聚合方式需由任务目标确定。

### 5.2 核心结论

SAC 的损失来自最大熵目标：软价值备份将未来熵计入 Critic，软策略改进将当前熵计入 Actor，温度调节控制两者的权衡。重参数化把这些目标连接到随机策略的参数梯度。

## 6. 关联与参考

### 6.1 相关知识

- [强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)：贝尔曼方程与策略梯度。
- [DDPG](./04_ddpg.md)、[TD3](./05_td3.md)：确定性策略梯度与双 Q 目标。
- [PPO](./03_ppo.md)：轨迹优势驱动的随机策略更新。
- [多价值网络与价值估计](../01_rl-foundations/05_multi-critic.md)：多个价值估计对损失与梯度的影响。
- MBPO：以模型生成的短轨迹补充经验池，相关条目仍在撰写中。

### 6.2 参考资料

1. Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html), ICML 2018。
2. Haarnoja et al., [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), 2018。
3. OpenAI Spinning Up, [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)。
4. CleanRL, [SAC 实现](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)。
